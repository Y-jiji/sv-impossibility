import numpy as np
import scipy.spatial as spatial

def knn_shapley(k: int, input_tra: np.array, label_tra: np.array, input_val: np.array, label_val: np.array):
    """
    R. Jia's algorithm for KNN Shapley Values. 
    Original KNN-Shapley proposed in http://www.vldb.org/pvldb/vol12/p1610-jia.pdf. 
    INPUT: 
        k: k-nearest neighbours k
        input_tra: training dataset input, shape [N, D]
        label_tra: training dataset label, shape [N]
        input_val: validation dataset input, shape [M, D]
        label_val: validation dataset label, shape [M]
    OUTPUT: 
        Shapley Values for each training datapoint. 
    """
    N, _ = input_tra.shape
    M, _ = input_val.shape
    # sort values by distance to valuation datum for each valutation datum
    a_sort = spatial.distance_matrix(input_val, input_tra).argsort(1)
    arange = np.arange(0, M).reshape(M, 1)
    # eqtest[i, j] = label_val[i] == label_tra[idsort[i, j]]
    # sv[i, N-1] = eqtest[i, N-1] / N
    # sv[i, j] = sv[i, j+1] + (eqtest[i, j] - eqtest[i, j+1]) / max(K, j+1)
    eqtest = 1.0 * (label_val.reshape(M, 1) == label_tra)[arange, a_sort]
    eqdiff = np.zeros_like(eqtest)
    eqdiff[:,  N-1] = eqtest[:, N-1] / N 
    eqdiff[:, :N-1] = (eqtest[:, :N-1] - eqtest[:, 1:]) / np.maximum(k, np.arange(1, N))
    sv = np.flip(np.flip(eqdiff, 1).cumsum(1), 1)
    # output[i, idsort[i, j]] = sv[i, j]
    output = np.zeros_like(sv)
    output[arange, a_sort] = sv
    return output.sum(0)

def knn_predict(k: int, input_tra: np.array, label_tra: np.array, input_val: np.array):
    """
    predict validation labels for each data point. 
    INPUT:
        k: k-nearest neighbours k
        input_tra: training dataset input, shape [N, D]
        label_tra: training dataset label, shape [N]
        input_val: validation dataset input, shape [M, D]
    """
    a_sort = spatial.distance_matrix(input_val, input_tra).argsort(1)
    labels = label_tra[a_sort[:, :k]]
    counts = (labels.reshape(*labels.shape, 1) == np.arange(0, labels.max()+1)).sum(1)
    return counts.argmax(1)

def knn_alter_validation(
        k: int, size_subset: int,
        input_tra: np.array, label_tra: np.array, 
        input_val: np.array, label_val: np.array
    ):
    """
    Change validation set for a given k-nearest neighbour model s.t. :  
    -- In terms of Shapely values, the same training subset still contains elements with largest Shapley values. 
    -- Valuation results change. 
    Current implementation is given by relabeling. 
    The construction heuristic theoretically guarantees if there exists a wider 'gap' of Shapley values between selected elements and unselected elements, then there exists a relabeling such that validation method is different. 
    INPUT: 
        k: k-nearest neighbours k
        size_subset: the selected dataset size
        input_tra: training dataset input, shape [N, D]
        label_tra: training dataset label, shape [N]
        input_val: validation dataset input, shape [M, D]
        label_val: validation dataset label, shape [M]
    OUTPUT: 
        a different validation data set, with input and label, satisfying desired properties. 
    """
    import pulp # linear programming package
    M = label_val.shape[0]
    N = label_tra.shape[0]
    L = int(label_val.max())+1
    assert int(label_val.min()) == 0
    # sv for each training sample
    score_tra = knn_shapley(k, input_tra, label_tra, input_val, label_val)
    # selected indices
    index_sel = sorted(list(score_tra.argsort(0)[N-size_subset:N]))
    print("initial gap: ", score_tra[index_sel].min() - score_tra[score_tra.argsort(0)[:N-size_subset]].max())
    # for selected indices, relabel inputs with predicted labels w.r.t. selected index
    lp = pulp.LpProblem("label-construction", pulp.LpMaximize)
    label_var = pulp.LpVariable.dict("label-category", (range(M), range(L)), cat=pulp.LpBinary)
    for i in range(M):
        for j in range(L):
            if label_val[i] == j:
                label_var[i, j].setInitialValue(1)
            else:
                label_var[i, j].setInitialValue(0)
    for i in range(M):
        lp += pulp.lpSum([label_var[i, j] for j in range(L)]) == 1
    # shapley values computed for each validation instance, and each value of label
    # aggregate shapley values to each training instance
    argsort = spatial.distance_matrix(input_val, input_tra).argsort(1)
    sv_list = pulp.LpVariable.dict("sv-list", (range(N), range(M)))
    for j in range(M):
        sv_list[argsort[j, N-1], j] = (label_var[j, label_tra[argsort[j, N-1]]] * (1 / N))
        for i in range(N-2, -1, -1):
            sv_list[argsort[j, i], j] = (sv_list[argsort[j, i+1], j]
                + (label_var[j, label_tra[argsort[j, i]]]    * (1/max(i+1, k)))
                - (label_var[j, label_tra[argsort[j, i+1]]]  * (1/max(i+1, k))))
    sv = pulp.LpVariable.dict("sv", range(N))
    for i in range(N):
        sv[i] = pulp.lpSum([sv_list[i, j] for j in range(M)])
    # split training instance to two halves, and create two variables: 
    # -- one for the minimal value from selected subset
    # -- another for the maximal value from its complement
    a = pulp.LpVariable("min-sv-selected")
    b = pulp.LpVariable("max-sv-unselected")
    a.setInitialValue(min([pulp.value(sv[i]) for i in index_sel]))
    for i in index_sel:
        lp += (a <= sv[i])
    b.setInitialValue(max([pulp.value(sv[i]) for i in set(range(N)).difference(index_sel)]))
    for i in set(range(N)).difference(index_sel):
        lp += (b >= sv[i])
    lp += (a >= b)
    lp += (a - b, "gap")
    print("initial gap: ", pulp.value(a) - pulp.value(b), "(sanity check)")
    # solve linear programming, and create results
    lp.solve(pulp.getSolver("COIN_CMD", msg=False, warmStart=True))
    print("final gap: ", pulp.value(a) - pulp.value(b), f"({pulp.LpStatus[lp.status]})")
    label_new = np.array([[pulp.value(label_var[i, j]) for j in range(L)] for i in range(M)]).argmax(1)
    index_new = sorted(list(knn_shapley(k, input_tra, label_tra, input_val, label_new).argsort(0)[N-size_subset:N]))
    assert index_new == index_sel
    return input_val, label_new

if __name__ == '__main__':
    N = 800
    M = 100
    S = 50
    drift = np.array([0.0, 0.0])
    data = {
        'input_tra': (input_tra := np.random.randn(N, 2) + drift),
        'label_tra': (label_tra := 1 * (input_tra[:, 0] > -input_tra[:, 1])),
        'input_val': (input_val := np.random.randn(M, 2) + drift),
        'label_val': (label_val := 1 * (input_val[:, 0] > -input_val[:, 1])),
    }
    label_new = knn_alter_validation(1, S, **data)