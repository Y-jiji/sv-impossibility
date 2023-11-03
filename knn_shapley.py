import numpy as np
import scipy.spatial as spatial

def knn_shapley(K: int, input_tra: np.array, label_tra: np.array, input_val: np.array, label_val: np.array):
    """
    R. Jia's algorithm for KNN Shapley Values. 
    Original KNN-Shapley proposed in http://www.vldb.org/pvldb/vol12/p1610-jia.pdf. 
    INPUT: 
        K: K-nearest neighbours K
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
    eqdiff[:, :N-1] = (eqtest[:, :N-1] - eqtest[:, 1:]) / np.maximum(K, np.arange(1, N))
    sv = np.flip(np.flip(eqdiff, 1).cumsum(1), 1)
    # output[i, idsort[i, j]] = sv[i, j]
    output = np.zeros_like(sv)
    output[arange, a_sort] = sv
    return output.sum(0)

def knn_predict(K: int, input_tra: np.array, label_tra: np.array, input_val: np.array):
    """
    predict validation labels for each data point. 
    INPUT:
        K: K-nearest neighbours K
        input_tra: training dataset input, shape [N, D]
        label_tra: training dataset label, shape [N]
        input_val: validation dataset input, shape [M, D]
    """
    a_sort = spatial.distance_matrix(input_val, input_tra).argsort(1)
    labels = label_tra[a_sort[:, :K]]
    counts = (labels.reshape(*labels.shape, 1) == np.arange(0, labels.max()+1)).sum(1)
    return counts.argmax(1)

def knn_alter_validation(
        K: int, S: int,
        input_tra: np.array, label_tra: np.array, 
        input_val: np.array, label_val: np.array
    ):
    """
    Change validation set for a given K-nearest neighbour model s.t. : 
    -- In terms of Shapely values, the same training subset still contains elements with largest Shapley values. 
    -- Valuation results change. 
    Current implementation is given by relabeling. 
    The construction heuristic guarantees the top-S SV elements doesn't change. 
    INPUT: 
        K: K-nearest neighbours K
        S: the selected dataset size
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
    score_tra = knn_shapley(K, input_tra, label_tra, input_val, label_val)
    # selected indices
    index_sel = sorted(list(score_tra.argsort(0)[N-S:N]))
    print("initial gap: ", score_tra[index_sel].min() - score_tra[score_tra.argsort(0)[:N-S]].max())
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
                + (label_var[j, label_tra[argsort[j, i]]]    * (1/max(i+1, K)))
                - (label_var[j, label_tra[argsort[j, i+1]]]  * (1/max(i+1, K))))
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
    # maximize the similarity between heuristic labeling and valid labeling, 
    #   i.e. maximize our data selection performance
    label_heu = knn_predict(K, input_tra[index_sel], label_tra[index_sel], input_val)
    lp += (pulp.lpSum([label_var[j, label_heu[j]] for j in range(M)]), "similarity")
    print("initial gap: ", pulp.value(a) - pulp.value(b), "(sanity check)")
    # solve linear programming, and create results
    lp.solve(pulp.getSolver("COIN_CMD", msg=False, warmStart=True))
    print("solver status: ", pulp.LpStatus[lp.status])
    label_new = np.array([[pulp.value(label_var[i, j]) for j in range(L)] for i in range(M)]).argmax(1)
    index_new = sorted(list(knn_shapley(K, input_tra, label_tra, input_val, label_new).argsort(0)[N-S:N]))
    assert index_new == index_sel
    return input_val, label_new

def expriment_1_MNIST(K: int, S: int):
    """
    Perform validation data alternating on MNIST dataset. 
    -- Split dataset into training dataset and validation dataset. 
    -- Extract features for each MNIST datapoint. 
    INPUT: 
        K: K-nearest neighbours K
        S: the selected dataset size
    OUTPUT: 
        TODO: what visualization?
    """
    pass

def experiment_1_CIFAR(K: int, S: int):
    pass

if __name__ == '__main__':
    N = 1000
    M = 100
    S = 100
    drift = np.array([0.0, 0.0])
    input_tra = np.random.randn(N, 2) + drift
    label_tra = 1 * (input_tra[:, 0] > -input_tra[:, 1])
    input_val = np.random.randn(M, 2) + drift
    label_val = 1 * (input_val[:, 0] > -input_val[:, 1])
    input_new, label_new = knn_alter_validation(1, S, input_tra, label_tra, input_val, label_val)
    print("label similarity", (label_new == label_val).sum() / M)
    s0 = knn_shapley(5, input_tra, label_tra, input_val, label_val).argsort(0)[N-S:]
    s1 = knn_shapley(5, input_tra, label_tra, input_new, label_new).argsort(0)[N-S:]
    r00 = ((knn_predict(5, input_tra, label_tra, input_val) == label_val) * 1).sum() / M
    r01 = ((knn_predict(5, input_tra, label_tra, input_new) == label_new) * 1).sum() / M
    print("original accuracy", f'{r00:.04}', f'{r01:.04}')
    r10 = ((knn_predict(5, input_tra[s0], label_tra[s0], input_val) == label_val) * 1).sum() / M
    r11 = ((knn_predict(5, input_tra[s1], label_tra[s1], input_new) == label_new) * 1).sum() / M
    print("relative accuracy", f'{r10 / r00:.04}', f'{r11 / r01:.04}')
