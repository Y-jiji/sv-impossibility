import numpy as np
import scipy.spatial as spatial
import torch as t

def dist(x1: t.Tensor, x2: t.Tensor) -> t.Tensor:
    return t.einsum("ij, ij -> i", x1, x1).unsqueeze(-1) + t.einsum("ij, ij -> i", x2, x2) - 2 * t.einsum("ij, kj -> ik", x1, x2)

def knn_shapley(K: int, input_tra: t.Tensor, label_tra: t.Tensor, input_val: t.Tensor, label_val: t.Tensor):
    """
    R. Jia's algorithm for KNN Shapley Values. 
    Original KNN-Shapley is proposed in http://www.vldb.org/pvldb/vol12/p1610-jia.pdf. 
    INPUT: 
        K: K-nearest neighbours K
        input_tra: training dataset input, shape [N, D]
        label_tra: training dataset label, shape [N]
        input_val: validation dataset input, shape [M, D]
        label_val: validation dataset label, shape [M]
    OUTPUT: 
        Shapley Values for each training datapoint. 
    """
    N, D = input_tra.shape
    M, D = input_val.shape
    # sort values by distance to valuation datum for each valutation datum
    a_sort = dist(input_val, input_tra).argsort(1)
    arange = t.arange(0, M).reshape(M, 1)
    # eqtest[i, j] = label_val[i] == label_tra[idsort[i, j]]
    # sv[i, N-1] = eqtest[i, N-1] / N
    # sv[i, j] = sv[i, j+1] + (eqtest[i, j] - eqtest[i, j+1]) / max(K, j+1)
    eqtest = 1.0 * (label_val.reshape(M, 1) == label_tra)[arange, a_sort]
    eqdiff = t.zeros_like(eqtest)
    eqdiff[:,  N-1] = eqtest[:, N-1] / N
    eqdiff[:, :N-1] = (eqtest[:, :N-1] - eqtest[:, 1:]) / t.maximum(t.tensor(K), t.arange(1, N))
    sv = t.flip(t.flip(eqdiff, (1,)).cumsum(1), (1,))
    # output[i, idsort[i, j]] = sv[i, j]
    output = t.zeros_like(sv)
    output[arange, a_sort] = sv
    return output.sum(0)

def knn_predict(K: int, input_tra: t.Tensor, label_tra: t.Tensor, input_val: t.Tensor):
    """
    predict validation labels for each data point. 
    INPUT:
        K: K-nearest neighbours K
        input_tra: training dataset input, shape [N, D]
        label_tra: training dataset label, shape [N]
        input_val: validation dataset input, shape [M, D]
    """
    N, D = input_tra.shape
    M, D = input_val.shape
    a_sort = dist(input_val, input_tra).argsort(1)
    labels = label_tra[a_sort[:, :K]]
    counts = (labels.reshape(*labels.shape, 1) == t.arange(0, labels.max()+1)).sum(1)
    return counts.argmax(1)

def knn_alter_validation(
        K: int, S: int,
        input_tra: t.Tensor, label_tra: t.Tensor, 
        input_val: t.Tensor, label_val: t.Tensor
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
    from tqdm import tqdm
    N, D = input_tra.shape
    M, D = input_val.shape
    L = int(label_val.max())+1
    assert int(label_val.min()) == 0
    # sv for each training sample
    score_tra = knn_shapley(K, input_tra, label_tra, input_val, label_val)
    # selected indices
    index_sel = sorted(score_tra.argsort(0)[N-S:N].tolist())
    print("initial gap: ", score_tra[index_sel].min().item() - score_tra[score_tra.argsort(0)[:N-S]].max().item())
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
    argsort = dist(input_val, input_tra).argsort(1)
    sv_list = pulp.LpVariable.dict("sv-list", (range(N), range(M)))
    for j in tqdm(range(M), "write constraints"):
        sv_list[argsort[j, N-1].item(), j] = (label_var[j, label_tra[argsort[j, N-1]].item()] * (1 / N))
        for i in range(N-2, -1, -1):
            sv_list[argsort[j, i].item(), j] = (sv_list[argsort[j, i+1].item(), j]
                + (label_var[j, label_tra[argsort[j, i].item()].item()]    * (1/max(i+1, K)))
                - (label_var[j, label_tra[argsort[j, i+1].item()].item()]  * (1/max(i+1, K))))
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
    lp += (pulp.lpSum([label_var[j, label_heu[j].item()] for j in range(M)]), "valuation-performance")
    print("initial gap: ", pulp.value(a) - pulp.value(b), "(sanity check)")
    # solve linear programming, and create results
    print('trying to find the best relabeling (20 secs)')
    lp.solve(pulp.getSolver("COIN_CMD", msg=False, warmStart=True, timeLimit=20))
    print("solver status: ", pulp.LpStatus[lp.status])
    label_new = t.tensor([[pulp.value(label_var[i, j]) for j in range(L)] for i in range(M)]).argmax(1)
    index_new = sorted(list(knn_shapley(K, input_tra, label_tra, input_val, label_new).argsort(0)[N-S:N]))
    assert index_new == index_sel, f"{index_new}\n{index_sel}"
    return input_val, label_new

def experiment_1_SYNTH(N: int, M: int, K: int, S: int):
    drift = t.tensor([0.0, 0.0])
    input_tra = t.randn(N, 2).to(t.float64) + drift.to(t.float64)
    label_tra = 1 * (input_tra[:, 0] > -input_tra[:, 1])
    input_val = t.randn(M, 2).to(t.float64) + drift.to(t.float64)
    label_val = 1 * (input_val[:, 0] > -input_val[:, 1])
    input_new, label_new = knn_alter_validation(K, S, input_tra, label_tra, input_val, label_val)
    print("label similarity", (label_new == label_val).sum() / M)
    s0 = knn_shapley(K, input_tra, label_tra, input_val, label_val).argsort(0)[N-S:]
    s1 = knn_shapley(K, input_tra, label_tra, input_new, label_new).argsort(0)[N-S:]
    r00 = ((knn_predict(K, input_tra, label_tra, input_val) == label_val) * 1).sum() / M
    r01 = ((knn_predict(K, input_tra, label_tra, input_new) == label_new) * 1).sum() / M
    print("original accuracy", f'{r00:.04}', f'{r01:.04}')
    r10 = ((knn_predict(K, input_tra[s0], label_tra[s0], input_val) == label_val) * 1).sum() / M
    r11 = ((knn_predict(K, input_tra[s1], label_tra[s1], input_new) == label_new) * 1).sum() / M
    print("relative accuracy", f'{r10 / r00:.04}', f'{r11 / r01:.04}')

def experiment_1_CIFAR(K: int, S: int):
    """
    Perform validation data alternating on MNIST dataset. 
    -- Split dataset into training dataset and validation dataset. 
    -- Extract features for each MNIST datapoint. 
    INPUT: 
        K: K-nearest neighbours K
        S: the selected dataset size
    OUTPUT: 
        PCA visualization of all photos
    """
    import torchvision
    import matplotlib.pyplot as plt
    from skimage.feature import hog
    from sklearn.decomposition import PCA
    from random import shuffle
    import tqdm
    dataset = (
        torchvision.datasets.CIFAR10("_data", download=True, train=True) +
        torchvision.datasets.CIFAR10("_data", download=True, train=False)
    )
    def load(index: t.tensor):
        xs, ys = [], []
        for i in tqdm.tqdm(index, "preprocessing"):
            x, y = dataset[i]
            xs.append(hog(np.array(x) / 255, channel_axis=2))
            ys.append(y)
        return t.tensor(np.array(xs)), t.tensor(ys)
    A = len(dataset)
    B = 2000
    N = (B*3)//5
    M = (B*1)//5
    T = (B*1)//5
    index_all = list(range(A))
    shuffle(index_all)
    index_all = index_all[:B]
    index_tra = index_all[0:N]
    index_val = index_all[N:N+M]
    index_tes = index_all[N+M:N+M+T]
    # train, validate and test on original dataset
    input_tra, label_tra = load(index_tra)
    input_val, label_val = load(index_val)
    input_tes, label_tes = load(index_tes)
    sv = knn_shapley(K, input_tra, label_tra, input_val, label_val)
    select = sv.argsort(0)[N-S:]
    print('==')
    print('original dataset')
    print('acc:', (label_tes == knn_predict(K, input_tra, label_tra, input_tes)).sum().item() / T, '(all data)')
    print('acc:', (label_tes == knn_predict(K, input_tra[select], label_tra[select], input_tes)).sum().item() / T, '(after selection)')
    # construct label-altered validation set and testing
    print('==')
    print('altered dataset')
    input_val, label_val = knn_alter_validation(K, S, input_tra, label_tra, input_val, label_val)
    sv = knn_shapley(K, input_tra, label_tra, input_val, label_val)
    select = sv.argsort(0)[N-S:]
    label_tes = knn_predict(K, input_val, label_val, input_tes)
    print('acc:', (label_tes == knn_predict(K, input_tra, label_tra, input_tes)).sum().item() / T, '(all data)')
    print('acc:', (label_tes == knn_predict(K, input_tra[select], label_tra[select], input_tes)).sum().item() / T, '(after selection)')

if __name__ == '__main__':
    # experiment_1_SYNTH(1000, 100, 5, 100)
    experiment_1_CIFAR(5, 100)