import torch as t
import warnings
warnings.filterwarnings('ignore')

def dist(x1: t.Tensor, x2: t.Tensor) -> t.Tensor:
    return t.einsum("ij, ij -> i", x1, x1).unsqueeze(-1) + t.einsum("ij, ij -> i", x2, x2) - 2 * t.einsum("ij, kj -> ik", x1, x2)

def knn_shapley(K: int, input_tra: t.Tensor, label_tra: t.Tensor, input_val: t.Tensor, label_val: t.Tensor):
    """
    R. Jia's algorithm for KNN Shapley values. 
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

def data_shapley(S: int, input_tra: t.Tensor, label_tra: t.Tensor, input_val: t.Tensor, label_val: t.Tensor):
    """
    A. Ghorbani's algorithm for Shapley values. 
    INPUT: 
        S: # of permutation samples
        input_tra: training dataset input, shape [N, D]
        label_tra: training dataset label, shape [N]
        input_val: validation dataset input, shape [M, D]
        label_val: validation dataset label, shape [M]
    OUTPUT:
        Shapley Values for each training datapoint. 
    """
    N, D = input_tra.shape
    perms = t.rand((S, N)).argsort(-1)
    print(perms.shape)
    pass

def knn_predict(K: int, input_tra: t.Tensor, label_tra: t.Tensor, input_val: t.Tensor):
    """
    predict validation labels for each data point. (with k-nn model)
    INPUT:
        K: K-nearest neighbours K
        input_tra: training dataset input, shape [N, D]
        label_tra: training dataset label, shape [N]
        input_val: validation dataset input, shape [M, D]
    """
    a_sort = dist(input_val, input_tra).argsort(1)
    labels = label_tra[a_sort[:, :K]]
    counts = (labels.reshape(*labels.shape, 1) == t.arange(0, labels.max()+1)).sum(1)
    return counts.argmax(1)

def reg_predict(C: int, input_tra: t.Tensor, label_tra: t.Tensor, input_val: t.Tensor):
    """
    predict validatiaon labels for each data point. (with regression model)
    INPUT: 
        C: C parameter (1/λ, where λ is the l2 penality)
        input_tra: training dataset input, shape [N, D]
        label_tra: training dataset label, shape [N]
        input_val: validation dataset input, shape [M, D]
    """
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=C)
    model.fit(input_tra, label_tra)
    output = model.predict(input_val)
    return t.tensor(output, device=input_tra.device)

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
    # for selected indices, relabel inputs with predicted labels w.r.t. selected index
    lp = pulp.LpProblem("label-construction", pulp.LpMaximize)
    label_var = pulp.LpVariable.dict("label-category", (range(M), range(L)), lowBound=0, upBound=1)
    # label_var[i][j] = 1: for i th instance, label is j
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
    # sv_list[i, j] : for the j th validation instance, shapley value for i th training instance is sv_list[i, j]. 
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
    argsort = score_tra.argsort(0)
    for i in range(N-1):
        lp += (sv[argsort[i].item()] <= sv[argsort[i+1].item()])
    # maximize the similarity between heuristic labeling and valid labeling, 
    #   i.e. maximize our data selection performance
    label_heu = knn_predict(K, input_tra[index_sel], label_tra[index_sel], input_val)
    lp += (pulp.lpSum([label_var[j, label_heu[j].item()] for j in range(M)]), "valuation-performance")
    # solve linear programming, and create results
    print('trying to find the best relabeling (20 secs)')
    lp.solve(pulp.getSolver("COIN_CMD", msg=False, warmStart=True))
    print("solver status: ", pulp.LpStatus[lp.status])
    label_new = t.tensor([[pulp.value(label_var[i, j]) for j in range(L)] for i in range(M)]).argmax(1)
    return input_val, label_new

def experiment_1_SYNTH(N: int, M: int, K: int, S: int):
    """
    Perform 
    """
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

def load_CIFAR(N: int, M: int, T: int):
    """
    Load CIFAR 10 data set and preprocessing. 
    INPUT: 
        N: training set size
        M: validation set size
        T: test set size
    OUTPUT: 
        (input_tra, label_tra), (input_val, label_val), (input_tes, label_tes)
        input_tra: training dataset input, shape [N, D]
        label_tra: training dataset label, shape [N]
        input_val: validation dataset input, shape [M, D]
        label_val: validation dataset label, shape [M]
        input_tes: test dataset input, shape [T, D]
        label_tes: test dataset label, shape [T]
    """
    import torch.utils.data as data
    import torchvision
    import torchvision.transforms as transforms
    from tqdm import tqdm
    from random import shuffle, seed
    seed(114514)
    dataset = (
        torchvision.datasets.CIFAR10("_data", download=True, train=True) +
        torchvision.datasets.CIFAR10("_data", download=True, train=False)
    )
    resnet = torchvision.models.resnet50().to('cuda:0')
    @t.no_grad()
    def load(index: t.tensor):
        tt = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.to('cuda:0'))
        ])
        subset = data.Subset(dataset, index)
        xs, ys = [], []
        collate = lambda xs: (t.stack([tt(x) for x, y in xs], 0), t.tensor([y for x, y in xs], device='cpu'))
        for x, y in tqdm(data.DataLoader(subset, 16, collate_fn=collate), 'preprocessing'):
            xs.append(resnet(x).to('cpu'))
            ys.append(y)
        return t.concat(xs), t.concat(ys)
    A = len(dataset)
    index_all = list(range(A))
    shuffle(index_all)
    index_all = index_all[:N+M+T]
    index_tra = index_all[0:N]
    index_val = index_all[N:N+M]
    index_tes = index_all[N+M:N+M+T]
    # train, validate and test on original dataset
    input_tra, label_tra = load(index_tra)
    input_val, label_val = load(index_val)
    input_tes, label_tes = load(index_tes)
    return (input_tra, label_tra), (input_val, label_val), (input_tes, label_tes)

def load_OPENML(N: int, M: int, T: int, FILE_NAME: str):
    """
    Load an openml dataset and split it. 
    INPUT: 
        N: training set size
        M: validation set size
        T: test set size
    OUTPUT: 
        (input_tra, label_tra), (input_val, label_val), (input_tes, label_tes)
        input_tra: training dataset input, shape [N, D]
        label_tra: training dataset label, shape [N]
        input_val: validation dataset input, shape [M, D]
        label_val: validation dataset label, shape [M]
        input_tes: test dataset input, shape [T, D]
        label_tes: test dataset label, shape [T]
    """
    import pickle
    from random import shuffle, seed
    seed(114514)
    with open(f"OpenML_datasets/{FILE_NAME}", "rb") as f:
        data_dict = pickle.load(f)
    x = t.tensor(data_dict['X_num'])
    y = t.tensor(data_dict['y'])
    @t.no_grad()
    def load(index: t.Tensor):
        return x[index], y[index]
    A = x.shape[0]
    if N+M+T > A: 
        N, M, T = A*N//(N+M+T), A*M//(N+M+T), A*T//(N+M+T)
    index_all = list(range(A))
    shuffle(index_all)
    index_all = index_all[:N+M+T]
    index_tra = index_all[0:N]
    index_val = index_all[N:N+M]
    index_tes = index_all[N+M:N+M+T]
    # train, validate and test on original dataset
    input_tra, label_tra = load(index_tra)
    input_val, label_val = load(index_val)
    input_tes, label_tes = load(index_tes)
    return (input_tra, label_tra), (input_val, label_val), (input_tes, label_tes)

def experiment_1(K: int, S: list[int], predict: object, dataset: tuple[tuple, tuple, tuple]):
    """
    Experiment 1 try to demonstrate when validation set changes, the data selection performance may change drastically even if Shapley value don't change. Therefore, Shapley value may not be a good indicator for data selection. 
    Perform validation data alternating on MNIST dataset. 
    -- Split dataset into training dataset and validation dataset. 
    -- Extract features for each MNIST datapoint. 
    INPUT: 
        K: K-nearest neighbours K
        S: the selected dataset size
        predict: function with format input_tra, label_tra, input_val -> label_val
    OUTPUT: 
        PCA visualization of all photos
    """
    from scipy.stats import spearmanr
    # train, validate and test on original dataset
    input_tra, label_tra = dataset[0]
    input_val, label_val = dataset[1]
    input_tes, label_tes = dataset[2]
    N = input_tra.shape[0]
    T = input_tes.shape[0]
    sv_0 = knn_shapley(K, input_tra, label_tra, input_val, label_val)
    select = sv_0.argsort(0)[N-S:]
    # use knn as proximal construction method
    print('==')
    print('original dataset')
    label_tes = predict(input_val, label_val, input_tes)
    print('acc:', (label_tes == predict(input_tra, label_tra, input_tes)).sum().item() / T, '(all data)')
    print('acc:', (label_tes == predict(input_tra[select], label_tra[select], input_tes)).sum().item() / T, '(after selection)')
    # construct label-altered validation set and testing
    print('==')
    print('altered dataset')
    input_val, label_val = knn_alter_validation(K, S, input_tra, label_tra, input_val, label_val)
    # label_val = predict(input_tra[select], label_tra[select], input_val)
    sv_1 = knn_shapley(K, input_tra, label_tra, input_val, label_val)
    print('spearman:', spearmanr(sv_0, sv_1))
    select = sv_1.argsort(0)[N-S:]
    label_tes = predict(input_val, label_val, input_tes)
    print('acc:', (label_tes == predict(input_tra, label_tra, input_tes)).sum().item() / T, '(all data)')
    print('acc:', (label_tes == predict(input_tra[select], label_tra[select], input_tes)).sum().item() / T, '(after selection)')

def experiment_2_CIFAR(K: int, S: int):
    """
    Experiment 2 is about failure patterns of Shapley values. 
    -- Add a few samples in training set and validation set. 
    -- These samples are corrupted. 
    -- If we only add samples to training set, then the labels don't change and have very low SVs. 
    -- If we also add samples to validation set, the corrupted samples are selected with very high SVs. 
    If we have a pair of traning set and a validation set. 
    INPUT:
        TODO
    OUTPUT:
        TODO
    """
    pass

if __name__ == '__main__':
    experiment_1(
        5, 100, 
        lambda x_tra, y_tra, x_val: reg_predict(1, x_tra, y_tra, x_val), 
        dataset=load_OPENML(800, 400, 400, "phoneme_1489.pkl")
    )