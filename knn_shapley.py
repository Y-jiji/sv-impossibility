import torch as t
import warnings
warnings.filterwarnings('ignore')

t.manual_seed(9999)

def dist(x1: t.Tensor, x2: t.Tensor) -> t.Tensor:
    return (
        t.einsum("...ij, ...ij -> ...i", x1, x1).unsqueeze(-1) 
        + t.einsum("...ij, ...ij -> ...i", x2, x2).unsqueeze(-2)
        - 2 * t.einsum("...ij, ...kj -> ...ik", x1, x2)
    )

def knn_shapley(K: int, input_tra: t.Tensor, label_tra: t.Tensor, input_val: t.Tensor, label_val: t.Tensor, keep=False):
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
    a_sort = dist(input_val, input_tra).argsort(-1)
    arange = t.arange(0, M).reshape(M, 1)
    # eqtest[i, j] = label_val[i] == label_tra[idsort[i, j]]
    # sv[i, N-1] = eqtest[i, N-1] / N
    # sv[i, j] = sv[i, j+1] + (eqtest[i, j] - eqtest[i, j+1]) / max(K, j+1)
    eqtest = 1.0 * (label_val.reshape(M, 1) == label_tra)[..., arange, a_sort]
    eqdiff = t.zeros_like(eqtest)
    eqdiff[...,  N-1] = eqtest[..., N-1] / N
    eqdiff[..., :N-1] = (eqtest[..., :N-1] - eqtest[..., 1:]) / t.maximum(t.tensor(K), t.arange(1, N))
    sv = t.flip(t.flip(eqdiff, (-1,)).cumsum(-1), (-1,))
    # output[i, idsort[i, j]] = sv[i, j]
    output = t.zeros_like(sv)
    output[arange, a_sort] = sv
    if not keep:
        return output.sum(-2)
    else:
        return output

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
    labels = label_tra[a_sort[..., :, :K]]
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
    try:
        model = LogisticRegression(C=C)
        model.fit(input_tra, label_tra)
        output = model.predict(input_val)
        return t.tensor(output, device=input_tra.device)
    except:
        return t.zeros(input_val.shape[0]).fill_(label_tra[0])

def knn_alter_validation(
        K: int, S: int, 
        input_tra: t.Tensor, label_tra: t.Tensor, 
        input_val: t.Tensor, label_val: t.Tensor,
        minimize: bool = True,
    ):
    """
    Change validation set for a given K-nearest neighbour model s.t. : 
    -- In terms of Shapely values, the same training subset still contains elements with largest Shapley values. 
    -- Valuation results change. 
    Current implementation is given by relabeling. 
    INPUT: 
        K: K-nearest neighbours K
        S: selected subset S
        input_tra: training dataset input, shape [N, D]
        label_tra: training dataset label, shape [N]
        input_val: validation dataset input, shape [M, D]
        label_val: validation dataset label, shape [M]
    OUTPUT: 
        a different validation data set, with input and label, satisfying desired properties. 
    """
    import pulp # linear programming package
    from tqdm import tqdm
    from random import random, seed
    seed(114514)
    N, D = input_tra.shape
    M, D = input_val.shape
    lp = pulp.LpProblem("alter-validation", pulp.LpMaximize)
    sv_list = knn_shapley(K, input_tra, label_tra, input_val, label_val, keep=True)
    sv_rank = sv_list.sum(0).argsort(0)
    index_var = pulp.LpVariable.dict("selection-index", range(M))
    for i in range(M):
        pulp.LpVariable.setInitialValue(index_var[i], 0.5)
    for i in range(M):
        lp += index_var[i] <= 1.0
        lp += index_var[i] >= 0.0
    index_trg = knn_predict(K, input_tra[sv_rank[-S:]], label_tra[sv_rank[-S:]], input_val) == label_val
    index_trg = index_trg.float()
    sv_alte = [pulp.lpSum([sv_list[i, j] * index_var[i] for i in range(M)]) for j in tqdm(range(N), "compute shapley values")]
    gap = pulp.LpVariable.dict("gap", range(N-1), 0.0)
    tot = pulp.lpSum(gap)
    for j in range(N-1):
        lp += sv_alte[sv_rank[j]] <= sv_alte[sv_rank[j+1]] + gap[j]
    lp += (pulp.lpSum([index_var[i] for i in range(M)]) >= M//4)
    lp += (pulp.lpSum([index_var[i] for i in range(M)]) <= M*3//4)
    lp += (1 if minimize else -1) * pulp.lpSum(
        [index_var[i] * index_trg[i] + (1-index_var[i]) * (1-index_trg[i])
         for i in range(M)]) - tot * 1000
    lp.solve(solver=pulp.GLPK(msg=0))
    index_new = t.tensor([index_var[i].value() for i in range(M)])
    index_new = index_new > t.rand(100, *index_new.shape)
    return index_new

def test_knn_alter_validation_sv_eq():
    K = 15
    data = load_RANDOM(1000, 200, 200, 10)
    input_new, label_new = knn_alter_validation(K, 400, *data[0], *data[1], *data[2])
    sv1 = knn_shapley(K, *data[0], *data[1])
    sv0 = knn_shapley(K, *data[0], input_new, label_new)

def load_RANDOM(N: int, M: int, T: int, D: int):
    """
    Generate random data set. 
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
    from random import shuffle, seed
    seed(114514)
    index_all = list(range(N+M+T))
    shuffle(index_all)
    index_tra = index_all[0:N]
    index_val = index_all[N:N+M]
    index_tes = index_all[N+M:N+M+T]
    input_all = t.rand(N+M+T, D)
    label_all = t.randint(0, 10, (N+M+T, ))
    return (
        (input_all[index_tra], label_all[index_tra]), 
        (input_all[index_val], label_all[index_val]), 
        (input_all[index_tes], label_all[index_tes]))

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
    t.manual_seed(114514)
    dataset = (
        torchvision.datasets.CIFAR10("_data", download=True, train=True) +
        torchvision.datasets.CIFAR10("_data", download=True, train=False)
    )
    resnet = torchvision.models.resnet50().to('cuda:0')
    @t.no_grad()
    def load(index: t.tensor):
        tt = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.to('cuda:0'))
        ])
        subset = data.Subset(dataset, index)
        xs, ys = [], []
        collate = lambda xs: (t.stack([tt(x) for x, y in xs], 0), t.tensor([y for x, y in xs], device='cpu'))
        for x, y in tqdm(data.DataLoader(subset, 16, collate_fn=collate), 'preprocessing'):
            xs.append(resnet(x).to('cpu'))
            ys.append(y)
        return t.concat(xs), t.concat(ys), subset
    A = len(dataset)
    index_all = list(range(A))
    # N, M, T = A*N//(N+M+T), A*M//(N+M+T), A*T//(N+M+T)
    shuffle(index_all)
    index_all = index_all[:N+M+T]
    index_tra = index_all[0:N]
    index_val = index_all[N:N+M]
    index_tes = index_all[N+M:N+M+T]
    # train, validate and test on original dataset
    input_tra, label_tra, p = load(index_tra)
    input_val, label_val, _ = load(index_val)
    input_tes, label_tes, _ = load(index_tes)
    return (input_tra, label_tra), (input_val, label_val), (input_tes, label_tes), p

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
    t.manual_seed(114514)
    print(f'load {FILE_NAME}')
    with open(f"OpenML_datasets/{FILE_NAME}", "rb") as f:
        data_dict = pickle.load(f)
    x = t.tensor(data_dict['X_num'])
    y = t.tensor(data_dict['y'])
    @t.no_grad()
    def load(index: t.Tensor):
        return x[index], y[index]
    A = x.shape[0]
    index_all = list(range(A))
    shuffle(index_all)
    A = x.shape[0] // 4
    index_all = index_all[:A]
    N, M, T = A*N//(N+M+T), A*M//(N+M+T), A*T//(N+M+T)
    index_all = index_all[:N+M+T]
    index_tra = index_all[0:N]
    index_val = index_all[N:N+M]
    index_tes = index_all[N+M:N+M+T]
    # train, validate and test on original dataset
    input_tra, label_tra = load(index_tra)
    input_val, label_val = load(index_val)
    return (input_tra, label_tra), (input_val, label_val)

def experiment_1(K: int, S: list[int], predict: object, dataset: tuple[tuple, tuple]):
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
        valutation result correspondent to selected data. 
    """
    # train, validate and test on original dataset
    input_tra, label_tra = dataset[0]
    input_val, label_val = dataset[1]
    N = input_tra.shape[0]
    S = [int(x*N/max(S)) for x in S]
    rand = t.rand_like(label_val * 1.0)
    # construct label-altered validation set and testing
    def compute(index):
        input_alt, label_alt = input_val[index], label_val[index]
        sv_2 = knn_shapley(K, input_tra, label_tra, input_alt, label_alt)
        T = input_alt.shape[0]
        select = sv_2.argsort(-1)
        result_2 = []
        for s in S:
            acc = (label_alt == predict(input_tra[select[N-s:]], label_tra[select[N-s:]], input_alt)).sum().item() / T
            result_2.append(acc)
        sv_3 = sv_2[t.argsort(t.rand(*sv_2.shape), dim=-1)]
        select = sv_3.argsort(-1)
        result_3 = []
        for s in S:
            acc = (label_alt == predict(input_tra[select[N-s:]], label_tra[select[N-s:]], input_alt)).sum().item() / T
            result_3.append(acc)
        return result_2, result_3, sv_2, sv_3
    label_val = knn_predict(K, input_tra, label_tra, input_val)
    index_max = knn_alter_validation(
        K, min(S), input_tra, label_tra, input_val, label_val, minimize=False)
    result_2, result_3, sv_2, sv_3 = [], [], [], []
    for index in index_max:
        _result_2, _result_3, _sv_2, _sv_3 = compute(index)
        result_2.append(_result_2)
        result_3.append(_result_3)
        sv_2.append(_sv_2)
        sv_3.append(_sv_3)
    label_val = (
        (rand <  0.5) * t.randint_like(label_val, label_val.max()) + 
        (rand >= 0.5) * knn_predict(K, input_tra, label_tra, input_val)
    )
    index_min = knn_alter_validation(
        K, min(S), input_tra, label_tra, input_val, label_val, minimize=True)
    result_0, result_1, sv_0, sv_1 = [], [], [], []
    for index in index_min:
        _result_0, _result_1, _sv_0, _sv_1 = compute(index)
        result_0.append(_result_0)
        result_1.append(_result_1)
        sv_0.append(_sv_0)
        sv_1.append(_sv_1)
    return S, result_0, result_1, result_2, result_3, sv_0, sv_1, sv_2, sv_3

def experiment_2(K: int, dataset: tuple[tuple, tuple, tuple]):
    """
    Experiment 2 is about failure patterns of Shapley values. 
    -- Add a few samples in training set and validation set. 
    -- These samples are corrupted. 
    -- If we only add samples to training set, then the labels don't change and have very low SVs. 
    -- If we also add samples to validation set, the corrupted samples are selected with very high SVs. 
    If we have a pair of traning set and a validation set. 
    INPUT:
        K: K-nearest neighbours K
        S: the number of different shapley value groups
        N: the size of each group
    OUTPUT:
        pic: a list of pictures
        sv: a list of shapley values, correpsonding to each picture
    """
    input_tra, label_tra = dataset[0]
    input_val, label_val = dataset[1]
    pic = dataset[3]
    for p in pic: print(p)
    sv = knn_shapley(K, input_tra, label_tra, input_val, label_val)
    sv_distribution = []
    asort = sv.argsort()
    for i in range(10):
        sv_distribution.append((1.0 * (label_tra[asort] == i)).cumsum(0))
        print((1.0 * (label_tra[asort] == i)).cumsum(0).shape)
    return sv_distribution

def run_experiment_1():
    import matplotlib.pyplot as plt
    files = \
    """
    phoneme_1489.pkl
    2dplanes_727.pkl
    cpu_act_761.pkl
    vehicle_sensIT_357.pkl
    pol_722.pkl
    wind_847.pkl
    Click_prediction_small_1218.pkl
    default-of-credit-card-clients_42477.pkl
    APSFailure_41138.pkl
    """
    import pickle
    S = [i+1 for i in range(20)]
    data = experiment_1(15, S, 
        lambda x_tra, y_tra, x_val: knn_predict(15, x_tra, y_tra, x_val), 
        dataset=load_CIFAR(1200, 400, 400)
    )
    with open(f"_plotdata/cifar", 'wb') as f:
        pickle.dump((data), f)
    for dataset in files.strip().splitlines():
        dataset = dataset.strip()
        S = [i+1 for i in range(20)]
        data = experiment_1(15, S, 
            lambda x_tra, y_tra, x_val: knn_predict(15, x_tra, y_tra, x_val), 
            dataset=load_OPENML(6, 2, 2, dataset)
        )
        with open(f"_plotdata/{dataset}", 'wb') as f:
            pickle.dump((data), f)

def illustrate_experiment_1():
    import matplotlib.pyplot as plt
    files = \
    """
    cifar
    2dplanes_727.pkl
    cpu_act_761.pkl
    vehicle_sensIT_357.pkl
    pol_722.pkl
    wind_847.pkl
    Click_prediction_small_1218.pkl
    default-of-credit-card-clients_42477.pkl
    APSFailure_41138.pkl
    phoneme_1489.pkl
    """
    import pickle
    for dataset in files.strip().splitlines():
        dataset = dataset.strip()
        with open(f"_plotdata/{dataset}", 'rb') as f:
            # S: selected dataset size
            # result_0: accuracy when selecting with original validation set
            # result_1: accuracy when selection with randomized validation set
            # result_2: accuracy when selection with altered validation set
            # result_3: accuracy when selection with randomized validation set
            # sv_0: shapley value for each training datum by original validation set
            # sv_1: shapley value for each training datum by randomized validation set
            # sv_2: shapley value for each training datum by altered validation set
            # sv_3: shapley value for each training datum by randomized validation set
            S, result_0, result_1, result_2, result_3, sv_0, sv_1, sv_2, sv_3 = pickle.load(f)
            result_0, result_1, result_2, result_3 = t.tensor(result_0), t.tensor(result_1), t.tensor(result_2), t.tensor(result_3)
        plt.subplot(1, 2, 1)
        plt.title(dataset + '-original')
        plt.plot(S, result_0.mean(0), label='original', color='red')
        plt.fill_between(S, result_0.mean(0)-result_0.std(0), result_0.mean(0)+result_0.std(0), alpha=0.5, color='red')
        plt.plot(S, result_1.mean(0), label='random', color='blue')
        plt.fill_between(S, result_1.mean(0)-result_1.std(0), result_1.mean(0)+result_1.std(0), alpha=0.5, color='blue')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.title(dataset + '-altered')
        plt.plot(S, result_2.mean(0), label='altered', color='red')
        plt.plot(S, result_3.mean(0), label='random', color='blue')
        plt.fill_between(S, result_2.mean(0)-result_2.std(0), result_2.mean(0)+result_2.std(0), alpha=0.5, color='red')
        plt.fill_between(S, result_3.mean(0)-result_3.std(0), result_3.mean(0)+result_3.std(0), alpha=0.5, color='blue')
        plt.legend()
        plt.savefig('fig/' + dataset.split('.')[0] + '-accuracy.png')
        plt.gcf().clear()
        # sort by original shapley value and fill
        plt.title(dataset)
        sv_0 = t.stack(sv_0).mean(0)
        sv_2 = t.stack(sv_2).mean(0)
        asort = sv_0.argsort()
        plt.fill_between(range(sv_0.shape[0]), sv_0[asort], alpha=0.7, label='original', color='red')
        plt.fill_between(range(sv_0.shape[0]), sv_2[asort], alpha=0.7, label='altered', color='blue')
        plt.legend()
        plt.savefig('fig/' + dataset.split('.')[0] + '-shapley-value.png')
        plt.gcf().clear()

def run_experiment_2():
    # progressively add more deer in validation
    # check knn
    import matplotlib.pyplot as plt
    K = 15
    distribution = experiment_2(K, load_CIFAR(5000, 1000, 100))
    label_map = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck',
    }
    for i in range(10):
        print(distribution[i].shape)
        plt.plot(range(distribution[i].shape[0]), distribution[i], 
                         label=label_map[i], alpha=0.5)
    plt.legend()
    plt.savefig('fig/' + 'experiment-no-legend-2.png', dpi=200)

if __name__ == '__main__':
    run_experiment_1()
    # illustrate_experiment_1()
    # run_experiment_2()