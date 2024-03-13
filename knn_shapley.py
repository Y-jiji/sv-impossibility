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
    A = x.shape[0]
    index_all = index_all[:A]
    N, M, T = min(999, A*N//(N+M+T)), min(999, A*M//(N+M+T)), A*T//(N+M+T)
    index_all = index_all[:N+M+T]
    index_tra = index_all[0:N]
    index_val = index_all[N:N+M]
    index_tes = index_all[N+M:N+M+T]
    # train, validate and test on original dataset
    input_tra, label_tra = load(index_tra)
    input_val, label_val = load(index_val)
    return (input_tra, label_tra), (input_val, label_val)

def experiment_1(K: int, S: list[int], dataset: tuple[tuple, tuple]):
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
    import pulp # linear programming package
    from tqdm import tqdm
    import string
    import random
    input_tra, label_tra = dataset[0]
    input_val, label_val = dataset[1]
    M, D = input_val.shape
    N, D = input_tra.shape
    sv = knn_shapley(K, input_tra, label_tra, input_val, label_val, True).to('cpu')
    assert (*sv.shape,) == (M, N)
    asort = sv.sum(-2).argsort(-1, descending=True).to('cpu')
    rperm = t.randperm(N, device=input_val.device).to('cpu')
    results = []
    for s in map(lambda x: int(N * x), tqdm(S)):
        print(M, N)
        prediction = knn_predict(K, input_tra[rperm[:s]], label_tra[rperm[:s]], input_val)
        z_rnd = 1 * (label_val == prediction)
        prediction = knn_predict(K, input_tra[asort[:s]], label_tra[asort[:s]], input_val)
        z_trg = 1 * (label_val == prediction)
        z_rnd_mean = z_rnd.float().mean().item()
        z_trg_mean = z_trg.float().mean().item() 
        z_sel = pulp.LpVariable.dict(f"selection", range(M), cat="Binary")
        for i in range(M): z_sel[i].setInitialValue(1.0)
        lp = pulp.LpProblem(f"alter-validation", pulp.LpMaximize)
        sv_sum = [pulp.lpSum([z_sel[j] * sv[j, i] for j in range(M)]) for i in range(N)]
        bound = pulp.LpVariable(f"bound", cat="Continuous")
        for i in asort[:s]: lp += sv_sum[i] >= bound
        for i in asort[s:]: lp += sv_sum[i] <= bound
        objective = pulp.lpSum(
            [z_sel[j] * 1 * (z_trg[j] - z_rnd[j]) for j in range(M)]
            if z_trg_mean < z_rnd_mean else
            [z_sel[j] * 1 * (z_rnd[j] - z_trg[j]) for j in range(M)]
        )
        lp += (objective >= 0)
        lp += objective
        lp.solve(pulp.CPLEX_PY(warmStart=True, gapRel=0.5))
        z_sel = t.tensor([z_sel[j].value() * 1.0 for j in range(M)], device=z_trg.device)
        z_trg_alt_mean = (((z_trg * z_sel).mean() + 1e-12) / (z_sel.mean() + 1e-12)).item()
        z_rnd_alt_mean = (((z_rnd * z_sel).mean() + 1e-12) / (z_sel.mean() + 1e-12)).item()
        print(results)
        results.append((s, z_trg_mean, z_rnd_mean, z_trg_alt_mean, z_rnd_alt_mean))
    return results

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
    asort = sv.argsort(descending=True)
    sv_distribution = (1.0 * (label_tra[asort] == t.arange(10).unsqueeze(-1))).cumsum(1)
    return sv_distribution / sv_distribution.max(0, keepdim=True).values

def run_experiment_1():
    import matplotlib.pyplot as plt
    # 2dplanes_727.pkl
    # Click_prediction_small_1218.pkl
    # CreditCardFraudDetection_42397.pkl
    # phoneme_1489.pkl
    # vehicle_sensIT_357.pkl
    # APSFailure_41138.pkl  
    # cpu_act_761.pkl
    # default-of-credit-card-clients_42477.pkl
    files = \
    """
    pol_722.pkl
    wind_847.pkl
    """
    import pickle
    S = [0.1*i for i in range(1, 10)]
    for dataset in files.strip().splitlines():
        dataset = dataset.strip()
        data = experiment_1(5, S, dataset=load_OPENML(6, 2, 0, dataset))
        with open(f"_plotdata/{dataset}", 'wb') as f:
            pickle.dump(data, f)

def illustrate_experiment_1():
    import matplotlib.pyplot as plt
    import matplotlib
    files = \
    """
    phoneme_1489.pkl
    vehicle_sensIT_357.pkl
    APSFailure_41138.pkl  
    cpu_act_761.pkl
    pol_722.pkl
    wind_847.pkl
    """
    font = {'size': 10}
    matplotlib.rc('font', **font)
    plt.rcParams["font.family"] = 'serif'
    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)
    matplotlib.rc('axes', labelsize=20)    # fontsize for xlabel and ylabel
    matplotlib.rc('axes', titlesize=20)    # fontsize for title
    # Click_prediction_small_1218.pkl
    # default-of-credit-card-clients_42477.pkl
    import pickle
    plt.rcParams['legend.fontsize'] = 20
    fig = plt.figure(figsize=(18, 12), dpi=150)
    for i, dataset in enumerate(files.strip().splitlines()):
        dataset = dataset.strip().split(".")[0]
        with open(f"_plotdata/{dataset}.pkl", "rb") as f:
            data = pickle.load(f)
            S, z_trg_mean, z_rnd_mean, z_trg_alt_mean, z_rnd_alt_mean = list(zip(*data))
        plt.subplot(2, 3, i+1)
        S = t.tensor([0.1*i for i in range(1, 10)])
        # 
        plt.plot(S, z_trg_mean, label="$v_1$, shapley", color='blue')
        plt.plot(S, z_rnd_mean, label="$v_1$, random", color='blue', linestyle='--')
        # 
        z_trg_alt_mean = t.tensor(z_trg_alt_mean)
        z_rnd_alt_mean = t.tensor(z_rnd_alt_mean)
        neq = (z_trg_alt_mean < 1.0).argwhere()[:, 0]
        plt.plot(S[neq], z_trg_alt_mean[neq], label="$v_2$, shapley", color='red')
        plt.plot(S[neq], z_rnd_alt_mean[neq], label="$v_2$, random", color='red', linestyle='--')
        plt.xticks(ticks=S, labels=['{:,.0%}'.format(x.item()) for x in S], rotation=30)
        plt.xlabel('selected training set')
        plt.ylabel('validation accuracy')
        plt.title(label=f'{" ".join(dataset.split("_")[:-1])}')
        plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('fig/experiment-1-all.png')

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
        plt.plot(range(distribution[i].shape[0]), distribution[i], 
                 label=label_map[i], alpha=0.5)
    plt.legend()
    plt.savefig('fig/' + 'experiment-no-legend-2.png', dpi=200)

if __name__ == '__main__':
    # run_experiment_1()
    illustrate_experiment_1()
