import torch

from models.krr import KernelRidgeRegression
from models.sntk import StructureBasedNeuralTangentKernel
import torch.nn.functional as F
from utils import match_loss, regularization, row_normalize_tensor, preprocess_cora, preprocess_cora_norm, \
    to_sparse_coo_tensor, kalman_filter
import deeprobust.graph.utils as utils

import numpy as np

from models.gcn import GCN
from models.parametrized_adj import PGE

from torch_sparse import SparseTensor
from models.fft import *

from scipy.sparse import coo_matrix


class STGC:

    def __init__(self, data, args, device='cuda', timeStep=10, **kwargs):
        self.data = data
        self.args = args
        self.device = device
        self.timeStep = timeStep

        # n = data.nclass * args.nsamples
        n = int(data.feat_train.shape[0] * args.reduction_rate)
        # from collections import Counter; print(Counter(data.labels_train))

        d = data.feat_train.shape[1]
        self.nnodes_syn = n
        self.feat_syn = nn.Parameter(torch.FloatTensor(1, n, timeStep, d).to(device))
        self.pge = PGE(nfeat=d, nnodes=n, device=device, args=args).to(device)
        in_features = self.feat_syn.shape[-1]
        out_features = in_features
        K = 3
        self.stcgcf = STCGCF(in_features, out_features, K).to(self.device)
        self.fftconv = FFTConv(out_features, out_features).to(self.device)
        self.labels_syn = torch.LongTensor(self.generate_labels_syn(data)).to(device)

        self.reset_parameters()
        # self.optimizer_feat = torch.optim.Adam([
        #                         {"params":self.feat_syn},
        #                         {"params":self.pge.parameters()},
        #                         {"params":self.stcgcf.parameters()},
        #                         {"params":self.fftconv.parameters()}], lr=0.0001)
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
        print('adj_syn:', (n, n), 'feat_syn:', self.feat_syn.shape)

    def reset_parameters(self):
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))

    def generate_labels_syn(self, data):
        from collections import Counter
        counter = Counter(data.labels_train)
        num_class_dict = {}
        n = len(data.labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x: x[1])
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}
        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                num_class_dict[c] = int(n * self.args.reduction_rate) - sum_
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]
            else:
                num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
                sum_ += num_class_dict[c]
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict
        return labels_syn

    def test_with_val(self, verbose=True):
        res = []

        data, device = self.data, self.device
        feat_syn, pge, labels_syn = self.feat_syn.detach(), \
            self.pge, self.labels_syn

        # with_bn = True if args.dataset in ['ogbn-arxiv'] else False
        model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=0.5,
                    weight_decay=5e-4, nlayers=2,
                    nclass=data.nclass, device=device).to(device)

        if self.args.dataset in ['ogbn-arxiv']:
            model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=0.5,
                        weight_decay=0e-4, nlayers=2, with_bn=False,
                        nclass=data.nclass, device=device).to(device)

        adj_syn = pge.inference(feat_syn)
        args = self.args

        if self.args.save:
            torch.save(adj_syn, f'saved_ours/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
            torch.save(feat_syn, f'saved_ours/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')

        if self.args.lr_adj == 0:
            n = len(labels_syn)
            adj_syn = torch.zeros((n, n))

        model.fit_with_val(feat_syn, adj_syn, labels_syn, data,
                           train_iters=600, normalize=True, verbose=False)

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()

        labels_train = torch.LongTensor(data.labels_train).cuda()
        output = model.predict(data.feat_train, data.adj_train)
        loss_train = F.nll_loss(output, labels_train)
        acc_train = utils.accuracy(output, labels_train)
        if verbose:
            print("Train set results:",
                  "loss= {:.4f}".format(loss_train.item()),
                  "accuracy= {:.4f}".format(acc_train.item()))
        res.append(acc_train.item())

        # Full graph
        output = model.predict(data.feat_full, data.adj_full)
        loss_test = F.nll_loss(output[data.idx_test], labels_test)
        acc_test = utils.accuracy(output[data.idx_test], labels_test)
        res.append(acc_test.item())
        if verbose:
            print("Test set results:",
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))
        return res

    # # fixme
    # device = torch.device('cpu')
    #
    # model = STGNN(in_channels=feat_sub.shape[1], hidden_channels=16, lstm_hidden_channels=32,
    #           out_channels=len(set(labels.tolist()))).to(device)
    # train_edge_index,_ = geoutils.from_scipy_sparse_matrix(toCooMetric(adj_sub))
    # feat_sub = preprocess_cora(feat_sub,10)
    # feat_sub= feat_sub.permute(1,2,0)
    # time_steps = feat_sub.shape[2]
    # train_data_list = []
    # for t in range(time_steps):
    #     x = torch.tensor(feat_sub[:, :, t], dtype=torch.float)  # 形状为 (140, 1433)
    #     y = torch.tensor(labels_syn, dtype=torch.long)  # 形状为 (140,)
    #     data_t = Data(x=x, edge_index=train_edge_index, y=y)
    #     train_data_list.append(data_t)
    # batch_size = 2
    # train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # print(f"Number of samples in dataset: {len(train_data_list)}")
    # print(f"Batch size: {batch_size}")
    #
    # for epoch in range(1, 20):
    #     avg_loss = train(train_loader, model, optimizer, device)
    #     acc = test(train_loader, model, device)
    #     print(f'Epoch: {epoch:03d}, Average Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}')
    def train(self, verbose=True):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        args = self.args
        data = self.data
        SNTK = StructureBasedNeuralTangentKernel(K=2, L=2, scale='average').to(device)
        ridge = torch.tensor(1e0).to(device)
        KRR = KernelRidgeRegression(SNTK.nodes_gram, ridge).to(device)
        feat_syn, pge, labels_syn = self.feat_syn, self.pge, self.labels_syn
        features, adj, labels = data.feat_train, data.adj_train, data.labels_train

        syn_class_indices = self.syn_class_indices

        features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        n_class = len(torch.unique(labels))
        labels = F.one_hot(labels, n_class).to(torch.float32)
        labels_syn = F.one_hot(labels_syn, n_class).to(torch.float32)

        feat_sub, adj_sub = self.get_sub_adj_feat(features)
        self.feat_syn.data.copy_(preprocess_cora_norm(feat_sub, num_time_steps=self.timeStep))

        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = utils.normalize_adj_tensor(adj)

        # 全图邻接矩阵
        adj = adj_norm
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
                           value=adj._values(), sparse_sizes=adj.size()).t()
        # 时空特征（全部）
        features = self.get_sample(features, False)

        adj_syn = pge(self.feat_syn[0, :, 0, :])
        adj_syn_norm = utils.normalize_adj_tensor(adj_syn, sparse=False)
        laplacian_adj = self.normalized_laplacian(adj.to_dense())
        laplacian_adj_syn_norm = self.normalized_laplacian(adj_syn_norm.detach())
        adj_syn_norm = to_sparse_coo_tensor(adj_syn_norm)

        self.pge = PGE(nfeat=7, nnodes=140, device=device, args=args).to(device)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
        pge = self.pge

        for it in range(args.epochs):

            compress_model = GCN(7, 64, 7, 3, device='cuda')
            compress_model.train()
            orgin_model = GCN(features.shape[3], 64, 7, 3, device='cuda')
            orgin_model.train()

            optimizer_model = torch.optim.Adam([
                {"params": self.stcgcf.parameters()},
                {"params": self.fftconv.parameters()}
            ]
                , lr=args.lr_model)


            loss = None
            H = self.stcgcf(feat_syn, laplacian_adj_syn_norm)

            # 进行快速傅里叶变换（FFT）和卷积操作

            H_out = self.fftconv(H)
            batch = []
            for b in range(features.shape[0]):
                time = []
                for t in range(features.shape[2]):
                    x = KRR.forward(features[b, :, t, :], H_out[b, :, t, :], labels, labels_syn,
                                    to_sparse_coo_tensor(adj.to_dense()), adj_syn_norm)
                    time.append(x)
                batch.append(torch.stack(time).permute(1, 0, 2))
            batch = torch.stack(batch)
            # 长短期权重
            alpha = kalman_filter(batch)
            outer_loop, inner_loop = get_loops(args)
            loss_avg = 0

            orgin_optimizer = torch.optim.Adam(orgin_model.parameters(), args.lr_feat)
            compress_optimizer = torch.optim.Adam(compress_model.parameters(), args.lr_feat)
            # todo 应该写dataloader来读取batch
            for ol in range(outer_loop):
                adj_syn = pge(batch[0, :, 0, :])
                adj_syn_norm = utils.normalize_adj_tensor(adj_syn, sparse=False)
                feat_syn_norm = feat_syn

                loss = torch.tensor(0.0).to(self.device)
                for c in range(data.nclass):
                    batch_size, n_id, adjs = data.retrieve_class_sampler(
                        c, adj, transductive=True, args=args)
                    if args.nlayers == 1:
                        adjs = [adjs]

                    adjs = [adj.to(self.device) for adj in adjs]
                    output = orgin_model.forward(features[0, :, 0, :], adj)
                    loss_real = F.mse_loss(output, labels)

                    gw_real = torch.autograd.grad(loss_real, list(orgin_model.parameters()))
                    gw_real = list((_.detach().clone() for _ in gw_real))
                    output_syn = compress_model.forward(batch[0, :, 0, :], adj_syn_norm)

                    ind = syn_class_indices[c]
                    loss_syn = F.mse_loss(
                        output_syn[ind[0]: ind[1]],
                        labels_syn[ind[0]: ind[1]])
                    gw_syn = torch.autograd.grad(loss_syn, list(compress_model.parameters()), create_graph=True)
                    coeff = self.num_class_dict[c] / max(self.num_class_dict.values())
                    loss += coeff * match_loss(gw_syn, gw_real, args, device=self.device)

                loss_avg += loss.item()
                # TODO: regularize
                if args.alpha > 0:
                    loss_reg = args.alpha * regularization(adj_syn, utils.tensor2onehot(labels_syn))
                else:
                    loss_reg = torch.tensor(0)

                loss = loss + loss_reg

                # update sythetic graph
                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                loss.backward()
                if it % 50 < 10:
                    self.optimizer_pge.step()
                    optimizer_model.step()
                else:
                    self.optimizer_feat.step()

                if args.debug and ol % 5 == 0:
                    print('Gradient matching loss:', loss.item())

                if ol == outer_loop - 1:
                    # print('loss_reg:', loss_reg.item())
                    # print('Gradient matching loss:', loss.item())
                    break

                feat_syn_inner = batch[0, :, 0, :].detach()
                adj_syn_inner = pge.inference(feat_syn_inner)
                adj_syn_inner_norm = utils.normalize_adj_tensor(adj_syn_inner, sparse=False)
                feat_syn_inner_norm = feat_syn_inner
                for j in range(inner_loop):
                    compress_optimizer.zero_grad()
                    output_syn_inner = compress_model.forward(feat_syn_inner_norm, adj_syn_inner_norm)
                    loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
                    loss_syn_inner.backward()
                    # print(loss_syn_inner.item())
                    compress_optimizer.step()  # update gnn param

                    orgin_optimizer.zero_grad()
                    output_real_inner = orgin_model.forward(features[0, :, 0, :], adj)
                    loss_real_inner = F.nll_loss(output_real_inner, labels)
                    loss_real_inner.backward()
                    orgin_optimizer.step()

            loss_avg /= (data.nclass * outer_loop)
            if it % 50 == 0:
                print('Epoch {}, loss_avg: {}'.format(it, loss_avg))

            eval_epochs = [400, 600, 800, 1000, 1200, 1600, 2000, 3000, 4000, 5000]

            if verbose and it in eval_epochs:
                # if verbose and (it+1) % 50 == 0:
                res = []
                runs = 1 if args.dataset in ['ogbn-arxiv'] else 3
                for i in range(runs):
                    if args.dataset in ['ogbn-arxiv']:
                        res.append(self.test_with_val())
                    else:
                        res.append(self.test_with_val())

                res = np.array(res)
                print('Train/Test Mean Accuracy:',
                      repr([res.mean(0), res.std(0)]))

        a = 1

    # fixme 对称归一化拉普拉斯
    def normalized_laplacian(self, adj_matrix):
        R = adj_matrix.sum(0).cpu()
        R_sqrt = 1 / np.sqrt(R)
        D_sqrt = np.diag(R_sqrt)
        I = np.eye(R.shape[0])
        L = I - np.matmul(np.matmul(D_sqrt, adj_matrix.cpu()), D_sqrt).numpy()
        adj_matrix.to('cuda')
        return torch.from_numpy(L).to(self.device).to(torch.float32)

    # fixme 得到了四维数据
    def get_sample(self, features, syn):
        a = []
        for i in range(1):
            if syn:
                a.append(preprocess_cora_norm(features, num_time_steps=self.timeStep))
            else:
                a.append(preprocess_cora(features, num_time_steps=self.timeStep))
        return torch.stack(a)

    def get_sub_adj_feat(self, features):
        data = self.data
        args = self.args
        idx_selected = []

        from collections import Counter;
        counter = Counter(self.labels_syn.cpu().numpy())

        for c in range(data.nclass):
            tmp = data.retrieve_class(c, num=counter[c])
            tmp = list(tmp)
            idx_selected = idx_selected + tmp
        idx_selected = np.array(idx_selected).reshape(-1)
        features = features[self.data.idx_train][idx_selected]

        # adj_knn = torch.zeros((data.nclass*args.nsamples, data.nclass*args.nsamples)).to(self.device)
        # for i in range(data.nclass):
        #     idx = np.arange(i*args.nsamples, i*args.nsamples+args.nsamples)
        #     adj_knn[np.ix_(idx, idx)] = 1

        from sklearn.metrics.pairwise import cosine_similarity
        # features[features!=0] = 1
        k = 2
        sims = cosine_similarity(features.cpu().numpy())
        sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
        for i in range(len(sims)):
            indices_argsort = np.argsort(sims[i])
            sims[i, indices_argsort[: -k]] = 0
        adj_knn = torch.FloatTensor(sims).to(self.device)
        return features, adj_knn


def toCooMetric(adj):
    rows, cols = torch.where(adj)
    values = adj[rows, cols]

    # 将索引和值转换为 NumPy 数组
    rows = rows.cpu().numpy()
    cols = cols.cpu().numpy()
    values = values.cpu().numpy()

    # 使用 scipy 创建 COO 格式的稀疏矩阵
    num_nodes = adj.size(0)

    return coo_matrix((values, (rows, cols)), shape=(num_nodes, num_nodes))


def test(loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            out = out.view(-1, out.size(-1))
            pred = out.argmax(dim=1)
            correct += (pred == data.y.view(-1)).sum().item()
            total += data.y.size(0)
    return correct / total


def train(loader, model, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model.forward(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out.view(-1, out.size(2)), data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def get_loops(args):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if args.one_step:
        if args.dataset == 'ogbn-arxiv':
            return 5, 0
        return 1, 0
    if args.dataset in ['ogbn-arxiv']:
        return args.outer, args.inner
    if args.dataset in ['cora']:
        return 20, 15  # sgc
    if args.dataset in ['citeseer']:
        return 20, 15
    if args.dataset in ['physics']:
        return 20, 10
    else:
        return 20, 10

