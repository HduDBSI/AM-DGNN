from utility.word import CFG
import model.help as utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class GraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features    # 节点向量的特征维度
        self.out_features = out_features    # 经过 gat 之后的特征维度
        self.alpha = alpha    # leakyRelu参数
        self.concat = concat   # 对特征向量是否拼接

        # 定义可训练参数，也即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414) 
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        # 定义leakyRelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.to_dense().nonzero().t()

        

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

class GAT(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.alpha = alpha


        self.attentions = [GraphAttentionLayer(in_features, out_features, dropout=self.dropout, alpha=self.alpha, concat=True) for _ in range(self.nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        
    def forward(self, x, adj, num):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mean(torch.stack([att(x, adj) for att in self.attentions], dim=1), dim=1)
        e = x[:num, :]
        return e

class ut_GAT(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(ut_GAT, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.alpha = alpha


        self.attentions = [GraphAttentionLayer(in_features, out_features, dropout=self.dropout, alpha=self.alpha, concat=True) for _ in range(self.nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)


    def forward(self, embs_ut, ut_adj, num_user):
        e_ut = F.dropout(embs_ut, self.dropout, training=self.training)
        e_ut = torch.mean(torch.stack([att(e_ut, ut_adj) for att in self.attentions], dim=1), dim=1)
        eu = e_ut[:num_user, :]
        return eu


class it_GAT(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(it_GAT, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.alpha = alpha

        self.attentions = [GraphAttentionLayer(in_features, out_features, dropout=self.dropout, alpha=self.alpha, concat=True) for _ in range(self.nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, embs_it, it_adj, num_item):
        e_it = F.dropout(embs_it, self.dropout, training=self.training)
        e_it = torch.mean(torch.stack([att(e_it, it_adj) for att in self.attentions], dim=1), dim=1)
        ei = e_it[:num_item, :]
        return ei

class DisenTag(nn.Module):
    def __init__(self, data):
        super().__init__()
        self._config(CFG)
        self.num_user = data.num['user']
        self.num_item = data.num['item']
        self.num_tag = data.num['tag']
        self.num_list = [data.num['user'], data.num['item']]
        self.norm_ui_adj = utils.creat_ui_adj(data, self.use_tag, self.norm_type, \
                                        self.split_adj_k, self.device)
        self.norm_ut_adj = utils.creat_ut_adj(data, self.use_tag, self.norm_type, \
                                        self.split_adj_k, self.device)
        self.norm_it_adj = utils.creat_it_adj(data, self.use_tag, self.norm_type, \
                                        self.split_adj_k, self.device)

        self._init_weight()

        self.data = data

    def _config(self, config):
        self.dim_latent = config['dim_latent']
        self.num_layer = len(config['dim_layer_list'])
        self.dim_layer_list = [self.dim_latent] + config['dim_layer_list']
        self.device = config['device']
        self.reg = config['reg']
        self.factor_k = config['factor_k']
        self.iterate_k = config['iterate_k']
        self.dim_k = self.dim_latent // self.factor_k
        self.use_tag = config['use_tag']
        self.split_adj_k = config["split_adj_k"]
        self.norm_type = config['norm_type']
        self.cor_reg = config['cor_reg']
        self.loss_func = config['mul_loss_func']
        self.nheads = config['n_heads']
        self.dropout = config['dropout']
        self.alpha = config['alpha']


    def _init_weight(self):  # 初始化参数
        self.embed = nn.ParameterDict({
            "user": nn.Parameter(torch.Tensor(self.num_user, self.dim_latent)),
            "item": nn.Parameter(torch.Tensor(self.num_item, self.dim_latent)),
            "tag": nn.Parameter(torch.Tensor(self.num_tag, self.dim_latent)),
        })
        self.layer1 = nn.ModuleDict()  # nn.ModuleDict 是nn.module的容器，用于包装一组网络层，以索引方式调用网络层。
        for k in range(1):
            self.layer1.update({
                f'{k}': ut_GAT(self.dim_layer_list[k], self.dim_layer_list[k + 1], \
                                   self.dropout, self.alpha, self.nheads),
            })

        self.layer2 = nn.ModuleDict()  # nn.ModuleDict 是nn.module的容器，用于包装一组网络层，以索引方式调用网络层。
        for k in range(1):
            self.layer2.update({
                f'{k}': it_GAT(self.dim_layer_list[k], self.dim_layer_list[k + 1], \
                                   self.dropout, self.alpha, self.nheads),
            })

        # self.layer = nn.ModuleDict()
        # for k in range(1):
        #     self.layer.update({
        #         f'{k}': GAT(self.dim_layer_list[k], self.dim_layer_list[k + 1], \
        #                            self.dropout, self.alpha, self.nheads),
        #     })
    

        initializer = nn.init.xavier_uniform_
        for param in self.parameters():
            if isinstance(param, nn.Parameter):
                initializer(param)
            elif isinstance(param, nn.Conv2d):
                initializer(param.weight)


    def forward(self, out_A=False):
        eu = self.embed['user']
        ei = self.embed['item']
        et = self.embed['tag']

        embs_ut = torch.cat([eu, et], dim = 0)
        embs_it = torch.cat([ei, et], dim = 0)
        
        ut_adj = self.norm_ut_adj
        it_adj = self.norm_it_adj


        for i, layer1 in enumerate(self.layer1.values()):
            embs_u = layer1(embs_ut, ut_adj, self.num_user)


        for i, layer2 in enumerate(self.layer2.values()):
            embs_i = layer2(embs_it, it_adj, self.num_item)

        # for i, layer in enumerate(self.layer.values()):
        #     embs_u = layer(embs_ut, ut_adj, self.num_user)
        #     embs_i = layer(embs_it, it_adj, self.num_item)


        
        A_values = torch.ones(self.factor_k, self.norm_ui_adj._nnz(), device=self.device)
        ego_embed = torch.cat([embs_u, embs_i], dim=0)
        all_embed = [ego_embed]
        out_layer_A = []
        for k in range(self.num_layer):
            A_values, ego_embed, out_A_factor = self.iterate_update(A_values, ego_embed)

            all_embed.append(ego_embed)
            out_layer_A.append(out_A_factor)

        all_embed = torch.stack(all_embed, dim=1)
        all_embed = torch.mean(all_embed, dim=1)
        list_emb = torch.split(all_embed, self.num_list, dim=0)
        if out_A == True:
            return out_layer_A

        return list_emb

    def iterate_update(self, A_values, ego_embed):
        ego_split_emb = torch.split(ego_embed, self.dim_k, dim=1)
        layer_emb = []
        out_A_factor = []
        for t in range(self.iterate_k):
            A_score_list = []
            A_factor = torch.softmax(A_values, dim=0)

            for i in range(self.factor_k):
                adj, factor_emb, A_score = self.factor_update(A_factor[i], ego_split_emb[i], self.norm_ui_adj)
                A_score_list.append(A_score)
                if t == self.iterate_k - 1:
                    layer_emb.append(factor_emb)
                    out_A_factor.append(adj)

            A_score = torch.stack(A_score_list, dim=0)
            A_values += A_score

        layer_emb = torch.stack(layer_emb)
        layer_emb = F.normalize(layer_emb, p=2, dim=2)
        ego_embed = torch.cat(list(layer_emb), dim=1)
        return A_values, ego_embed, out_A_factor

    def factor_update(self, A_factor, ego_split_emb, norm_ui_adj):
        adj = torch.sparse_coo_tensor(norm_ui_adj._indices(), A_factor.detach().cpu(), \
            norm_ui_adj.shape, device=self.device)
        col_sum = torch.sparse.sum(adj, dim=1)
        val = 1 / torch.sqrt(col_sum._values())
        val[torch.isinf(val)] = 0.0
        D = torch.sparse_coo_tensor(col_sum._indices()[0].unsqueeze(0).repeat(2, 1), val, \
            norm_ui_adj.shape, device=self.device)
        factor_emb = torch.sparse.mm(D, ego_split_emb)
        factor_emb = torch.sparse.mm(adj, factor_emb)
        factor_emb = torch.sparse.mm(D, factor_emb)

        head, tail = norm_ui_adj._indices().to(self.device)
        h_emb = factor_emb[head]
        t_emb = ego_split_emb[tail]  # t_emb = factor_emb[tail]
        h_emb = F.normalize(h_emb, p=2, dim=1)
        t_emb = F.normalize(t_emb, p=2, dim=1)
        mut_emb = torch.mul(h_emb, torch.tanh(t_emb))
        A_score = torch.sum(mut_emb, dim=1)
        return adj, factor_emb, A_score

    def get_ego_embed(self):
        return self.embed['user'], self.embed['item'], self.embed['tag']

    def loss(self, batch_data):
        data, cor = batch_data
        users, pos_items, neg_items = data.T
        all_embs = self.forward()
        all_users, all_items = all_embs[:2]
        users_emb = all_users[users.long()]
        pos_emb = all_items[pos_items.long()]
        neg_emb = all_items[neg_items.long()]

        loss = utils.mul_loss(users_emb, pos_emb, neg_emb, self.loss_func)
        #---------------------reg-------------------------
        user_ego, item_ego = self.get_ego_embed()[:2]
        users_emb = user_ego[users.long()]
        pos_emb = item_ego[pos_items.long()]
        neg_emb = item_ego[neg_items.long()]
        reg_loss = utils.l2reg_loss(users_emb, pos_emb, neg_emb)
        # -------------------cor-------------------------
        # emb_list = []
        # cor_user, cor_item = cor[:2]
        # emb_list.append(all_users[cor_user.long()])
        # emb_list.append(all_items[cor_item.long()])
    
        # emb_list = all_embs
        # all_emb = torch.cat(emb_list, dim=0)
        # dim_k = int(all_emb.shape[1] / self.factor_k)
        # factor_emb = torch.split(all_emb, dim_k, dim=1)
        # cor_loss = utils.cor_loss(factor_emb, self.factor_k)

        return loss, self.reg * reg_loss #, self.cor_reg * cor_loss

    def predict_rating(self, users):
        all_users, all_items = self.forward()[:2]
        users_emb = all_users[users]
        items_emb = all_items
        rating = torch.nn.Sigmoid()(torch.matmul(users_emb, items_emb.t()))
        return rating

