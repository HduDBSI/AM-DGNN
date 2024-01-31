from utility.word import CFG
import model.help as utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


class Attention1(nn.Module):
    def __init__(self, in_features: int, atten_dim: int, dim_w: int):
        super().__init__()
        self.W_1 = nn.Parameter(torch.Tensor(in_features + dim_w, atten_dim))
        self.W_2 = nn.Parameter(torch.Tensor(in_features, atten_dim))
        self.b = nn.Parameter(torch.Tensor(1, atten_dim))
        self.v = nn.Parameter(torch.Tensor(1, atten_dim))
        self.act = nn.ReLU()

    def forward(self, ev, ej, ew, v_jw):
        zeroj = torch.zeros((1, ej.shape[1]), dtype=ej.dtype, device=ej.device)
        zerow = torch.zeros((1, ew.shape[1]), dtype=ew.dtype, device=ew.device)
        ej = torch.cat([zeroj, ej])
        ew = torch.cat([zerow, ew])
        v_j, v_w = v_jw
        k = v_j.shape[1]

        eNj = ej[v_j]
        eNw = ew[v_w]
        eNv = ev.unsqueeze(1).repeat(1, k, 1)
        #  A.unsqueeze(1)从A一个[M, N]到[M, 1, N]并沿第二维.repeat(1, K, 1)重复张量时间K

        eN_vw = torch.cat([eNv, eNw], dim=-1)
        av_j = torch.matmul(eN_vw, self.W_1) + torch.matmul(eNj, self.W_2) + self.b
        x = torch.matmul(self.act(av_j), self.v.T)
        a = torch.softmax(x, dim=1)
        # 注意力得分公式
        eN = torch.sum(a * eNj, dim=1)
        return eN

class BasicLayer(nn.Module):
    def __init__(self, in_features, out_features, atten_dim, weight_dim):
        super().__init__()

        self.atten1 = nn.ModuleDict()
        type_name = ["user", "item", "tag"]
        for i in range(3):
            self.atten1.update({f"{type_name[i]}": Attention1(in_features, atten_dim, weight_dim)})
        # attention2
        self.U = nn.Parameter(torch.Tensor(in_features, atten_dim))
        self.q = nn.Parameter(torch.Tensor(1, atten_dim))
        self.p = nn.Parameter(torch.Tensor(1, atten_dim))
        self.act = nn.ReLU()


    def _atten2(self, u, i, t):
        uit = torch.stack([u, i, t], dim=1)
        x = torch.matmul(uit, self.U) + self.q
        x = torch.matmul(self.act(x), self.p.T)
        b = torch.softmax(x, dim=1)
        x = b * uit
        x = x.sum(dim=1).squeeze(1)
        return x


    #Intention
    def _intent(self, ev, et, v_jw, j_tw):
        # zerot = torch.zeros((1, et.shape[1]), dtype=et.dtype, device=et.device)
        # et = torch.cat([zerot, et])
        j_t, _ = j_tw
        v_j, _ = v_jw
        k = v_j.shape[1]

        e_t = et[j_t[v_j-1]-1]
        e_v = ev.unsqueeze(1).unsqueeze(1).permute(0, 1, 3, 2)
        e_v = e_v.repeat(1, k, 1, 1)
        x = torch.matmul(e_t, e_v)
        a = torch.softmax(x, dim=2)
        e_j = torch.sum(a * e_t, dim=2).squeeze(2)
        emb = (ev.unsqueeze(1) + e_j).mean(dim=1)
        return emb


    def forward(self, eu, ei, et, ew, u_iw, u_tw, i_uw, i_tw, t_uw, t_iw):
        eu_uN = eu
        eu_iN = self.atten1["item"].forward(eu, ei, ew, u_iw)
        eu_tN = self.atten1["tag"].forward(eu, et, ew, u_tw)
        ei_iN = ei
        ei_uN = self.atten1["user"].forward(ei, eu, ew, i_uw)
        ei_tN = self.atten1["tag"].forward(ei, et, ew, i_tw)
        et_tN = et
        et_uN = self.atten1["user"].forward(et, eu, ew, t_uw)
        et_iN = self.atten1["item"].forward(et, ei, ew, t_iw)

        euN = self._atten2(eu_uN, eu_iN, eu_tN)
        eiN = self._atten2(ei_uN, ei_iN, ei_tN)
        etN = self._atten2(et_uN, et_iN, et_tN)

        # user_emb = self._intent(euN, etN, u_iw, i_tw)
        # item_emb = self._intent(eiN, etN, i_uw, u_tw)
        # tag_emb = etN
        #
        # return user_emb, item_emb, tag_emb
        return euN, eiN, etN

class TagGCN(nn.Module):
    def __init__(self, data):
        super().__init__()
        self._config(CFG)
        self.num_user = data.num['user']
        self.num_item = data.num['item']
        self.num_tag = data.num['tag']
        self.num_weight = data.num['weight']
        self.num_list = [data.num['user'], data.num['item'], data.num['tag']]
        self.norm_adj = utils.creat_adj(data, self.use_tag, self.norm_type, \
                                        self.split_adj_k, self.device)

        self._init_weight()

        self.data = data
        start = time.time()
        self.all_sample = data.get_sample_neighbor(self.neighbor_k)
        # data.get_all_neighbor() 全部邻居
        # data.get_sample_neighbor(self.neighbor_k) 采样部分邻居
        print(f"TagGCN got ready! [neighbor sample time {time.time()-start}]")

        # self.lightgcnLayer = LightGCN()

    def _config(self, config):
        self.dim_latent = config['dim_latent']
        self.dim_weight = config['dim_weight']
        self.num_layer = len(config['dim_layer_list'])
        self.dim_layer_list = [self.dim_latent] + config['dim_layer_list']
        self.dim_atten = config['dim_atten']
        self.message_drop_list = config['message_drop_list']
        self.device = config['device']
        self.neighbor_k = config['neighbor_k']
        self.reg = config['reg']
        self.transtag_reg = config['transtag_reg']
        self.use_tag = config['use_tag']
        self.split_adj_k = config["split_adj_k"]
        self.norm_type = config['norm_type']
        self.loss_func = config['mul_loss_func']
        self.margin = config['margin']
        self.node_drop = config['node_drop']

    def _init_weight(self):  # 初始化参数
        self.embed = nn.ParameterDict({
            "user": nn.Parameter(torch.Tensor(self.num_user, self.dim_latent)),
            "item": nn.Parameter(torch.Tensor(self.num_item, self.dim_latent)),
            "tag": nn.Parameter(torch.Tensor(self.num_tag, self.dim_latent)),
            "weight": nn.Parameter(torch.Tensor(self.num_weight, self.dim_weight)),
        })
        self.layer = nn.ModuleDict()  # nn.ModuleDict 是nn.module的容器，用于包装一组网络层，以索引方式调用网络层。
        for k in range(1):
            self.layer.update({
                f'{k}': BasicLayer(self.dim_layer_list[k], self.dim_layer_list[k + 1], \
                                   self.dim_atten, self.dim_weight),
            })

        initializer = nn.init.xavier_uniform_
        for param in self.parameters():
            if isinstance(param, nn.Parameter):
                initializer(param)
            elif isinstance(param, nn.Conv2d):
                initializer(param.weight)

    def sample(self):
        neighbor = []
        for adj_w in self.all_sample:
            indexs = np.arange(adj_w[0].shape[1])
            np.random.shuffle(indexs)
            nei = [x[:, :self.neighbor_k] for x in adj_w]
            nei = torch.tensor(nei, dtype=torch.long, device=self.device)
            neighbor.append(nei)
        return neighbor

    def forward(self):
        eu, ei = self.embed['user'], self.embed['item']
        et, ew = self.embed['tag'], self.embed['weight']
        embs_u, embs_i, embs_t = [eu], [ei], [et]
        for i, layer in enumerate(self.layer.values()):
        # layer =self.layer.values()
            neighbor = self.sample()
            u_iw, u_tw, i_uw, i_tw, t_uw, t_iw = neighbor
            eu, ei, et = layer(eu, ei, et, ew, u_iw, u_tw, i_uw, i_tw, t_uw, t_iw)
            eu = F.dropout(eu, p=self.message_drop_list[i], training=self.training)
            ei = F.dropout(ei, p=self.message_drop_list[i], training=self.training)
            et = F.dropout(et, p=self.message_drop_list[i], training=self.training)
            eu_norm = F.normalize(eu, p=2, dim=1)
            ei_norm = F.normalize(ei, p=2, dim=1)
            et_norm = F.normalize(et, p=2, dim=1)
            embs_u.append(eu_norm)
            embs_i.append(ei_norm)
            embs_t.append(et_norm)
        embs_u = torch.cat(embs_u, dim=1)
        embs_i = torch.cat(embs_i, dim=1)
        embs_t = torch.cat(embs_t, dim=1)

        norm_adj = utils.node_drop(self.norm_adj, self.node_drop, self.training)
        all_embed = torch.cat([embs_u, embs_i, embs_t], dim=0)
        all_embed_list = [all_embed]
        for k in range(self.num_layer):
            all_embed = utils.split_mm(norm_adj, all_embed)
            # all_embed = F.dropout(all_embed, p=self.message_drop_list[k], training=self.training)
            # norm_embed = F.normalize(all_embed, p=2, dim=1)  # 添加norm降低性能
            all_embed_list += [all_embed]

        all_embed = torch.mean(torch.stack(all_embed_list, dim=1), dim=1)
        list_embed = torch.split(all_embed, self.num_list, dim=0)
        return list_embed

    def get_ego_embed(self):
        return self.embed['user'], self.embed['item'], self.embed['tag']

    

    def loss(self, batch_data):
        users, pos_items, neg_items = batch_data.T
        all_users, all_items = self.forward()[:2]
        users_emb = all_users[users.long()]
        pos_emb = all_items[pos_items.long()]
        neg_emb = all_items[neg_items.long()]

        loss = utils.mul_loss(users_emb, pos_emb, neg_emb, self.loss_func)
        user_ego, item_ego = self.get_ego_embed()[:2]
        users_emb = user_ego[users.long()]
        pos_emb = item_ego[pos_items.long()]
        neg_emb = item_ego[neg_items.long()]
        reg_loss = utils.l2reg_loss(users_emb, pos_emb, neg_emb)

        return loss, self.reg * reg_loss

    # def transtag_loss(self, batch_data):
    #     user, tag, pos_item, neg_item = batch_data.T
    #     all_users, all_items, all_tags = self.get_ego_embed()  #self.forward()
    #     tag_emb = all_tags[tag.long()]
    #     user_emb = all_users[user.long()]
    #     pos_i_emb = all_items[pos_item.long()]
    #     neg_i_emb = all_items[neg_item.long()]
    #
    #     loss = utils.transtag_loss(user_emb, tag_emb, pos_i_emb, neg_i_emb, self.margin)
    #     reg_loss = utils.l2reg_loss(user_emb, tag_emb, pos_i_emb, neg_i_emb)
    #     return loss, self.transtag_reg * reg_loss

    def predict_rating(self, users):
        all_users, all_items = self.forward()[:2]
        users_emb = all_users[users]
        items_emb = all_items
        rating = torch.nn.Sigmoid()(torch.matmul(users_emb, items_emb.t()))
        return rating
