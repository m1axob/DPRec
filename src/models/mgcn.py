import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from common.combiner import CombinedFusion
from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph
from common.combine_qianwen import Combiner
from common.crossModalAttention import MultiHeadAttention
class MGCN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MGCN, self).__init__(config, dataset)
        self.sparse = True
        self.cl_loss = config['cl_loss']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.projection_dim=config['projection_dim']
        self.hidden_dim = config['hidden_dim']
        self.vt_loss= config['vt_loss']
        self.tempe=config['tempe']
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))

        self.norm_adj = self.get_adj_mat()
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)


        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,
                                                       norm_type='sym')
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda()

        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        self.softmax = nn.Softmax(dim=-1)

        self.query_common = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        self.gate_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_image_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_text_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.tau = 0.5

        self.combiner = CombinedFusion(self.embedding_dim, self.projection_dim, self.hidden_dim)
        self.cb = Combiner(self.embedding_dim,self.projection_dim,self.hidden_dim)
        self.img_projector = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.text_projector = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.modal_alignment = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        # self.user_image_embeds = torch.randn(self.n_users, self.embedding_dim)
        # self.user_text_embeds = torch.randn(self.n_users, self.embedding_dim)
        self.multihead_attn = MultiHeadAttention(self.embedding_dim).to(self.device)
    def pre_epoch_processing(self):
        pass
    def cross_modal_enhance(self, modal1_embeds, modal2_embeds):
       """跨模态注意力增强
       Args:
           modal1_embeds: 第一个模态的用户表征 [n_users, dim]
           modal2_embeds: 第二个模态的物品表征 [n_items, dim]
       Returns:
           enhanced_embeds: 增强后的用户表征 [n_users, dim]
       """
       # 计算跨模态注意力分数
       attention_scores = torch.mm(modal1_embeds, modal2_embeds.t())  # [n_users, n_items]
       attention_weights = F.softmax(attention_scores / math.sqrt(self.embedding_dim), dim=1)
       
       # 加权聚合另一个模态的信息
       enhanced_embeds = torch.mm(attention_weights, modal2_embeds)
       
       # 残差连接
       return modal1_embeds + 0.2 * enhanced_embeds
    # def build_user_user_sim(self):
    #     """基于用户交互历史构建用户相似度矩阵"""
    #     # 获取用户-物品交互矩阵 R
    #     user_item_matrix = self.R.to_dense()  # [n_users, n_items]
        
    #     # 计算用户交互的余弦相似度
    #     norm = torch.norm(user_item_matrix, p=2, dim=1, keepdim=True)
    #     normalized_matrix = user_item_matrix / (norm + 1e-8)
    #     user_sim = torch.mm(normalized_matrix, normalized_matrix.t())  # [n_users, n_users]
        
    #     # 可选: KNN稀疏化
    #     if self.knn_k > 0:
    #         # 保留每个用户的Top-K相似用户
    #         vals, cols = torch.topk(user_sim, k=self.knn_k, dim=1)
    #         rows = torch.arange(user_sim.size(0)).view(-1, 1).expand(-1, self.knn_k)
    #         user_sim = torch.zeros_like(user_sim)
    #         user_sim[rows.reshape(-1), cols.reshape(-1)] = vals.reshape(-1)
        
    #     return user_sim

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            # norm_adj = adj.dot(d_mat_inv)
            # print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, adj, train=False):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)

        # Behavior-Guided Purifier
        image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(image_feats))
        text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_t(text_feats))

        # User-Item View
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings
        # Item-Item View
        if self.sparse:
            for i in range(self.n_layers):
                image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
        else:
            for i in range(self.n_layers):
                image_item_embeds = torch.mm(self.image_original_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_user_embeds=self.cross_modal_enhance(image_user_embeds, text_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)

        if self.sparse:
            for i in range(self.n_layers):
                text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        else:
            for i in range(self.n_layers):
                text_item_embeds = torch.mm(self.text_original_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_user_embeds = self.cross_modal_enhance(text_user_embeds, image_item_embeds)
        
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        # Behavior-Aware Fuser
        # att_common = torch.cat([self.query_common(image_embeds), self.query_common(text_embeds)], dim=-1)
        # weight_common = self.softmax(att_common)
        # common_embeds = weight_common[:, 0].unsqueeze(dim=1) * image_embeds + weight_common[:, 1].unsqueeze(
        #     dim=1) * text_embeds
        # sep_image_embeds = image_embeds - common_embeds
        # sep_text_embeds = text_embeds - common_embeds
        # sep_mm_embeds = self.cb.forward(sep_image_embeds,sep_text_embeds)
        # side_embeds = (sep_mm_embeds+common_embeds)/2
        # image_prefer = self.gate_image_prefer(content_embeds)
        # text_prefer = self.gate_text_prefer(content_embeds)
        # sep_image_embeds = torch.multiply(image_prefer, sep_image_embeds)
        # sep_text_embeds = torch.multiply(text_prefer, sep_text_embeds)
        # side_embeds = (sep_image_embeds + sep_text_embeds + common_embeds) / 3
        
        side_embeds=self.cb.forward(image_embeds,text_embeds)
        all_embeds = content_embeds + side_embeds
        
        
        # 计算融合权重
        # alpha = self.fusion_gate(content_embeds)  # 行为信息的权重
        # beta = 1 - alpha  # 多模态信息的权重
        
        # # 加权融合
        # all_embeds = alpha * content_embeds + beta * side_embeds
        # all_embeds = all_embeds + 0.1 * (content_embeds + side_embeds)
        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds,image_embeds,text_embeds

        return all_embeddings_users, all_embeddings_items

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)



    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, side_embeds, content_embeds,image_embeds,text_embeds = self.forward(
            self.norm_adj, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)

        # noisy_image_embeds = self.add_noise(image_embeds, noise_type='gaussian', noise_level=0.1)
        # noisy_text_embeds = self.add_noise(text_embeds, noise_type='gaussian', noise_level=0.1)

        # image_embeds_user, image_embeds_items = torch.split(image_embeds, [self.n_users, self.n_items], dim=0)
        # text_embeds_user, text_embeds_items = torch.split(text_embeds, [self.n_users, self.n_items], dim=0)


        cl_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(
            side_embeds_users[users], content_embeds_user[users], 0.2)

        # image_user_embeds, image_item_embeds = torch.split(image_embeds, [self.n_users, self.n_items], dim=0)
        # text_user_embeds, text_item_embeds = torch.split(text_embeds, [self.n_users, self.n_items], dim=0)
        # user_vt_loss = self.align_vt(image_user_embeds, text_user_embeds)
        # # 2. 物品层面的模态对齐
        # item_vt_loss = self.align_vt(image_item_embeds, text_item_embeds)
        # 3. 整体表示的模态对齐
        vt_loss = self.modal_contrast_loss(image_embeds, text_embeds,self.tempe)
        
        return batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss+self.vt_loss*vt_loss
        




    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

    # def align_vt(self, embed1, embed2):
    #     emb1_var, emb1_mean = torch.var(embed1), torch.mean(embed1)
    #     emb2_var, emb2_mean = torch.var(embed2), torch.mean(embed2)

    #     vt_loss = (torch.abs(emb1_var - emb2_var) + torch.abs(emb1_mean - emb2_mean)).mean()

    #     return vt_loss

    def add_noise(self, embeds, noise_type='gaussian', noise_level=0.1):
        if noise_type == 'gaussian':
            noise = torch.randn_like(embeds) * noise_level
        elif noise_type == 'uniform':
            noise = (torch.rand_like(embeds) - 0.5) * 2 * noise_level
        else:
            raise ValueError("Unsupported noise type")
        return embeds + noise

    def content_id_contrastive(self, content_feature, id_feature):
        content_feature = torch.nn.functional.normalize(content_feature, p=2, dim=-1, eps=1e-5)
        id_feature = torch.nn.functional.normalize(id_feature, p=2, dim=-1, eps=1e-5)

        bs, _ = content_feature.size()
        neg_sample_id = torch.randint(0, bs, [bs, 16]) #.to(content_feature.device)
        neg_id_feat = id_feature[neg_sample_id]
        neg_content_feat = content_feature[neg_sample_id]
        id_samples = torch.cat([id_feature.unsqueeze(1), neg_id_feat], 1)
        content_samples = torch.cat([content_feature.unsqueeze(1), neg_content_feat], 1)

        c2i_score = torch.matmul(id_samples, content_feature.unsqueeze(2)).squeeze(2) / 0.5
        i2c_score = torch.matmul(content_samples, id_feature.unsqueeze(2)).squeeze(2) / 0.5

        label = torch.zeros([bs, ], dtype=torch.long).to(content_feature.device)
        c2i_loss = torch.nn.functional.cross_entropy(c2i_score, label)
        i2c_loss = torch.nn.functional.cross_entropy(i2c_score, label)
        cic_loss = 0.5 * (c2i_loss + i2c_loss)
        return cic_loss

    # def compute_itc_loss(self, image_embeds, text_embeds):
    #     """计算图像-文本对比损失(ITC)
    #     Args:
    #         image_embeds: 图像特征 [batch_size, dim]
    #         text_embeds: 文本特征 [batch_size, dim]
    #     Returns:
    #         loss_itc: 对比损失值
    #     """
    #     # 特征归一化
    #     image_embeds = F.normalize(image_embeds, dim=-1)
    #     text_embeds = F.normalize(text_embeds, dim=-1)
        
    #     # 计算相似度矩阵
    #     sim_i2t = torch.matmul(image_embeds, text_embeds.t()) / 0.1  # [batch_size, batch_size]
    #     sim_t2i = sim_i2t.t()
        
    #     # 构建标签(对角线为正样本)
    #     labels = torch.arange(image_embeds.size(0)).to(self.device)
        
    #     # 计算图像到文本和文本到图像的对比损失
    #     loss_i2t = F.cross_entropy(sim_i2t, labels)
    #     loss_t2i = F.cross_entropy(sim_t2i, labels)
        
    #     # 总的ITC损失
    #     loss_itc = (loss_i2t + loss_t2i) / 2
    #     return loss_itc
    # def align_uniform_loss(self, image_embeds, text_embeds, alpha=2, t=2):
    #     """对齐和均匀性损失
    #     Args:
    #         image_embeds: 图像特征 [batch_size, dim]
    #         text_embeds: 文本特征 [batch_size, dim]
    #         alpha: 对齐损失权重
    #         t: 均匀性损失的温度参数
    #     """
    #     # 特征归一化
    #     image_embeds = F.normalize(image_embeds, dim=-1)
    #     text_embeds = F.normalize(text_embeds, dim=-1)
        
    #     # 对齐损失 - 最小化配对样本的距离
    #     align_loss = torch.mean(torch.norm(image_embeds - text_embeds, dim=1) ** 2)
        
    #     # 均匀性损失 - 使特征分布更均匀
    #     img_uniform = torch.pdist(image_embeds, p=2).pow(2).mul(-t).exp().mean().log()
    #     txt_uniform = torch.pdist(text_embeds, p=2).pow(2).mul(-t).exp().mean().log()
    #     uniform_loss = (img_uniform + txt_uniform) / 2
        
    #     return alpha * align_loss - uniform_loss

    def align_vt(self, embed1, embed2):
        emb1_var, emb1_mean = torch.var(embed1), torch.mean(embed1)
        emb2_var, emb2_mean = torch.var(embed2), torch.mean(embed2)
    
        var_ratio = torch.abs(emb1_var / (emb2_var + 1e-6) - 1)
        mean_ratio = torch.abs(emb1_mean / (emb2_mean + 1e-6) - 1)
    
        # 添加L2正则化项
        var_l2 = torch.mean(emb1_var ** 2 + emb2_var ** 2)
    
        # 组合损失
        vt_loss = torch.mean(var_ratio + mean_ratio) + 1e-4 * var_l2
    
        return vt_loss
    # def content_id_contrastive(self, content_feature, id_feature):
    #     content_feature = self.cic_image_linear(content_feature)
    #     id_feature = self.cic_text_linear(id_feature)
    #     content_feature = torch.nn.functional.normalize(content_feature, p=2, dim=-1, eps=1e-5)
    #     id_feature = torch.nn.functional.normalize(id_feature, p=2, dim=-1, eps=1e-5)
    #
    #     bs, _ = content_feature.size()
    #     neg_sample_id = torch.randint(0, bs, [bs, 16]) #.to(content_feature.device)
    #     neg_id_feat = id_feature[neg_sample_id]
    #     neg_content_feat = content_feature[neg_sample_id]
    #     id_samples = torch.cat([id_feature.unsqueeze(1), neg_id_feat], 1)
    #     content_samples = torch.cat([content_feature.unsqueeze(1), neg_content_feat], 1)
    #
    #     c2i_score = torch.matmul(id_samples, content_feature.unsqueeze(2)).squeeze(2) / 0.1
    #     i2c_score = torch.matmul(content_samples, id_feature.unsqueeze(2)).squeeze(2) / 0.1
    #
    #     label = torch.zeros([bs, ], dtype=torch.long).to(content_feature.device)
    #     c2i_loss = torch.nn.functional.cross_entropy(c2i_score, label)
    #     i2c_loss = torch.nn.functional.cross_entropy(i2c_score, label)
    #     cic_loss = 0.5 * (c2i_loss + i2c_loss)
    #     return cic_loss
    #def modal_contrast_loss(self, image_embeds, text_embeds, temperature=0.1):
    def modal_contrast_loss(self, image_embeds, text_embeds, temperature):
        """轻量级的模态对比损失
        Args:
            image_embeds: 图像特征 [batch_size, dim]
            text_embeds: 文本特征 [batch_size, dim]
        """
        # 特征归一化
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        # 计算正样本对的相似度
        pos_sim = torch.sum(image_embeds * text_embeds, dim=-1)
        
        # 为每个样本随机采样负样本
        batch_size = len(image_embeds)
        neg_indices = torch.randperm(batch_size)[:min(batch_size, 128)]  # 限制负样本数量
        
        # 计算与负样本的相似度
        neg_sim_i2t = torch.matmul(image_embeds, text_embeds[neg_indices].t())
        neg_sim_t2i = torch.matmul(text_embeds, image_embeds[neg_indices].t())
        
        # 计算对比损失
        logits_i2t = torch.cat([pos_sim.unsqueeze(-1), neg_sim_i2t], dim=-1) / temperature
        logits_t2i = torch.cat([pos_sim.unsqueeze(-1), neg_sim_t2i], dim=-1) / temperature
        
        labels = torch.zeros(batch_size, device=image_embeds.device, dtype=torch.long)
        loss = (F.cross_entropy(logits_i2t, labels) + F.cross_entropy(logits_t2i, labels)) / 2
        
        return loss
