

from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
import torch_scatter as ts
import cv2
import numpy as np
from utils.common import (plot_image_pair,plot_keypoints)
import matplotlib.pyplot as plt
import matplotlib
from torch_cluster import knn_graph
import torch_geometric 


cross_or_self=0
l_img=0
r_img=0
head_num=0
data_g=0
cur_layer=0
c_img=0
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-2, keepdim=True)
        std = x.std(-2, keepdim=True)
        return torch.reshape(self.a_2, (1, -1, 1)) * ((x - mean) / (std + self.eps)) + torch.reshape(self.b_2, (1, -1, 1))

def MLP(channels: list, use_layernorm, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if use_layernorm:
                layers.append(LayerNorm(channels[i]))
            elif do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers, use_layernorm=False):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim], use_layernorm=use_layernorm)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query, key, value,adj=None):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    if adj is not None:
        #adj转为tensor
        adj=torch.from_numpy(adj).cuda()
        #拼接4个adj将(602,602)转为（4，602，602）
        adj=torch.stack([adj,adj,adj,adj])
        #增添一个维度，将（4，602，602）转为（1，4，602，602）
        adj=adj.unsqueeze(0)
        zero_vec=-1e12*torch.ones_like(scores)
        scores=torch.where(adj>0,scores,zero_vec)#将不相邻的点的注意力置为负无穷
    prob = torch.nn.functional.softmax(scores, dim=-1)
    vis_attention(prob,1)

    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

def vis_attention(prob,i):
    #将prob转为numpy
    prob=prob.cpu().detach().numpy()
    #获取0头的第i个特征点的注意力
    att_i=prob[0,0,i,:]
    global data_g,cur_layer,cross_or_self,c_img
    kpts0=data_g['keypoints0'][0].cpu().numpy()
    kpts1=data_g['keypoints1'][0].cpu().numpy()
    demo_1=cv2.imread('mine/demo_1.jpg')
    demo_2=cv2.imread('mine/demo_2.jpg')
    demo_1=cv2.resize(demo_1,(640, 480))
    demo_2=cv2.resize(demo_2,(640, 480))
    #bgr转rgb
    demo_1=cv2.cvtColor(demo_1,cv2.COLOR_BGR2RGB)
    demo_2=cv2.cvtColor(demo_2,cv2.COLOR_BGR2RGB)
    plot_image_pair([demo_1, demo_2])
    # plot_keypoints(kpts0, kpts1, color='k', ps=4)
    plot_keypoints(kpts0, kpts1, color='w', ps=2) 
    if cross_or_self=='self' and c_img=='l':
        #在ax[0]中绘制第i个特征点
        ax = plt.gcf().axes
        ax[0].scatter(kpts0[i, 0], kpts0[i, 1], c='r', s=4)
        for a_i in range(len(att_i)):
            if att_i[a_i]>(1/att_i.shape[0]):
                #显示编号
                # ax[0].text(kpts0[a_i, 0], kpts0[a_i, 1], a_i, fontsize=8)
                #设置att_i[i]值越大,利用sigmoid函数实现线宽越大
                sg=1/(1+np.exp(-att_i[a_i]))
                ax[0].plot([kpts0[i, 0], kpts0[a_i, 0]], [kpts0[i, 1], kpts0[a_i, 1]], c='r', linewidth=(sg))
                #sigmoid函数
                #np中的sigmoid函数
                # ax[0].plot([kpts0[i, 0], kpts0[a_i, 0]], [kpts0[i, 1], kpts0[a_i, 1]], c='r', linewidth=att_i[a_i]*5)

        #设置清晰度
        plt.savefig('mine_out/self_att_{}_curlayer{}.jpg'.format(i,cur_layer),dpi=300)
        plt.close()
        return 0
    if cross_or_self=='cross' and c_img=='l':
        fig = plt.gcf()        
        ax = plt.gcf().axes
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
        fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))
        
        ax[0].scatter(kpts0[i, 0], kpts0[i, 1], c='r', s=4)
        for a_i in range(len(att_i)):
            if att_i[a_i]>(1/att_i.shape[0]):
                #显示编号
                #设置att_i[i]值越大，连线越粗
                # ax[0].plot([kpts0[i, 0], kpts1[a_i, 0]], [kpts0[i, 1], kpts1[a_i, 1]], c='r', linewidth=att_i[a_i]*20)
                sg=1/(1+np.exp(-att_i[a_i]))
                fig.lines.append(matplotlib.lines.Line2D(
                    (fkpts0[i, 0], fkpts1[a_i, 0]), (fkpts0[i, 1], fkpts1[a_i, 1]), zorder=1,
                    transform=fig.transFigure, c='r', linewidth=sg))
        #设置清晰度
        plt.savefig('mine_out/cross_att_{}_curlayer{}.jpg'.format(i,cur_layer),dpi=300)
        plt.close()
        return 0
    
    return 0

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value,adj=None):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        if adj is None:
            x, _ = attention(query, key, value) 
        else:
            x, _ = attention(query, key, value, adj)    
        
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, use_layernorm=False):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim], use_layernorm=use_layernorm)
        nn.init.constant_(self.mlp[-1].bias, 0.0)
        
        
    def divide_graph(self, x_f, y_f,k_nn=3):
        #全为1
        #判断x_f和y_f维度是否相同
        if x_f.shape != y_f.shape:
            x_y=torch.cat([x_f, y_f])
            n_xy=x_y[0].permute(1,0)

            edge_index = knn_graph(n_xy, k=3, loop=False)
            Adj=torch_geometric.utils.to_scipy_sparse_matrix(edge_index)
            adj = Adj.toarray() 

            print('x_f.shape != y_f.shape,是cross attention')
            return 0
        else:#self-attention
            #(1,256)变为(256,1)
            n_x_f=x_f[0].permute(1,0)
            edge_index = knn_graph(n_x_f, k=k_nn, loop=False)
            #转coo稀疏矩阵
            Adj=torch_geometric.utils.to_scipy_sparse_matrix(edge_index)
            #转为numpy矩阵
            adj = Adj.toarray() 
            print('x_f.shape == y_f.shape,是self attention')
        return adj

    def forward(self, x, source):
        #全连接图的注意力传播
        message = self.attn(x, source, source)
        #对每个子图进行注意力传播,得到子图的特征向量 (特征点数*特征维度)
        for i in range(8,32,8):
            adj=self.divide_graph(x, source,i)
            #对每个子图进行注意力传播,得到子图的特征向量 (特征点数*特征维度)
            message = self.attn(x, source, source,adj)
            #拼接子图特征向量
            message=torch.cat([message, message], dim=1)   
            
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, use_layernorm=False):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4, use_layernorm=use_layernorm)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        layer_num=0
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
                continue
            global cross_or_self,cur_layer
            cross_or_self=name
            cur_layer=layer_num            
            # delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            global c_img
            c_img='l'
            delta0= layer(desc0, src0)
            c_img='r'
            delta1 = layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
            layer_num+=1
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """
    default_config = {
        'descriptor_dim': 256,
        'weights_path': None,
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
        'use_layernorm': False
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'], use_layernorm=self.config['use_layernorm'])

        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'], use_layernorm=self.config['use_layernorm'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(self.config['bin_value'] if 'bin_value' in self.config else 1.0))
        self.register_parameter('bin_score', bin_score)

        if self.config['weights_path']:
            weights = torch.load(self.config['weights_path'], map_location="cpu")
            if ('ema' in weights) and (weights['ema'] is not None):
                load_dict = weights['ema']
            elif 'model' in weights:
                load_dict = weights['model']
            else:
                load_dict = weights
            self.load_state_dict(load_dict)
            print('Loaded SuperGlue model (\"{}\" weights)'.format(
            self.config['weights_path']))

    def forward(self, data, **kwargs):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        if kwargs.get('mode', 'test') == "train":
            return self.forward_train(data)
        global data_g
        data_g=data
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }
        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape)

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, data['scores0'])
        desc1 = desc1 + self.kenc(kpts1, data['scores1'])

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])
        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return {
            'matches0': indices0, # use -1 for invalid match
            'matches1': indices1, # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }

    def forward_train(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        batch_size = data['image0'].shape[0]
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']

        kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape)

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, data['scores0'])
        desc1 = desc1 + self.kenc(kpts1, data['scores1'])

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])
        # Get the matches with score above "match_threshold".
        gt_indexes = data['matches']
        neg_flag = (gt_indexes[:, 1] == -1) | (gt_indexes[:, 2] == -1)
        loss_pre_components = scores[gt_indexes[:, 0], gt_indexes[:, 1], gt_indexes[:, 2]]
        loss_pre_components = torch.clamp(loss_pre_components, min=-100, max=0.0)
        loss_vector = -1 * loss_pre_components
        neg_index, pos_index = gt_indexes[:, 0][neg_flag], gt_indexes[:, 0][~neg_flag]
        # batched_loss = ts.scatter_mean(loss_vector, gt_indexes[:, 0])
        batched_pos_loss, batched_neg_loss = ts.scatter_mean(loss_vector[~neg_flag], pos_index, dim_size=batch_size), ts.scatter_mean(loss_vector[neg_flag], neg_index, dim_size=batch_size)
        pos_loss, neg_loss = self.config['pos_loss_weight']*batched_pos_loss.mean(), self.config['neg_loss_weight']*batched_neg_loss.mean()
        loss = pos_loss + neg_loss
        # batched_pos_loss, batched_neg_loss = (self.config['pos_loss_weight']*ts.scatter_mean(loss_vector[~neg_flag], pos_index, dim_size=batch_size)), (self.config['neg_loss_weight']*ts.scatter_mean(loss_vector[neg_flag], neg_index, dim_size=batch_size))
        # batched_loss = batched_pos_loss + batched_neg_loss
        # loss, pos_loss, neg_loss = batched_loss.mean(), batched_pos_loss.mean(), batched_neg_loss.mean()
        return loss, pos_loss, neg_loss