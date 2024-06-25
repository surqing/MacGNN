import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


import torch
import torch.nn as nn
import math

# 定义Dice类，用于在前向传播中实现特定的加权操作
class Dice(nn.Module):
    
    def __init__(self):
        super(Dice, self).__init__()
        self.alpha = nn.Parameter(torch.zeros((1, )))
        
    def forward(self, x):
        # 计算平均值和标准差，用于标准化输入x
        avg = x.mean(dim=0)
        std = x.std(dim=0)
        # 标准化x
        norm_x = (x - avg) / std
        # 计算sigmoid函数的值，用于加权
        p = torch.sigmoid(norm_x + 1e-8)

        # 返回加权后的结果
        return x.mul(p) + self.alpha * x.mul(1 - p)

# 定义NeighborAggregation类，用于邻居节点的聚合操作
class NeighborAggregation(nn.Module):

    def __init__(self, embed_dim=8, hidden_dim=8):
        super(NeighborAggregation, self).__init__()
        # 初始化Q、K、V的线性变换矩阵
        self.Q_w = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.K_w = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.V_w = nn.Linear(embed_dim, hidden_dim, bias=False)
        # 设置转换系数
        self.trans_d = math.sqrt(hidden_dim)
        # 初始化softmax函数，用于计算注意力权重
        self.get_score = nn.Softmax(dim=-1)

    def forward(self, query, key):
        # 对query和key进行线性变换
        trans_Q = self.Q_w(query)
        trans_K = self.K_w(key)
        trans_V = self.V_w(query)
        # 计算注意力权重
        score = self.get_score(torch.bmm(trans_Q, torch.transpose(trans_K, 1, 2)) / (self.trans_d))
        # 根据注意力权重聚合信息
        answer = torch.mul(trans_V, score)
        return answer

# 定义MacGNN类，用于实现图神经网络模型
class MacGNN(nn.Module):

    def __init__(self, field_dims, u_group_num, i_group_num, embed_dim, recent_len, tau=0.8, device='cpu'):
        super(MacGNN, self).__init__()
        # 初始化用户、物品、类别等的嵌入层
        self.user_embed = nn.Embedding(field_dims[0], embed_dim)
        self.item_embed = nn.Embedding(field_dims[1], embed_dim)
        self.cate_embed = nn.Embedding(field_dims[2], embed_dim)
        # 初始化用户和物品的宏观嵌入层
        self.u_macro_embed = nn.Embedding(u_group_num + 1, embed_dim)
        self.i_macro_embed = nn.Embedding(i_group_num + 1, embed_dim)
        # 初始化嵌入层的权重
        torch.nn.init.xavier_uniform_(self.user_embed.weight.data)
        torch.nn.init.xavier_uniform_(self.item_embed.weight.data)
        torch.nn.init.xavier_uniform_(self.cate_embed.weight.data)
        torch.nn.init.xavier_uniform_(self.u_macro_embed.weight.data)
        torch.nn.init.xavier_uniform_(self.i_macro_embed.weight.data)
        # 设置温度参数tau
        self.tau = tau
        # 初始化共享的邻居聚合模块
        self.u_shared_aggregator = NeighborAggregation(embed_dim, 2 * embed_dim)
        self.i_shared_aggregator = NeighborAggregation(embed_dim, 2 * embed_dim)
        # 设置用户和物品的组数、最近交互序列的长度等参数
        self.u_group_num = u_group_num + 1
        self.i_group_num = i_group_num + 1
        self.recent_len = recent_len
        # 初始化softmax函数，用于计算宏观权重
        self.macro_weight_func = nn.Softmax(dim=1)
        # 初始化用于索引的张量，并将其移动到指定设备
        self.u_gruop_slice = torch.arange(self.u_group_num, requires_grad=False).to(device)
        self.i_gruop_slice = torch.arange(self.i_group_num, requires_grad=False).to(device)
        # 初始化多层感知器模块，用于最后的预测
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 14, 200),
            Dice(),
            nn.Linear(200, 80),
            Dice(),
            nn.Linear(80, 1)
        )

    def forward(self, x):
        # 获取用户和物品的嵌入表示
        user_embedding = self.user_embed(x[:, 0])
        user_1ord_neighbor = x[:, 1: self.i_group_num + 1]
        user_2ord_neighbor = x[:, self.i_group_num + 1: self.i_group_num + self.u_group_num + 1]
        user_recent = x[:, self.i_group_num + self.u_group_num + 1: self.i_group_num + self.u_group_num + self.recent_len + 1]
        item_embedding = self.item_embed(x[:, self.i_group_num + self.u_group_num + self.recent_len + 1])
        item_1ord_neighbor = x[:, self.i_group_num + self.u_group_num + self.recent_len + 2: self.i_group_num + 2 * self.u_group_num + self.recent_len + 2]
        item_2ord_neighbor = x[:, self.i_group_num + 2 * self.u_group_num + self.recent_len + 2: 2 * self.i_group_num + 2 * self.u_group_num + self.recent_len + 2]
        item_recent = x[:, 2 * self.i_group_num + 2 * self.u_group_num + self.recent_len + 2:]

        # 创建用于批量操作的索引张量
        batch_u_gruop_slice = self.u_gruop_slice.expand(x.shape[0], self.u_group_num)
        batch_i_gruop_slice = self.i_gruop_slice.expand(x.shape[0], self.i_group_num)

        # 创建最近交互的掩码，用于后续加权操作
        user_recent_mask = (user_recent > 0).float().unsqueeze(-1)
        item_recent_mask = (item_recent > 0).float().unsqueeze(-1)
        
        # 计算各类邻居的权重
        user_1ord_weight = self.macro_weight_func(torch.log(user_1ord_neighbor.float() + 1) / self.tau).unsqueeze(-1)
        user_2ord_weight = self.macro_weight_func(torch.log(user_2ord_neighbor.float() + 1) / self.tau).unsqueeze(-1)
        item_1ord_weight = self.macro_weight_func(torch.log(item_1ord_neighbor.float() + 1) / self.tau).unsqueeze(-1)
        item_2ord_weight = self.macro_weight_func(torch.log(item_2ord_neighbor.float() + 1) / self.tau).unsqueeze(-1)

        # 获取各类邻居的嵌入表示
        user_1ord_embedding = self.i_macro_embed(batch_i_gruop_slice)
        user_2ord_embedding = self.u_macro_embed(batch_u_gruop_slice)
        item_1ord_embedding = self.u_macro_embed(batch_u_gruop_slice)
        item_2ord_embedding = self.i_macro_embed(batch_i_gruop_slice)
        user_recent_embedding = self.item_embed(user_recent)
        item_recent_embedding = self.user_embed(item_recent)

        # 使用邻居聚合模块对邻居嵌入进行转换和聚合
        u_1ord_trans_emb = self.i_shared_aggregator(user_1ord_embedding, item_embedding.unsqueeze(1))
        u_2ord_trans_emb = self.u_shared_aggregator(user_2ord_embedding, user_embedding.unsqueeze(1))
        i_1ord_trans_emb = self.u_shared_aggregator(item_1ord_embedding, user_embedding.unsqueeze(1))
        i_2ord_trans_emb = self.i_shared_aggregator(item_2ord_embedding, item_embedding.unsqueeze(1))
        user_recent_trans_emb = self.i_shared_aggregator(user_recent_embedding, item_embedding.unsqueeze(1))
        item_recent_trans_emb = self.u_shared_aggregator(item_recent_embedding, user_embedding.unsqueeze(1))

        # 根据权重对聚合结果进行加权求和
        user_1ord_ws = torch.mul(u_1ord_trans_emb, user_1ord_weight).sum(dim=1)
        user_2ord_ws = torch.mul(u_2ord_trans_emb, user_2ord_weight).sum(dim=1)
        item_1ord_ws = torch.mul(i_1ord_trans_emb, item_1ord_weight).sum(dim=1)
        item_2ord_ws = torch.mul(i_2ord_trans_emb, item_2ord_weight).sum(dim=1)
        user_recent_ws = torch.mul(user_recent_trans_emb, user_recent_mask).sum(dim=1)
        item_recent_ws = torch.mul(item_recent_trans_emb, item_recent_mask).sum(dim=1)

        # 将所有信息连接起来，输入到多层感知器中进行进一步处理
        concated = torch.hstack(
            [user_embedding, user_1ord_ws, user_2ord_ws, user_recent_ws, item_embedding, item_1ord_ws, item_2ord_ws, 
             item_recent_ws])
        output = self.mlp(concated)
        # 输出结果，使用sigmoid函数将结果限制在0到1之间
        output = torch.sigmoid(output)
        return output