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
    """
    Dice类初始化方法。
    初始化时，除了调用nn.Module的初始化方法，还定义了一个可学习的参数alpha。
    """
    
    def __init__(self):
        """
        初始化Dice类。
        """
        super(Dice, self).__init__()  # 调用nn.Module的初始化方法
        self.alpha = nn.Parameter(torch.zeros((1, )))  # 初始化alpha参数为0，作为一个可学习的参数
        
    def forward(self, x):
        """
        定义前向传播过程。
        
        参数:
        x: 输入的张量。
        
        返回:
        加权后的输入张量。
        """
        avg = x.mean(dim=0) # 计算前一层输出的均值
        std = x.std(dim=0) # 计算前一层输出的标准差
        norm_x = (x - avg) / std # 计算归一化后的输入张量
        p = torch.sigmoid(norm_x + 1e-8)

        return x.mul(p) + self.alpha * x.mul(1 - p)

# 定义NeighborAggregation类，用于邻居节点的聚合操作
class NeighborAggregation(nn.Module):
    """
    邻居聚合类，用于实现基于注意力机制的节点聚合操作。

    参数:
    embed_dim (int): 嵌入维度，默认为8。
    hidden_dim (int): 隐藏层维度，默认为8。
    """
    def __init__(self, embed_dim=8, hidden_dim=8):
        super(NeighborAggregation, self).__init__()
        # 定义线性变换矩阵，用于将输入的嵌入向量（查询、键、值）变换到隐空间
        self.Q_w = nn.Linear(embed_dim, hidden_dim, bias=False)  # 查询向量的线性变换
        self.K_w = nn.Linear(embed_dim, hidden_dim, bias=False)  # 键向量的线性变换
        self.V_w = nn.Linear(embed_dim, hidden_dim, bias=False)  # 值向量的线性变换
        
        # 计算缩放因子，用于缩放点积结果，防止梯度消失或爆炸
        self.trans_d = math.sqrt(hidden_dim) 
        
        # 定义softmax函数，用于计算注意力得分
        self.get_score = nn.Softmax(dim=-1) 

    def forward(self, query, key):
        """
        前向传播函数，计算注意力机制下的邻居节点聚合结果。

        参数:
        query (Tensor): 查询向量。
        key (Tensor): 关键向量。

        返回:
        Tensor: 聚合后的结果向量。
        """
        # 对查询向量进行线性变换
        trans_Q = self.Q_w(query) # (1024 x 21 x 20)，是将query(1024 × 21 × 10）映射到隐空间 (1024 x 21 x 20)
        # 对关键向量进行线性变换
        trans_K = self.K_w(key) # (1024 x 1 x 20)
        # 对查询向量进行第二次线性变换，得到值向量
        trans_V = self.V_w(query) # (1024 x 21 x 20)
        # 计算注意力权重，基于变换后的查询和关键向量的内积
        score = self.get_score(torch.bmm(trans_Q, torch.transpose(trans_K, 1, 2)) / (self.trans_d)) # (1024 x 21 x 1)，1024个用户，每个用户对21个物品的注意力得分
        # 根据注意力权重对值向量进行加权求和，得到聚合结果
        answer = torch.mul(trans_V, score) # (1024 x 21 x 20)
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
        # Xavier初始化方法使得权重服从均匀分布 [−6/sqrt(𝑛_𝑖𝑛+𝑛_𝑜𝑢𝑡),6/sqrt(𝑛_𝑖𝑛+𝑛_𝑜𝑢𝑡)]
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
        # self.u_shared_aggregator = NeighborAggregation(embed_dim, 128)
        # self.i_shared_aggregator = NeighborAggregation(embed_dim, 128)
        # 设置用户和物品的组数、最近交互序列的长度等参数
        self.u_group_num = u_group_num + 1
        self.i_group_num = i_group_num + 1
        self.recent_len = recent_len
        # 初始化softmax函数，用于计算宏观权重
        self.macro_weight_func = nn.Softmax(dim=1) # dim=1 表示对矩阵的行进行归一化，dim=0 表示对矩阵的列进行归一化
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
        # 获取一个批次中用户ID对应的嵌入表示 (1024 x 10)
        user_embedding = self.user_embed(x[:, 0]) 
        # 获取一个批次用户的一阶邻居（聚类物品）的边 (1024 × 21)
        user_1ord_neighbor = x[:, 1: self.i_group_num + 1] 
        # 获取用户2阶邻居（聚类用户）的边 (1024 × 20)，这个 20 是自己设定的聚类用户的个数
        user_2ord_neighbor = x[:, self.i_group_num + 1: self.i_group_num + self.u_group_num + 1] 
        # 获取用户-物品的交互记录 (1024 × 20)
        user_recent = x[:, self.i_group_num + self.u_group_num + 1: self.i_group_num + self.u_group_num + self.recent_len + 1]
        # 获取一个批次物品的嵌入表示 (1024 × 10)
        item_embedding = self.item_embed(x[:, self.i_group_num + self.u_group_num + self.recent_len + 1])
        # 获取物品的一阶邻居和二阶邻居
        item_1ord_neighbor = x[:, self.i_group_num + self.u_group_num + self.recent_len + 2: self.i_group_num + 2 * self.u_group_num + self.recent_len + 2]
        item_2ord_neighbor = x[:, self.i_group_num + 2 * self.u_group_num + self.recent_len + 2: 2 * self.i_group_num + 2 * self.u_group_num + self.recent_len + 2]
        # 获取物品-用户的交互记录
        item_recent = x[:, 2 * self.i_group_num + 2 * self.u_group_num + self.recent_len + 2:]

        # 将一维的用户组索引复制填充至二维为每个用户都分配一个索引 (1024 × 20)
        batch_u_gruop_slice = self.u_gruop_slice.expand(x.shape[0], self.u_group_num)
        # 同理创建物品组索引 (1024 × 21)
        batch_i_gruop_slice = self.i_gruop_slice.expand(x.shape[0], self.i_group_num)

        # 创建最近交互的掩码，用于后续加权操作
        user_recent_mask = (user_recent > 0).float().unsqueeze(-1) # (1024 × 20 × 1)
        item_recent_mask = (item_recent > 0).float().unsqueeze(-1) # (1024 × 20 × 1)
        
        # 计算各类邻居的权重
        user_1ord_weight = self.macro_weight_func(torch.log(user_1ord_neighbor.float() + 1) / self.tau).unsqueeze(-1) # (1024 × 21 × 1) self.macro_weight_func 对行进行归一化
        user_2ord_weight = self.macro_weight_func(torch.log(user_2ord_neighbor.float() + 1) / self.tau).unsqueeze(-1)
        item_1ord_weight = self.macro_weight_func(torch.log(item_1ord_neighbor.float() + 1) / self.tau).unsqueeze(-1)
        item_2ord_weight = self.macro_weight_func(torch.log(item_2ord_neighbor.float() + 1) / self.tau).unsqueeze(-1)

        # 获取各类邻居的嵌入表示
        user_1ord_embedding = self.i_macro_embed(batch_i_gruop_slice) # (1024 × 21 × 10) 1024个用户，每个用户与21个物品组有或者没有交互，每个物品组10个嵌入维度
        user_2ord_embedding = self.u_macro_embed(batch_u_gruop_slice) # (1024 × 20 × 10) 1024个用户，每个用户与20个用户组有或者没有交互，每个用户组10个嵌入维度
        item_1ord_embedding = self.u_macro_embed(batch_u_gruop_slice) # (1024 × 20 × 10) 1024个物品，每个物品与20个用户组有或者没有交互，每个用户组10个嵌入维度
        item_2ord_embedding = self.i_macro_embed(batch_i_gruop_slice) # (1024 × 21 × 10) 1024个物品，每个物品与21个物品组有或者没有交互，每个物品组10个嵌入维度
        user_recent_embedding = self.item_embed(user_recent) # (1024 × 20 × 10) 1024个用户，每个用户最近的20条与物品组交互的记录，每个物品10个嵌入维度
        item_recent_embedding = self.user_embed(item_recent) # (1024 × 20 × 10) 1024个物品，每个物品最近的20条与用户组交互的记录，每个用户10个嵌入维度

        # 使用邻居聚合模块对邻居嵌入进行转换和聚合
        u_1ord_trans_emb = self.i_shared_aggregator(user_1ord_embedding, item_embedding.unsqueeze(1)) # (1024 × 21 × 20)
        u_2ord_trans_emb = self.u_shared_aggregator(user_2ord_embedding, user_embedding.unsqueeze(1)) # (1024 × 20 × 20)
        i_1ord_trans_emb = self.u_shared_aggregator(item_1ord_embedding, user_embedding.unsqueeze(1)) # (1024 × 20 × 20)
        i_2ord_trans_emb = self.i_shared_aggregator(item_2ord_embedding, item_embedding.unsqueeze(1)) # (1024 × 21 × 20)
        user_recent_trans_emb = self.i_shared_aggregator(user_recent_embedding, item_embedding.unsqueeze(1)) # (1024 × 20 × 20)
        item_recent_trans_emb = self.u_shared_aggregator(item_recent_embedding, user_embedding.unsqueeze(1)) # (1024 × 20 × 20)

        # 根据权重对聚合结果进行加权求和
        user_1ord_ws = torch.mul(u_1ord_trans_emb, user_1ord_weight).sum(dim=1) # (1024 × 20)
        user_2ord_ws = torch.mul(u_2ord_trans_emb, user_2ord_weight).sum(dim=1) # (1024 × 20)
        item_1ord_ws = torch.mul(i_1ord_trans_emb, item_1ord_weight).sum(dim=1)
        item_2ord_ws = torch.mul(i_2ord_trans_emb, item_2ord_weight).sum(dim=1)
        user_recent_ws = torch.mul(user_recent_trans_emb, user_recent_mask).sum(dim=1)
        item_recent_ws = torch.mul(item_recent_trans_emb, item_recent_mask).sum(dim=1)

        # 将所有信息连接起来，输入到多层感知器中进行进一步处理
        concated = torch.hstack([user_embedding, user_1ord_ws, user_2ord_ws, user_recent_ws, item_embedding, item_1ord_ws, item_2ord_ws,  item_recent_ws]) # (1024 × 140)
        output = self.mlp(concated)
        # 输出结果，使用sigmoid函数将结果限制在0到1之间
        output = torch.sigmoid(output)
        return output