#!/usr/bin/env python
# coding: utf-8

import torch
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import DataLoader


class DatasetBuilder(torch.utils.data.Dataset):
    """
    自定义数据集类，用于构建包含用户和物品数据的PyTorch数据集。
    
    该类继承自torch.utils.data.Dataset，实现了数据集的迭代访问。
    
    参数:
    - data: 包含用户和物品数据的二维数组，最后一列是标签。
    - user_count: 用户的数量，用于定义用户字段的维度。
    - item_count: 物品的数量，用于定义物品字段的维度。
    """

    def __init__(self, data, user_count, item_count):
        """
        初始化数据集对象。
        
        将输入数据转换为Tensor格式，并存储用户和物品的维度信息。
        """
        # 将数据集转换为长整型Tensor，去除标签列
        self.x = torch.tensor(data[:, :-1], dtype=torch.long)
        # 将数据集最后一列（标签）转换为浮点型Tensor，并增加一个维度，变成一个二维张量
        self.y = torch.tensor(data[:, -1], dtype=torch.float).unsqueeze(1)
        # 存储用户和物品的维度信息
        self.field_dims = [user_count, item_count]
    
    def __len__(self):
        """
        返回数据集的大小，即数据集中的样本数量。
        
        返回:
        - int: 数据集的大小。
        """
        # 返回y的行数，即数据集样本的数量
        return self.y.shape[0]

    def __getitem__(self, index):
        """
        根据给定的索引获取数据集中的一个样本。
        
        参数:
        - index: 样本的索引。
        
        返回:
        - tuple: 包含一个用户和物品特征的Tensor以及一个标签的Tensor。
        """
        # 根据索引从x和y中获取对应的样本
        return self.x[index], self.y[index]
