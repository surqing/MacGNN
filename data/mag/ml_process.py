#!/usr/bin/env python
# coding: utf-8
"""
This is a example of MAG generation and data processing on the MovieLens dataset.
"""

import numpy as np
import csv
import pandas as pd
import random
import pickle
import copy
from scipy.sparse import csr_matrix

np.random.seed(2023)
random.seed(2023)

path = './ratings.dat'
reviews_df  = pd.read_csv(path,sep='::',header=None)
reviews_df.columns = ['user_id','item_id','rating','timestamp']
# 将rating列中的值转换为0或1，条件为大于等于4的值映射为1，其余为0
reviews_df.loc[:,'rating'] = reviews_df['rating'].map(lambda x: 1 if x >= 4 else 0)

# ID重新索引
def build_map(df, col_name):
    """
    为DataFrame中指定列的唯一值构建映射字典。
    
    参数:
    df: pandas.DataFrame
        包含待处理列的DataFrame。
    col_name: str
        df中需要其唯一值被映射到新索引的列名。
        
    返回:
    字典, 列表
        返回一个元组，包含一个字典和一个列表。字典映射原始唯一值到它们的新索引，
        列表则包含排序后的原始唯一值。
    """
    # 对指定列的唯一值进行排序并转换为列表
    key = sorted(df[col_name].unique().tolist())
    # 将唯一值映射到从1开始的相应索引号
    m = dict(zip(key, range(1, len(key) + 1)))
    # 用对应索引号替换指定列中的值
    df.loc[:, col_name] = df[col_name].map(lambda x: m[x])
    # 返回映射字典和唯一值列表
    return m, key


# 根据指定的列名，构建一个映射表，将列中的唯一值映射到一个连续的整数ID
# 这里使用'user_id'列来创建用户ID到整数ID的映射，以便在后续处理中更高效地引用用户
uid_map, uid_key = build_map(reviews_df, 'user_id')


path = './movies.dat'
meta_df  = pd.read_csv(path,sep='::',header=None)
meta_df.columns = ['item_id','title','genres']
# 选择保留'item_id'和'genres'列，去除'title'列
meta_df = meta_df[['item_id', 'genres']]
# 将'genres'列中的第一个类型提取出来作为新的'genres'值
meta_df.loc[:,'genres'] = meta_df['genres'].map(lambda x: x.split('|')[0])

vid_map, vid_key = build_map(meta_df, 'item_id')
cat_map, cat_key = build_map(meta_df, 'genres')

# 统计数据集的元素数量
user_count, item_count, cate_count, example_count =    len(uid_map), len(vid_map), len(cat_map), reviews_df.shape[0]
print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
      (user_count, item_count, cate_count, example_count))

# 根据item_id对元数据DataFrame进行排序，确保后续处理的顺序一致性
meta_df = meta_df.sort_values('item_id')
# 重置索引，去除旧的索引，以便于后续操作
meta_df = meta_df.reset_index(drop=True)

# 使用vid_map将reviews_df中的item_id转换为新的标识符
reviews_df['item_id'] = reviews_df['item_id'].map(lambda x: vid_map[x])
# 根据user_id和timestamp对评论数据进行排序，确保时间顺序的正确性
reviews_df = reviews_df.sort_values(['user_id', 'timestamp'])
# 重置索引，去除旧的索引，以便于后续操作
reviews_df = reviews_df.reset_index(drop=True)

# 通过元数据DataFrame的genres列生成类别列表，用于后续的分类处理
cate_list = [meta_df['genres'][i] for i in range(len(vid_map))]
# 将类别列表转换为numpy数组，并指定数据类型为int32，以优化计算性能
cate_list = np.array(cate_list, dtype=np.int32)

# 在类别数组的开始位置插入0，用于表示未分类或未知的类别
cate_list = np.insert(cate_list, 0, 0)

# 使用pickle模块以二进制写模式打开文件remap.pkl，用于存储数据框架和列表
with open('remap.pkl', 'wb') as f:
    # 将reviews_df数据框架dump到文件中，使用最高的协议版本
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)
    # 将cate_list列表dump到同一文件中，同样使用最高的协议版本
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    # 将用户数、物品数、类别数和样本数的元组dump到文件中
    pickle.dump((user_count, item_count, cate_count, example_count),
              f, pickle.HIGHEST_PROTOCOL)
    # 将vid_key、cat_key和uid_key的元组dump到文件中
    pickle.dump((vid_key, cat_key, uid_key), f, pickle.HIGHEST_PROTOCOL)

# 初始化正向和负向评价计数器
pos_cnt, neg_cnt = 0, 0

# 遍历每个用户的评论历史
for userId, hist in reviews_df.groupby('user_id'):
    # 获取用户评论的电影列表
    movie_list = hist['item_id'].tolist()
    # 获取用户评论的评分列表
    label_list = hist['rating'].tolist()

    # 累加正向评价数量
    pos_cnt += sum(label_list)
    # 累加负向评价数量
    neg_cnt += len(label_list) - sum(label_list)

# 计算训练集和测试集的时间截点
train_hist_time, test_time = list(np.quantile(reviews_df.timestamp, [0.80, 0.90]))
# 根据时间截点划分训练集和测试集
train_df = reviews_df[reviews_df.timestamp <= test_time]
test_df = reviews_df[reviews_df.timestamp > test_time]

# 检查划分后的数据集是否与原始数据集总和相等，确保划分正确
if train_df.shape[0]+test_df.shape[0] == reviews_df.shape[0]:
    print("Split Correct!")
else:
    print("Split Error!")

# 初始化用户训练数据集，重置索引以方便后续操作
user_train_df = train_df
user_train_df = user_train_df.reset_index(drop=True)

# 对物品训练数据集按物品ID和时间戳排序，重置索引以方便后续操作
item_train_df = train_df.sort_values(['item_id', 'timestamp'])
item_train_df = item_train_df.reset_index(drop=True)

# 根据训练历史时间筛选训练历史数据集
train_hist_df = reviews_df[reviews_df.timestamp <= train_hist_time]

# 筛选出评分不为0的训练历史数据集，用于正面反馈数据集
pos_train_hist_df = train_hist_df.drop(train_hist_df[train_hist_df['rating']==0].index)
# 筛选出评分不为0的训练数据集，用于正面反馈数据集
pos_train_df = train_df.drop(train_df[train_df['rating']==0].index)

# 定义最近交互记录的长度
recent_len = 20

# 构建用户正面反馈历史物品列表字典
pos_user_train_hist_dict = {}
recent_user_train_hist_dict = {}
for user_id, hist in pos_train_hist_df.groupby('user_id'):
    item_list = hist['item_id'].tolist()
    pos_user_train_hist_dict[user_id] = item_list
    recent_user_train_hist_dict[user_id] = item_list[-recent_len:]

# 构建用户所有历史物品列表字典，以及最近交互物品列表字典
pos_user_train_dict = {}
recent_user_train_dict = {}
for user_id, hist in pos_train_df.groupby('user_id'):
    item_list = hist['item_id'].tolist()
    pos_user_train_dict[user_id] = item_list
    recent_user_train_dict[user_id] = item_list[-recent_len:]

# 初始化用于存储正向训练历史数据的字典
pos_item_train_hist_dict = {}
# 初始化用于存储最近训练历史数据的字典
recent_item_train_hist_dict = {}
# 遍历正向训练历史数据框，按商品ID分组
for item_id, hist in pos_train_hist_df.groupby('item_id'):
    # 将用户ID列表存储到对应的商品ID下
    user_list = hist['user_id'].tolist()
    pos_item_train_hist_dict[item_id] = user_list
    # 存储每个商品ID的最近用户访问列表
    recent_item_train_hist_dict[item_id] = user_list[-recent_len:]

# 初始化用于存储正向训练数据的字典
pos_item_train_dict = {}
# 初始化用于存储最近训练数据的字典
recent_item_train_dict = {}
# 遍历正向训练数据框，按商品ID分组
for item_id, hist in pos_train_df.groupby('item_id'):
    # 将用户ID列表存储到对应的商品ID下
    user_list = hist['user_id'].tolist()
    pos_item_train_dict[item_id] = user_list
    # 存储每个商品ID的最近用户访问列表
    recent_item_train_dict[item_id] = user_list[-recent_len:]

# 根据时间戳筛选训练评估数据框
train_eval_df = reviews_df[(reviews_df.timestamp > train_hist_time) & (reviews_df.timestamp <= test_time)]

# 初始化用于构建训练历史矩阵的行和列列表
train_hist_row = []
train_hist_col = []
# 遍历正向训练用户历史字典，收集用户-商品对
for user in list(pos_user_train_hist_dict.keys()):
    for item in pos_user_train_hist_dict[user]:
        train_hist_row.append(user)
        train_hist_col.append(item)

# 初始化训练历史矩阵的权重为1
train_hist_edge = np.ones(len(train_hist_row))
# 将行和列列表转换为数组
train_hist_row = np.array(train_hist_row)
train_hist_col = np.array(train_hist_col)
# 使用稀疏矩阵格式构建训练历史矩阵
train_hist_mat = csr_matrix((train_hist_edge, (train_hist_row, train_hist_col)), shape=(user_count+1, item_count+1))

# 初始化类别列表，用于后续处理
i_cluster_list = cate_list

# 初始化行索引和列索引列表
train_hist_ic_row = []
train_hist_ic_col = []

# 遍历聚类列表，填充行索引和列索引
for item in range(len(i_cluster_list)):
    train_hist_ic_row.append(item)
    train_hist_ic_col.append(i_cluster_list[item])

# 初始化边缘权重数组，所有值为1
train_hist_ic_edge = np.ones(len(train_hist_ic_row))

# 将行索引和列索引转换为numpy数组
train_hist_ic_row = np.array(train_hist_ic_row)
train_hist_ic_col = np.array(train_hist_ic_col)

# 构建稀疏矩阵，表示训练历史中类别出现的次数
train_hist_ic_mat = csr_matrix((train_hist_ic_edge, (train_hist_ic_row, train_hist_ic_col)), shape=(item_count+1, len(cat_map)+1))

# 计算一阶偏好矩阵，即考虑了物品流行度和类别流行度的矩阵
train_hist_u_1ord_mat = train_hist_mat*train_hist_ic_mat

# 将一阶偏好矩阵转换为密集矩阵形式
train_hist_u_1ord_mat_dense = train_hist_u_1ord_mat.todense()


from sklearn import preprocessing
from sklearn.cluster import KMeans

# 对用户一级序列表进行L2标准化处理，以便进行聚类分析
train_hist_u_1ord_mat_normalized = preprocessing.normalize(train_hist_u_1ord_mat_dense, norm='l2')

# 使用KMeans算法进行聚类，设置随机状态以保证结果可复现
cluster_fit = KMeans(n_clusters=20, random_state=0).fit(train_hist_u_1ord_mat_normalized)

# 将聚类结果保存到文件中
with open('uc_cluster_kmeans.pkl', 'wb') as f:
    pickle.dump(cluster_fit.labels_, f, pickle.HIGHEST_PROTOCOL) # uid, iid, time(sorted)

# 从文件中加载聚类结果
with open('./uc_cluster_kmeans.pkl', 'rb') as f:
    u_cluster_list = pickle.load(f, encoding='latin1')

# 初始化用户聚类的行和列索引列表
train_hist_uc_row = []
train_hist_uc_col = []

# 根据用户聚类结果填充行和列索引
for user in range(len(u_cluster_list)):
    train_hist_uc_row.append(user)
    train_hist_uc_col.append(u_cluster_list[user])

# 创建用户聚类的稀疏矩阵
train_hist_uc_edge = np.ones(len(train_hist_uc_row))
train_hist_uc_row = np.array(train_hist_uc_row)
train_hist_uc_col = np.array(train_hist_uc_col)
train_hist_uc_mat = csr_matrix((train_hist_uc_edge, (train_hist_uc_row, train_hist_uc_col)), shape=(user_count+1, len(set(u_cluster_list))))

# 计算用户二级序列表和物品二级序列表
train_hist_u_2ord_mat = train_hist_mat*(train_hist_mat.T*train_hist_uc_mat)
train_hist_i_2ord_mat = train_hist_mat.T*(train_hist_mat*train_hist_ic_mat)

# 将二级序列表转换为密集矩阵
train_hist_u_2ord_mat_dense = train_hist_u_2ord_mat.todense()
train_hist_i_2ord_mat_dense = train_hist_i_2ord_mat.todense()

# 计算用户一级序列表和物品一级序列表
train_hist_u_1ord_mat = train_hist_mat*train_hist_ic_mat
train_hist_i_1ord_mat = train_hist_mat.T*train_hist_uc_mat

# 将一级序列表转换为密集矩阵
train_hist_u_1ord_mat_dense = train_hist_u_1ord_mat.todense()
train_hist_i_1ord_mat_dense = train_hist_i_1ord_mat.todense()

# 将矩阵转换为数组形式方便后续处理
train_hist_u_1ord_mat_dense_arr = train_hist_u_1ord_mat_dense.A
train_hist_u_2ord_mat_dense_arr = train_hist_u_2ord_mat_dense.A
train_hist_i_1ord_mat_dense_arr = train_hist_i_1ord_mat_dense.A
train_hist_i_2ord_mat_dense_arr = train_hist_i_2ord_mat_dense.A

# 重置训练集评估数据框的索引
train_eval_df = train_eval_df.reset_index(drop=True)

# 初始化训练数据列表
train_data = []

# 遍历训练评估数据框的每一行
for idx, row in train_eval_df.iterrows():
    # 每处理100000行数据打印一次进度
    if idx % 100000 == 0:
        print("now have processed %d"%idx)
    # 提取当前行的用户和物品ID
    now_user = row[0]
    now_item = row[1]
    # 如果用户或物品在训练历史字典中不存在，则跳过当前行
    if (now_user not in pos_user_train_hist_dict.keys()) or (now_item not in pos_item_train_hist_dict.keys()):
        continue
    # 提取当前行的标签（用户对物品的评分）
    now_label = row[2]
    # 提取用户的一阶和二阶历史行为序列
    now_user_1hop = train_hist_u_1ord_mat_dense_arr[now_user]
    now_user_2hop = train_hist_u_2ord_mat_dense_arr[now_user]
    # 提取物品的一阶和二阶历史行为序列
    now_item_1hop = train_hist_i_1ord_mat_dense_arr[now_item]
    now_item_2hop = train_hist_i_2ord_mat_dense_arr[now_item]
    # 初始化用户和物品的最近行为序列列表
    user_recent = []
    user_recent.extend(recent_user_train_hist_dict[now_user])
    # 如果最近行为序列长度不足，用0进行填充
    if len(user_recent) < recent_len:
        pad = [0 for i in range(recent_len-len(user_recent))]
        user_recent.extend(pad)
    item_recent = []
    item_recent.extend(recent_item_train_hist_dict[now_item])
    if len(item_recent) < recent_len:
        pad = [0 for i in range(recent_len-len(item_recent))]
        item_recent.extend(pad)
    # 将用户ID、用户行为序列、物品ID、物品行为序列和标签合并为一个序列
    # 这个序列用于表示一个训练样本
    now_train_seq = np.concatenate([np.array([now_user]), now_user_1hop, now_user_2hop, user_recent, np.array([now_item]), now_item_1hop, now_item_2hop, item_recent, np.array([now_label])], axis=0)
    # 将当前样本序列添加到训练数据列表中
    train_data.append(now_train_seq)


# 将训练数据转换为NumPy数组，以便进行后续处理
train_data = np.array(train_data)

# 初始化用户-物品交互的训练集行和列的列表
train_row = []
train_col = []
# 遍历训练数据中的用户和物品，填充训练集的行和列
for user in list(pos_user_train_dict.keys()):
    for item in pos_user_train_dict[user]:
        train_row.append(user)
        train_col.append(item)

# 初始化训练集矩阵的边缘值为1，用于构造稀疏矩阵
train_edge = np.ones(len(train_row))
# 将行和列转换为NumPy数组，并使用csr矩阵格式构造训练集矩阵
train_row = np.array(train_row)
train_col = np.array(train_col)
train_mat = csr_matrix((train_edge, (train_row, train_col)), shape=(user_count+1, item_count+1))

# 初始化用户聚类的行和列的列表
train_uc_row = []
train_uc_col = []
# 遍历用户聚类列表，填充用户聚类矩阵的行和列
for user in range(len(u_cluster_list)):
    train_uc_row.append(user)
    train_uc_col.append(u_cluster_list[user])

# 初始化用户聚类矩阵的边缘值为1
train_uc_edge = np.ones(len(train_uc_row))
# 将行和列转换为NumPy数组，并使用csr矩阵格式构造用户聚类矩阵
train_uc_row = np.array(train_uc_row)
train_uc_col = np.array(train_uc_col)

# 初始化物品聚类的行和列的列表
train_ic_row = []
train_ic_col = []
# 遍历物品聚类列表，填充物品聚类矩阵的行和列
for item in range(len(i_cluster_list)):
    train_ic_row.append(item)
    train_ic_col.append(i_cluster_list[item])

# 初始化物品聚类矩阵的边缘值为1
train_ic_edge = np.ones(len(train_ic_row))
# 将行和列转换为NumPy数组，并使用csr矩阵格式构造物品聚类矩阵
train_ic_row = np.array(train_ic_row)
train_ic_col = np.array(train_ic_col)

train_uc_mat = csr_matrix((train_uc_edge, (train_uc_row, train_uc_col)), shape=(user_count+1, len(set(u_cluster_list))))
train_ic_mat = csr_matrix((train_ic_edge, (train_ic_row, train_ic_col)), shape=(item_count+1, len(cat_map)+1))

# 计算用户二阶相似度矩阵
train_u_2ord_mat = train_mat*(train_mat.T*train_uc_mat)
# 计算物品二阶相似度矩阵
train_i_2ord_mat = train_mat.T*(train_mat*train_ic_mat)

# 将稀疏矩阵转换为密集矩阵，以便进行后续计算
train_u_2ord_mat_dense = train_u_2ord_mat.todense()
train_i_2ord_mat_dense = train_i_2ord_mat.todense()

# 计算用户一阶相似度矩阵
train_u_1ord_mat = train_mat*train_ic_mat
# 计算物品一阶相似度矩阵
train_i_1ord_mat = train_mat.T*train_uc_mat

# 将稀疏矩阵转换为密集矩阵，以便进行后续计算
train_u_1ord_mat_dense = train_u_1ord_mat.todense()
train_i_1ord_mat_dense = train_i_1ord_mat.todense()

# 将密集矩阵转换为数组形式，以方便后续处理
train_u_1ord_mat_dense_arr = train_u_1ord_mat_dense.A
train_u_2ord_mat_dense_arr = train_u_2ord_mat_dense.A
train_i_1ord_mat_dense_arr = train_i_1ord_mat_dense.A
train_i_2ord_mat_dense_arr = train_i_2ord_mat_dense.A


# 重置测试数据集的索引，以便后续处理
test_df = test_df.reset_index(drop=True)

# 初始化测试数据列表，用于存储处理后的测试数据样本
test_data = []

# 遍历测试数据集的每一行，行索引为idx，行数据为row
for idx, row in test_df.iterrows():
    # 每处理100000条数据打印一次进度
    if idx % 100000 == 0:
        print("now have processed %d"%idx)
    # 提取当前处理的用户和物品ID
    now_user = row[0]
    now_item = row[1]
    # 如果当前用户或物品在训练集中不存在，则跳过当前数据样本
    if (now_user not in pos_user_train_dict.keys()) or (now_item not in pos_item_train_dict.keys()):
        continue
    # 提取当前数据样本的标签（评价分数）
    now_label = row[2]
    # 根据用户ID和物品ID提取用户的一阶和二阶关联信息
    now_user_1hop = train_u_1ord_mat_dense_arr[now_user]
    now_user_2hop = train_u_2ord_mat_dense_arr[now_user]
    now_item_1hop = train_i_1ord_mat_dense_arr[now_item]
    now_item_2hop = train_i_2ord_mat_dense_arr[now_item]
    # 根据用户ID和物品ID提取用户的最近评价记录
    user_recent = []
    user_recent.extend(recent_user_train_dict[now_user])
    # 如果用户的最近评价记录不足recent_len，则用0填充至recent_len长度
    if len(user_recent) < recent_len:
        pad = [0 for i in range(recent_len-len(user_recent))]
        user_recent.extend(pad)
    # 根据用户ID和物品ID提取物品的最近被评价记录
    item_recent = []
    item_recent.extend(recent_item_train_dict[now_item])
    # 如果物品的最近被评价记录不足recent_len，则用0填充至recent_len长度
    if len(item_recent) < recent_len:
        pad = [0 for i in range(recent_len-len(item_recent))]
        item_recent.extend(pad)
    # 将用户ID、用户一阶关联、用户二阶关联、用户最近评价记录、物品ID、物品一阶关联、物品二阶关联、物品最近被评价记录和标签拼接为一个序列
    # 这个序列代表了一个完整的测试数据样本
    now_test_seq = np.concatenate([np.array([now_user]), now_user_1hop, now_user_2hop, user_recent, np.array([now_item]), now_item_1hop, now_item_2hop, item_recent, np.array([now_label])], axis=0)
    # 将当前处理好的测试数据样本添加到测试数据列表中
    test_data.append(now_test_seq)

# 将测试数据列表转换为numpy数组，以便后续处理和存储
test_data = np.array(test_data)
# 打印测试数据的形状，用于确认数据规模和结构
test_data.shape

# 计算用户聚类的数量和物品聚类的数量
u_cluster_num = len(set(u_cluster_list))
i_cluster_num = len(cat_map)

# 使用pickle将训练数据、测试数据、类别列表、用户聚类列表、物品聚类列表以及相关计数保存到文件中
# 这便于后续模型的训练和评估
with open('../ml-10m.pkl', 'wb') as f:
    pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(u_cluster_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(i_cluster_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count, u_cluster_num, i_cluster_num), f, pickle.HIGHEST_PROTOCOL)