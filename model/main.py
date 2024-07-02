#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import DataLoader
import torch.nn as nn
import sklearn.metrics as metrics
from gauc_metric import cal_group_auc
import tqdm
import matplotlib.pyplot as plt
from utils import DatasetBuilder
from macgnn import MacGNN
import argparse
import random
import os


############################get-args####################################
def parse_args():
    parser = argparse.ArgumentParser(description='model trainning')
    parser.add_argument('--dataset_name', type=str, default='ml-10m', choices=['ml-10m', 'elec', 'kuairec'])
    parser.add_argument('--model_name', default='cgi')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--early_epoch', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--sample_size', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=10)
    parser.add_argument('--save_path', default='chkpt')
    parser.add_argument('--record_path', default='record')
    parser.add_argument('--use_gpu', default=True, help='Whether to use CUDA')
    parser.add_argument('--cuda_id', type=int, default=0, help='CUDA id')
    parser.add_argument('--seq_len', type=int, default=100, help='user hist len')
    parser.add_argument('--short_len', type=int, default=20, help='user hist len')
    parser.add_argument('--recent_len', type=int, default=20, help='user hist len')
    parser.add_argument('--runs', type=int, default=1, help='model runs')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--test_iter', type=int, default=50)

    return parser.parse_args()


def set_seed(seed, cuda):
    print('Set Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopper(object):
    """
    早期停止类，用于在训练过程中监控模型性能，并根据设定的条件决定是否停止训练。
    """
    
    def __init__(self, num_trials, save_path):
        """
        初始化早期停止器。
        
        设置最大尝试次数、当前尝试次数、最佳AUC和最佳对数损失的初始值，
        以及模型最佳性能的保存路径。
        """
        self.num_trials = num_trials  # 允许的最大尝试次数
        self.trial_counter = 0  # 当前的尝试次数
        self.best_auc = 0.0  # 最好的AUC值，初始化为0.0
        self.best_logloss = 1000000  # 最好的对数损失，初始化为一个非常大的数
        self.save_path = save_path  # 模型最佳性能的保存路径

    def is_continuable(self, model, auc, log_loss):
        """
        判断模型训练是否应该继续。

        这个方法根据当前模型的AUC和log_loss与之前记录的最佳值进行比较，来决定是否应该继续训练模型。
        如果当前模型的AUC超过了之前的最佳AUC，则更新最佳AUC和log_loss，并保存当前模型，然后重置尝试次数并返回True。
        如果当前模型的AUC没有超过最佳AUC，但尝试次数还没有达到设定的最大尝试次数，则增加尝试次数并返回True。
        如果尝试次数已经达到或超过了最大尝试次数，则返回False，表示应该停止训练。

        参数:
        model: 当前的模型。
        auc: 当前模型的AUC值。
        log_loss: 当前模型的log_loss值。

        返回:
        True如果应该继续训练，否则返回False。
        """
        # 如果当前AUC超过最佳AUC，则更新最佳log_loss、最佳AUC和尝试次数，并保存当前模型，然后返回True
        if auc > self.best_auc:
            self.best_logloss = log_loss
            self.best_auc = auc
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        # 如果当前AUC未超过最佳AUC，但尝试次数还没达到上限，则增加尝试次数并返回True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        # 如果尝试次数已经达到或超过上限，则返回False，表示停止训练
        else:
            return False


# model training
def train(model, optimizer, train_data_loader, test_data_loader, criterion, device, early_stopper, epochs=10, test_iter=50, log_interval=20):
    """
    训练模型。
    
    参数:
    model -- 要训练的模型。
    optimizer -- 优化器，用于更新模型参数。
    train_data_loader -- 训练数据的加载器。
    test_data_loader -- 测试数据的加载器。
    criterion -- 损失函数，用于计算模型预测与目标值之间的差异。
    device -- 模型训练所在的设备（CPU或GPU）。
    early_stopper -- 早停策略对象，用于根据验证集性能决定是否停止训练。
    epochs -- 训练的轮数，默认为10。
    test_iter -- 每隔多少迭代进行一次测试，默认为50。
    log_interval -- 日志输出间隔，默认为20。
    
    返回:
    无
    """
    total_loss = 0.0  # 训练过程中的总损失
    tk0 = tqdm.tqdm(train_data_loader, smoothing=0, mininterval=1.0)  # 使用tqdm显示训练进度条
    now_iter = 0  # 当前迭代次数
    break_flag = False  # 是否中断训练的标志
    
    for epo in range(epochs):
        for i, (fields, target) in enumerate(tk0):
            model.train()  # 将模型设置为训练模式
            fields, target = fields.to(device), target.to(device)  # 将数据移动到训练设备
            y = model(fields)

            loss = criterion(y, target.float())  # 计算损失

            model.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            total_loss += loss.item()  # 累加当前损失
            now_iter += 1
            
            # 每隔test_iter迭代进行一次测试
            if now_iter % test_iter == 0:
                # testing
                auc, log_losses, _ = evaluation(model, test_data_loader, device, use_gauc=False)
                # 根据早停策略检查是否应该停止训练
                if not early_stopper.is_continuable(model, auc, log_losses):
                    print(f'validation: best auc: {early_stopper.best_auc}, best logloss: {early_stopper.best_logloss}')
                    break_flag = True
                    break
            
            # 每隔log_interval迭代输出一次日志
            if (i+1) % log_interval == 0:
                # display
                tk0.set_postfix(loss=total_loss / log_interval)  # 更新进度条的损失显示
                total_loss = 0  # 重置总损失
        
        if break_flag:
            break


# 模型评估
def evaluation(model, data_loader, device, use_gauc=False):
    """
    对模型进行评估，计算ROC AUC、Log Loss以及可选的Group AUC。
    
    参数:
    model: 需要评估的模型实例。
    data_loader: 数据加载器，用于获取评估数据集。
    device: 指定模型运行的设备（如CPU或GPU）。
    use_gauc: 布尔值，表示是否计算Group AUC，默认为False。
    
    返回:
    auc: ROC AUC得分。
    log_loss: Log Loss得分。
    gauc: 如果use_gauc为True，则返回Group AUC得分，否则为None。
    """
    # 设置模型为评估模式
    model.eval()
    
    # 初始化目标标签、预测结果及用户ID列表
    targets, predicts, user_id_list = list(), list(), list()
    
    # 在不进行梯度计算的上下文中评估模型
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            # 将数据移至指定设备
            fields, target = fields.to(device), target.to(device)
            
            # 模型前向传播得到预测值
            y = model(fields)
            
            # 收集用户ID、目标标签和预测结果
            user_id_list.extend(fields[:, 0].tolist())
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    
    # 计算Group AUC，如果use_gauc为True
    gauc = None
    if use_gauc:
        gauc = cal_group_auc(targets, predicts, user_id_list)
    
    # 将目标标签和预测结果转换为numpy数组以进行评估
    targets = np.array(targets)
    predicts = np.array(predicts)
    
    # 计算并返回ROC AUC和Log Loss
    return metrics.roc_auc_score(targets, predicts), metrics.log_loss(targets, predicts), gauc
    

def write_to_file(file_path, content):
    """
    将内容写入指定的文件。如果文件夹不存在，则创建文件夹。

    参数:
    file_path (str): 文件的路径
    content (str): 要写入文件的内容
    """
    # 获取文件的目录
    directory = os.path.dirname(file_path)
    
    # 如果目录不存在，则创建目录
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 将内容写入文件
    with open(file_path, 'a') as file:
        file.write(content)

def main():
    """
    程序的主入口函数。
    解析命令行参数，初始化模型训练所需的配置和数据集，进行模型训练和评估。
    """
    
    # 解析命令行参数
    args = parse_args()
    print(args)

    # 手动设置工作目录，确保在程序运行时，当前工作目录为当前脚本所在的目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 从命令行参数中读取各种配置
    embed_dim = args.embed_dim
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    epoch = args.epoch
    trials = args.early_epoch
    batch_size = args.batch_size
    # 根据是否使用GPU和可用性决定设备
    device = torch.device("cuda:%d" % (args.cuda_id) if (torch.cuda.is_available() and args.use_gpu) else "cpu")
    save_path = args.save_path
    record_path = args.record_path
    model_name = args.model_name
    dataset_name = args.dataset_name
    seq_len = args.seq_len
    short_len = args.short_len
    recent_len = args.recent_len
    sample_size = args.sample_size

    # 设置随机种子以保证可复现性
    set_seed(args.seed, args.use_gpu)

    # 根据数据集名称加载不同数据集
    if args.dataset_name == 'ml-10m':
        with open(f'../data/{dataset_name}.pkl', 'rb') as f:
            train_set = np.array(pickle.load(f, encoding='latin1')) 
            test_set  = np.array(pickle.load(f, encoding='latin1')) 
            # # 输出内容类型和结构
            # print("Train set type:", type(train_set))
            # print("Test set type:", type(test_set))
            # print("Train set shape:", train_set.shape)
            # print("Test set shape:", test_set.shape)

            # # 如果你想查看具体内容，可以直接打印或查看某个样本
            # print("Train set example:", train_set[0])
            # print("Test set example:", test_set[0])
            cate_list = torch.tensor(pickle.load(f, encoding='latin1')).to(device)
            u_cluster_list = torch.tensor(pickle.load(f, encoding='latin1')).to(device)
            i_cluster_list = torch.tensor(pickle.load(f, encoding='latin1')).to(device)
            user_count, item_count, cate_count, u_cluster_num, i_cluster_num = pickle.load(f)
    elif args.dataset_name == 'elec':
        with open(f'../data/{dataset_name}.pkl', 'rb') as f:
            train_set = np.array(pickle.load(f, encoding='latin1'))  
            test_set = np.array(pickle.load(f, encoding='latin1')) 
            cate_list = torch.tensor(pickle.load(f, encoding='latin1')).to(device)  
            u_cluster_list = torch.tensor(pickle.load(f, encoding='latin1')).to(device)
            i_cluster_list = torch.tensor(pickle.load(f, encoding='latin1')).to(device)
            user_count, item_count, cate_count, u_cluster_num, i_cluster_num = pickle.load(f)
    elif args.dataset_name == 'kuairec':
        with open(f'../data/{dataset_name}.pkl', 'rb') as f:
            train_set = np.array(pickle.load(f, encoding='latin1'))  
            test_set = np.array(pickle.load(f, encoding='latin1'))  
            cate_list = torch.tensor(pickle.load(f, encoding='latin1')).to(device) 
            u_cluster_list = torch.tensor(pickle.load(f, encoding='latin1')).to(device)
            i_cluster_list = torch.tensor(pickle.load(f, encoding='latin1')).to(device)
            user_count, item_count, cate_count, u_cluster_num, i_cluster_num = pickle.load(f)

    # 计算数据集的大小和类别数量
    train_size = (u_cluster_num+i_cluster_num+recent_len+1+1)*2
    test_size = (u_cluster_num+i_cluster_num+recent_len+1+1)*2
    u_cluster_num -= 1

    # 定义字段维度，包括用户、物品和类别的数量
    field_dims = [user_count + 1, item_count + 1, cate_count + 1]  # idx-0 for padding

    # 使用数据集构建器构建训练集和测试集
    train_data = DatasetBuilder(data=train_set, user_count=user_count, item_count=item_count)
    test_data = DatasetBuilder(data=test_set, user_count=user_count, item_count=item_count)

    # 创建数据加载器
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)
   
    # 初始化性能指标列表
    auc_runs = []
    logloss_runs = []
    gauc_runs = []

    # 进行多次运行以收集性能指标
    for now_run in range(args.runs):
        # 如果不是单次运行，则每次运行重新设置随机种子
        if args.runs != 1:
            set_seed(now_run, args.use_gpu)

        # 打印当前运行的信息
        print("###########now run: %d##############" % now_run)
        print("use dataset: " + dataset_name)

        # 根据模型名称创建模型
        print("now model: " + model_name)
        if model_name == 'macgnn':
            model = MacGNN(field_dims=field_dims, u_group_num=u_cluster_num, i_group_num=i_cluster_num,
                        embed_dim=embed_dim, recent_len=recent_len, tau=args.tau, device=device).to(device)
        else:
            raise Exception("no model selected!")

        # 定义损失函数和优化器
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # 定义早停策略
        early_stopper = EarlyStopper(num_trials=trials, save_path=f'{model_name}_{dataset_name}.pt')

        # 训练模型
        train(model, optimizer, train_data_loader, test_data_loader, criterion, device, early_stopper, epochs=args.epoch, test_iter=args.test_iter, log_interval=20)
        
        # 加载最佳模型
        model = torch.load(f'{model_name}_{dataset_name}.pt').to(device)
        # 评估模型性能
        auc, log_losses, gauc = evaluation(model, test_data_loader, device, use_gauc=True)
        
        # 记录性能指标
        print(f'test auc: {auc}, test logloss: {log_losses}, test gauc: {gauc}')
        auc_runs.append(auc)
        logloss_runs.append(log_losses)
        gauc_runs.append(gauc)

    # 计算性能指标的平均值和标准差
    auc_mean, auc_std = np.mean(np.array(auc_runs), axis=0), np.std(np.array(auc_runs), axis=0)
    logloss_mean, logloss_std = np.mean(np.array(logloss_runs), axis=0), np.std(np.array(logloss_runs), axis=0)
    gauc_mean, gauc_std = np.mean(np.array(gauc_runs), axis=0), np.std(np.array(gauc_runs), axis=0)

    # 打印性能指标的平均值和标准差
    print("Test AUC: "+str(auc_mean)+" ± "+str(auc_std))
    print("Test GAUC: "+str(gauc_mean)+" ± "+str(gauc_std))
    print("Test Logloss: "+str(logloss_mean)+" ± "+str(logloss_std))

    # 将结果写入文件
    result_file_path = f'./result/{model_name}/{dataset_name}.txt'
    write_to_file(result_file_path, f'args: {args}\n')
    write_to_file(result_file_path, f'Test AUC: {(auc_mean*100):.2f}±{(auc_std*100):.2f}\n')
    write_to_file(result_file_path, f'Test GAUC: {(gauc_mean*100):.2f}±{(gauc_std*100):.2f}\n')
    write_to_file(result_file_path, f'Test Logloss: {(logloss_mean*100):.2f}±{(logloss_std*100):.2f}\n\n\n')

if __name__ == '__main__':
    main()