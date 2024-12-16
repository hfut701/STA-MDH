import torch
import numpy as np
import scipy.sparse as sp
# from torchvision import datasets
from collections import namedtuple
# from torchvision import datasets, transforms
import pickle as pk
import os
from dataset import MyDataset,ParkDataset,ppmiDataset
import pandas as pd
import math

def getwasserstein(m1,v1,m2,v2,mode='nosquare'):
    w=0
    bl=len(m1)
    for i in range(bl):
        tw=0
        tw+=(np.sum(np.square(m1[i].cpu().numpy()-m2[i].cpu().numpy())))
        tw+=(np.sum(np.square(np.sqrt(v1[i].cpu().numpy())- np.sqrt(v2[i].cpu().numpy()))))
        if mode=='square':
            w+=tw
        else:
            w+=math.sqrt(tw)
    return w

def get_weight_matrix1(bnmlist,bnvlist,clients):
    client_num=len(bnmlist)
    weight_m=np.zeros((client_num,client_num))
    for i in range(client_num):
        for j in range(client_num):
            if i==j:
                weight_m[i,j]=0
            else:
                tmp=getwasserstein(bnmlist[i],bnvlist[i],bnmlist[j],bnvlist[j])
                if tmp==0:
                    weight_m[i,j]=100000000000000
                else:
                    weight_m[i,j]=1/tmp
    weight_s=np.sum(weight_m,axis=1)
    weight_s=np.repeat(weight_s,clients).reshape((clients,clients))
    # args.model_momentum = 0.5
    weight_m=(weight_m/weight_s)*(1-0.5)
    for i in range(client_num):
        weight_m[i,i]=0.5
    return weight_m


def load_park(args):
    folder_path = r"./park"

    # folder_path = r"./park_all"
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    train_batches = []
    test_batches = []
    avg_score =[]
    test_set = []
    index =0
    # # hebing
    # df_list = []
    # for filename in file_paths:
    #     df = pd.read_csv(filename, index_col=None, header=0)
    #     df_list.append(df)
    #
    # # 将所有 DataFrame 合并为一个 DataFrame
    # merged_df = pd.concat(df_list, axis=0, ignore_index=True)
    #
    # # 将合并后的 DataFrame 写入一个新的 csv 文件
    # merged_df.to_csv('./user_data/user.csv', index=False)

    for file_path in file_paths:
        print(file_path)
        print("client",index)
        index = index +1
        train_dataset = ParkDataset(file_path, train=True)

        test_dataset = ParkDataset(file_path, train=False)



        df = pd.read_csv(file_path)
        col_mean = df['total_UPDRS'].mean()

        avg_score.append(int(col_mean))
        train_batches.append(Batches(train_dataset,
                                    args.batch_size, shuffle=False, device=args.device, drop_last=False))

        test_batches.append(Batches(test_dataset,
                                    args.batch_size, shuffle=False, device=args.device, drop_last=False))

    
    std = np.array(avg_score).std()

    A = np.zeros((args.clients, args.clients))

    # prepare A
    link_list = []
    for i in range(len(avg_score)):
        for j in range(i,len(avg_score)):
            if avg_score[i]-avg_score[j]< std:
                link_list.append([i, j])

    link_sample = list(range(len(link_list)))

    link_idx = np.random.choice(link_sample, int(args.edge_frac * len(link_list)), replace=False)

    for idx in link_idx:
        # A[link_list[idx][0], link_list[idx][1]] = A[link_list[idx][0], link_list[idx][1]] + 1
        A[link_list[idx][0], link_list[idx][1]] = 1
        A[link_list[idx][1], link_list[idx][0]] = 1


    overall_tbatches = Batches(test_set, args.batch_size, shuffle=False,
                               device=args.device, drop_last=False)

    return train_batches, test_batches, A, overall_tbatches


def load_ppmi(args):
    folder_path = r"./ppmi"


    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    train_batches = []
    test_batches = []
    avg_score = []
    test_set = []
    index = 0

    score_avg ={}

    for file_path in file_paths:




        train_dataset = ppmiDataset(file_path, train=True)

        test_dataset = ppmiDataset(file_path, train=False)

        df = pd.read_csv(file_path)
        col_mean = df['score'].mean()

        avg_score.append(int(col_mean))
        train_batches.append(Batches(train_dataset,
                                     args.batch_size, shuffle=False, device=args.device, drop_last=False))

        test_batches.append(Batches(test_dataset,
                                    args.batch_size, shuffle=False, device=args.device, drop_last=False))

        score_avg[index] = col_mean
        index = index + 1


    std = np.array(avg_score).std()

    A = np.zeros((args.clients, args.clients))

    # prepare A
    link_list = []
    for i in range(len(avg_score)):
        for j in range(i, len(avg_score)):
            if avg_score[i] - avg_score[j] < std:
                link_list.append([i, j])

    link_sample = list(range(len(link_list)))

    link_idx = np.random.choice(link_sample, int(args.edge_frac * len(link_list)), replace=False)

    for idx in link_idx:
        A[link_list[idx][0], link_list[idx][1]] = 1
        A[link_list[idx][1], link_list[idx][0]] = 1

    overall_tbatches = Batches(test_set, args.batch_size, shuffle=False,
                               device=args.device, drop_last=False)

    return train_batches, test_batches, A, overall_tbatches

def load_crosscheck(args):
    folder_path = r"./user_data"
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    train_batches = []
    test_batches = []
    avg_score =[]
    test_set = []
    index =0


    for file_path in file_paths:

        index = index +1
        train_dataset = MyDataset(file_path, train=True)

        test_dataset = MyDataset(file_path, train=False)



        df = pd.read_csv(file_path)
        col_mean = df['ema_score'].mean()
        avg_score.append(int(col_mean))
        train_batches.append(Batches(train_dataset,
                                    args.batch_size, shuffle=False, device=args.device, drop_last=False))
        
        test_batches.append(Batches(test_dataset,
                                    args.batch_size, shuffle=False, device=args.device, drop_last=False))


    std = np.array(avg_score).std()

    A = np.zeros((args.clients, args.clients))

    # prepare A
    link_list = []
    for i in range(len(avg_score)):
        for j in range(i,len(avg_score)):
            if avg_score[i]-avg_score[j]< std:
                link_list.append([i, j])

    link_sample = list(range(len(link_list)))

    link_idx = np.random.choice(link_sample, int(args.edge_frac * len(link_list)), replace=False)

    for idx in link_idx:
        A[link_list[idx][0], link_list[idx][1]] = 1
        A[link_list[idx][1], link_list[idx][0]] = 1


    overall_tbatches = Batches(test_set, args.batch_size, shuffle=False,
                               device=args.device, drop_last=False)

    return train_batches, test_batches, A, overall_tbatches




class Batches():
    def __init__(self, dataset, batch_size, shuffle, device, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.device = device
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle,
            drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        if self.device is not None:
            return ({'input': x.to(self.device), 'target': y.to(self.device).long()} for (x, y) in self.dataloader)
        else:
            return ({'input': x, 'target': y.long()} for (x, y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)


