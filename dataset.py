#!/usr/bin/env python
# coding=utf-8
import os

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset



class MyDataset(Dataset):
    def __init__(self, data_path, max_length_day=5, train=True):
        super(MyDataset, self).__init__()

        df = pd.read_csv(data_path)
        # 创建 k 个矩阵
        k = df['ema_CALM'].notnull().sum()

        matrices = []
        new_col_data = []
        a = 0

        for i in range(len(df)):
            if pd.isnull(df.loc[i, 'ema_CALM']):
                new_col_data.append(a)
            else:
                new_col_data.append(a)
                a = a + 1

        df['index'] = new_col_data

        # 特征
        col_1 = df.iloc[:, 3:51]
        col_2 = df.iloc[:, 65:]

        # ema总分
        score = df.iloc[:, 64].dropna()
        # # neg总分
        # score = df.iloc[:, 61]
        # # pos总分
        # score = df.iloc[:, 62]
        score = score.reset_index(drop=True)

        feature = pd.concat([col_1, col_2], axis=1)
        
        grouped = feature.groupby('index')
        

        for group_name, group_df in grouped:
            list1 = []
            group_df.fillna(group_df.mean(), inplace=True)  # 填充均值
            tmp = np.array(group_df.drop('index', axis=1))



            for i in range(len(tmp)):
                a = tmp[i].reshape(18, 4)

                list1.append(a.tolist())
                
            if len(matrices) < k:
                matrices.append(list1)

        self.max_length_day = max_length_day

        if train:
            self.matrices = matrices[:int(len(matrices)*0.7)]
            self.score = score.tolist()[:int(len(matrices)*0.7)]
        else:
            self.matrices = matrices[int(len(matrices) * 0.7):]
            self.score = score.tolist()[int(len(matrices) * 0.7):]



    def __len__(self):
        return len(self.score)

    def __getitem__(self, item):


        score = self.score[item]
        data = self.matrices[item]

        if len(data) < self.max_length_day:
            extended_sentences = [[[-1 for _ in range(4)] for _ in range(18)] for _ in
                                  range(self.max_length_day - len(data))]
            data.extend(extended_sentences)

        data = data[-self.max_length_day:]

        return torch.tensor(data,dtype=torch.float32), torch.tensor([score],dtype=torch.float32)

class ParkDataset(Dataset):
    def __init__(self, data_path, max_length_day=3, train=True):
        super(ParkDataset, self).__init__()

        df = pd.read_csv(data_path)
        k = df['total_UPDRS'].notnull().sum()//3

        matrices = []
        new_col_data = []
        a = 0

        for i in range(1,len(df)+1):
            if i%3!=0:
                new_col_data.append(a)
            else:
                new_col_data.append(a)
                a = a + 1

        df['index'] = new_col_data


        score = df.iloc[:, 5].dropna()[::3][1:]




        score = score.reset_index(drop=True)
        print("score",score.mean())

        feature = df.iloc[:, 6:]


        grouped = feature.groupby('index')
        for group_name, group_df in grouped:
            list1 = []
            group_df.fillna(group_df.mean(), inplace=True)  # 填充均值
            tmp = np.array(group_df.drop('index', axis=1))



            for i in range(len(tmp)):
                a = tmp[i].reshape(16, 1)

                list1.append(a.tolist())
             
            if len(matrices) < k:
                matrices.append(list1)

        self.max_length_day = max_length_day

        if train:
            self.matrices = matrices[:int(len(matrices)*0.7)]
            self.score = score.tolist()[:int(len(matrices)*0.7)]

        else:
            self.matrices = matrices[int(len(matrices) * 0.7):]
            self.score = score.tolist()[int(len(matrices) * 0.7):]



    def __len__(self):
        return len(self.score)

    def __getitem__(self, item):


        score = self.score[item]
        data = self.matrices[item]

        if len(data) < self.max_length_day:
            extended_sentences = [[[-1 for _ in range(1)] for _ in range(16)] for _ in
                                  range(self.max_length_day - len(data))]
            data.extend(extended_sentences)
        # 从后往前截取
        data = data[-self.max_length_day:]

        return torch.tensor(data,dtype=torch.float32), torch.tensor([score],dtype=torch.float32)


class ppmiDataset(Dataset):
    def __init__(self, data_path, max_length_day=6, train=True):
        super(ppmiDataset, self).__init__()
        print('path',data_path)
        df = pd.read_csv(data_path)
        # 创建 k 个矩阵
        k = df['score'].notnull().sum() // 6

        matrices = []
        new_col_data = []
        a = 0

        for i in range(1, len(df) + 1):
            if i % 6 != 0:
                new_col_data.append(a)
            else:
                new_col_data.append(a)
                a = a + 1

        df['index'] = new_col_data


        score = df.iloc[:, 5].dropna()[::6][1:]

        score = score.reset_index(drop=True)
        
        print("score", score.mean())

        feature = df.iloc[:, 6:]

        grouped = feature.groupby('index')
        for group_name, group_df in grouped:
            list1 = []
            group_df.fillna(group_df.mean(), inplace=True)  # 填充均值
            tmp = np.array(group_df.drop('index', axis=1))

            for i in range(len(tmp)):
                a = tmp[i].reshape(17, 1)

                list1.append(a.tolist())

            if len(matrices) < k:
                matrices.append(list1)

        self.max_length_day = max_length_day

        if train:
            self.matrices = matrices[:int(len(matrices) * 0.7)]
            self.score = score.tolist()[:int(len(matrices) * 0.7)]

        else:
            self.matrices = matrices[int(len(matrices) * 0.7):]
            self.score = score.tolist()[int(len(matrices) * 0.7):]

    def __len__(self):
        return len(self.score)

    def __getitem__(self, item):

        score = self.score[item]
        data = self.matrices[item]

        if len(data) < self.max_length_day:
            extended_sentences = [[[-1 for _ in range(1)] for _ in range(17)] for _ in
                                  range(self.max_length_day - len(data))]
            data.extend(extended_sentences)
        # 从后往前截取
        data = data[-self.max_length_day:]

        return torch.tensor(data, dtype=torch.float32), torch.tensor([score], dtype=torch.float32)

