import glob
import logging
import os
import random
import warnings
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import torch
from data.dataset import FutsData
from torch.utils.data import DataLoader, Dataset
from utils.tools import StandardScaler

warnings.filterwarnings('ignore')
logger = logging.getLogger("__name__")

class Dataset_MTS(Dataset):
    def __init__(self, root_path, data_path='ETTh1.csv', flag='train', size=None, 
                  data_split = [0.7, 0.1, 0.2], scale=True, scale_statistic=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.in_len = size[0]
        self.out_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.scale = scale
        #self.inverse = inverse
        
        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.scale_statistic = scale_statistic
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        if (self.data_split[0] > 1):
            train_num = self.data_split[0]; val_num = self.data_split[1]; test_num = self.data_split[2];
        else:
            train_num = int(len(df_raw)*self.data_split[0]); 
            test_num = int(len(df_raw)*self.data_split[2])
            val_num = len(df_raw) - train_num - test_num; 
        border1s = [0, train_num - self.in_len, train_num + val_num - self.in_len]
        border2s = [train_num, train_num+val_num, train_num + val_num + test_num]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            if self.scale_statistic is None:
                self.scaler = StandardScaler()
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
            else:
                self.scaler = StandardScaler(mean = self.scale_statistic['mean'], std = self.scale_statistic['std'])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.in_len
        r_begin = s_end
        r_end = r_begin + self.out_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y
    
    def __len__(self):
        return len(self.data_x) - self.in_len- self.out_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

# TODO: check if data normalization is needed.
class Dataset_Futs(Dataset):
    def __init__(self, root_dir: str, pattern: str, in_len: int, out_len: int):
        super(Dataset_Futs, self).__init__()
        self.out_len = out_len
        self.data = FutsData(root_dir, pattern, in_len, out_len)
        self.IDs = self.data.all_IDs # list of data IDs, but also mapping between integer index and ID
        self.feature_df = self.data.feature_df
        self.labels_df = self.data.labels_df

    def __getitem__(self, ind):
        """
        For a given integer index, returns the corresponding (seq_length, feat_dim) array and a noise mask of same shape
        Args:
            ind: integer index of sample in dataset
        Returns:
            X: (seq_length, feat_dim) tensor of the multivariate time series corresponding to a sample
            y: (num_labels,) tensor of labels (num_labels > 1 for multi-task models) for each sample
            ID: ID of sample
        """
        X = self.feature_df.loc[self.IDs[ind][0] : self.IDs[ind][1]].values  # (seq_length, feat_dim) array
        y = self.labels_df.loc[self.IDs[ind][1]].values  # (num_labels,) array

        return torch.from_numpy(X), torch.from_numpy(y)#, self.IDs[ind][0]

    def __len__(self):
        return len(self.IDs)

class Dataset_Futs_Pretrain(Dataset):
    def __init__(self, root_dir: str, pattern: str, out_len: int):
        super(Dataset_Futs_Pretrain, self).__init__()
        self.out_len = out_len
        self.data = FutsData(root_dir, pattern, out_len) 
        self.IDs = self.data.all_IDs # list of data IDs, but also mapping between integer index and ID
        self.feature_df = self.data.feature_df
        self.labels_df = self.data.labels_df

    def __getitem__(self, ind):
        """
        For a given integer index, returns the corresponding (seq_length, feat_dim) array and a noise mask of same shape
        Args:
            ind: integer index of sample in dataset
        Returns:
            X: (seq_length, feat_dim) tensor of the multivariate time series corresponding to a sample
            y: (num_labels,) tensor of labels (num_labels > 1 for multi-task models) for each sample
            ID: ID of sample
        """
        X = self.feature_df.loc[self.IDs[ind][0] : self.IDs[ind][1]].values  # (feature_seq_length, feat_dim) array
        y = self.feature_df.loc[self.IDs[ind][1]+1 : self.IDs[ind][1] + self.out_len].values  # (pred_seq_len, feat_dim) array

        return torch.from_numpy(X), torch.from_numpy(y)#, self.IDs[ind][0]

    def __len__(self):
        return len(self.IDs)
