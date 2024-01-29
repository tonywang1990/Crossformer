import glob
import logging
import os
import random
import warnings
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from utils.tools import StandardScaler

warnings.filterwarnings('ignore')
logger = logging.getLogger("__main__")


class BaseData(object):
    def set_num_processes(self, n_proc):
        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())

class FutsData(BaseData):
    """
    Dataset class for Machine dataset.
    Attributes:
        all_df: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(
        self,
        root_dir,
        split,
        pattern,
        file_list=None,
        n_proc=1,
        limit_size=None,
        config=None,
    ):
        self.max_seq_len = 1024
        self.lookahead = 40
        # process features
        feature_df = self.get_feature_data(os.path.join(root_dir, pattern.format(split=split)))
        num_rows = feature_df.shape[0]
        # process labels
        labels_df = self.get_label_data(feature_df, lookahead=self.lookahead, seq_len=self.max_seq_len)
        # all_IDs uses a compressed representation: i-th position in all_ID maps to (start, end) of the feature_df and start of the label_df.
        self.all_IDs = [[i, i+self.max_seq_len-1] for i in range(0,num_rows-self.lookahead-self.max_seq_len)]
        self.all_df = feature_df
        self.labels_df = labels_df
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = random.sample(self.all_IDs, k=limit_size)

        self.feature_names = list(self.all_df.columns)
        self.feature_df = self.all_df[self.feature_names]

    def _preprocess(self, df: pd.DataFrame):
        valid_col = [c for c in df.columns if c.startswith("book_valid_field")]
        assert len(valid_col) == 1, df.columns
        valid_col = valid_col[0]
        df = df[df[valid_col] > 0]
        useful_cols = [c for c in df.columns if c.startswith("bookdata")]
        df = df[useful_cols]
        return df

    def get_feature_data(self, pattern: str):
        datas = []
        logger.info(f"loading data from {pattern}")
        for file in sorted(glob.glob(pattern)):
            if "xy" in file:
                continue
            df = pd.read_parquet(file)
            data = self._preprocess(df)
            datas.append(data)
        logger.info(f"number of files loaded: {len(datas)}")
        if len(datas) != 0:
            data = pd.concat(datas)
            data = data.reset_index(drop=True)
        else:
            data = None
        return data

    def get_feature_data_xl(self, pattern: str):
        """
        Same result as get_feature_data, but data is stored in np.memmap so can load large dataset.
        """
        logger.info(f"preprocssing data from {pattern}")
        num_rows = 0
        for file in sorted(glob.glob(pattern)):
            if "xy" in file:
                continue
            # date = file.split(".")[-2]
            df = pd.read_parquet(file)
            df = self._preprocess(df)
            num_rows += df.shape[0]
            num_cols = df.shape[1]
        logger.info(f"Total data size needs to load: {num_rows} x {num_cols}")

        split = "train" if "train" in pattern else "val"
        path = f"/workspace/futs/data/{split}.bin"

        # If file already exists and matches, directly return without importing again.
        if os.path.isfile(path):
            arr = np.memmap(path, dtype=float, mode="r")
            if arr.shape == (num_rows, num_cols):
                logger.info(f"loaded data from preprocessed file {path}")
                return data.pd.DataFrame(arr, copy=False), None

        # If files don't exist, create and import data.
        arr = np.memmap(path, dtype=float, mode="w+", shape=(num_rows, num_cols))
        i = 0
        for file in sorted(glob.glob(pattern)):
            if "xy" in file:
                continue
            # date = file.split(".")[-2]
            df = pd.read_parquet(file)
            df = self._preprocess(df)
            arr[i : i + df.shape[0], :] = df.values
            i += df.shape[0]
        logger.info(f"loaded data from {pattern}")
        data = pd.DataFrame(arr, copy=False)
        arr.flush()  # save to disk
        return data, None

    def get_label_data(self, feature_df: pd.DataFrame, lookahead: int, seq_len: int):
        """
        Compute label value based on feature_df.
        TODO:
        Current data is normalized with rolling window. Need to figure out how to unnormalize to calculate the right correlation. 
        """
        def _extract_label(data_df: pd.DataFrame, lookahead: int, seq_len:int):
            BID = "bid_0"
            ASK = "ask_0"
            idx1 = [i for i, n in enumerate(data_df.columns) if BID in n]
            idx2 = [i for i, n in enumerate(data_df.columns) if ASK in n]
            assert len(idx1) == 1 and len(idx2) == 1, f"{idx1}, {idx2}"
            # sample the dataframe with an offset of lookahead(40) + seq_len(1024) - 1
            label_df = (
                data_df.iloc[lookahead+seq_len-1:, idx1[0]] + data_df.iloc[lookahead+seq_len-1:, idx2[0]]
            ) / 2
            label_df = label_df.reset_index(drop=True)
            return label_df.to_frame().astype(np.float32)

        label_df = _extract_label(feature_df, lookahead, seq_len)
        #assert max(label_df.index) == max(feature_df.index)
        return label_df