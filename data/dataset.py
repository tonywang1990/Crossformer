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

warnings.filterwarnings("ignore")
logger = logging.getLogger("__main__")

FEATURE_COLS = [
    "bookdata::book=book_UR::data_name=bid_0",
    "bookdata::book=book_UR::data_name=bid_1",
    "bookdata::book=book_UR::data_name=bid_2",
    "bookdata::book=book_UR::data_name=bid_3",
    "bookdata::book=book_UR::data_name=bid_4",
    "bookdata::book=book_UR::data_name=bid_size_0",
    "bookdata::book=book_UR::data_name=bid_size_1",
    "bookdata::book=book_UR::data_name=bid_size_2",
    "bookdata::book=book_UR::data_name=bid_size_3",
    "bookdata::book=book_UR::data_name=bid_size_4",
    "bookdata::book=book_UR::data_name=ask_0",
    "bookdata::book=book_UR::data_name=ask_1",
    "bookdata::book=book_UR::data_name=ask_2",
    "bookdata::book=book_UR::data_name=ask_3",
    "bookdata::book=book_UR::data_name=ask_4",
    "bookdata::book=book_UR::data_name=ask_size_0",
    "bookdata::book=book_UR::data_name=ask_size_1",
    "bookdata::book=book_UR::data_name=ask_size_2",
    "bookdata::book=book_UR::data_name=ask_size_3",
    "bookdata::book=book_UR::data_name=ask_size_4",
    "bookdata::book=book_UR::data_name=buy_size",
    "bookdata::book=book_UR::data_name=buy_price",
    "bookdata::book=book_UR::data_name=sell_size",
    "bookdata::book=book_UR::data_name=sell_price",
]

VALIDATION_COL = "book_valid_field::book=book_UR"

TS_COLS = ['ts']

LABEL_COLS = [
    #"extdata::book=book_UR::data_name=forward_return_vwap_10s",
    #"extdata::book=book_UR::data_name=forward_return_vwap_60s",
    "extdata::book=book_UR::data_name=forward_return_vwap_600s",
    #"extdata::book=book_UR::data_name=forward_return_vwap_1800s",
]

ALL_COLS = FEATURE_COLS + LABEL_COLS + TS_COLS 


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
        pattern,
        in_len,
        out_len,
        file_list=None,
        n_proc=1,
        limit_size=None,
        config=None,

    ):
        self.max_seq_len = in_len
        # process features
        data_df = self.read_data(os.path.join(root_dir, pattern))
        # for large size of data that don't fit into memory
        #data_df = self.read_data_xl(os.path.join(root_dir, pattern))
        num_rows = data_df.shape[0]
        # process labels
        feature_df = data_df[FEATURE_COLS]
        labels_df = data_df[LABEL_COLS]
        ts_df = data_df[TS_COLS]
        # all_IDs uses a compressed representation: i-th position in all_ID maps to (start, end) of the feature_df and start of the label_df.
        self.all_IDs = [
            [i-self.max_seq_len+1, i]
            for i in range(self.max_seq_len-1, num_rows-out_len)
        ]
        self.all_df = feature_df
        self.labels_df = labels_df
        self.ts_df = ts_df
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = random.sample(self.all_IDs, k=limit_size)

        self.feature_names = list(self.all_df.columns)
        self.feature_df = self.all_df[self.feature_names]

    def _validate(self, df: pd.DataFrame):
        assert VALIDATION_COL in df.columns.to_list()
        df = df[df[VALIDATION_COL] > 0]
        return df

    def read_data(self, pattern: str):
        datas = []
        logger.info(f"loading data from {pattern}")
        for file in sorted(glob.glob(pattern)):
            if "xy" in file:
                continue
            df = pd.read_parquet(file)
            df = self._validate(df)
            data = df[ALL_COLS]
            datas.append(data)
        logger.info(f"number of files loaded: {len(datas)}")
        if len(datas) != 0:
            data = pd.concat(datas)
            data = data.reset_index(drop=True)
        else:
            data = None
        return data

    def read_data_xl(self, pattern: str):
        """
        Same result as get_feature_data, but data is stored in np.memmap so can load large dataset.
        """
        logger.info(f"preprocssing data from {pattern}")
        num_rows = 0
        num_cols = len(ALL_COLS)
        for file in sorted(glob.glob(pattern)):
            if "xy" in file:
                continue
            df = pd.read_parquet(file)
            df = self._validate(df)
            #data = df[FEATURE_COLS]
            num_rows += data.shape[0]
        logger.info(f"Total data size needs to load: {num_rows} x {num_cols}")

        split = "train" if "train" in pattern else "val"
        path = f"/workspace/futs/data/{split}.bin"

        # If file already exists and matches, directly return without importing again.
        if os.path.isfile(path):
            arr = np.memmap(path, dtype=float, mode="r")
            if arr.shape == (num_rows, num_cols):
                logger.info(f"loaded data from preprocessed file {path}")
                return data.pd.DataFrame(arr, copy=False)

        # If files don't exist, create and import data.
        arr = np.memmap(path, dtype=float, mode="w+", shape=(num_rows, num_cols))
        i = 0
        for file in sorted(glob.glob(pattern)):
            if "xy" in file:
                continue
            # date = file.split(".")[-2]
            df = pd.read_parquet(file)
            df = self._validate(df)
            data = df[FEATURE_COLS + LABEL_COLS]
            arr[i : i + df.shape[0], :] = data.values
            i += data.shape[0]
        logger.info(f"loaded data from {pattern}")
        data = pd.DataFrame(arr, copy=False)
        arr.flush()  # save to disk
        return data
