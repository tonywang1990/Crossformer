import json
import os
import pickle
import time
import warnings
import logging

import numpy as np
import torch
import torch.nn as nn
from cross_models.cross_former import Crossformer
from data.data_loader import Dataset_Futs, Dataset_Futs_Pretrain  # , Dataset_MTS
from torch import optim
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, count_parameters, Logger

from cross_exp.exp_basic import Exp_Basic

warnings.filterwarnings("ignore")
#logger = logging.getLogger("__main__")


class Exp_crossformer(Exp_Basic):
    def __init__(self, args, setting):
        super(Exp_crossformer, self).__init__(args)
        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.logger = Logger(os.join(path, "logfile"))

    def _build_model(self):
        model = Crossformer(
            self.args.input_dim,
            self.args.output_dim,
            self.args.in_len,
            self.args.out_len,
            self.args.seg_len,
            self.args.win_size,
            self.args.factor,
            self.args.d_model,
            self.args.d_ff,
            self.args.n_heads,
            self.args.e_layers,
            self.args.dropout,
            self.args.baseline,
            self.device,
        ).float()
        logger = self.logger
        logger.log("Model:\n{}".format(model))
        logger.log("Total number of parameters: {}".format(count_parameters(model)))
        logger.log(
            "Trainable parameters: {}".format(count_parameters(model, trainable=True))
        )
        if self.args.load_model:
            saved_model_path = os.path.join("./checkpoints", self.args.load_model)
            model.load_state_dict(torch.load(saved_model_path), strict=False)
            logger.log("Loaded model from {saved_model_path}.")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args
        print(args)

        if flag == "test" or "val":
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
        # data_set = Dataset_MTS(
        #    root_path=args.root_path,
        #    data_path=args.data_path,
        #    flag=flag,
        #    size=[args.in_len, args.out_len],
        #    data_split = args.data_split,
        # )
        if flag == "train":
            data_path = args.train_path
        elif flag == "val":
            data_path = args.val_path

        data_set = Dataset_Futs(
            root_dir=args.root_path,
            pattern=data_path,
            in_len=args.in_len,
            out_len=args.out_len,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )
        self.logger.log(f"data_loader number of batches = {len(data_loader)}")

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def calc_corr(self, targets, preds) -> float:
        # Flatten
        targets = np.concatenate(targets, axis=0).flatten()
        preds = np.concatenate(preds, axis=0).flatten()
        # calcualte price change
        return np.corrcoef(preds, targets)[0, 1]

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        preds, tgts = [], []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                pred, true = self._process_one_batch(vali_data, batch_x, batch_y)
                tgts += true.detach().cpu()
                preds += pred.detach().cpu()
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss.detach().item())
        total_loss = np.average(total_loss)
        corr = self.calc_corr(tgts, preds)
        self.model.train()
        return total_loss, corr

    def train(self, setting):
        logger = self.logger
        path = self.path

        with open(os.path.join(path, "args.json"), "w") as f:
            json.dump(vars(self.args), f, indent=True)

        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        # test_data, test_loader = self._get_data(flag = 'test')

        # TODO: check if data normalization is needed
        # scale_statistic = {'mean': train_data.scaler.mean, 'std': train_data.scaler.std}
        # with open(os.path.join(path, "scale_statistic.pkl"), 'wb') as f:
        #    pickle.dump(scale_statistic, f)

        sampling_ratio = 0.1
        train_steps = int(len(train_loader) * sampling_ratio)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            train_loss = []
            logger.log(
                f"Training: Sampling Ratio = {sampling_ratio}, #batches per epoch = {train_steps}"
            )

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                if i > train_steps:
                    break

                model_optim.zero_grad()
                pred, true = self._process_one_batch(train_data, batch_x, batch_y)

                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if i % 100 == 0:
                    logger.log(
                        f"\tTraining epoch {epoch:2d}: {i / train_steps * 100:5.1f}% | batch {i:5d} of {train_steps:5d} | train loss: {loss.item():5.3f}"
                    )

                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss, vali_corr = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)

            logger.log(
                "Epoch: {0:2d}, batches: {1:5d}, time {2:5.1f}s | Train Loss: {3:5.3f} | Vali Loss: {4:5.3f} | Vali Corr {5:5.3f} ".format(
                    epoch + 1,
                    train_steps,
                    time.time() - epoch_time,
                    train_loss,
                    vali_loss,
                    vali_corr,
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                logger.log("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # best_model_path = path + "/" + "checkpoint.pth"
        # self.model.load_state_dict(torch.load(best_model_path))
        # state_dict = (
        #    self.model.module.state_dict()
        #    if isinstance(self.model, DataParallel)
        #    else self.model.state_dict()
        # )#
        # torch.save(state_dict, path + "/" + "checkpoint.pth")

        return self.model

    def test(self, setting, save_pred=False, inverse=False):
        test_data, test_loader = self._get_data(flag="test")

        self.model.eval()

        preds = []
        trues = []
        metrics_all = []
        instance_num = 0

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                pred, true = self._process_one_batch(
                    test_data, batch_x, batch_y, inverse
                )
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = (
                    np.array(
                        metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())
                    )
                    * batch_size
                )
                metrics_all.append(batch_metric)
                if save_pred:
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())

        metrics_all = np.stack(metrics_all, axis=0)
        metrics_mean = metrics_all.sum(axis=0) / instance_num

        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print("mse:{}, mae:{}".format(mse, mae))

        np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
        if save_pred:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            np.save(folder_path + "pred.npy", preds)
            np.save(folder_path + "true.npy", trues)

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, inverse=False):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        outputs = self.model(batch_x)

        if inverse:
            outputs = dataset_object.inverse_transform(outputs)
            batch_y = dataset_object.inverse_transform(batch_y)

        # TODO: add args to select between point prediction and vector prediciton
        # outputs = self._point_prediction(outputs).reshape(batch_y.shape)

        return outputs, batch_y

    def eval(self, setting, save_pred=False, inverse=False):
        # evaluate a saved model
        args = self.args
        # data_set = Dataset_MTS(
        #    root_path=args.root_path,
        #    data_path=args.data_path,
        #    flag='test',
        #    size=[args.in_len, args.out_len],
        #    data_split = args.data_split,
        #    scale = True,
        #    scale_statistic = args.scale_statistic,
        # )
        data_set, data_loader = self._get_data(flag="val")

        self.model.eval()

        preds = []
        trues = []
        metrics_all = []
        instance_num = 0

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(data_loader):
                pred, true = self._process_one_batch(
                    data_set, batch_x, batch_y, inverse
                )
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = (
                    np.array(
                        metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())
                    )
                    * batch_size
                )
                metrics_all.append(batch_metric)
                if save_pred:
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())

        metrics_all = np.stack(metrics_all, axis=0)
        metrics_mean = metrics_all.sum(axis=0) / instance_num

        # result save
        folder_path = "./checkpoints/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print("mse:{}, mae:{}".format(mse, mae))

        np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
        if save_pred:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            np.save(folder_path + "pred.npy", preds)
            np.save(folder_path + "true.npy", trues)

        return mae, mse, rmse, mape, mspe
