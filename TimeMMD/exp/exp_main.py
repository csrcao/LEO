import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from data_provider.data_factory import data_provider
from einops import rearrange
from exp.exp_basic import Exp_Basic
from torch import optim
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from aurora.modeling_aurora import AuroraForPrediction
from aurora.configuration_aurora import AuroraConfig

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_path = self.args.model_path
        config_path = os.path.join(model_path, 'config.json')
        config = AuroraConfig.from_json_file(config_path)
        model = AuroraForPrediction._from_config(config)

        return model.cuda()

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):

        self.model.eval()
        trues = []
        preds = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_input_ids, batch_attention_mask, batch_token_type_ids) in enumerate(
                    vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_input_ids = batch_input_ids.to(self.device)
                batch_attention_mask = batch_attention_mask.to(self.device)
                batch_token_type_ids = batch_token_type_ids.to(self.device)
                finite_row = torch.isfinite(batch_y.view(batch_y.size(0), -1)).all(dim=1)  # [B]
                if not finite_row.all():
                    batch_x = batch_x[finite_row]
                    batch_y = batch_y[finite_row]
                    batch_input_ids = batch_input_ids[finite_row]
                    batch_attention_mask = batch_attention_mask[finite_row]
                    batch_token_type_ids = batch_token_type_ids[finite_row]
                    if batch_y.size(0) == 0:
                        continue

                n_vars = batch_x.shape[-1]
                input.append(batch_x.detach().cpu().numpy())
                batch_x = rearrange(batch_x, "b l c -> (b c) l")
                batch_input_ids = batch_input_ids.repeat(n_vars, 1)
                batch_attention_mask = batch_attention_mask.repeat(n_vars, 1)
                batch_token_type_ids = batch_token_type_ids.repeat(n_vars, 1)
                pred_y = self.model.generate(inputs=batch_x, text_input_ids=batch_input_ids,
                                             text_attention_mask=batch_attention_mask,
                                             text_token_type_ids=batch_token_type_ids,
                                             inference_token_len=self.args.inference_token_len,
                                             max_output_length=self.args.pred_len, max_text_token_length=500,
                                             num_samples=10)

                samples = rearrange(pred_y, "(b c) s l -> s b l c", c=n_vars)
                output = samples.mean(0)

                preds.append(output.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

                # print("---------")
                # print(output.shape)

            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)

            mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        self.model.train()
        return mse, mae

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        total_time = 0

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            torch.cuda.synchronize()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_input_ids, batch_attention_mask, batch_token_type_ids) in enumerate(
                    train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_input_ids = batch_input_ids.to(self.device)
                batch_attention_mask = batch_attention_mask.to(self.device)
                batch_token_type_ids = batch_token_type_ids.to(self.device)

                n_vars = batch_x.shape[-1]
                batch_x = rearrange(batch_x, "b l c -> (b c) l")
                batch_y = rearrange(batch_y, "b l c -> (b c) l")
                batch_input_ids = batch_input_ids.repeat(n_vars, 1)
                batch_attention_mask = batch_attention_mask.repeat(n_vars, 1)
                batch_token_type_ids = batch_token_type_ids.repeat(n_vars, 1)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_ids=batch_x, text_input_ids=batch_input_ids,
                                             text_attention_mask=batch_attention_mask,
                                             text_token_type_ids=batch_token_type_ids,
                                             labels=batch_y,
                                             max_output_length=self.args.pred_len)
                        loss = outputs.loss
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(input_ids=batch_x, text_input_ids=batch_input_ids,
                                         text_attention_mask=batch_attention_mask,
                                         text_token_type_ids=batch_token_type_ids,
                                         labels=batch_y,
                                         max_output_length=self.args.pred_len)
                    loss = outputs.loss
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()


                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()


            torch.cuda.synchronize()
            per_epoch_time = time.time() - epoch_time
            total_time = total_time + per_epoch_time
            print("Epoch: {} cost time: {}".format(epoch + 1, per_epoch_time))

            train_loss = np.average(train_loss)
            vali_loss, _ = self.vali(vali_data, vali_loader, criterion)
            test_mse, test_mae = self.vali(test_data, test_loader, criterion)
            self.model.train()

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test MSE: {4:.7f} Test MAE: {5:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_mse, test_mae))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, None, epoch + 1, self.args)

        print('total train time:', total_time)
        self.model.load_state_dict(early_stopping.ckpt)

        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        preds = []
        trues = []
        self.model.eval()
        with (torch.no_grad()):
            for i, (batch_x, batch_y, batch_input_ids, batch_attention_mask, batch_token_type_ids) in enumerate(
                    test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_input_ids = batch_input_ids.to(self.device)
                batch_attention_mask = batch_attention_mask.to(self.device)
                batch_token_type_ids = batch_token_type_ids.to(self.device)

                finite_row = torch.isfinite(batch_y.view(batch_y.size(0), -1)).all(dim=1)  # [B]
                if not finite_row.all():
                    batch_x = batch_x[finite_row]
                    batch_y = batch_y[finite_row]
                    batch_input_ids = batch_input_ids[finite_row]
                    batch_attention_mask = batch_attention_mask[finite_row]
                    batch_token_type_ids = batch_token_type_ids[finite_row]
                    if batch_y.size(0) == 0:
                        continue

                n_vars = batch_x.shape[-1]
                batch_x = rearrange(batch_x, "b l c -> (b c) l")
                batch_input_ids = batch_input_ids.repeat(n_vars, 1)
                batch_attention_mask = batch_attention_mask.repeat(n_vars, 1)
                batch_token_type_ids = batch_token_type_ids.repeat(n_vars, 1)

                pred_y = self.model.generate(inputs=batch_x, text_input_ids=batch_input_ids,
                                             text_attention_mask=batch_attention_mask,
                                             text_token_type_ids=batch_token_type_ids,
                                             inference_token_len=self.args.inference_token_len,
                                             max_output_length=self.args.pred_len, max_text_token_length=500,
                                             num_samples=10)
                samples = rearrange(pred_y, "(b c) s l -> s b l c", c=n_vars)
                output = samples.mean(0)

                preds.append(output.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

                # print("---------")
                # print(output.shape)

            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)

            mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
            print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))

