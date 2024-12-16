import threading
import datetime
import torch
import time
import numpy as np
from BResidual import BResidual,CNN
from optimiser import SGD
from torch.optim import Adam
from util import sd_matrixing, PiecewiseLinear, trainable_params, StatsLogger
from hierachical_att_model import HierAttNet,LSTM,MLP,normalCNN,CNNLSTMModel,HierAttNetnofeature,HierAttNetnotime,TransformerModel
from sklearn.metrics import r2_score,max_error,mean_squared_log_error

def msle_loss(y_pred, y_true):
    epsilon = 1e-10  # 定义一个小正数
    log_pred = torch.where(y_pred >= 0, torch.log(y_pred + 1), torch.log(epsilon))
    log_true = torch.where(y_true >= 0, torch.log(y_true + 1), torch.log(epsilon))
    msle = torch.mean((log_pred - log_true)**2)  # 计算均方误差
    return msle



class Cifar10FedEngine:
    def __init__(self, args, dataloader, global_param, server_param, local_param,
                 outputs, cid, tid, mode, server_state, client_states):
        self.args = args
        self.dataloader = dataloader

        self.global_param = global_param
        self.server_param = server_param
        self.local_param = local_param
        self.server_state = server_state
        self.client_state = client_states

        self.client_id = cid
        self.outputs = outputs
        self.thread = tid

        self.mode = mode

        self.model = self.prepare_model()
        # self.threadLock = threading.Lock()

        self.m1, self.m2, self.m3, self.reg1, self.reg2 = None, None, None, None, None

    def prepare_model(self):

        if self.args.dataset == "crosscheck" or self.args.dataset == "park" or self.args.dataset == "ppmi":


            if self.args.dataset == 'park':

                model = HierAttNet(feature_hidden_size=512, time_hidden_size=20, batch_size=8,
                                   output_size=1).to(self.args.device)

            elif self.args.dataset == 'ppmi':
                model = HierAttNet(feature_hidden_size=512, time_hidden_size=20, batch_size=8,
                                   output_size=1).to(self.args.device)

            else:
                model = HierAttNet(feature_hidden_size=512, time_hidden_size=20, batch_size=8,
                                   output_size=1).to(self.args.device)


        else:
            print("Unknown model type ... ")
            model = None

        model.set_state(self.global_param, self.server_param,self.args.agg)
        return model

    def run(self):
        self.model.to(self.args.device)

        output, fea_att, time_att, l1_regularization_list = self.client_run()


        self.free_memory()

        return output, fea_att, time_att

    def client_run(self):


        opt = Adam(trainable_params(self.model), lr=0.01, weight_decay=5e-4* self.args.batch_size
                   ) #our

        mean_loss = []
        mean_rmse = []
        mean_mse = []
        mean_msle = []
        mean_mape = []
        t1 = time.time()
        c_state = None

        if self.mode == "Train":
            # training process
            for epoch in range(self.args.client_epochs):

                stats,fea_att, time_att, l1_regularization_list = self.batch_run(True, opt.step)


                mean_loss.append(stats['loss'].cpu().detach())
                mean_rmse.append(stats['rmse'].cpu().detach())

                mean_msle.append(stats['msle'].cpu().detach())
                mean_mape.append(stats['mape'].cpu().detach())


        elif self.mode == "Test":
            # validation process

            stats, fea_att, time_att,l1_regularization_list = self.batch_run(True, opt.step)


            mean_loss.append(stats['loss'].cpu().detach())
            mean_rmse.append(stats['rmse'].cpu().detach())
            mean_mse.append(stats['mse'].cpu().detach())

            mean_msle.append(stats['msle'].cpu().detach())
            mean_mape.append(stats['mape'].cpu().detach())


        time_cost = time.time() - t1


        log = self.mode + ' - Thread: {:03d}, Client: {:03d}. Average Loss: {:.4f},' \
                          ' Average rmse: {:.4f}, Average mse: {:.4f},, Average r2: {:.4f}, Average max erro: {:.4f},Average map: {:.4f},Average mape: {:.4f},Average msle: {:.4f},Total Time Cost: {:.4f}'
        self.logger(log.format(self.thread, self.client_id, np.mean(mean_loss), np.mean(mean_rmse), np.mean(mean_mse),np.mean(mean_r2),np.mean(mean_cf),np.mean(mean_map),np.mean(mean_mape),np.mean(mean_msle),
                               time_cost), False)

        self.model.to("cpu")
        output = {"params": self.model.get_state(),
                  "time": time_cost,
                  "loss": np.mean(mean_loss),
                  "rmse": np.mean(mean_rmse),
                  "mse": np.mean(mean_mse),
                  "msle": np.mean(mean_msle),
                  "mape": np.mean(mean_mape),
                  "client_state": self.client_state,
                  "c_state": c_state}


        return output, fea_att, time_att,l1_regularization_list



    def batch_run(self, training, optimizer_step=None, stats=None):
        # stats = stats or StatsLogger(('loss', 'correct'))
        stats = {}
        criterion = torch.nn.L1Loss(reduction='mean')
        ma = torch.nn.MSELoss(reduction='mean')
        self.model.train(training)
        for batch in self.dataloader:
            # print("batch",batch['input'].size())

            output, fea_att, time_att, l1_regularization_list= self.model(batch)


            if training:

                loss = criterion(output, batch["target"])
                stats['loss'] = loss
                stats['mse'] = ma(output, batch["target"])
                stats['rmse'] = torch.sqrt(stats['mse'])

                stats['r2'] = 0
                stats['cf'] = max_error(batch["target"].detach().cpu().numpy(), output.detach().cpu().numpy())

                stats['mape'] = torch.mean(torch.where(batch["target"].float() != 0, torch.abs(
                    (batch["target"].float() - output.float()) / batch["target"].float()),
                                                       torch.zeros_like(batch["target"].float()))) * 100
                stats['msle'] = ma(torch.log(output + 16), torch.log(batch["target"] + 16))  #
                stats['map'] = torch.mean(torch.where(batch["target"].float() != 0,
                                                      (batch["target"].float() - output.float()) / batch[
                                                          "target"].float(),
                                                      torch.zeros_like(batch["target"].float()))) * 100




                loss.backward()
                optimizer_step()
                self.model.zero_grad()
            else:


                loss = criterion(output, batch["target"])
                stats['loss'] = loss
                stats['mse'] = ma(output, batch["target"])
                stats['rmse'] = torch.sqrt(stats['mse'])
                stats['r2'] = 0
                stats['cf'] = max_error(batch["target"].detach().cpu().numpy(), output.detach().cpu().numpy())

                stats['mape'] = torch.mean(torch.where(batch["target"].float() != 0, torch.abs(
                    (batch["target"].float() - output.float()) / batch["target"].float()),
                                                     torch.zeros_like(batch["target"].float()))) * 100
                stats['msle'] = ma(torch.log(output+ 16), torch.log(batch["target"] + 16))  #

                stats['map'] = torch.mean(torch.where(batch["target"].float() != 0,
                                                      (batch["target"].float() - output.float() )/ batch[
                                                          "target"].float(),
                                                      torch.zeros_like(batch["target"].float()))) * 100



            return stats,fea_att, time_att,l1_regularization_list


    def criterion(self, loss, mode):
        if self.args.agg == "avg":
            pass
        elif self.args.reg > 0 and mode != "PerTrain" and self.args.clients != 1:
            self.m1 = sd_matrixing(self.model.get_state()[0]).reshape(1, -1).to(self.args.device)
            self.m2 = sd_matrixing(self.server_param).reshape(1, -1).to(self.args.device)
            self.m3 = sd_matrixing(self.global_param).reshape(1, -1).to(self.args.device)
            self.reg1 = torch.nn.functional.pairwise_distance(self.m1, self.m2, p=2)
            self.reg2 = torch.nn.functional.pairwise_distance(self.m1, self.m3, p=2)
            loss = loss + 0.3 * self.reg1 + 0.3 * self.reg2
        return loss

    def free_memory(self):
        if self.m1 is not None:
            self.m1.to("cpu")
        if self.m2 is not None:
            self.m2.to("cpu")
        if self.m3 is not None:
            self.m3.to("cpu")
        if self.reg1 is not None:
            self.reg1.to("cpu")
        if self.reg2 is not None:
            self.reg2.to("cpu")

        torch.cuda.empty_cache()

    def logger(self, buf, p=False):
        if p:
            print(buf)
        # self.threadLock.acquire()
        with open(self.args.logDir, 'a+') as f:
            f.write(str(datetime.datetime.now()) + '\t' + buf + '\n')
        # self.threadLock.release()
