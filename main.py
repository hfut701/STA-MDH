import torch
import random
import copy
import numpy as np
import time
from BResidual import BResidual,CNN
from options import arg_parameter
from data_util import load_cifar10, load_mnist,load_crosscheck,get_weight_matrix1,load_park,load_ppmi
from federated import Cifar10FedEngine
from hierachical_att_model import HierAttNet,LSTM,MLP,normalCNN,CNNLSTMModel,HierAttNetnofeature,HierAttNetnotime, TransformerModel
from aggregator import parameter_aggregate, read_out
from util import *
import pandas as pd


def main(args):
    args.device = torch.device(args.device)
    batch_size = 8

    print("Prepare data and model...")
    if args.dataset =='park':
        args.clients = 42

        train_batches, test_batches, A, overall_tbatches = load_park(args)


    elif args.dataset =='ppmi':
        args.clients = 127

        train_batches, test_batches, A, overall_tbatches = load_ppmi(args)

    else:
        train_batches, test_batches, A, overall_tbatches = load_crosscheck(args)



    if args.dataset =='park':

        model = HierAttNet(feature_hidden_size=512, time_hidden_size=20, batch_size=batch_size,
                           output_size=1).to(args.device)

    elif args.dataset =='ppmi':
        model = HierAttNet(feature_hidden_size=512, time_hidden_size=20, batch_size=batch_size,
                           output_size=1).to(args.device)

    else:
        model = HierAttNet(feature_hidden_size=512, time_hidden_size=20, batch_size=batch_size,
                           output_size=1).cuda()


    print("Parameter holders")
    w_server, w_local = model.get_state()
    w_server = [w_server] * args.clients

    w_local = [w_local] * args.clients
    global_model = copy.deepcopy(w_server)
    personalized_model = copy.deepcopy(w_server)

    server_state = None
    client_states = [None] * args.clients

    print2file(str(args), args.logDir, True)
    nParams = sum([p.nelement() for p in model.parameters()])
    print2file('Number of model parameters is ' + str(nParams), args.logDir, True)

    print("Start Training...")
    num_collaborator = max(int(args.client_frac * args.clients), 1)
    weight_m = 0

    besy_round =0
    best_loss = 100000000
    best_mse = 0
    best_rmse = 0
    best_r2 = 0
    best_cf = 0
    best_map = 0
    best_mape = 0
    best_msle = 0

    best_loss_dict = {}
    best_mse_dict = {}
    best_rmse_dict = {}
    best_r2_dict = {}
    best_cf_dict = {}
    best_map_dict = {}
    best_msle_dict ={}
    best_mape_dict = {}

    best_fea_dict = {}
    best_time_dict = {}

    for com in range(1, args.com_round + 1):

        selected_user = np.random.choice(range(args.clients), num_collaborator, replace=False)
        print("selected_user",len(selected_user))
        train_time = []
        train_loss = []
        train_rmse = []
        train_mse = []
        train_msle = []
        train_mape = []
        for c in selected_user:
            engine = Cifar10FedEngine(args, copy.deepcopy(train_batches[c]), global_model[c], personalized_model[c],
                                          w_local[c], {}, c, 0, "Train", server_state, client_states[c])

            outputs, fea_att, time_att = engine.run()



            w_local[c] = copy.deepcopy(outputs['params'][0])
            personalized_model[c] = copy.deepcopy(outputs['params'][0])

            train_time.append(outputs["time"])
            train_loss.append(outputs["loss"])
            train_rmse.append(outputs["rmse"])
            train_mse.append(outputs["mse"])
            train_mape.append(outputs["mape"])
            train_msle.append(outputs["msle"])
            client_states[c] = outputs["c_state"]


        mtrain_time = np.mean(train_time)
        mtrain_loss = np.mean(train_loss)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_mse= np.mean(train_mse)
        mtrain_mape = np.mean(train_mape)
        mtrain_msle = np.mean(train_msle)
        log = 'Communication Round: {:03d}, Train Loss: {:.4f},' \
            ' Train RMSE: {:.4f}, Train MSE: {:.4f}, train mape: {:.4f},train msle: {:.4f},Training Time: {:.4f}'
        print2file(log.format(com,  mtrain_loss, mtrain_rmse,mtrain_mse, mtrain_mape,mtrain_msle,mtrain_time),
                args.logDir, True)


        # Validation
        if com % args.valid_freq == 0:

            all_vtime = []
            all_vloss = []
            all_vrmse = []
            all_vmse = []

            all_vmsle = []
            all_vmape = []
            client_vloss = {}
            client_vrmse = {}
            client_vmse = {}
            client_vmsle = {}
            client_vmape = {}


            # for c in range(args.clients):
            for c in selected_user:
                client_vloss[c] = []
                client_vrmse[c] = []
                client_vmse[c] = []
                client_vmsle[c] = []
                client_vmape[c] = []
                if c not in best_loss_dict.keys():
                    best_loss_dict[c] = 10000000
                tengine = Cifar10FedEngine(args, test_batches[c], personalized_model[c], personalized_model[c],
                                           w_local[c], {}, c, 0, "Test", server_state, client_states[c])


                outputs,tfea_att,ttime_att = tengine.run()


                client_vloss[c].append(outputs["loss"])
                client_vrmse[c].append(outputs["rmse"])
                client_vmse[c].append(outputs["mse"])
                client_vmsle[c].append(outputs["msle"])
                client_vmape[c].append(outputs["mape"])

                all_vtime.append(outputs["time"])
                all_vloss.append(outputs["loss"])
                all_vrmse.append(outputs["rmse"])
                all_vmse.append(outputs["mse"])
                all_vmsle.append(outputs["msle"])
                all_vmape.append(outputs["mape"])


                if best_loss_dict[c] > np.mean( client_vloss[c]):
                    best_loss_dict[c] = np.mean( client_vloss[c])
                    best_mse_dict[c] = np.mean(  client_vmse[c])
                    best_rmse_dict[c] = np.mean( client_vrmse[c])
                    best_mape_dict[c] = np.mean(client_vmape[c])
                    best_msle_dict[c] = np.mean( client_vmsle[c])


            all_log = 'AllValidation Round: {:03d}, Valid Loss: {:.4f}, ' \
                         'Valid rmse: {:.4f},Valid mse: {:.4f}, Valid mape: {:.4f},Valid msle: {:.4f},Test Time: {:.4f}'
            print2file(all_log.format(com, np.mean(all_vloss), np.mean(all_vrmse),np.mean(all_vmse),np.mean(all_vmape),np.mean(all_vmsle),
                                      np.mean(all_vtime)), args.logDir, True)


            if best_loss > np.mean(all_vloss):
                besy_round= com
                best_loss = np.mean(all_vloss)
                best_mse = np.mean(all_vmse)
                best_rmse = np.mean(all_vrmse)
                best_msle = np.mean(all_vmsle)
                best_mape = np.mean(all_vmape)

        if com != args.com_round + 1 and args.agg != 'central' :
            # Server aggregation
            t1 = time.time()


            if len(selected_user) ==args.clients:
                personalized_model, client_states, server_state,topk_predict = \
                    parameter_aggregate(args, A, personalized_model, global_model, server_state, client_states, selected_user,
                                        non_part,
                                        weight_m)

            else:
                personalized_model_select, client_states, server_state,topk_predict = \
                    parameter_aggregate(args, A, personalized_model, global_model, server_state, client_states,
                                        selected_user,
                                        non_part,
                                        weight_m)
                for ind,i in enumerate(selected_user):
                    personalized_model[i] = copy.deepcopy(personalized_model_select[ind])

            t2 = time.time()
            log = 'Communication Round: {:03d}, Aggregation Time: {:.4f} secs'
            print2file(log.format(com, (t2 - t1)), args.logDir, True)

            # Readout for global model
            global_model = read_out(personalized_model, args.device)
        else:
            print("bujuhe")
    log = 'best Round: {:03d},best,  Loss: {:.4f},' \
          '  RMSE: {:.4f},  RMSE: {:.4f},max mape: {:.4f},max msle: {:.4f}'
    print2file(log.format( besy_round,best_loss, best_rmse, best_mse,best_r2,best_cf, best_map, best_mape,best_msle),
               args.logDir, True)


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True
    option = arg_parameter()
    initial_environment(option.seed)
    main(option)

    print("Everything so far so good....")