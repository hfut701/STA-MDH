import copy
import torch
import os
import pickle as pk
from util import sd_matrixing,sd_matrixing1,sd_matrixing_other
from data_util import normalize_adj
from GraphConstructor import GraphConstructor
from optimiser import FedProx
import numpy as np
from scipy import linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv



def parameter_aggregate1(args, A, w_server, cluster):
    # update global weights
    new_s_state = None
    new_c_state = [None] * args.clients
    print("args.agg",args.agg)
    w_server = average_dic1(w_server, cluster)
    # w_server = [w_server] * args.clients
    # personalized_model = copy.deepcopy(w_server)
    personalized_model = []
    for i in range(args.clients):
        for j in range(len(cluster)):
            if i in cluster[j]:
                personalized_model.append(w_server[j])

    return personalized_model, new_c_state, new_s_state



def parameter_aggregate(args, A, w_server, global_model, server_state, client_states, active_idx,worng_client,weight_m):
    # update global weights
    new_s_state = None
    new_c_state = [None] * args.clients
    print("args.agg",args.agg)
    if args.agg == 'avg' or args.agg == "prox" or args.agg == "scaf":
        w_server = average_dic(w_server, args.device)
        w_server = [w_server] * args.clients
        personalized_model = copy.deepcopy(w_server)
    elif  args.agg == 'avg_rej':
        w_server = average_dic_rej(w_server, worng_client)
        w_server = [w_server] * args.clients
        personalized_model = copy.deepcopy(w_server)
    elif  args.agg == 'flop':
        personalized_model = average_flop(w_server)
        # personalized_model = copy.deepcopy(w_server)
    elif  args.agg == 'fedap':
        personalized_model = average_fedap(w_server,weight_m)
        # personalized_model = copy.deepcopy(w_server)
    elif args.agg == "att":
        w_server = att_dic(w_server, global_model[0], args.device)
        w_server = [w_server] * args.clients
        personalized_model = copy.deepcopy(w_server)

    elif args.agg == "graph" or args.agg == "graph_v2" or args.agg == "graph_v3" or args.agg == "gat" or args.agg == "attation_gat" or args.agg =="attation_gcn" or args.agg == "attation"or args.agg == "fen_attation_gat" or args.agg == "fen_attation":

        personalized_model,topk_predict = graph_dic(w_server, A, args)




    elif args.agg == "scaffold":
        new_s_state, new_c_state = scaffold_update(server_state, client_states, active_idx, args)
        w_server = average_dic(w_server, args.device)
        w_server = [w_server] * args.clients
        personalized_model = copy.deepcopy(w_server)

    else:
        personalized_model = None
        exit('Unrecognized aggregation.')

    return personalized_model, new_c_state, new_s_state,topk_predict

def average_dic1(model_dic, cluster, dp=0.001):
    w_avg ={}
    print("cluster",cluster)
    print("type",type(model_dic[0]))
    for j in range(len(cluster)):
        w_avg[j] = copy.deepcopy(model_dic[cluster[j][0]])
        if len(cluster[j]) ==1:
            continue
        for k in w_avg[j].keys():
            for i in cluster[j]:
                if i == cluster[j][0] :
                    continue
                w_avg[j][k] = w_avg[j][k].data.clone().detach() + model_dic[i][k].data.clone().detach()
            w_avg[j][k] = w_avg[j][k].data.clone().detach().div(torch.tensor(len(cluster[j]))) + torch.mul(torch.randn(w_avg[j][k].shape), dp)
    return w_avg

def average_dic(model_dic, device, dp=0.001):
    dp = 0.2
    w_avg = copy.deepcopy(model_dic[0])
    for k in w_avg.keys():
        for i in range(1, len(model_dic)):
            w_avg[k] = w_avg[k].data.clone().detach() + model_dic[i][k].data.clone().detach()
        w_avg[k] = w_avg[k].data.clone().detach().div(len(model_dic)) + torch.mul(torch.randn(w_avg[k].shape), dp)
    return w_avg



def euclidean_distance(x, y, sigma=1.0):
    # return 1/(torch.sqrt(torch.sum((x - y) ** 2)) + 1e-6)
    # distance = torch.sum((x - y) ** 2)
    # return torch.exp(-distance / (2 * sigma ** 2))
    return torch.sqrt(torch.sum((x - y) ** 2))




def graph_dic(models_dic, pre_A, args):
    keys = []
    key_shapes = []
    param_metrix = []
    param_metrix_fea = []
    param_metrix_time = []
    param_metrix_predict = []

    for key, param in models_dic[0].items():
        keys.append(key)
        key_shapes.append(list(param.data.shape))


    for model in models_dic:
        p_fea,p_time,p_predict = sd_matrixing1(model)
        param_metrix_fea.append(p_fea.clone().detach())
        param_metrix_time.append(p_time.clone().detach())
        param_metrix_predict.append(p_predict.clone().detach())

    param_metrix_fea = torch.stack(param_metrix_fea)
    param_metrix_time = torch.stack(param_metrix_time)
    param_metrix_predict = torch.stack(param_metrix_predict)

    for model in models_dic:
        param_metrix.append(sd_matrixing(model).clone().detach())

    param_metrix = torch.stack(param_metrix)



    # constract adj
    subgraph_size = min(args.subgraph_size, args.clients)
    A = generate_adj(param_metrix, args, subgraph_size).cpu().detach().numpy()
    A = normalize_adj(A)
    A = torch.tensor(A)

    for i in A.nonzero():
        A[i[0],i[1]] =1
        A[i[1],i[0]] = 1
    A = normalize_adj(A)
    A = torch.tensor(A)



    A = normalize_adj(torch.from_numpy(pre_A))


    # Aggregating




    dist_metrix_fea1 = torch.zeros((len(param_metrix_fea), len(param_metrix_fea)))
    dist_metrix_time1 = torch.zeros((len(param_metrix_time), len(param_metrix_time)))
    dist_metrix_predict1 = torch.zeros((len(param_metrix_predict), len(param_metrix_predict)))
    for i in range(len(param_metrix_fea)):
        for j in range(i + 1, len(param_metrix_fea)):
            dist_fea = torch.nn.functional.cosine_similarity(
                param_metrix_fea[i].view(1, -1), param_metrix_fea[j].view(1, -1)).item()

            dist_time = torch.nn.functional.cosine_similarity(
                param_metrix_time[i].view(1, -1), param_metrix_time[j].view(1, -1)).item()

            dist_predict = torch.nn.functional.cosine_similarity(
                param_metrix_predict[i].view(1, -1), param_metrix_predict[j].view(1, -1)).item()


            # Store distances in the distance matrices
            dist_metrix_fea1[i][j] = dist_metrix_fea1[j][i] = dist_fea
            dist_metrix_time1[i][j] = dist_metrix_time1[j][i] = dist_time
            dist_metrix_predict1[i][j] = dist_metrix_predict1[j][i] = dist_predict

        # Find the top 15 similar matrices for each category
        if args.dataset =='ppmi':
            top_k_fea = np.argsort(dist_metrix_fea1, axis=1)[:, :40]
            top_k_time = np.argsort(dist_metrix_time1, axis=1)[:, :40]
            top_k_predict = np.argsort(dist_metrix_predict1, axis=1)[:, :40]
        else:
            if args.dataset =='park':
                top_k_fea = np.argsort(dist_metrix_fea1, axis=1)[:, :15]
                top_k_time = np.argsort(dist_metrix_time1, axis=1)[:, :15]
                top_k_predict = np.argsort(dist_metrix_predict1, axis=1)[:, :15]
            else:
                top_k_fea = np.argsort(dist_metrix_fea1, axis=1)[:, :15]
                top_k_time = np.argsort(dist_metrix_time1, axis=1)[:, :15]
                top_k_predict = np.argsort(dist_metrix_predict1, axis=1)[:, :15]



        dist_metrix_fea = torch.zeros((len(param_metrix_fea), len(param_metrix_fea)))
        dist_metrix_time = torch.zeros((len(param_metrix_time), len(param_metrix_time)))
        dist_metrix_predict = torch.zeros((len(param_metrix_predict), len(param_metrix_predict)))
        for i in range(len(param_metrix_fea)):
            dist_metrix_fea[i, top_k_fea[i]] = dist_metrix_fea1[i, top_k_fea[i]]
            dist_metrix_time[i, top_k_time[i]] = dist_metrix_time1[i, top_k_time[i]]
            dist_metrix_predict[i, top_k_predict[i]] = dist_metrix_predict1[i, top_k_predict[i]]

        dist_metrix_fea = normalize_adj(dist_metrix_fea)


        aggregated_param_fea = torch.mm(torch.from_numpy(dist_metrix_fea), param_metrix_fea)
        for i in range(args.layers - 1):
            aggregated_param_fea = torch.mm(torch.from_numpy(dist_metrix_fea), aggregated_param_fea)
        param_metrix_fea = (args.serveralpha * aggregated_param_fea) + ((1 - args.serveralpha) * param_metrix_fea)

        # time
        dist_metrix_time = normalize_adj(dist_metrix_time)


        aggregated_param_time = torch.mm(torch.from_numpy(dist_metrix_time), param_metrix_time)
        for i in range(args.layers - 1):
            aggregated_param_time = torch.mm(torch.from_numpy(dist_metrix_time), aggregated_param_time)
        param_metrix_time = (args.serveralpha * aggregated_param_time) + ((1 - args.serveralpha) * param_metrix_time)

        # predict

        dist_metrix_predict = normalize_adj(dist_metrix_predict)



        aggregated_param_predict = torch.mm(torch.from_numpy(dist_metrix_predict), param_metrix_predict)
        for i in range(args.layers - 1):
            aggregated_param_predict = torch.mm(torch.from_numpy(dist_metrix_predict), aggregated_param_predict)
        param_metrix_predict = (args.serveralpha * aggregated_param_predict) + ((1 - args.serveralpha) * param_metrix_predict)
        # ''' fea,time,predict'''
        #




        for i in range(len(models_dic)):

            index =0
            pointer = 0
            for k in range(len(keys)):
                if index == 11 or index ==20:
                    pointer = 0
                num_p = 1
                for n in key_shapes[k]:
                    num_p *= n
                if index <11:

                    models_dic[i][keys[k]] = param_metrix_fea[i][pointer:pointer + num_p].reshape(key_shapes[k])
                elif index>=11  and index<20:

                    models_dic[i][keys[k]] = param_metrix_time[i][pointer:pointer + num_p].reshape(key_shapes[k])
                else:

                    models_dic[i][keys[k]] = param_metrix_predict[i][pointer:pointer + num_p].reshape(key_shapes[k])
                pointer += num_p
                index = index + 1




    return models_dic


def scaffold_update(server_state, client_states, active_ids, args):
    active_clients = [client_states[i] for i in active_ids]
    c_delta = []
    cc = [client_state["c_i_delta"] for client_state in active_clients]
    for ind in range(len(server_state["c"])):
        # handles the int64 and float data types jointly
        c_delta.append(
            torch.mean(torch.stack([c_i_delta[ind].float() for c_i_delta in cc]), dim=0).to(server_state["c"][ind].dtype)
        )
    c_delta = tuple(c_delta)

    c = []
    for param_1, param_2 in zip(server_state["c"], c_delta):
        c.append(param_1 + param_2 * args.clients * args.client_frac / args.clients)

    c = tuple(c)

    new_server_state = {
        "global_round": server_state["global_round"] + 1,
        "c": c
    }

    new_client_state = [{
        "global_round": new_server_state["global_round"],
        "model_delta": None,
        "c_i": client["c_i"],
        "c_i_delta": None,
        "c": server_state["c"]
    } for client in client_states]

    return new_server_state, new_client_state


def generate_adj(param_metrix, args, subgraph_size):
    dist_metrix = torch.zeros((len(param_metrix), len(param_metrix)))
    for i in range(len(param_metrix)):
        for j in range(len(param_metrix)):
            dist_metrix[i][j] = torch.nn.functional.pairwise_distance(
                param_metrix[i].view(1, -1), param_metrix[j].view(1, -1), p=2).clone().detach()
    dist_metrix = torch.nn.functional.normalize(dist_metrix).to(args.device)

    gc = GraphConstructor(args.clients, subgraph_size, args.node_dim,
                          args.device, args.adjalpha).to(args.device)
    idx = torch.arange(args.clients).to(args.device)



    optimizer = torch.optim.SGD(gc.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for e in range(args.gc_epoch):
        optimizer.zero_grad()
        adj = gc(idx)
        adj = torch.nn.functional.normalize(adj)

        loss = torch.nn.functional.mse_loss(adj, dist_metrix)
        loss.backward()
        optimizer.step()

    adj = gc.eval(idx).to("cpu")

    if args.dataset=='ppmi':

        adj = adj * (adj >= torch.topk(adj, 40)[0][..., -1].unsqueeze(-1)).float()
        
    else:
        if args.dataset == 'park':

            adj = adj * (adj >= torch.topk(adj, 15)[0][..., -1].unsqueeze(-1)).float()
        else:

            adj = adj * (adj >= torch.topk(adj, 15)[0][..., -1].unsqueeze(-1)).float()
        

    adj = adj +adj.T
    adj = torch.where(adj >0, torch.tensor(1.0), adj)
    return adj


def generate_adj_select(param_metrix, args, subgraph_size):
    dist_metrix = torch.zeros((len(param_metrix), len(param_metrix)))
    for i in range(len(param_metrix)):
        for j in range(len(param_metrix)):
            dist_metrix[i][j] = torch.nn.functional.pairwise_distance(
                param_metrix[i].view(1, -1), param_metrix[j].view(1, -1), p=2).clone().detach()
    dist_metrix = torch.nn.functional.normalize(dist_metrix).to(args.device)



    gc = GraphConstructor(len(param_metrix), subgraph_size, args.node_dim,
                          args.device, args.adjalpha).to(args.device)
    idx = torch.arange(len(param_metrix)).to(args.device)

    optimizer = torch.optim.SGD(gc.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for e in range(args.gc_epoch):
        optimizer.zero_grad()
        adj = gc(idx)
        adj = torch.nn.functional.normalize(adj)

        loss = torch.nn.functional.mse_loss(adj, dist_metrix)
        loss.backward()
        optimizer.step()

    adj = gc.eval(idx).to("cpu")


    adj = adj * (adj >= torch.topk(adj, int(len(param_metrix) * 0.5))[0][..., -1].unsqueeze(-1)).float()

    # adj = torch.triu(adj) + torch.triu(adj, 1).T

    adj = adj + adj.T
    adj = torch.where(adj > 0, torch.tensor(1.0), adj)
    return adj

def read_out(personalized_models, device):
    # average pooling as read out function
    global_model = average_dic(personalized_models, device, 0)
    return [global_model] * len(personalized_models)
