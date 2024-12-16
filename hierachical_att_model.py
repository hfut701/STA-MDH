#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from feature_att_model import FeatureAttNet
from time_att_model import TimeAttNet


class HierAttNet(nn.Module):
    def __init__(self, feature_hidden_size, time_hidden_size, batch_size, output_size):
        super(HierAttNet, self).__init__()
        self.batch_size = batch_size
        self.feature_hidden_size = feature_hidden_size
        self.time_hidden_size = time_hidden_size
        self.feature_att_net = FeatureAttNet(input_size=4, hidden_size=feature_hidden_size, num_layers=1) #crosscheck
        # self.feature_att_net = FeatureAttNet(input_size=1, hidden_size=feature_hidden_size, num_layers=1)#parkson
        # self.feature_att_net = FeatureAttNet(input_size=4, hidden_size=1028, num_layers=1)
        self.time_att_net = TimeAttNet(input_size=feature_hidden_size, hidden_size=time_hidden_size, num_layers=1)
        self.linear = nn.Linear(time_hidden_size//2, output_size) #正常



    def forward(self, input):
        output_list = []
        feature_list = []

        input = input['input'].permute(1, 0, 2, 3)

        l1_regularization_list = []

        for i in input:

            fan_output,feature_attention,l1_regularization = self.feature_att_net(i.permute(1, 0, 2))

            l1_regularization_list.append(l1_regularization)
            output_list.append(fan_output)
            feature_list.append(feature_attention)
        output = torch.cat(output_list, 0)
        feature_list = torch.cat(feature_list, 0)
        tan_output,time_attention = self.time_att_net(output)

        tan_output = tan_output[0]
        #
        predict = self.linear(tan_output)


        return predict,feature_list,time_attention, l1_regularization_list

    def half(self):

        return self

    def get_state(self, mode="full"):
        return self.state_dict(), []

    def set_state(self, w_server, w_local, mode="full"):


        self.load_state_dict(w_local)


