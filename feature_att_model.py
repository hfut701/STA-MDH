import torch
import torch.nn as nn


def matrix_mul(input, weight, bias=False):
    feature_list = []
    for feature in input:

        if feature.size(0) == 40:
            feature = feature.unsqueeze(0)
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)
    return torch.cat(feature_list, 0).squeeze()


def element_wise_mul(input1, input2):
    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        # print(feature_2)
        # print(feature_2.shape)
        # print(feature_2.numel())
        if feature_2.numel() == 1:
            # feature_2 = feature_2.reshape(1, 1)
            feature_2 = feature_2.unsqueeze(0)
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)
    # a = torch.sum(output, 0)
    # b = a.unsqueeze(0)
    return torch.sum(output, 0).unsqueeze(0)


class FeatureAttNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(FeatureAttNet, self).__init__()
        self.mlp = nn.Linear(input_size, hidden_size)
        self.mlp1 = nn.Linear(hidden_size, hidden_size//2)
        self.att = nn.Linear(hidden_size//2, hidden_size // 4)
        self.att1 = nn.Linear(hidden_size // 4, 1)
        self.feature_w1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.feature_b1 = nn.Parameter(torch.Tensor(1, hidden_size))
        self.feature_w2 = nn.Parameter(torch.Tensor( hidden_size, 1))
        self._create_weights(mean=0.0, std=0.05)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)

        # self.mlp = nn.Linear(72, hidden_size) #fearture
    def _create_weights(self, mean=0.0, std=0.05):
        self.feature_w1.data.normal_(mean, std)
        self.feature_b1.data.normal_(mean, std)
        self.feature_w2.data.normal_(mean, std)

    def forward(self, input):

        output1 = self.mlp(input)


        h_output = self.mlp1(output1 )

        output = self.att(h_output)

        output  = torch.tanh(output )

        output = self.att1(output)


        output_softmax = self.softmax(output.squeeze())

        l1_regularization = torch.norm( output_softmax, p=1)

        output1 = element_wise_mul(h_output, output_softmax)
        if output_softmax.dim()==1:
            output_softmax = output_softmax.unsqueeze(0)

        if input.size(1)==1:
            return output1, output_softmax,l1_regularization
        else:
            return output1,output.squeeze().permute(1, 0),l1_regularization

