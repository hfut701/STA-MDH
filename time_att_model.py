import torch
import torch.nn as nn
from feature_att_model import matrix_mul, element_wise_mul


class TimeAttNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TimeAttNet, self).__init__()
        self.mlp = nn.Linear(input_size//2, hidden_size)
        self.mlp1 = nn.Linear(hidden_size, hidden_size//2)
        self.att = nn.Linear(hidden_size//2, 1)

        self.time_w1 = nn.Parameter(torch.Tensor( hidden_size,  hidden_size))
        self.time_b1 = nn.Parameter(torch.Tensor(1, hidden_size))
        self.time_w2 = nn.Parameter(torch.Tensor( hidden_size, 1))
        self._create_weights(mean=0.0, std=0.05)
        self.tanh = nn.Tanh()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(dim=0)

    def _create_weights(self, mean=0.0, std=0.05):
        self.time_w1.data.normal_(mean, std)
        self.time_b1.data.normal_(mean, std)
        self.time_w2.data.normal_(mean, std)

    def forward(self, input):


        h_output = self.mlp(input )

        h_output = self.mlp1(h_output)


        output = self.att(h_output)

        output_softmax = self.softmax(output.squeeze())
        output = element_wise_mul(h_output, output_softmax)

        return output, output_softmax
