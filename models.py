import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class GenerativeModel(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.output_size = 256
        self.layers = nn.ModuleList([nn.Conv1d(self.output_size, self.output_size, 2, stride=2) for i in range(int(np.log2(self.input_size)))])
        
    def forward(self, x, saves):
        if saves is None:
            saves = []
            x = x.reshape(1, 256, -1)
            for i, layer in enumerate(self.layers):
                new_layer = nn.Conv1d(layer.in_channels, layer.out_channels, 2, dilation=pow(2, i))
                new_layer.load_state_dict(layer.state_dict())
                x = F.tanh(new_layer(x))
                saves.append(list(torch.chunk(x, x.shape[-1], dim=2)[-pow(2, i+1):]))
            saves = saves[:-1]
        else:
            new_saves = []
            x = self.layers[0](x[-2:].reshape(1, 256, -1))
            for i, layer in enumerate(self.layers[1:]):
                new_saves.append(saves[i-1][1:] + [x])
                x = F.tanh(layer(torch.cat((saves[i-1][0], x), dim=2)))
        x = x.reshape(1, 256)
        return (x, saves)
