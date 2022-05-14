import torch
import torch.nn as nn
from utils.network import Net
from torch.nn.parameter import Parameter
from utils.hypernet import HyperNetwork, HyperNetwork1


class HyperNetwork(nn.Module):

    def __init__(self, z_dim=512, n_classes=48, hidden_size=64):
        super(HyperNetwork, self).__init__()
        print('Using HyperNetwork')
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.w1 = Parameter(torch.fmod(torch.randn(512, 48).cuda(),1))
        self.b1 = Parameter(torch.fmod(torch.randn(48).cuda(),1))
        self.w2 = Parameter(torch.fmod(torch.randn(48, 768).cuda(),1))
        self.b2 = Parameter(torch.fmod(torch.randn(768).cuda(),1))
        self.w3 = Parameter(torch.fmod(torch.randn(16, hidden_size).cuda(),1))
        self.b3 = Parameter(torch.fmod(torch.randn(hidden_size).cuda(),1))


    def forward(self, z):
        # print('z:', min(z), max(z))
        h1 = torch.matmul(z, self.w1) + self.b1
        h1 = torch.matmul(h1, self.w2) + self.b2
        h1 = h1.view(self.n_classes, 16)
        layer_norm1 = nn.LayerNorm([16])
        h1 = layer_norm1(h1).cuda()
        # h1 = torch.nn.functional.normalize(h1)
        h1 = torch.matmul(h1, self.w3) + self.b3
        kernel = h1.view(self.n_classes, self.hidden_size)
        layer_norm2 = nn.LayerNorm([self.hidden_size])
        kernel = kernel(h1)
        # kernel = torch.nn.functional.normalize(kernel)
        # print('kernel:', min(min(kernel[i]) for i in range(48)), max(max(kernel[i]) for i in range(48)))
        return kernel

z = Parameter(torch.fmod(torch.randn(512).cuda(),2))
model = HyperNetwork()
model.cuda()
output = model(z)
print(output.shape)