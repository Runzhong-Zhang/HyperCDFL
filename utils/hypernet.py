#!/usr/bin/python3.7

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from datetime import datetime
import sys
import pdb


class HyperNetwork_layernorm(nn.Module):

    def __init__(self, z_dim=512, n_classes=48, hidden_size=64):
        super(HyperNetwork_layernorm, self).__init__()
        print('Using HyperNetwork_layernorm')
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.w1 = Parameter(torch.fmod(torch.randn(512, 48).cuda(),1))
        self.b1 = Parameter(torch.fmod(torch.randn(48).cuda(),1))
        self.w2 = Parameter(torch.fmod(torch.randn(48, 768).cuda(),1))
        self.b2 = Parameter(torch.fmod(torch.randn(768).cuda(),1))
        self.w3 = Parameter(torch.fmod(torch.randn(16, hidden_size).cuda(),1))
        self.b3 = Parameter(torch.fmod(torch.randn(hidden_size).cuda(),1))
        self.layer_norm1 = nn.LayerNorm([16])
        self.layer_norm1.cuda()
        self.layer_norm2 = nn.LayerNorm([self.hidden_size])
        self.layer_norm2.cuda()


    def forward(self, z):
        # print('z:', min(z), max(z))
        h1 = torch.matmul(z, self.w1) + self.b1
        h1 = torch.matmul(h1, self.w2) + self.b2
        h1 = h1.view(self.n_classes, 16)
        h1 = self.layer_norm1(h1)
        h1 = torch.matmul(h1, self.w3) + self.b3
        kernel = h1.view(self.n_classes, self.hidden_size)
        kernel = self.layer_norm2(kernel)
        # print('kernel:', min(min(kernel[i]) for i in range(48)), max(max(kernel[i]) for i in range(48)))
        return kernel


class HyperNetwork1(nn.Module):

    def __init__(self, z_dim=512, n_classes=48, hidden_size=64):
        super(HyperNetwork1, self).__init__()
        print('Using HyperNetwork1')
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.w1 = Parameter(torch.fmod(torch.randn(8, self.n_classes).cuda(),2))
        self.b1 = Parameter(torch.fmod(torch.randn(self.n_classes).cuda(),2))


    def forward(self, z):
        print('z:', min(z), max(z))
        z = z.view(64, 8)
        h1 = torch.matmul(z, self.w1) + self.b1
        kernel = h1.view(self.n_classes, self.hidden_size)
        print('kernel:', min(min(kernel[i]) for i in range(48)), max(max(kernel[i]) for i in range(48)))
        return kernel


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
        h1 = torch.nn.functional.normalize(h1)
        h1 = torch.matmul(h1, self.w3) + self.b3
        kernel = h1.view(self.n_classes, self.hidden_size)
        kernel = torch.nn.functional.normalize(kernel)
        # print('kernel:', min(min(kernel[i]) for i in range(48)), max(max(kernel[i]) for i in range(48)))
        return kernel


if __name__ == '__main__':
    a = datetime.now()
    net = HyperNetwork1()
    input = torch.rand(512).cuda()
    output = net(input)
    print(output.shape)
    b = datetime.now()
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))

    a = datetime.now()
    net = HyperNetwork()
    input = torch.rand(512).cuda()
    output = net(input)
    print(output.shape)
    b = datetime.now()
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
