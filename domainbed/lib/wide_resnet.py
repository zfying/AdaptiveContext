# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
From https://github.com/meliketoy/wide-resnet.pytorch
"""

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes, kernel_size=1, stride=stride,
                    bias=True), )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    """Wide Resnet with the softmax layer chopped off"""
    def __init__(self, input_shape, depth, widen_factor, dropout_rate):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        # print('| Wide-Resnet %dx%d' % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(input_shape[0], nStages[0])
        self.layer1 = self._wide_layer(
            wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(
            wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(
            wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.layer4 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn2 = nn.BatchNorm2d(nStages[3])
        self.n_outputs = nStages[3]

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, feat_ext = None):
        if feat_ext is None:
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.relu(self.bn1(out))
            out = F.avg_pool2d(out, 16)
            return out.view(-1, self.n_outputs)
        else:
            add_feat = []
            out = self.conv1(x)
            out = self.layer1(out)
            if 'layer1' in feat_ext:
                add_feat.append(out)
            out = self.layer2(out)
            if 'layer2' in feat_ext:
                add_feat.append(out)
            out = self.layer3(out)
            out = F.relu(self.bn1(out))
            out = F.avg_pool2d(out, 16)
            out = out.view(-1, self.n_outputs)
            if 'layer3' in feat_ext:
                add_feat.append(out)
            return out, add_feat

# def WideResnet_VAE(nn.Module):
#     def __init__(self, input_shape, depth, widen_factor, dropout_rate, feat_ext = None):
#         super().__init__()
        
#         # params
#         self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = 1024, 1024, 256
        
#         # encode
#         self.wide_resnet = Wide_ResNet(input_shape, depth, widen_factor, dropout_rate, feat_ext)
#         self.fc1 = nn.Linear(2048, self.fc_hidden1)
#         self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
#         self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
#         self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
#         # Latent vectors mu and sigma
#         self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)      # output = CNN embedding latent variables
#         self.fc3_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables
        
#         # decode
#         self.convTrans6 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
#                                padding=self.pd4),
#             nn.BatchNorm2d(32, momentum=0.01),
#             nn.ReLU(inplace=True),
#         )
#         self.convTrans7 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
#                                padding=self.pd3),
#             nn.BatchNorm2d(8, momentum=0.01),
#             nn.ReLU(inplace=True),
#         )

#         self.convTrans8 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
#                                padding=self.pd2),
#             nn.BatchNorm2d(3, momentum=0.01),
#             nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
#         )
    
    
#     def encode(self, x):
#         x = self.wide_resnet(x)  # ResNet

#         # FC layers
#         x = self.bn1(self.fc1(x))
#         x = self.relu(x)
#         x = self.bn2(self.fc2(x))
#         x = self.relu(x)
#         # x = F.dropout(x, p=self.drop_p, training=self.training)
#         mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
#         return mu, logvar
    
#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = logvar.mul(0.5).exp_()
#             eps = Variable(std.data.new(std.size()).normal_())
#             return eps.mul(std).add_(mu)
#         else:
#             return mu
        
#     def decode(self, z):
#         x = self.relu(self.fc_bn4(self.fc4(z)))
#         x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 4, 4)
#         x = self.convTrans6(x)
#         x = self.convTrans7(x)
#         x = self.convTrans8(x)
#         x = F.interpolate(x, size=(64, 64), mode='bilinear')
#         return x

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         x_reconst = self.decode(z)

#         return x_reconst, z, mu, logvar
    
# def loss_function(recon_x, x, mu, logvar):
#     # MSE = F.mse_loss(recon_x, x, reduction='sum')
#     MSE = F.binary_cross_entropy(recon_x, x, reduction='sum')
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return MSE + KLD