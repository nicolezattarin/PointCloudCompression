import open3d as o3d
import os
import torch
import torch.nn as nn
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOR_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOR_DIR)
from utils import batch_quat2mat, batch_transform


class PointNet(nn.Module):
    def __init__(self, in_dim, gn, mlps=[64, 128, 1024]):
        super(PointNet, self).__init__()
        self.indices = dict()
        self.backbone = nn.Sequential()
        self.dims = dict()
        for i, out_dim in enumerate(mlps):
            self.backbone.add_module(f'pointnet_conv_{i}',
                                     nn.Conv1d(in_dim, out_dim, 1, 1, 0))

            if gn:
                self.backbone.add_module(f'pointnet_gn_{i}',
                                    nn.GroupNorm(8, out_dim))
            self.backbone.add_module(f'pointnet_relu_{i}',
                                     nn.ReLU(inplace=True))
            in_dim = out_dim

    def forward(self, x):
        x = self.backbone(x)
        # x, _ = torch.max(x, dim=2)
        return x

class PointNetInverse(nn.Module):
    def __init__(self, in_dim, gn, mlps=[1024, 128, 64]):
        # use deconvolutional layers
        super(PointNetInverse, self).__init__()
        self.backbone = nn.Sequential()
        for i, out_dim in enumerate(mlps):
            self.backbone.add_module(f'pointnet_conv_{i}',
                                     nn.ConvTranspose1d(in_dim, out_dim, 1, 1, 0))
            if gn:
                self.backbone.add_module(f'pointnet_gn_{i}',
                                    nn.GroupNorm(8, out_dim))
            self.backbone.add_module(f'pointnet_relu_{i}',
                                     nn.ReLU(inplace=True))
            in_dim = out_dim
        
        self.backbone.add_module(f'pointnet_conv_{i+1}', nn.ConvTranspose1d(in_dim, 3, 1, 1, 0))

    def forward(self, x):
        x = self.backbone(x)
        return x


class Autoencode_1 (nn.Module):


    def __init__(self, gn, in_dim, in_dim2=2048,mlps=[64, 128, 1024]):
        super(Autoencoder_1, self).__init__()
        self.in_dim = in_dim
        self.encoder = PointNet(in_dim=in_dim, gn=gn, mlps=mlps)
        self.decoder = PointNetInverse(in_dim=mlps[-1], gn=gn, mlps=mlps[::-1])
        
    def forward(self, x):
        encoded = self.encoder(x)
        out = self.decoder(encoded)
    
        return encoded, out



from torch import nn

class Autoencoder(nn.Module):
    def linear_block_en(self, flatten_dim, out_dims=[32, 16, 8]):
        layers = []
        in_dim = flatten_dim
        for out_dim in out_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        return nn.Sequential(*layers)


    def linear_block_de(self, in_dims=[8, 16, 32], flatten_dim=64, unflatten_dim=(64,64)):
        layers = []
        in_dim = in_dims[0]
        for out_dim in in_dims[1:]:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, flatten_dim))
        layers.append(nn.Unflatten(1, unflatten_dim))

        return nn.Sequential(*layers)


    def __init__(self):
        super(Autoencoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 16, kernel_size=3)
        self.maxpool1 = nn.MaxPool1d(kernel_size=5, return_indices=True)
    
        self.encoder1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(16, 32,kernel_size=3, padding='same'),
        )
        self.maxpool2 = nn.MaxPool1d(kernel_size=5, return_indices=True)

        self.encoder2 = nn.Sequential(
            nn.Tanh(),
            nn.Conv1d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
        )
        # flatten 
        self.flatten = nn.Flatten()

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.maxunpool1 = nn.MaxUnpool1d(kernel_size=5)

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose1d(32,16,kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.unmaxunpool2 = nn.MaxUnpool1d(kernel_size=5)
        self.unconv1 = nn.ConvTranspose1d(16,3,kernel_size=3)

        self.float()

    def forward(self, x):
        # print ("input shape: ", x.shape)

        x = self.conv1(x)
        outputsize1 = x.shape
        x,indices1 = self.maxpool1(x)
        # print ("maxpool1")
        # print (x.shape)
        
        x = self.encoder1(x)
        # print ("encoder1")
        # print (x.shape)

        outsize2 = x.shape
        x,indices2 = self.maxpool2(x)
        x = self.encoder2(x)

        unflatten_dim = x.shape[1:]

        # flatten
        x = self.flatten(x)
        flatten_dim = x.shape[1]
        coding = self.linear_block_en(flatten_dim)(x)
        # unflatten
        x = self.linear_block_de(flatten_dim=flatten_dim, unflatten_dim=unflatten_dim) (coding)


        x = self.decoder2(x)

        x = self.unmaxunpool2(x, output_size = outsize2, indices=indices2)

        x = self.decoder1(x)


        x = self.maxunpool1(x, output_size = outputsize1, indices=indices1)


        x = self.unconv1(x)

        output = nn.Tanh()(x)

        return coding, output
