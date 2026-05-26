#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import torch
import torch.nn.functional as F
from ....codebase import utils as ut
from torch import autograd, nn, optim
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    
def dag_right_linear(input, weight, bias=None):
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        ret = torch.addmm(bias, input, weight.t())
    else:
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        ret = output
    return ret
    
def dag_left_linear(input, weight, bias=None):
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        ret = torch.addmm(bias, input, weight.t())
    else:
        output = weight.matmul(input)
        if bias is not None:
            output += bias
        ret = output
    return ret



class MaskLayer(nn.Module):
    def __init__(self, z_dim, concept=4, z2_dim=4):
        super().__init__()
        self.z_dim = z_dim
        self.z2_dim = z2_dim
        self.concept = concept
        
        self.elu = nn.ELU()
        self.net1 = nn.Sequential(
            nn.Linear(z2_dim , 32),
            nn.ELU(),
            nn.Linear(32, z2_dim),
        )
        self.net2 = nn.Sequential(
            nn.Linear(z2_dim , 32),
            nn.ELU(),
            nn.Linear(32, z2_dim),
        )
        self.net3 = nn.Sequential(
            nn.Linear(z2_dim , 32),
            nn.ELU(),
          nn.Linear(32, z2_dim),
        )
        self.net4 = nn.Sequential(
            nn.Linear(z2_dim , 32),
            nn.ELU(),
            nn.Linear(32, z2_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(z2_dim , 32),
            nn.ELU(),
            nn.Linear(32, z2_dim),
        )
    def masked(self, z):
        z = z.view(-1, self.z_dim)
        z = self.net(z)
        return z
   
    def masked_sep(self, z):
        z = z.view(-1, self.z_dim)
        z = self.net(z)
        return z
   
    def mix(self, z):
        zy = z.view(-1, self.concept*self.z2_dim)
        if self.z2_dim == 1:
            zy = zy.reshape(zy.size()[0],zy.size()[1],1)
            if self.concept ==4:
                zy1, zy2, zy3, zy4= zy[:,0],zy[:,1],zy[:,2],zy[:,3]
            elif self.concept ==3:
                zy1, zy2, zy3= zy[:,0],zy[:,1],zy[:,2]
        else:
            if self.concept ==4:
                zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim//self.concept, dim = 1)
            elif self.concept ==3:
                zy1, zy2, zy3= torch.split(zy, self.z_dim//self.concept, dim = 1)
        rx1 = self.net1(zy1)
        rx2 = self.net2(zy2)
        rx3 = self.net3(zy3)
        if self.concept ==4:
            rx4 = self.net4(zy4)
            h = torch.cat((rx1,rx2,rx3,rx4), dim=1)
        elif self.concept ==3:
            h = torch.cat((rx1,rx2,rx3), dim=1)
        #print(h.size())
        return h


class CausalLayer(nn.Module):
    def __init__(self, z_dim, concept=4,z1_dim=4):
        super().__init__()
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.concept = concept
        
        self.elu = nn.ELU()
        self.net1 = nn.Sequential(
            nn.Linear(z1_dim , 32),
            nn.ELU(),
            nn.Linear(32, z1_dim),
        )
        self.net2 = nn.Sequential(
            nn.Linear(z1_dim , 32),
            nn.ELU(),
            nn.Linear(32, z1_dim),
        )
        self.net3 = nn.Sequential(
            nn.Linear(z1_dim , 32),
            nn.ELU(),
          nn.Linear(32, z1_dim),
        )
        self.net4 = nn.Sequential(
            nn.Linear(z1_dim , 32),
            nn.ELU(),
            nn.Linear(32, z1_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(z_dim , 128),
            nn.ELU(),
            nn.Linear(128, z_dim),
        )
   
    def calculate(self, z, v):
        z = z.view(-1, self.z_dim)
        z = self.net(z)
        return z, v
   
    def masked_sep(self, z, v):
        z = z.view(-1, self.z_dim)
        z = self.net(z)
        return z,v
   
    def calculate_dag(self, z, v):
        zy = z.view(-1, self.concept*self.z1_dim)
        if self.z1_dim == 1:
            zy = zy.reshape(zy.size()[0],zy.size()[1],1)
            zy1, zy2, zy3, zy4= zy[:,0],zy[:,1],zy[:,2],zy[:,3]
        else:
            zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim//self.concept, dim = 1)
        rx1 = self.net1(zy1)
        rx2 = self.net2(zy2)
        rx3 = self.net3(zy3)
        rx4 = self.net4(zy4)
        h = torch.cat((rx1,rx2,rx3,rx4), dim=1)
        #print(h.size())
        return h,v
   
   
class Attention(nn.Module):
  def __init__(self, in_features, bias=False):
    super().__init__()
    self.M =  nn.Parameter(torch.nn.init.normal_(torch.zeros(in_features,in_features), mean=0, std=1))
    self.sigmd = torch.nn.Sigmoid()
    #self.M =  nn.Parameter(torch.zeros(in_features,in_features))
    #self.A = torch.zeros(in_features,in_features).to(device)
    
  def attention(self, z, e):
    a = z.matmul(self.M).matmul(e.permute(0,2,1))
    a = self.sigmd(a)
    #print(self.M)
    A = torch.softmax(a, dim = 1)
    e = torch.matmul(A,e)
    return e, A
    
class DagLayer(nn.Linear):
    def __init__(self, in_features, out_features,i = False, bias=False, initial=True,causal_discover=False):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.i = i
        self.a = torch.zeros(out_features,out_features)
        self.a = self.a
        if initial:
            #self.a[0][1], self.a[0][2], self.a[0][3] = 1,1,1
            self.a[0][1], self.a[0][2], self.a[0][3] = 1,1,1
            self.a[1][2], self.a[1][3] = 1,1
        if causal_discover:
            self.A = nn.Parameter(self.a)
        else:
            self.A = self.a  

        #self.A = nn.Parameter(self.a)
        
        # self.b = torch.eye(out_features)
        # self.b = self.b
        # self.B = nn.Parameter(self.b)
        
        self.I = nn.Parameter(torch.eye(out_features))
        self.I.requires_grad=False
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
    def mask_z(self,x):
        A = self.A.to(x.device)
        #if self.i:
        #    x = x.view(-1, x.size()[1], 1)
        #    x = torch.matmul((self.B+0.5).t().int().float(), x)
        #    return x
        x = torch.matmul(A.t(), x)
        return x
        
    def mask_u(self,x):
        A = self.A.to(x.device)
        #if self.i:
        #    x = x.view(-1, x.size()[1], 1)
        #    x = torch.matmul((self.B+0.5).t().int().float(), x)
        #    return x
        x = x.view(-1, x.size()[1], 1)
        x = torch.matmul(A.t(), x)
        return x
        
    def inv_cal(self, x,v):
        A = self.A.to(x.device)
        if x.dim()>2:
            x = x.permute(0,2,1)
        x = F.linear(x, self.I - A, self.bias)
       
        if x.dim()>2:
            x = x.permute(0,2,1).contiguous()
        return x,v

    def calculate_dag(self, x, v):
        #print(self.A)
        #x = F.linear(x, torch.inverse((torch.abs(self.A))+self.I), self.bias)
        '''
        transpose here using the matrix porperty
        tranpose(AB) = B_T * A_T

        '''
        A = self.A.to(x.device)
        if x.dim()>2:
            #transpose
            x = x.permute(0,2,1)
        # equation 1
        x = F.linear(x, torch.inverse(self.I - A.t()), self.bias) 
        #print(x.size())
       
        if x.dim()>2:
            x = x.permute(0,2,1).contiguous()
        return x,v
        
    def calculate_cov(self, x, v):
        #print(self.A)
        A = self.A.to(x.device)
        v = ut.vector_expand(v)
        #x = F.linear(x, torch.inverse((torch.abs(self.A))+self.I), self.bias)
        x = dag_left_linear(x, torch.inverse(self.I - A), self.bias)
        v = dag_left_linear(v, torch.inverse(self.I - A), self.bias)
        v = dag_right_linear(v, torch.inverse(self.I - A), self.bias)
        #print(v)
        return x, v
        
    def calculate_gaussian_ini(self, x, v):
        A = self.A.to(x.device)
        print(self.A)
        
        #x = F.linear(x, torch.inverse((torch.abs(self.A))+self.I), self.bias)
        
        if x.dim()>2:
            x = x.permute(0,2,1)
            v = v.permute(0,2,1)
        x = F.linear(x, torch.inverse(self.I - A), self.bias)
        v = F.linear(v, torch.mul(torch.inverse(self.I - A),torch.inverse(self.I - A)), self.bias)
        if x.dim()>2:
            x = x.permute(0,2,1).contiguous()
            v = v.permute(0,2,1).contiguous()
        return x, v
    #def encode_
    def forward(self, x):
        A = self.A.to(x.device)
        x = x * torch.inverse((A)+self.I)
        return x
    def calculate_gaussian(self, x, v):
        A = self.A.to(x.device)
        print(self.A)
        #x = F.linear(x, torch.inverse((torch.abs(self.A))+self.I), self.bias)
        
        if x.dim()>2:
            x = x.permute(0,2,1)
            v = v.permute(0,2,1)
        x = dag_left_linear(x, torch.inverse(self.I - A), self.bias)
        v = dag_left_linear(v, torch.inverse(self.I - A), self.bias)
        v = dag_right_linear(v, torch.inverse(self.I - A), self.bias)
        if x.dim()>2:
            x = x.permute(0,2,1).contiguous()
            v = v.permute(0,2,1).contiguous()
        return x, v
    #def encode_
    def forward(self, x):
        A = self.A.to(x.device)
        x = x * torch.inverse((A)+self.I)
        return x
      
class ConvEncoder(nn.Module):
    def __init__(self, out_dim=None):
        super().__init__()
        # init 96*96
        self.conv1 = torch.nn.Conv2d(3, 32, 4, 2, 1) # 48*48
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2, 1, bias=False) # 24*24
        self.conv3 = torch.nn.Conv2d(64, 1, 4, 2, 1, bias=False)
        #self.conv4 = torch.nn.Conv2d(128, 1, 1, 1, 0) # 54*44
   
        self.LReLU = torch.nn.LeakyReLU(0.2, inplace=True)
        self.convm = torch.nn.Conv2d(1, 1, 4, 2, 1)
        self.convv = torch.nn.Conv2d(1, 1, 4, 2, 1)
        self.mean_layer = nn.Sequential(
            torch.nn.Linear(8*8, 16)
            ) # 12*12
        self.var_layer = nn.Sequential(
            torch.nn.Linear(8*8, 16)
            )
        # self.fc1 = torch.nn.Linear(6*6*128, 512)
        self.conv6 = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),
            nn.ReLU(True),
            nn.Conv2d(256,128 , 1)
        )

    def encode(self, x):
        x = self.LReLU(self.conv1(x))
        x = self.LReLU(self.conv2(x))
        x = self.LReLU(self.conv3(x))
        #x = self.LReLU(self.conv4(x))
        #print(x.size())
        hm = self.convm(x)
        #print(hm.size())
        hm = hm.view(-1, 8*8)
        hv = self.convv(x)
        hv = hv.view(-1, 8*8)
        mu, var = self.mean_layer(hm), self.var_layer(hv)
        var = F.softplus(var) + 1e-8
        #var = torch.reshape(var, [-1, 16, 16])
        #print(mu.size())
        return  mu, var
    def encode_simple(self,x):
        x = self.conv6(x)
        m,v = ut.gaussian_parameters(x, dim=1)
        #print(m.size())
        return m,v
class ConvDecoder(nn.Module):
    def __init__(self, out_dim = None):
        super().__init__()
   
        self.net6 = nn.Sequential(
                nn.Conv2d(16, 128, 1),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(128, 64, 4),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(64, 64, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(32, 32, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(32, 32, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(32, 3, 4, 2, 1)
        )
   
    def decode_sep(self,x):
        return None
   
    def decode(self, z):
        z = z.view(-1, 16, 1, 1)
        z = self.net6(z)
        return z

class ConvDec(nn.Module):
  def __init__(self, out_dim = None):
    super().__init__()
    self.concept = 4
    self.z1_dim = 16
    self.z_dim = 64
    self.net1 = ConvDecoder()
    self.net2 = ConvDecoder()
    self.net3 = ConvDecoder()
    self.net4 = ConvDecoder()
    self.net5 = nn.Sequential(
            nn.Linear(16, 512),
          nn.BatchNorm1d(512),
          nn.Linear(512, 1024),
          nn.BatchNorm1d(1024)
     )
    self.net6 = nn.Sequential(
          nn.Conv2d(16, 128, 1),
      nn.LeakyReLU(0.2),
      nn.ConvTranspose2d(128, 64, 4),
      nn.LeakyReLU(0.2),
      nn.ConvTranspose2d(64, 64, 4, 2, 1),
      nn.LeakyReLU(0.2),
      nn.ConvTranspose2d(64, 32, 4, 2, 1),
      nn.LeakyReLU(0.2),
      nn.ConvTranspose2d(32, 32, 4, 2, 1),
      nn.LeakyReLU(0.2),
      nn.ConvTranspose2d(32, 32, 4, 2, 1),
      nn.LeakyReLU(0.2),
      nn.ConvTranspose2d(32, 3, 4, 2, 1)
        )
        
  def decode_sep(self, z, u, y=None):
    z = z.view(-1, self.concept*self.z1_dim)
    zy = z if y is None else torch.cat((z, y), dim=1)
    zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim//self.concept, dim = 1)
    rx1 = self.net1.decode(zy1)
    #print(rx1.size())
    rx2 = self.net2.decode(zy2)
    rx3 = self.net3.decode(zy3)
    rx4 = self.net4.decode(zy4)
    z = (rx1+rx2+rx3+rx4)/4
    return z
    
  def decode(self, z, u, y=None):
    z = z.view(-1, self.concept*self.z1_dim, 1, 1)
    z = self.net6(z)
    #print(z.size())
    
    return z

class Encoder(nn.Module):
    def __init__(self, z_dim, channel=4, y_dim=4):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.channel = channel
        self.fc1 = nn.Linear(self.channel*96*96, 300)
        self.fc2 = nn.Linear(300+y_dim, 300)
        self.fc3 = nn.Linear(300, 300)
        self.fc4 = nn.Linear(300, 2 * z_dim)
        self.LReLU = nn.LeakyReLU(0.2, inplace=True)
        self.net = nn.Sequential(
            nn.Linear(self.channel*96*96, 900),
            nn.ELU(),
            nn.Linear(900, 300),
            nn.ELU(),
            nn.Linear(300, 2 * z_dim),
        )

    def conditional_encode(self, x, l):
        x = x.view(-1, self.channel*96*96)
        x = F.elu(self.fc1(x))
        l = l.view(-1, 4)
        x = F.elu(self.fc2(torch.cat([x, l], dim=1)))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        m, v = ut.gaussian_parameters(x, dim=1)
        return m,v

    def encode(self, x, y=None):
        xy = x if y is None else torch.cat((x, y), dim=1)
        xy = xy.view(-1, self.channel*96*96)
        h = self.net(xy)
        m, v = ut.gaussian_parameters(h, dim=1)
        #print(self.z_dim,m.size(),v.size())
        return m, v
   
   
class Decoder_DAG(nn.Module):
    def __init__(self, z_dim, concept, z1_dim, channel = 4, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.concept = concept
        self.y_dim = y_dim
        self.channel = channel
        #print(self.channel)
        self.elu = nn.ELU()
        self.net1 = nn.Sequential(
            nn.Linear(z1_dim + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 1024),
            nn.ELU(),
            nn.Linear(1024, self.channel*96*96)
        )
        self.net2 = nn.Sequential(
            nn.Linear(z1_dim + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 1024),
            nn.ELU(),
            nn.Linear(1024, self.channel*96*96)
        )
        self.net3 = nn.Sequential(
            nn.Linear(z1_dim + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 1024),
            nn.ELU(),
            nn.Linear(1024, self.channel*96*96)
        )
        self.net4 = nn.Sequential(
            nn.Linear(z1_dim + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 1024),
            nn.ELU(),
            nn.Linear(1024, self.channel*96*96)
        )
        self.net5 = nn.Sequential(
            nn.ELU(),
            nn.Linear(1024, self.channel*96*96)
        )
   
        self.net6 = nn.Sequential(
            nn.Linear(z_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 1024),
            nn.ELU(),
            nn.Linear(1024, 1024),
            nn.ELU(),
            nn.Linear(1024, self.channel*96*96)
        )
    def decode_condition(self, z, u):
        #z = z.view(-1,3*4)
        z = z.view(-1, 3*4)
        z1, z2, z3 = torch.split(z, self.z_dim//4, dim = 1)
        #print(u[:,0].reshape(1,u.size()[0]).size())
        rx1 = self.net1(torch.transpose(torch.cat((torch.transpose(z1, 1,0), u[:,0].reshape(1,u.size()[0])), dim = 0), 1, 0))
        rx2 = self.net2(torch.transpose(torch.cat((torch.transpose(z2, 1,0), u[:,1].reshape(1,u.size()[0])), dim = 0), 1, 0))
        rx3 = self.net3(torch.transpose(torch.cat((torch.transpose(z3, 1,0), u[:,2].reshape(1,u.size()[0])), dim = 0), 1, 0))
   
        h = self.net4( torch.cat((rx1,rx2, rx3), dim=1))
        return h

    def decode_mix(self, z):
        z = z.permute(0,2,1)
        z = torch.sum(z, dim = 2, out=None) 
        #print(z.contiguous().size())
        z = z.contiguous()
        h = self.net1(z)
        return h
   
    def decode_union(self, z, u, y=None):
        
        z = z.view(-1, self.concept*self.z1_dim)
        zy = z if y is None else torch.cat((z, y), dim=1)
        if self.z1_dim == 1:
            zy = zy.reshape(zy.size()[0],zy.size()[1],1)
            zy1, zy2, zy3, zy4 = zy[:,0],zy[:,1],zy[:,2],zy[:,3]
        else:
            zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim//self.concept, dim = 1)
        rx1 = self.net1(zy1)
        rx2 = self.net2(zy2)
        rx3 = self.net3(zy3)
        rx4 = self.net4(zy4)
        h = self.net5((rx1+rx2+rx3+rx4)/4)
        return h,h,h,h,h
   
    def decode(self, z, u , y = None):
        z = z.view(-1, self.concept*self.z1_dim)
        h = self.net6(z)
        return h, h,h,h,h
    
    def decode_sep(self, z, u, y=None):
        z = z.view(-1, self.concept*self.z1_dim)
        zy = z if y is None else torch.cat((z, y), dim=1)
            
        if self.z1_dim == 1:
            zy = zy.reshape(zy.size()[0],zy.size()[1],1)
            if self.concept ==4:
                zy1, zy2, zy3, zy4= zy[:,0],zy[:,1],zy[:,2],zy[:,3]
            elif self.concept ==3:
                zy1, zy2, zy3= zy[:,0],zy[:,1],zy[:,2]
        else:
            if self.concept ==4:
                zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim//self.concept, dim = 1)
            elif self.concept ==3:
                zy1, zy2, zy3= torch.split(zy, self.z_dim//self.concept, dim = 1)
        rx1 = self.net1(zy1)
        rx2 = self.net2(zy2)
        rx3 = self.net3(zy3)
        if self.concept ==4:
            rx4 = self.net4(zy4)
            h = (rx1+rx2+rx3+rx4)/self.concept
        elif self.concept ==3:
            h = (rx1+rx2+rx3)/self.concept
        
        return h,h,h,h,h
   
    def decode_cat(self, z, u, y=None):
        z = z.view(-1, 4*4)
        zy = z if y is None else torch.cat((z, y), dim=1)
        zy1, zy2, zy3, zy4 = torch.split(zy, 1, dim = 1)
        rx1 = self.net1(zy1)
        rx2 = self.net2(zy2)
        rx3 = self.net3(zy3)
        rx4 = self.net4(zy4)
        h = self.net5( torch.cat((rx1,rx2, rx3, rx4), dim=1))
        return h
   
   
class Decoder(nn.Module):
    def __init__(self, z_dim, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 4*96*96)
        )

    def decode(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)
        return self.net(zy)

class Classifier(nn.Module):
    def __init__(self, y_dim):
        super().__init__()
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(784, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, y_dim)
        )

    def classify(self, x):
        return self.net(x)



from torch.distributions import MultivariateNormal
from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor')

class GaussianConvEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 num_vars=4,
                 **kwargs) -> None:
        super().__init__()
        
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.in_channels = in_channels
        self.num_vars = num_vars

        modules = []
        if hidden_dims is None:
            if self.num_vars == 4:
                hidden_dims = [16, 32, 32, 64, 64, 128]
                ## hidden_dims = [16, 32, 32, 64, 64, 128,128]
            elif self.num_vars == 2:
                hidden_dims = [16, 32, 64, 128]
                
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        # for i in range(len(hidden_dims) - 1):
        #     modules.append(
        #         nn.Sequential(
        #             nn.ConvTranspose2d(hidden_dims[i],
        #                                hidden_dims[i + 1],
        #                                kernel_size=3,
        #                                stride = 2,
        #                                padding=1,
        #                                output_padding=1),
        #             nn.BatchNorm2d(hidden_dims[i + 1]),
        #             nn.LeakyReLU())
        #     )
    
    def gaussian_parameters(self, h, dim=-1):
        """
        Converts generic real-valued representations into mean and variance
        parameters of a Gaussian distribution

        Args:
            h: tensor: (batch, ..., dim, ...): Arbitrary tensor
            dim: int: (): Dimension along which to split the tensor for mean and
                variance

        Returns:z
            m: tensor: (batch, ..., dim / 2, ...): Mean
            v: tensor: (batch, ..., dim / 2, ...): Variance
        """
        m, h = torch.split(h, h.size(dim) // 2, dim=dim)
        v = F.softplus(h) + 1e-8
        
        return m, v


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        log_var = F.softplus(log_var) + 1e-8
        
        return [mu, log_var]
    
    


class GaussianConvEncoderClf(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 num_vars=4,
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.in_channels = in_channels
        self.num_vars = num_vars

        modules = []
        if hidden_dims is None:
            if self.num_vars == 4:
                hidden_dims = [16, 32, 32, 64, 64, 128]
            elif self.num_vars == 2:
                hidden_dims = [16, 32, 64, 128]
                
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
        
        self.fc = nn.Linear(hidden_dims[-1]*4, 1)

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
    
    def gaussian_parameters(self, h, dim=-1):
        """
        Converts generic real-valued representations into mean and variance
        parameters of a Gaussian distribution

        Args:
            h: tensor: (batch, ..., dim, ...): Arbitrary tensor
            dim: int: (): Dimension along which to split the tensor for mean and
                variance

        Returns:z
            m: tensor: (batch, ..., dim / 2, ...): Mean
            v: tensor: (batch, ..., dim / 2, ...): Variance
        """
        m, h = torch.split(h, h.size(dim) // 2, dim=dim)
        v = F.softplus(h) + 1e-8
        
        return m, v


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        log_var = F.softplus(log_var) + 1e-8
        
        return [mu, log_var]
    
    def forward(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        
        out = self.fc(result)
        
        return out
    
    


# class MLP(nn.Module):
#     """ a simple 4-layer MLP """

#     def __init__(self, latent_dim, num_var):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.num_var = num_var

#         self.net = nn.Sequential(
#             nn.Linear(self.latent_dim // self.num_var, self.latent_dim),
#             nn.LeakyReLU(),
#             nn.Linear(self.latent_dim, self.latent_dim // self.num_var)
#         )

#     def forward(self, x):
#         return self.net(x)

class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, latent_dim, num_var,middle_dim=64,use_bias=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_var = num_var

        self.net = nn.Sequential(
            nn.Linear(self.latent_dim // self.num_var, middle_dim,bias=use_bias),
            nn.LeakyReLU(),
            nn.Linear(middle_dim, middle_dim,bias=use_bias),
            nn.LeakyReLU(),
            nn.Linear(middle_dim, self.latent_dim // self.num_var,bias=use_bias),
        )

    def forward(self, x):
        return self.net(x)


class CausalModeling(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 num_var=None,
                 learn = False,
                 use_bias=False,
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.num_var = num_var

        if learn:
            self.a = torch.zeros(self.num_var, self.num_var)
            self.A = nn.Parameter(self.a)
        else:
            self.A = torch.tensor([[0, 1], [0, 0]])

        self.nonlinearities = nn.ModuleDict()

        for i in range(self.num_var):
            self.nonlinearities[str(i)] = MLP(latent_dim=latent_dim, num_var=num_var,use_bias=use_bias)
        
        # self.nonlinearity1 = nn.Sequential(
        #     nn.Linear(self.latent_dim // self.num_var, latent_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(latent_dim, self.latent_dim // self.num_var)
        # )

        # self.nonlinearity2 = nn.Sequential(
        #     nn.Linear(self.latent_dim // self.num_var, latent_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(latent_dim, self.latent_dim // self.num_var)
        # )

        # self.nonlinearity3 = nn.Sequential(
        #     nn.Linear(self.latent_dim // self.num_var, latent_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(latent_dim, self.latent_dim // self.num_var)
        # )
        
        # self.nonlinearity4 = nn.Sequential(
        #     nn.Linear(self.latent_dim // self.num_var, latent_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(latent_dim, self.latent_dim // self.num_var)
        # )

    def causal_masking(self, u, A):
        u = u.reshape(-1, self.num_var, self.latent_dim // self.num_var)

        z_pre = torch.matmul(A.t().to(u.device), u)
        
        return z_pre

    def nonlinearity_add_back_noise(self, u, z_pre,result_v_indices=[]):
        # z_pre = z_pre.reshape(-1, self.num_var, self.latent_dim // self.num_var)
        u = u.reshape(-1, self.num_var, self.latent_dim // self.num_var)
        z_post = torch.zeros(u.shape)

        for i in range(self.num_var):
            if len(result_v_indices)>0 and i in result_v_indices:
                z_post[:, i, :] = self.nonlinearities[str(i)](z_pre[:, i, :])
            else:
                z_post[:, i, :] = self.nonlinearities[str(i)](z_pre[:, i, :]) + u[:, i, :]
        # for i in range(self.num_var):
        #     if len(result_v_indices)>0 and i in result_v_indices:

        #     z_post[:, i, :] = self.nonlinearities[str(i)](z_pre[:, i, :]) + u[:, i, :]

        # for i in range(self.num_var):
        # z_post[:, 0, :] = self.nonlinearity1(z_pre[:, 0, :]) + u[:, 0, :]
        # z_post[:, 1, :] = self.nonlinearity2(z_pre[:, 1, :]) + u[:, 1, :]
        # z_post[:, 2, :] = self.nonlinearity3(z_pre[:, 2, :]) + u[:, 2, :]
        # z_post[:, 3, :] = self.nonlinearity4(z_pre[:, 3, :]) + u[:, 3, :]

        return z_post
        #return z_post.reshape(-1, self.num_var*(self.latent_dim // self.num_var))


# class MLP_MASK(nn.Module):
#     """ a simple 4-layer MLP """

#     def __init__(self, nin, nout, nh):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(nin, nh),
#             nn.ReLU(),
#             nn.Linear(nh, nh),
#             nn.ReLU(),
#             nn.Linear(nh, nout),
#             nn.Sigmoid(),
#         )
#     def forward(self, x, mask):
#         return self.net(x * mask)
    
    
# self.s_cond = nn.ModuleDict()
# self.t_cond = nn.ModuleDict()

# for i in range(self.dim):
#     self.s_cond[str(i)] = net_class(self.dim*self.k, self.k, nh)

# for i in range(self.dim):
#     self.t_cond[str(i)] = net_class(self.dim*self.k, self.k, nh)
    
# Causal Normalizing Flow
class MultivariateCausalFlow(nn.Module):
    def __init__(self, dim, k, nh=100):
        super().__init__()
        self.dim = dim
        self.k = k


        # self.s_cond = net_class(self.dim*self.k, self.k, 100)
        self.s_cond = nn.Sequential(
                nn.Linear(self.dim*self.k, nh),
                nn.ReLU(),
                nn.Linear(nh, nh),
                nn.ReLU(),
                nn.Linear(nh, self.k),
                nn.Sigmoid(),
            )
        # self.t_cond = net_class(self.dim*self.k, self.k, 100)
        self.t_cond = nn.Sequential(
                nn.Linear(self.dim*self.k, nh),
                nn.ReLU(),
                nn.Linear(nh, nh),
                nn.ReLU(),
                nn.Linear(nh, self.k),
                nn.Sigmoid(),
            )

    def flow(self, e, C):
        e = e.reshape(-1, 2, 256)
        total_dims = e.shape[1]*e.shape[2]
        log_det = torch.zeros(e.size(0)).to(e.device)
        # p_logprob = th.zeros(e.size(0)).to(e.device)
        batch_size = e.shape[0]
        z = torch.zeros(e.shape).to(e.device)
        
        
        for i in range(self.dim):    
            if 1 in C[:, i]: # does it have any parents (z_3)
                mask = C[:, i].repeat(self.k, 1).T.reshape(total_dims).to(e.device)
            elif 1 not in C[:, i]: # doesnt have parents
                mask = torch.zeros(total_dims).to(e.device)
            
            # compute slope and offset
            s = self.s_cond(z.reshape(-1, total_dims) * mask).reshape(batch_size, self.k) # slope
            t = self.t_cond(z.reshape(-1, total_dims) * mask).reshape(batch_size, self.k) # offset

            # slope and offset transformation (affine transformation)
            z[:, i, :] = torch.exp(s) * e[:, i, :].reshape(batch_size, self.k) + t
  
            log_det += torch.sum(s, dim=1) # dz / de
            

        return [z.reshape(-1, 512), log_det]
    
    def reverse(self, z, C):
        z = z.reshape(-1, 2, 256)
        prior = MultivariateNormal(torch.ones(z.shape[1]*z.shape[2]).to(z.device), th.eye(z.shape[1]*z.shape[2]).to(z.device))
        total_dims = z.shape[1]*z.shape[2]
        log_det = torch.zeros(z.size(0)).to(z.device)
        # p_logprob = th.zeros(z.size(0)).to(z.device)
        batch_size = z.shape[0]
        e = torch.zeros(batch_size, self.dim, self.k).to(z.device)
        
        
        for i in range(self.dim):
            
            if 1 in C[:, i]: # does it have any parents (z_3)
                # mask = self.C[:, i].reshape(self.dim).to(device) # [1, 1, 0, 0]
                mask = C[:, i].repeat(self.k, 1).T.reshape(total_dims).to(e.device)
            elif 1 not in C[:, i]: # doesnt have parents
                mask = torch.zeros(total_dims).to(e.device)
            
            # compute slope and offset
            s = self.s_cond(z.reshape(-1, total_dims) * mask).reshape(batch_size, self.k) # slope
            t = self.t_cond(z.reshape(-1, total_dims) * mask).reshape(batch_size, self.k) # offset
            
        
            # slope and offset transformation (affine transformation)
            e[:, i, :] = torch.exp(-s) * (z[:, i, :].reshape(batch_size, self.k) - t)

            log_det -= torch.sum(s, dim=1) # dz / de

        
        p_log_prob = prior.log_prob(e.reshape(-1, z.shape[1]*z.shape[2]))
            
        return [log_det, p_log_prob]
    

# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def kl_normal(qm, qv, pm, pv):
        """
        Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
        sum over the last dimension

        Args:
            qm: tensor: (batch, dim): q mean
            qv: tensor: (batch, dim): q variance
            pm: tensor: (batch, dim): p mean
            pv: tensor: (batch, dim): p variance

        Return:
            kl: tensor: (batch,): kl between each sample
        """
        element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
        kl = element_wise.sum(-1)

        return kl


def reparameterize(m, v):
    """
    Reparameterization Trick.
    """
    sample = torch.randn(m.size()).to(m.device)
    z = m + (v**0.5)*sample
    
    return z


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
