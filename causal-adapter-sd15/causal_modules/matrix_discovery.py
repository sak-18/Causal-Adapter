#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import scipy
#device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")



def zero_module(module):
    #return module 
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class DispatcherLayer3(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        adjacency_p=2.0,
        mask=None,
        use_gumbel=False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.adjacency_p = adjacency_p
        self.mask=None
        # if mask is not None:
        #     self.register_buffer("mask", torch.tensor(mask).float())
        # else:
        #     self.register_buffer("mask", torch.ones((in_dim, out_dim)))

        self._weight = nn.Parameter(torch.zeros(in_dim, out_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim, hidden_dim))
        self.reset_parameters_bounded_eigenvalues()

    @property
    def weight(self):
        if self.mask is not None:
            return self._weight * self.mask[:, :, None]
        return self._weight

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, in_dim)
        Returns:
            torch.Tensor: output tensor of shape (batch_size, out_dim, hidden_dim)
        """
        
        x = torch.einsum("ni, ioh -> noh", x, self.weight) + self.bias
        return x

    @torch.no_grad()
    def reset_parameters(self):
        self.reset_parameters_bounded_eigenvalues()

    @torch.no_grad()
    def reset_parameters_bounded_eigenvalues(self, scale=1.0):
        bound = scale / self.in_dim / self.hidden_dim ** (1.0 / self.adjacency_p)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def get_adjacency_matrix(self):
        return torch.linalg.vector_norm(self.weight, dim=2, ord=self.adjacency_p)

    def __repr__(self):
        return (
            f"DispatcherLayer("
            f"in_dim={self.in_dim}, "
            f"out_dim={self.out_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"adjacency_p={self.adjacency_p}"
            f")"
        )

class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            #nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DAG_discovery3(nn.Module):
    
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
        dims: List[int]=[4,768,1], 
        bias: bool = True,
        mask=None,
        dag_penalty_flavor = "none",
        
    ):
        super().__init__()
        self.embed_dim = 128
        self.dims, self.d = [dims[0],self.embed_dim,1], dims[0]
        self.I = torch.eye(self.d)

        if dag_penalty_flavor == "none":
            # Need to mask out identity to prevent learning self-loops
            if mask is not None:
                mask = (
                    mask.astype(bool) & (1 - np.eye(self.d)).astype(bool)
                ).astype(int)
            else:
                #mask = 1 - np.eye(self.d)
                #mask = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,0],[1,1,0,0]])
                #mask = np.array([[0,0,0,0,0,0],[0,0,1,0,1,0],[0,0,0,1,1,1],[0,0,1,0,1,1],[0,1,1,1,0,1],[0,0,1,1,1,0]])
                #mask = np.array([[1,0,0,0,0,0],[0,1,1,0,1,0],[0,1,1,1,1,1],[0,0,1,1,1,1],[0,1,1,1,1,1],[0,0,1,1,1,1]])
                #mask = np.array([[1,0,0,0,0,0],[0,1,1,0,1,0],[0,1,1,1,1,1],[0,0,1,1,1,1],[0,1,1,1,1,1],[0,0,1,1,1,1]]) *(1 - np.eye(self.d))
                mask = None
        self.embed_linear1 = DispatcherLayer3(self.d,self.d,self.dims[1],mask=mask)
        self.layer_norm = nn.BatchNorm1d(self.d)
        #self.layer_norm = nn.LayerNorm((self.dims[0], self.dims[1]))
        self.nonlinearities = nn.ModuleDict()

        for i in range(self.d):
            self.nonlinearities[str(i)] = MLP(self.dims[1],1024)
        # fc2: local linear layers

        self.conv_up = nn.Sequential(
            nn.Conv2d(self.d, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
        )
        self.conv_out = nn.Conv2d(128, conditioning_embedding_channels, kernel_size=3, padding=1)
        
        # self.conv_out = zero_module(
        #     nn.Conv2d(128, conditioning_embedding_channels, kernel_size=3, padding=1)
        # )


    def h_func(self, s: float = 1.0) -> torch.Tensor:
        r"""
        Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG

        Parameters
        ----------
        s : float, optional
            Controls the domain of M-matrices, by default 1.0

        Returns
        -------
        torch.Tensor
            A scalar value of the log-det acyclicity function :math:`h(\Theta)`.
        """
        
        A =  self.embed_linear1.get_adjacency_matrix() ** 2
        h = -torch.slogdet(s * self.I.to(A.device) - A)[1] + self.d * np.log(s)
        return h

    def fc1_l1_reg(self) -> torch.Tensor:
        r"""
        Takes L1 norm of the weights in the first fully-connected layer

        Returns
        -------
        torch.Tensor
            A scalar value of the L1 norm of first FC layer. 
        """
        # l2
        A = self.embed_linear1.get_adjacency_matrix() ** 2 
        return torch.sum(A)
        


    # x is input attributes (bs,a_dim), embed embeddings [a_dim,768]
    def forward(self, attr,embed):
        device = attr.device

        x = self.embed_linear1(attr)
        
        x = x.view(-1, self.dims[0], self.dims[1])
        #x = self.layer_norm(F.sigmoid(x))
        x= self.layer_norm(x)
        #x= F.leaky_relu(x)

        x_pred = torch.zeros(x.shape[0],x.shape[1],1024)
        x_pred = x_pred.type(x.dtype).to(x.device)
        for i in range(self.d):
            x_pred[:, i, :] = self.nonlinearities[str(i)](x[:, i, :])
        # (bs, a_dim,32,32)
        z= x_pred.view(x_pred.shape[0],x_pred.shape[1],32,32) 
        # aim_dim->128->320 
        z= self.conv_up(z)
        z= self.conv_out(z) 

        # compute loss
        #L = self.h_func(s=1.0)+0.001*self.fc1_l1_reg()
        L = self.h_func(s=1.0)
        return z,L

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        r"""
        Computes the induced weighted adjacency matrix W from the first FC weights.
        Intuitively each edge weight :math:`(i,j)` is the *L2 norm of the functional influence of variable i to variable j*.

        Returns
        -------
        np.ndarray
            :math:`(d,d)` weighted adjacency matrix 
        """
        A =  self.embed_linear1.get_adjacency_matrix()
        W = A.cpu().detach().numpy()  # [i, j]
        return W



class DAG_discovery2(nn.Module):
    
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
        dims: List[int]=[4,768,1], 
        bias: bool = True
        
    ):
        super().__init__()
        self.embed_dim = 768
        self.dims, self.d = [dims[0],self.embed_dim,1], dims[0]
        self.I = torch.eye(self.d)
        
        
        self.embed_linear1 = nn.Linear(self.d,self.d * self.dims[1],bias=True)
        nn.init.zeros_(self.embed_linear1.weight)
        nn.init.zeros_(self.embed_linear1.bias)
        # 1024=32*32
        
        self.layer_norm = nn.BatchNorm1d(self.d)
        # self.compress_emb = nn.Sequential(nn.Tanh(),nn.Linear( self.embed_dim, 1))

        # layers = []
        # for l in range(len(dims) - 2):
        #     layers.append(LocallyConnected(self.d, self.dims[l + 1], self.dims[l + 2], bias=True))
        # self.fc2 = nn.ModuleList(layers)

        # self.up_emb = nn.Sequential(nn.Linear(1,1024))
        self.up_emb = nn.Sequential(nn.Linear(self.dims[1], 1024)
                                         ,nn.LeakyReLU())
        # fc2: local linear layers

        self.conv_up = nn.Sequential(
            nn.Conv2d(self.d, 128, kernel_size=3, padding=1),
        )
        self.conv_out = nn.Conv2d(128, conditioning_embedding_channels, kernel_size=3, padding=1)
        
        # self.conv_out = zero_module(
        #     nn.Conv2d(128, conditioning_embedding_channels, kernel_size=3, padding=1)
        # )

    def matrix_poly(self, matrix, d,device):
        x = torch.eye(d).to(device)+ torch.div(matrix.to(device), d).to(device)
        return torch.matrix_power(x, d)

    def _h_A(self,A, m,device):
        expm_A = self.matrix_poly(A*A, m,device)
        h_A = torch.trace(expm_A) - m
        return h_A
    def log_mse_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        n, d = target.shape
        loss = 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))
        return loss

    def h_func(self, s: float = 1.0) -> torch.Tensor:
        r"""
        Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG

        Parameters
        ----------
        s : float, optional
            Controls the domain of M-matrices, by default 1.0

        Returns
        -------
        torch.Tensor
            A scalar value of the log-det acyclicity function :math:`h(\Theta)`.
        """
        fc1_weight = self.embed_linear1.weight
        fc1_weight = fc1_weight.view(self.d, -1, self.d)
        A = torch.sum(fc1_weight ** 2, dim=1).t()  # [i, j]
        h = -torch.slogdet(s * self.I.to(A.device) - A)[1] + self.d * np.log(s)
        return h

    def fc1_l1_reg(self) -> torch.Tensor:
        r"""
        Takes L1 norm of the weights in the first fully-connected layer

        Returns
        -------
        torch.Tensor
            A scalar value of the L1 norm of first FC layer. 
        """
        fc1_weight = self.embed_linear1.weight
        fc1_weight = fc1_weight.view(self.d, -1, self.d)
        A = torch.sum(fc1_weight ** 2, dim=1).t()  # [i, j]
        return torch.sum(A)
    


    # x is input attributes (bs,a_dim), embed embeddings [a_dim,768]
    def forward(self, attr,embed):
        device = attr.device

        x = self.embed_linear1(attr)
        #x = self.layer_norm(x)
        # (bs,a_dim,768)
        x = x.view(-1, self.dims[0], self.dims[1])
        x= self.layer_norm(x)
        
        x_pred = x
        # (bs,a_dim,1)
        # x_pred = self.compress_emb(x)


        # (bs, a_dim,32,32)
        z= self.up_emb(x_pred).view(x.shape[0],x.shape[1],32,32) 
        # aim_dim->128->320 
        z= self.conv_up(z)
        z= self.conv_out(z) 

        # compute loss

        #L = F.mse_loss(attr,x_pred.squeeze(2))+self.h_func(A_matrix,s=1.0)+self.fc1_l1_reg(A_matrix)
        #L = 10*self.h_func(s=1.0)+0.005*self.fc1_l1_reg()
        L = self.h_func(s=1.0)
        #L = self.h_func(s=1.0)+self.log_mse_loss(x_pred.squeeze(2),attr)
        # vae_img_representatation size (bs,320,32,32)
        return z,L

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        r"""
        Computes the induced weighted adjacency matrix W from the first FC weights.
        Intuitively each edge weight :math:`(i,j)` is the *L2 norm of the functional influence of variable i to variable j*.

        Returns
        -------
        np.ndarray
            :math:`(d,d)` weighted adjacency matrix 
        """
        fc1_weight = self.embed_linear1.weight
        fc1_weight = fc1_weight.view(self.d, -1, self.d)  
        A = torch.sum(fc1_weight ** 2, dim=1).t() 
        W = torch.sqrt(A)
        W = W.cpu().detach().numpy()  # [i, j]
        return W


class DAG_discovery4(nn.Module):
    
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
        dims: List[int]=[4,768,1], 
        bias: bool = True,
        mask=None,
        dag_penalty_flavor = "none",
        
    ):
        super().__init__()
        self.embed_dim = 128
        self.dims, self.d = [dims[0],self.embed_dim,1], dims[0]
        self.I = torch.eye(self.d)

        if dag_penalty_flavor == "none":
            # Need to mask out identity to prevent learning self-loops
            if mask is not None:
                mask = (
                    mask.astype(bool) & (1 - np.eye(self.d)).astype(bool)
                ).astype(int)
            else:
                #mask = 1 - np.eye(self.d)
                #mask = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,0],[1,1,0,0]])
                #mask = np.array([[0,0,0,0,0,0],[0,0,1,0,1,0],[0,0,0,1,1,1],[0,0,1,0,1,1],[0,1,1,1,0,1],[0,0,1,1,1,0]])
                #mask = np.array([[1,0,0,0,0,0],[0,1,1,0,1,0],[0,1,1,1,1,1],[0,0,1,1,1,1],[0,1,1,1,1,1],[0,0,1,1,1,1]])
                mask = np.array([[1,0,0,0,0,0],[0,1,1,0,1,0],[0,1,1,1,1,1],[0,0,1,1,1,1],[0,1,1,1,1,1],[0,0,1,1,1,1]]) *(1 - np.eye(self.d))
        self.embed_linear1 = DispatcherLayer3(self.d,self.d,self.dims[1],mask=mask)
        self.layer_norm = nn.BatchNorm1d(self.d)
        self.nonlinearities = nn.ModuleDict()

        self.power_grad = PowerIterationGradient(
                self.embed_linear1.get_adjacency_matrix(),
                self.d,
                n_iter=15,
            )

        for i in range(self.d):
            self.nonlinearities[str(i)] = MLP(self.dims[1],1024)
        # fc2: local linear layers

        self.conv_up = nn.Sequential(
            nn.Conv2d(self.d, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
        )
        self.conv_out = nn.Conv2d(128, conditioning_embedding_channels, kernel_size=3, padding=1)
        
        # self.conv_out = zero_module(
        #     nn.Conv2d(128, conditioning_embedding_channels, kernel_size=3, padding=1)
        # )


    def h_func(self, s: float = 1.0) -> torch.Tensor:
        r"""
        Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG

        Parameters
        ----------
        s : float, optional
            Controls the domain of M-matrices, by default 1.0

        Returns
        -------
        torch.Tensor
            A scalar value of the log-det acyclicity function :math:`h(\Theta)`.
        """
        
        A =  self.embed_linear1.get_adjacency_matrix() ** 2
        h = -torch.slogdet(s * self.I.to(A.device) - A)[1] + self.d * np.log(s)
        return h

    def fc1_l1_reg(self) -> torch.Tensor:
        r"""
        Takes L1 norm of the weights in the first fully-connected layer

        Returns
        -------
        torch.Tensor
            A scalar value of the L1 norm of first FC layer. 
        """
        # l2
        A = self.embed_linear1.get_adjacency_matrix() ** 2 
        return torch.sum(A)

    def dag_reg_power_grad(self):
        grad, A = self.power_grad.compute_gradient(self.embed_linear1.get_adjacency_matrix())
        # with torch.no_grad():
        #     grad = grad - A * (grad * A).sum() / ((A**2).sum() + 1e-6) / 2
        # grad = grad + torch.eye(self.in_dim)
        h_val = (grad.detach() * A).sum()
        return h_val
        


    # x is input attributes (bs,a_dim), embed embeddings [a_dim,768]
    def forward(self, attr,embed):
        device = attr.device

        x = self.embed_linear1(attr)
        
        x = x.view(-1, self.dims[0], self.dims[1])
        #x = self.layer_norm(F.sigmoid(x))
        x= self.layer_norm(x)
        

        x_pred = torch.zeros(x.shape[0],x.shape[1],1024)
        x_pred = x_pred.type(x.dtype).to(x.device)
        for i in range(self.d):
            x_pred[:, i, :] = self.nonlinearities[str(i)](x[:, i, :])
        # (bs, a_dim,32,32)
        z= x_pred.view(x_pred.shape[0],x_pred.shape[1],32,32) 
        # aim_dim->128->320 
        z= self.conv_up(z)
        z= self.conv_out(z) 

        # compute loss
        #L = self.h_func(s=1.0)+0.001*self.fc1_l1_reg()
        L = self.dag_reg_power_grad()
        return z,L

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        r"""
        Computes the induced weighted adjacency matrix W from the first FC weights.
        Intuitively each edge weight :math:`(i,j)` is the *L2 norm of the functional influence of variable i to variable j*.

        Returns
        -------
        np.ndarray
            :math:`(d,d)` weighted adjacency matrix 
        """
        A =  self.embed_linear1.get_adjacency_matrix()
        W = A.cpu().detach().numpy()  # [i, j]
        return W


def normalize(v):
    return v / torch.linalg.vector_norm(v)


class PowerIterationGradient(nn.Module):
    def __init__(self, init_adj_mtx, d, n_iter=5):
        super().__init__()
        self.d = d
        self.n_iter = n_iter

        self._dummy_param = nn.Parameter(
            torch.zeros(1), requires_grad=False
        )  # Used to track device

        self.register_buffer("u", None)
        self.register_buffer("v", None)

        self.init_eigenvect(init_adj_mtx)

    @property
    def device(self):
        return self._dummy_param.device

    def init_eigenvect(self, adj_mtx):
        self.u, self.v = torch.ones(size=(2, self.d), device=self.device)
        self.u = normalize(self.u)
        self.v = normalize(self.v)
        self.iterate(adj_mtx, self.n_iter)

    def iterate(self, adj_mtx, n=2):
        with torch.no_grad():
            A = adj_mtx + 1e-6
            for _ in range(n):
                self.one_iteration(A)

    def one_iteration(self, A):
        """One iteration of power method"""
        self.u = normalize(A.T @ self.u)
        self.v = normalize(A @ self.v)

    def compute_gradient(self, adj_mtx):
        """Gradient eigenvalue"""
        A = adj_mtx  # **2
        # fixed penalty
        self.iterate(A, self.n_iter)
        # self.init_eigenvect(adj_mtx)
        grad = self.u[:, None] @ self.v[None] / (self.u.dot(self.v) + 1e-6)
        # grad += torch.eye(self.d)
        # grad += A.T
        return grad, A


class DAG_discovery5(nn.Module):
    
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
        dims: List[int]=[4,768,1], 
        bias: bool = True,
        mask=None,
        dag_penalty_flavor = "none",
        
    ):
        super().__init__()
        self.embed_dim = 128
        self.dims, self.d = [dims[0],self.embed_dim,1], dims[0]
        self.I = torch.eye(self.d)

        if dag_penalty_flavor == "none":
            # Need to mask out identity to prevent learning self-loops
            if mask is not None:
                mask = (
                    mask.astype(bool) & (1 - np.eye(self.d)).astype(bool)
                ).astype(int)
            else:
                #mask = 1 - np.eye(self.d)
                #mask = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,0],[1,1,0,0]])
                #mask = np.array([[1,0,0,0,0,0],[0,1,1,0,1,0],[0,1,1,1,1,1],[0,0,1,1,1,1],[0,1,1,1,1,1],[0,0,1,1,1,1]])
                #mask = np.array([[1,0,0,0,0,0],[0,1,1,0,1,0],[0,1,1,1,1,1],[0,0,1,1,1,1],[0,1,1,1,1,1],[0,0,1,1,1,1]]) *(1 - np.eye(self.d))
                mask = None
        self.embed_linear1 = DispatcherLayer3(self.d,self.d,self.dims[1],mask=mask)
        self.layer_norm = nn.BatchNorm1d(self.d)
        #self.layer_norm = nn.LayerNorm((self.dims[0], self.dims[1]))
        self.nonlinearities = LinearParallel(self.dims[1],1024,self.dims[0])
        # self.nonlinearities = nn.ModuleDict()

        # for i in range(self.d):
        #     self.nonlinearities[str(i)] = MLP(self.dims[1],1024)
        # fc2: local linear layers

        self.conv_up = nn.Sequential(
            nn.Conv2d(self.d, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
        )
        self.conv_out = nn.Conv2d(128, conditioning_embedding_channels, kernel_size=3, padding=1)
        
        # self.conv_out = zero_module(
        #     nn.Conv2d(128, conditioning_embedding_channels, kernel_size=3, padding=1)
        # )


    def h_func(self, s: float = 1.0) -> torch.Tensor:
        r"""
        Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG

        Parameters
        ----------
        s : float, optional
            Controls the domain of M-matrices, by default 1.0

        Returns
        -------
        torch.Tensor
            A scalar value of the log-det acyclicity function :math:`h(\Theta)`.
        """
        
        A =  self.embed_linear1.get_adjacency_matrix() ** 2
        h = -torch.slogdet(s * self.I.to(A.device) - A)[1] + self.d * np.log(s)
        return h

    def fc1_l1_reg(self) -> torch.Tensor:
        r"""
        Takes L1 norm of the weights in the first fully-connected layer

        Returns
        -------
        torch.Tensor
            A scalar value of the L1 norm of first FC layer. 
        """
        # l2
        A = self.embed_linear1.get_adjacency_matrix() ** 2 
        return torch.sum(A)
    
    def log_mse_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        n, d = target.shape
        loss = 0.5 * d * (1 / n) * torch.sum((output - target) ** 2)
        return loss.mean()


    # x is input attributes (bs,a_dim), embed embeddings [a_dim,768]
    def forward(self, attr,embed):
        device = attr.device

        x = self.embed_linear1(attr)
        
        x = x.view(-1, self.dims[0], self.dims[1])
        #x = self.layer_norm(F.sigmoid(x))
        x= self.layer_norm(x)
        #x= F.leaky_relu(x)
        # (bs,a_dim,1024)
        x_pred = self.nonlinearities(x)
        # (bs, a_dim,32,32)
        z= x_pred.view(x_pred.shape[0],x_pred.shape[1],32,32) 
        # aim_dim->128->320 
        z= self.conv_up(z)
        z= self.conv_out(z) 

        # compute loss
        #L = self.h_func(s=1.0)+0.001*self.fc1_l1_reg()
        #L = self.h_func(s=1.0)
        L = self.h_func(s=1.0)+ 0.1*self.log_mse_loss(x_pred.mean(-1),attr)
        return z,L

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        r"""
        Computes the induced weighted adjacency matrix W from the first FC weights.
        Intuitively each edge weight :math:`(i,j)` is the *L2 norm of the functional influence of variable i to variable j*.

        Returns
        -------
        np.ndarray
            :math:`(d,d)` weighted adjacency matrix 
        """
        A =  self.embed_linear1.get_adjacency_matrix()
        W = A.cpu().detach().numpy()  # [i, j]
        return W

    
class LinearParallel(nn.Module):
    def __init__(self, in_dim, out_dim, parallel_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.parallel_dim = parallel_dim

        self.weight = nn.Parameter(torch.zeros(parallel_dim, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(parallel_dim, out_dim))
        self.reset_parameters()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, parallel_dim, in_dim)
        Returns:
            torch.Tensor: output tensor of shape (batch_size, parallel_dim, out_dim)
        """
        x = torch.einsum("npi, pio -> npo", x, self.weight) + self.bias
        return x

    @torch.no_grad()
    def reset_parameters(self):
        bound = 1.0 / self.in_dim**0.5
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def __repr__(self):
        return f"LinearParallel(in_dim={self.in_dim}, out_dim={self.out_dim}, parallel_dim={self.parallel_dim})"
