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
from typing import Literal
#device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")


import torch

def bin_array(num: torch.Tensor, m: int = None, reverse: bool = False):

    if reverse:
        if num.dim() == 1:
            num = num.unsqueeze(dim=0)
        # num: shape (bs, m), binary vectors like [0, 1, 1, 0]
        # Output: shape (bs,), scalar representation
        bs, width = num.shape
        weights = 2 ** torch.arange(width - 1, -1, -1, dtype=torch.float32, device=num.device)
        return torch.sum(num * weights, dim=1)
    else:
        # num: shape (bs,), each element is an integer to be encoded to binary
        if m is None:
            m = int(torch.ceil(torch.log2(num.max().float() + 1)).item())
        bs = num.shape[0]
        powers = 2 ** torch.arange(m - 1, -1, -1, device=num.device)
        num = num.unsqueeze(1).long()
        return ((num & powers) > 0).float()

# for processing the slice
def ordinal_array(num: torch.Tensor, m: int = 10, reverse: bool = False, scale: float = 1.0):
    if reverse:
        if num.dim() == 1:
            num = num.unsqueeze(dim=0)
        # num: (bs, 10), values like [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        # Output: (bs,), counting number of 1s per row
        return scale * torch.count_nonzero(num, dim=1).to(num.dtype)
    else:
        # num: (bs,) — scalar levels like [3, 5, 7], build ordinal encoding
        bs = num.shape[0]
        device = num.device
        out = torch.zeros((bs, m), dtype=torch.float32, device=device)

        for i in range(bs):
            count = int(num[i].item())
            if count > 0:
                out[i, m - count : m] = scale  # fill right-aligned ones
        return out


def get_activation(activation: Literal["relu", "sigmoid", "tanh", "linear","leakyrelu"]):
    """
    Args:
        activation (str): activation function name
    Returns:
        torch.nn.Module: activation function
    """
    if activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sin":
        return torch.sin()
    elif activation == "linear":
        return nn.Identity()
    elif activation == "leakyrelu":
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unknown activation function: {activation}")


def zero_module(module):
    #return module 
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

def transform_attributes(tensor: torch.Tensor, reverse: bool = False) -> torch.Tensor:
    """
    Converts between 16-dimensional and 6-dimensional feature representations, preserving field order:
    [apoE (2 or scalar), age, sex, brain_vol, ventricle_vol, slice (10 or scalar)]

    Forward (reverse=False): 
        Input:  (bs, 16)
        Output: (bs, 6)

    Reverse (reverse=True):
        Input:  (bs, 6)
        Output: (bs, 16)
    """
    if not reverse:
        # Forward: (bs, 16) → (bs, 6)
        apoE_labels = bin_array(tensor[:, 0:2], reverse=True)                     # (bs,)
        slice_labels = ordinal_array(tensor[:, -10:], reverse=True)              # (bs,)
        other_features = tensor[:, 2:6]                                           # [age, sex, brain_vol, ventricle_vol] → (bs, 4)

        return torch.cat([
            apoE_labels.unsqueeze(1),           # (bs, 1)
            other_features,                     # (bs, 4)
            slice_labels.unsqueeze(1)           # (bs, 1)
        ], dim=1)  # (bs, 6)

    else:
        # Reverse: (bs, 6) → (bs, 16)
        bs = tensor.shape[0]
        apoE_labels = tensor[:, 0].long()                                        # (bs,)
        other_features = tensor[:, 1:5]                                          # (bs, 4)
        slice_labels = tensor[:, 5].long()                                       # (bs,)

        # Encode apoE and slice
        apoE_encoding = bin_array(apoE_labels, m=2, reverse=False)              # (bs, 2)
        slice_encoding = ordinal_array(slice_labels, m=10, reverse=False)       # (bs, 10)

        return torch.cat([
            apoE_encoding,                    # (bs, 2)
            other_features,                   # (bs, 4)
            slice_encoding                    # (bs, 10)
        ], dim=1)  # (bs, 16)




class DispatcherLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        adjacency_p=2.0,
        mask=None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.adjacency_p = adjacency_p

        

        if mask is not None:
            self.register_buffer("mask", torch.tensor(mask).float())
        else:
            self.register_buffer("mask", torch.ones((in_dim, out_dim)))

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
        if self._weight.device.type == 'meta':
            print("Skipping initialization on meta device.")
            return
        bound = scale / self.in_dim / self.hidden_dim ** (1.0 / self.adjacency_p)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def get_adjacency_matrix(self):
        # same as Dagma sum(pow)
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
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim // self.num_var, 1,bias=use_bias),
        )

    def forward(self, x):
        return self.net(x)

    
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
        if self.weight.device.type == 'meta':
            print("Skipping initialization on meta device.")
            return
        bound = 1.0 / self.in_dim**0.5
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def __repr__(self):
        return f"LinearParallel(in_dim={self.in_dim}, out_dim={self.out_dim}, parallel_dim={self.parallel_dim})"


class ControlNetConditioningEmbedding(nn.Module):
    
    def __init__(
        self,
        in_dim,
        hidden_dims=16,
        activation='leakyrelu',
        adjacency_p: float = 2.0,
        mask=None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.activation = get_activation(activation)
        self.adjacency_p = adjacency_p

        if mask is not None:
            mask = (
                mask.astype(bool) & (1 - np.eye(self.in_dim)).astype(bool)
            ).astype(int)
        else:
            mask = 1 - np.eye(self.in_dim)

        
        

        self.dispatcher_layer = DispatcherLayer(
                self.in_dim,
                self.in_dim,
                hidden_dims,
                adjacency_p=self.adjacency_p,
                mask=mask,
            )

        self.identity = torch.eye(self.in_dim)
        self.nonlinearities = nn.ModuleDict()
        for i in range(self.in_dim):
            self.nonlinearities[str(i)] = MLP(latent_dim=self.hidden_dims*self.in_dim
                                                ,num_var=self.in_dim,use_bias=False)
        # if self.model_variance_flavor == "nn":
        #     self.var_layer = LinearParallel(dims[-1], 1, self.in_dim)

        self.reset_parameters()    
    
    def update_mask(self,A):
        if isinstance(A, torch.Tensor):
            # Create identity matrix on the same device
            eye = torch.eye(self.in_dim, device=A.device)

            # Convert to bool and apply mask logic
            mask = (A.bool() & (~eye.bool())).int()

            # Assign to device and save
            self.mask = mask.to(self.device)
            self.dispatcher_layer.mask = self.mask
        else:
            mask = (
                    A.astype(bool) & (1 - np.eye(self.in_dim)).astype(bool)
            ).astype(int)
            self.mask = torch.tensor(mask).to(self.device)

            self.dispatcher_layer.mask = self.mask

    def get_adjacency_matrix(self):
        return self.dispatcher_layer.get_adjacency_matrix()

    @torch.no_grad()
    def reset_parameters(self):
        self.dispatcher_layer.reset_parameters()

    def get_result_variable_indices(self,A):
        reason_variable_indices = []
        result_variable_indices = []

        for i in range(A.size(1)):
            col = A[:, i]

            if torch.all(col == 0):
                reason_variable_indices.append(i)
            if torch.any(col != 0):
                result_variable_indices.append(i)
        
        return reason_variable_indices, result_variable_indices

    @property
    def device(self):
        return next(self.parameters()).device
    
    def inference(self,label,sample=False,intervention_indx=None,intervention_values=None,update_one_concept=False):
        
        device = self.device
        mu= label.to(device).clone()
        bs = mu.shape[0]
        if intervention_indx !=None and intervention_values!=None:
            reason_v_indices, result_v_indices = self.get_result_variable_indices(self.mask)

            if intervention_indx !=None:
                # 0-3
                if len(reason_v_indices)>0 and intervention_indx in reason_v_indices:
                    
                    mu[:, intervention_indx] = intervention_values

            z_pre = self.dispatcher_layer(mu)        # [bs, num_var, 16]
            z_post = torch.zeros_like(mu).to(device) # [bs, num_var]

            for i in range(self.in_dim):
                z_i = z_pre[:, i, :]  # [bs, 16]

                # Pass through MLP or Linear to get scalar output: [bs]
                z_i_out = self.nonlinearities[str(i)](z_i).squeeze(-1)  # [bs]

                if len(result_v_indices) > 0 and i in result_v_indices:
                    z_post[:, i] = z_i_out
                else:
                    z_post[:, i] = mu[:, i]
            
            z=z_post
            # if sample is False:
            #     z = mu
            if intervention_indx !=None:
                # 0-3
                if len(result_v_indices)>0 and intervention_indx in result_v_indices:
                    z = label.to(device)
                    z[:, intervention_indx] = intervention_values
                    
            output = z.unsqueeze(2)
        else:
            # do reconstruction
            output = mu.unsqueeze(2)
        return output,None
       

    def forward(self, label):
        
        device = self.device
        # follow paper use the final three axis to generate image      
        mu = label.to(device)
        
        #mu = label.to(device)
        loss =0
        output = mu.unsqueeze(2)
        return output,loss

    def pretrain(self, label):
        
        device = self.device
        mu = label.to(device)

        z_pre = self.dispatcher_layer(mu)        # [bs, num_var, 16]
        z_post = torch.zeros_like(mu).to(device) # [bs, num_var]
        
        reason_v_indices, result_v_indices = self.get_result_variable_indices(self.mask)

        for i in range(self.in_dim):
            z_i = z_pre[:, i, :]  # [bs, 16]

            # Pass through MLP or Linear to get scalar output: [bs]
            z_i_out = self.nonlinearities[str(i)](z_i).squeeze(-1)  # [bs]

            if len(result_v_indices) > 0 and i in result_v_indices:
                z_post[:, i] = z_i_out
            else:
                z_post[:, i] = mu[:, i]
        
    
        # Only keep the columns (variables) in result_v_indices
        if len(result_v_indices) > 0:
            # Stack the losses for selected variable indices: shape [bs, len(result_v_indices)]

            # Compute element-wise MSE loss: [bs, num_var]
            loss = F.mse_loss(z_post[:, result_v_indices], mu[:, result_v_indices], reduction='none')
            
            # Aggregate, e.g., mean over all selected elements
        else:
            loss = torch.tensor(0.0, device=mu.device)

        output = z_post.unsqueeze(2)
        return output,loss
