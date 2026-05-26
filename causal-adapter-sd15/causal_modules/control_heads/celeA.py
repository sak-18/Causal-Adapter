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

        #mask=None

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
        #x = torch.einsum("nih, ioh -> noh", x, self.weight) + self.bias
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

class MLP_discovery(nn.Module):
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
            nn.Linear(self.latent_dim // self.num_var, 768,bias=use_bias),
            #nn.Linear(self.latent_dim // self.num_var, 1,bias=use_bias),
        )

    def forward(self, x):
        return self.net(x)

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
    
    def inference(self,label,sample=False,intervention_indx=None,intervention_values=None,disentangle=False):
        
        device = self.device
        mu= label.to(device).clone()
        bs = mu.shape[0]
        if intervention_indx !=None and intervention_values!=None:
            if disentangle==True:
                mu[:, intervention_indx] = intervention_values
                output = mu.unsqueeze(2)
            else:
                # do intervention
                reason_v_indices, result_v_indices = self.get_result_variable_indices(self.mask)

                if intervention_indx !=None:
                    # 0-3
                    if len(reason_v_indices)>0 and intervention_indx in reason_v_indices:
                        # start_truncate = intervention_indx
                        # end_truncate = intervention_indx+1
                        mu[:, intervention_indx] = intervention_values

                z_pre = self.dispatcher_layer(mu)        # [bs, num_var, 16]
                z_post = torch.zeros_like(mu).to(device) # [bs, num_var]

                for i in range(self.in_dim):
                    z_i = z_pre[:, i, :]  # [bs, 16]

                    # Pass through MLP or Linear to get scalar output: [bs]
                    z_i_out = self.nonlinearities[str(i)](z_i).squeeze(-1)  # [bs]

                    if len(result_v_indices) > 0 and i in result_v_indices:
                        z_post[:, i] = torch.sigmoid(z_i_out)
                    else:
                        z_post[:, i] = mu[:, i]
                
                z=z_post

                if intervention_indx !=None:
                    # 0-3
                    if len(result_v_indices)>0 and intervention_indx in result_v_indices:
                        # start_truncate = intervention_indx
                        # end_truncate = intervention_indx+1
                        z = label.to(device)
                        
                        z[:, intervention_indx] = intervention_values
                        #z_post[:, start_truncate:end_truncate] = intervention_values
                    
                        
                
                preds = (z > 0.5).float()  # [bs, num_var]
                # do OOD editing
                if intervention_indx is not None:
                    if len(result_v_indices) > 0 and intervention_indx in result_v_indices:
                        if intervention_indx in [2]:
                            # Find all rows where label[:, 1] == 0
                            row_index = torch.where(label[:, 1] == 0)[0]
                            preds[row_index, intervention_indx] = -1


                output = preds.unsqueeze(2)  # [bs, num_var, 1]
        else:
            # do reconstruction
            output = mu.unsqueeze(2)
        return output,None

    def forward(self, label):
        
        device = self.device
        mu = label.to(device)

        loss = 0
        output = mu.unsqueeze(2)
        # or return prob?
        #output = probs.unsqueeze(2)
        return output, loss

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
                z_post[:, i] = torch.sigmoid(z_i_out)
            else:
                z_post[:, i] = mu[:, i]
        
        # Raw logits
        
        probs = z_post  # [bs, num_var]

        # Compute binary cross-entropy loss using sigmoid probabilities
        if len(result_v_indices) > 0:
            selected_probs = probs[:, result_v_indices]     # [bs, num_result_vars]
            targets = mu[:, result_v_indices]               # [bs, num_result_vars]
            
            #loss = F.mse_loss(z_post[:, result_v_indices], mu[:, result_v_indices], reduction='none')
            loss = F.binary_cross_entropy(selected_probs, targets, reduction='none')
        else:
            loss = torch.tensor(0.0, device=mu.device)

        # Threshold at 0.5 to get hard binary predictions
        preds = (probs >= 0.5).float()  # [bs, num_var]
        # Return predictions and loss
        # Optionally you could also return probs if needed
        output = preds.unsqueeze(2)  # [bs, num_var, 1]
        #output = mu.unsqueeze(2)
        # or return prob?
        #output = probs.unsqueeze(2)
        return output, loss

# record for original causal discovery
class ControlNetConditioningEmbedding_discovery(nn.Module):
    
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
                mask=None,
            )

        self.identity = torch.eye(self.in_dim)
        self.layer_norm = nn.BatchNorm1d(self.in_dim)
        #self.sigmoid = nn.Sigmoid()
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

    def l1_reg_dispatcher(self):
        # maybe change to abs of the collapsed weights (sum over hidden dim)
    
        return torch.sum(torch.abs(self.dispatcher_layer.weight))

    def l2_reg_all_weights(self):
        return sum(
            [
                torch.sum(p**2)
                for p_name, p in self.named_parameters()
                if p.requires_grad and (p_name != "layers.0.gumbel_adjacency.log_alpha")
            ]
        )

    @torch.no_grad()
    def reset_parameters(self):
        self.dispatcher_layer.reset_parameters()

    @property
    def device(self):
        return next(self.parameters()).device 
    
    def dag_reg(self):
        A = self.get_adjacency_matrix() ** 2
        h = -torch.slogdet(self.identity.to(self.device) - A)[1]
        return h

    def forward(self, label):
        
        device = self.device
        mu = label.to(device)
        
        z_pre = self.dispatcher_layer(mu)        # [bs, num_var, 16]
        z_pre = self.layer_norm(z_pre)
        z_post = torch.zeros_like(mu).to(device) # [bs, num_var]
        
        for i in range(self.in_dim):
            z_i = z_pre[:, i, :]  # [bs, 16]
            # Pass through MLP or Linear to get scalar output: [bs]
            z_i_out = self.nonlinearities[str(i)](z_i).squeeze(-1)  # [bs]            
            z_post[:, i] = z_i_out

        output = z_post.unsqueeze(2)
        
        dag_loss = self.dag_reg()
        regular_loss = self.l1_reg_dispatcher()
        #loss = self.dag_reg()+0.0005*self.l1_reg_dispatcher()+0.0005*self.l2_reg_all_weights()
        return output,(dag_loss,regular_loss)

    def inference(self,label,sample=False,intervention_indx=None,intervention_values=None,disentangle=False):
        
        device = self.device
        mu = label.to(device)
        
        z_pre = self.dispatcher_layer(mu)        # [bs, num_var, 16]
        #z_pre = nn.Sigmoid()(z_pre)
        z_pre = self.layer_norm(z_pre)
        z_post = torch.zeros_like(mu).to(device) # [bs, num_var]
        
        for i in range(self.in_dim):
            z_i = z_pre[:, i, :]  # [bs, 16]
            # Pass through MLP or Linear to get scalar output: [bs]
            z_i_out = self.nonlinearities[str(i)](z_i).squeeze(-1)  # [bs]            
            z_post[:, i] = z_i_out

        output = z_post.unsqueeze(2)
        
        dag_loss = self.dag_reg()
        regular_loss = self.l1_reg_dispatcher()
        #loss = self.dag_reg()+0.0005*self.l1_reg_dispatcher()+0.0005*self.l2_reg_all_weights()
        return output,(dag_loss,regular_loss)




# all with 1 input 
# class ControlNetConditioningEmbedding_discovery(nn.Module):
    
#     def __init__(
#         self,
#         in_dim,
#         hidden_dims=16,
#         activation='leakyrelu',
#         adjacency_p: float = 2.0,
#         mask=None,
#     ):
#         super().__init__()
#         self.in_dim = in_dim
#         self.hidden_dims = hidden_dims
#         self.activation = get_activation(activation)
#         self.adjacency_p = adjacency_p

#         if mask is not None:
#             mask = (
#                 mask.astype(bool) & (1 - np.eye(self.in_dim)).astype(bool)
#             ).astype(int)
#         else:
#             mask = 1 - np.eye(self.in_dim)
        
#         self.dispatcher_layer = DispatcherLayer(
#                 self.in_dim,
#                 self.in_dim,
#                 hidden_dims,
#                 adjacency_p=self.adjacency_p,
#                 mask=None,
#             )

#         self.identity = torch.eye(self.in_dim)
#         self.layer_norm = nn.BatchNorm1d(self.in_dim)
#         #self.sigmoid = nn.Sigmoid()
#         self.nonlinearities = nn.ModuleDict()
#         for i in range(self.in_dim):
#             self.nonlinearities[str(i)] = MLP(latent_dim=self.hidden_dims*self.in_dim
#                                                 ,num_var=self.in_dim,use_bias=False)
#         # if self.model_variance_flavor == "nn":
#         #     self.var_layer = LinearParallel(dims[-1], 1, self.in_dim)

#         self.reset_parameters()    
    
#     def update_mask(self,A):
#         if isinstance(A, torch.Tensor):
#             # Create identity matrix on the same device
#             eye = torch.eye(self.in_dim, device=A.device)

#             # Convert to bool and apply mask logic
#             mask = (A.bool() & (~eye.bool())).int()

#             # Assign to device and save
#             self.mask = mask.to(self.device)
#             self.dispatcher_layer.mask = self.mask
#         else:
#             mask = (
#                     A.astype(bool) & (1 - np.eye(self.in_dim)).astype(bool)
#             ).astype(int)
#             self.mask = torch.tensor(mask).to(self.device)

#             self.dispatcher_layer.mask = self.mask

#     def get_adjacency_matrix(self):
#         return self.dispatcher_layer.get_adjacency_matrix()

#     def l1_reg_dispatcher(self):
#         # maybe change to abs of the collapsed weights (sum over hidden dim)
    
#         return torch.sum(torch.abs(self.dispatcher_layer.weight))

#     def l2_reg_all_weights(self):
#         return sum(
#             [
#                 torch.sum(p**2)
#                 for p_name, p in self.named_parameters()
#                 if p.requires_grad and (p_name != "layers.0.gumbel_adjacency.log_alpha")
#             ]
#         )

#     @torch.no_grad()
#     def reset_parameters(self):
#         self.dispatcher_layer.reset_parameters()

#     @property
#     def device(self):
#         return next(self.parameters()).device 
    
#     def dag_reg(self):
#         A = self.get_adjacency_matrix() ** 2
#         h = -torch.slogdet(self.identity.to(self.device) - A)[1]
#         return h

#     def forward(self, label):
        
#         device = self.device
#         #mu = label.to(device)
#         mu = torch.ones_like(label)

#         z_pre = self.dispatcher_layer(mu)        # [bs, num_var, 16]
#         z_pre = self.layer_norm(z_pre)
#         z_post = torch.zeros_like(mu).to(device) # [bs, num_var]
        
#         for i in range(self.in_dim):
#             z_i = z_pre[:, i, :]  # [bs, 16]
#             # Pass through MLP or Linear to get scalar output: [bs]
#             z_i_out = self.nonlinearities[str(i)](z_i).squeeze(-1)  # [bs]            
#             z_post[:, i] = z_i_out

#         output = z_post.unsqueeze(2)
        
#         dag_loss = self.dag_reg()
#         regular_loss = self.l1_reg_dispatcher()
#         #loss = self.dag_reg()+0.0005*self.l1_reg_dispatcher()+0.0005*self.l2_reg_all_weights()
#         return output,(dag_loss,regular_loss)

#     def inference(self,label,sample=False,intervention_indx=None,intervention_values=None,disentangle=False):
        
#         device = self.device
#         mu = label.to(device)
        
#         z_pre = self.dispatcher_layer(mu)        # [bs, num_var, 16]
#         #z_pre = nn.Sigmoid()(z_pre)
#         z_pre = self.layer_norm(z_pre)
#         z_post = torch.zeros_like(mu).to(device) # [bs, num_var]
        
#         for i in range(self.in_dim):
#             z_i = z_pre[:, i, :]  # [bs, 16]
#             # Pass through MLP or Linear to get scalar output: [bs]
#             z_i_out = self.nonlinearities[str(i)](z_i).squeeze(-1)  # [bs]            
#             z_post[:, i] = z_i_out

#         output = z_post.unsqueeze(2)
        
#         dag_loss = self.dag_reg()
#         regular_loss = self.l1_reg_dispatcher()
#         #loss = self.dag_reg()+0.0005*self.l1_reg_dispatcher()+0.0005*self.l2_reg_all_weights()
#         return output,(dag_loss,regular_loss)

'''Commentted, use textual embeddings for causal discovery'''
# class ControlNetConditioningEmbedding_discovery(nn.Module):
    
#     def __init__(
#         self,
#         in_dim,
#         hidden_dims=16,
#         activation='leakyrelu',
#         adjacency_p: float = 2.0,
#         mask=None,
#     ):
#         super().__init__()
#         hidden_dims = 768
#         self.in_dim = 7
#         self.hidden_dims = hidden_dims
#         self.activation = get_activation(activation)
#         self.adjacency_p = adjacency_p

#         if mask is not None:
#             mask = (
#                 mask.astype(bool) & (1 - np.eye(self.in_dim)).astype(bool)
#             ).astype(int)
#         else:
#             mask = 1 - np.eye(self.in_dim)
        
#         self.dispatcher_layer = DispatcherLayer(
#                 self.in_dim,
#                 self.in_dim,
#                 hidden_dims,
#                 adjacency_p=self.adjacency_p,
#                 mask=None,
#             )

#         self.identity = torch.eye(self.in_dim)
#         self.layer_norm = nn.BatchNorm1d(self.in_dim)
#         #self.sigmoid = nn.Sigmoid()
#         self.nonlinearities = nn.ModuleDict()
#         for i in range(self.in_dim):
#             self.nonlinearities[str(i)] = MLP_discovery(latent_dim=self.hidden_dims*self.in_dim
#                                                 ,num_var=self.in_dim,use_bias=False)
#         # if self.model_variance_flavor == "nn":
#         #     self.var_layer = LinearParallel(dims[-1], 1, self.in_dim)

#         self.reset_parameters()    
    
#     def update_mask(self,A):
#         if isinstance(A, torch.Tensor):
#             # Create identity matrix on the same device
#             eye = torch.eye(self.in_dim, device=A.device)

#             # Convert to bool and apply mask logic
#             mask = (A.bool() & (~eye.bool())).int()

#             # Assign to device and save
#             self.mask = mask.to(self.device)
#             self.dispatcher_layer.mask = self.mask
#         else:
#             mask = (
#                     A.astype(bool) & (1 - np.eye(self.in_dim)).astype(bool)
#             ).astype(int)
#             self.mask = torch.tensor(mask).to(self.device)

#             self.dispatcher_layer.mask = self.mask

#     def get_adjacency_matrix(self):
#         return self.dispatcher_layer.get_adjacency_matrix()

#     def l1_reg_dispatcher(self):
#         # maybe change to abs of the collapsed weights (sum over hidden dim)
    
#         return torch.sum(torch.abs(self.dispatcher_layer.weight))

#     def l2_reg_all_weights(self):
#         return sum(
#             [
#                 torch.sum(p**2)
#                 for p_name, p in self.named_parameters()
#                 if p.requires_grad and (p_name != "layers.0.gumbel_adjacency.log_alpha")
#             ]
#         )

#     @torch.no_grad()
#     def reset_parameters(self):
#         self.dispatcher_layer.reset_parameters()

#     @property
#     def device(self):
#         return next(self.parameters()).device 
    
#     def dag_reg(self):
#         A = self.get_adjacency_matrix() ** 2
#         h = -torch.slogdet(self.identity.to(self.device) - A)[1]
#         return h

#     def forward(self, label):
        
#         device = self.device
#         mu = label.to(device)
#         #mu = torch.zeros_like(label).to(device)

#         z_pre = self.dispatcher_layer(mu)        # [bs, num_var, 16]
#         z_pre = self.layer_norm(z_pre)
#         z_post = torch.zeros_like(mu).to(device) # [bs, num_var]
        
#         for i in range(self.in_dim):
#             z_i = z_pre[:, i, :]  # [bs, 16]
#             # Pass through MLP or Linear to get scalar output: [bs]
#             z_i_out = self.nonlinearities[str(i)](z_i).squeeze(-1)  # [bs]            
#             z_post[:, i] = z_i_out

#         output = z_post
        
#         dag_loss = self.dag_reg()
#         regular_loss = self.l1_reg_dispatcher()
#         #loss = self.dag_reg()+0.0005*self.l1_reg_dispatcher()+0.0005*self.l2_reg_all_weights()
#         return output,(dag_loss,regular_loss)

#     def inference(self,label,sample=False,intervention_indx=None,intervention_values=None,disentangle=False):
        
#         device = self.device
#         mu = label.to(device)
        
#         z_pre = self.dispatcher_layer(mu)        # [bs, num_var, 16]
#         #z_pre = nn.Sigmoid()(z_pre)
#         z_pre = self.layer_norm(z_pre)
#         z_post = torch.zeros_like(mu).to(device) # [bs, num_var]
        
#         for i in range(self.in_dim):
#             z_i = z_pre[:, i, :]  # [bs, 16]
#             # Pass through MLP or Linear to get scalar output: [bs]
#             z_i_out = self.nonlinearities[str(i)](z_i).squeeze(-1)  # [bs]            
#             z_post[:, i] = z_i_out

#         output = z_post
        
#         dag_loss = self.dag_reg()
#         regular_loss = self.l1_reg_dispatcher()
#         #loss = self.dag_reg()+0.0005*self.l1_reg_dispatcher()+0.0005*self.l2_reg_all_weights()
#         return output,(dag_loss,regular_loss)