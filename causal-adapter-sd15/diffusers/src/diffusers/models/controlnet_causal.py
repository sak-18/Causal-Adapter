# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from ..configuration_utils import ConfigMixin, register_to_config
from ..loaders.single_file_model import FromOriginalModelMixin
from ..utils import BaseOutput, logging
from .attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from .embeddings import TextImageProjection, TextImageTimeEmbedding, TextTimeEmbedding, TimestepEmbedding, Timesteps
from .modeling_utils import ModelMixin
from .unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    DownBlock2D,
    UNetMidBlock2D,
    UNetMidBlock2DCrossAttn,
    get_down_block,
)
from .unets.unet_2d_condition import UNet2DConditionModel
from causal_modules.model import nns as causalnn
import causal_modules
from causal_modules.pretraining import load_dataset_model
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class ControlNetOutput(BaseOutput):
    """
    The output of [`ControlNetModel`].

    Args:
        down_block_res_samples (`tuple[torch.Tensor]`):
            A tuple of downsample activations at different resolutions for each downsampling block. Each tensor should
            be of shape `(batch_size, channel * resolution, height //resolution, width // resolution)`. Output can be
            used to condition the original UNet's downsampling activations.
        mid_down_block_re_sample (`torch.Tensor`):
            The activation of the middle block (the lowest sample resolution). Each tensor should be of shape
            `(batch_size, channel * lowest_resolution, height // lowest_resolution, width // lowest_resolution)`.
            Output can be used to condition the original UNet's middle block activation.
    """

    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor


class LinearParallel(nn.Module):
    def __init__(self, in_dim, out_dim, parallel_dim, init_method="kaiming"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.parallel_dim = parallel_dim
        self.init_method = init_method

        # Initialize weight tensor
        self.weight = nn.Parameter(torch.empty(parallel_dim, in_dim, out_dim))

        # Apply the selected initialization method
        self.reset_parameters()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, parallel_dim, in_dim)
        Returns:
            torch.Tensor: output tensor of shape (batch_size, parallel_dim, out_dim)
        """
        x = torch.einsum("npi,pio->npo", x, self.weight)
        return x

    @torch.no_grad()
    def reset_parameters(self):
        """ Initialize weights using the best method for deep learning stability. """
        if self.init_method == "kaiming":
            nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')
        elif self.init_method == "xavier":
            nn.init.xavier_uniform_(self.weight)
        elif self.init_method == "orthogonal":
            for i in range(self.parallel_dim):
                nn.init.orthogonal_(self.weight[i])  # Orthogonal initialization for stability
        else:
            bound = 1.0 / self.in_dim ** 0.5
            nn.init.uniform_(self.weight, -bound, bound)  # Default uniform initialization

    def __repr__(self):
        return f"LinearParallel(in_dim={self.in_dim}, out_dim={self.out_dim}, parallel_dim={self.parallel_dim}, init_method={self.init_method})"


# causalmoduleV2(diffAuto)
class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
        causal_discover:bool=False,
        num_causal_concepts:int = 4,
        
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(block_out_channels[0])  # Batch norm after first conv

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.BatchNorm2d(channel_in))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))
            self.blocks.append(nn.BatchNorm2d(channel_out)) 
        #320 
        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

        # causal Modeling
        self.num_num_causal_concepts = num_causal_concepts
        self.causal_latent_dim_N = 16
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_mu = nn.Linear(block_out_channels[-1], self.causal_latent_dim_N*self.num_num_causal_concepts)
        self.fc_var = nn.Linear(block_out_channels[-1], self.causal_latent_dim_N*self.num_num_causal_concepts)
        self.scale = np.array([[0,1],[0,1],[0,1],[0,1]])
        self.causal_mask = causalnn.CausalModeling(latent_dim=self.causal_latent_dim_N*self.num_num_causal_concepts, num_var=self.num_num_causal_concepts, learn=False)
        self.causal_discover = causal_discover
        self.A = None    
        # if self.causal_discover:
        #     self.A = nn.Parameter(1-torch.eye((4)))
        self.up_emb = nn.Sequential(nn.Linear(self.causal_latent_dim_N*self.num_num_causal_concepts,block_out_channels[-1]),
                                     nn.SiLU())
    def reparameterize(self,m, v):
        """
        Reparameterization Trick.
        """
        sample = torch.randn(m.size()).to(m.device)
        z = m + (v**0.5)*sample

        return z
    
    def set_A_martix(self,A):
        self.A = A

    def prior(self, scale, label, dim):
        mean = torch.ones(label.size()[0],label.size()[1],dim)
        var = torch.ones(label.size()[0],label.size()[1],dim)
        for i in range(label.size()[0]):
            for j in range(label.size()[1]):
                mul = (float(label[i][j])-scale[j][0])/(scale[j][1]-0)
                mean[i][j] = mul
        return mean, var

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

    def matrix_poly(self, matrix, d,device):
        x = torch.eye(d).to(device)+ torch.div(matrix.to(device), d).to(device)
        return torch.matrix_power(x, d)

    def _h_A(self,A, m,device):
        expm_A = self.matrix_poly(A*A, m,device)
        h_A = torch.trace(expm_A) - m
        return h_A
        

    def kl_normal(self, qm, qv, pm, pv):
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
    
    def inference(self, conditioning, sample=False,intervention_indx=None,intervention_values=None,label=None):
        embedding = self.conv_in(conditioning)
        embedding = self.bn_in(embedding)
        embedding = F.leaky_relu(embedding)

        # Apply each block (conv + batch norm + activation)
        for i in range(0, len(self.blocks), 4):  # Step by 4 because each block has 4 layers (conv, bn, conv, bn)
            embedding = self.blocks[i](embedding)  # First conv
            embedding = self.blocks[i+1](embedding)  # First batch norm
            embedding = F.leaky_relu(embedding)
            embedding = self.blocks[i+2](embedding)  # Second conv (with stride 2)
            embedding = self.blocks[i+3](embedding)  # Second batch norm
            embedding = F.leaky_relu(embedding)
        
       
        bs, channel, height,weight = embedding.shape
        # causal modeling
        device = embedding.device
        embedding = self.adaptive_pool(embedding)
        embedding = embedding.view(embedding.size(0), -1)
        
        mu = self.fc_mu(embedding)
        log_var = self.fc_var(embedding)
        var = F.softplus(log_var) + 1e-8

        if label is not None:
            x_expanded = label.unsqueeze(-1).repeat(1, 1, self.causal_latent_dim_N)  # shape: (bs, 4, 16)
            x_repeated = x_expanded.reshape(bs, self.causal_latent_dim_N*self.num_num_causal_concepts)
            mu=x_repeated       # shape: (bs, 64)
        if sample==True:
            mu = torch.randn((1,mu.size()[1])).to(mu.device)
            mu = mu.repeat(bs,1)

        if intervention_indx !=None:
            # 0-3
            reason_v_indices, result_v_indices = self.get_result_variable_indices(self.A)
            if len(reason_v_indices)>0 and intervention_indx in reason_v_indices:
                start_truncate = intervention_indx*self.causal_latent_dim_N
                end_truncate = (intervention_indx*self.causal_latent_dim_N)+self.causal_latent_dim_N
                mu[:, start_truncate:end_truncate] = torch.ones((embedding.shape[0], self.causal_latent_dim_N)) * intervention_values
                #z=torch.randn(z.shape).to(device)
        ## edit on mu
        # if intervention_indx !=None:
            
        #     start_truncate = intervention_indx*16
        #     end_truncate = (intervention_indx*16)+16
        #     mu[:, start_truncate:end_truncate] = torch.ones((embedding.shape[0], 16)) * intervention_values
        #     #z=torch.randn(z.shape).to(device)

        z_pre = self.causal_mask.causal_masking(mu, self.A)
        
        z_post = self.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(device)
        
        z_post=z_post.reshape(-1,self.causal_latent_dim_N*self.num_num_causal_concepts)
       
        if intervention_indx!=None or sample==True:
            var = torch.ones(var.shape).to(device)
        
        if intervention_indx !=None:
            # 0-3
            if len(result_v_indices)>0 and intervention_indx in result_v_indices:
                start_truncate = intervention_indx*self.causal_latent_dim_N
                end_truncate = (intervention_indx*self.causal_latent_dim_N)+self.causal_latent_dim_N
                z_post[:, start_truncate:end_truncate] = torch.ones((embedding.shape[0], self.causal_latent_dim_N)) * intervention_values
                

        z = self.reparameterize(z_post, var * 0.001)
        
        z = self.up_emb(z)
        
        # Step 1: unsqueeze to add two new dimensions, turning (bs, 256) -> (bs, 256, 1, 1)
        z_expanded = z.unsqueeze(2).unsqueeze(3)

        # Step 2: expand to shape (bs, 256, 12, 12) by repeating the values
        z_reshaped = z_expanded.expand(-1, -1, height,weight)
                
        output = self.conv_out(z_reshaped)
        return output,None




    def forward(self, conditioning, label):
        embedding = self.conv_in(conditioning)
        embedding = self.bn_in(embedding)
        embedding = F.leaky_relu(embedding)

        # Apply each block (conv + batch norm + activation)
        for i in range(0, len(self.blocks), 4):  # Step by 4 because each block has 4 layers (conv, bn, conv, bn)
            embedding = self.blocks[i](embedding)  # First conv
            embedding = self.blocks[i+1](embedding)  # First batch norm
            embedding = F.leaky_relu(embedding)
            embedding = self.blocks[i+2](embedding)  # Second conv (with stride 2)
            embedding = self.blocks[i+3](embedding)  # Second batch norm
            embedding = F.leaky_relu(embedding)

        # if self.causal_discover is False:
        #     self.A = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],dtype=embedding.dtype).to(embedding.device)

        bs, channel, height,weight = embedding.shape
        # causal modeling
        device = embedding.device
        embedding = self.adaptive_pool(embedding)
        embedding = embedding.view(embedding.size(0), -1)
             
        mu = self.fc_mu(embedding)
        log_var = self.fc_var(embedding)
        var = F.softplus(log_var) + 1e-8
        

        z_pre = self.causal_mask.causal_masking(mu, self.A)

        
        z_post = self.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(device)
        
        z_post=z_post.reshape(-1,self.causal_latent_dim_N*self.num_num_causal_concepts)

        z = self.reparameterize(z_post, var * 0.001)
        
        z = self.up_emb(z)
        
        # Step 1: unsqueeze to add two new dimensions, turning (bs, 256) -> (bs, 256, 1, 1)
        z_expanded = z.unsqueeze(2).unsqueeze(3)

        # Step 2: expand to shape (bs, 256, 12, 12) by repeating the values
        z_reshaped = z_expanded.expand(-1, -1, height,weight)
                

        # compute loss
        num_vars= self.num_num_causal_concepts
        zero_mean = torch.zeros(mu.shape).to(device)
        unit_var = torch.ones(var.shape).to(device)
        # [bs,4,1280]
        y_prior_mean, y_var = self.prior(self.scale, label, dim=mu.shape[1] // num_vars)

        kld = 0.0
        
        kld = self.kl_normal(mu, var, zero_mean, unit_var) # for standard Gaussian

        for i in range(num_vars):
            
            kld = kld + self.kl_normal(z_post.reshape(-1, num_vars, mu.shape[1] // num_vars)[:, i, :], 
                                    unit_var.reshape(-1, num_vars, mu.shape[1] // num_vars)[:, i, :], 
                                    y_prior_mean[:, i, :].to(device), 
                                    unit_var.reshape(-1, num_vars, mu.shape[1] // num_vars)[:, i, :])
        
        
        # For training the Causal Matrix A: 
        if self.causal_discover:  
            h_a = self._h_A(self.A, self.A.size()[0],device)
            kld = kld + 3*h_a + 0.5*h_a*h_a

        #embedding = self.conv_out(embedding)
        #return embedding  
        # bs 320
        output = self.conv_out(z_reshaped)
        return output,kld

# causalmoduleV2(diffAuto) not add back result factors
class ControlNetConditioningEmbedding_v2(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
        causal_discover:bool=False,
        num_causal_concepts:int = 4,
        
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(block_out_channels[0])  # Batch norm after first conv

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.BatchNorm2d(channel_in))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))
            self.blocks.append(nn.BatchNorm2d(channel_out)) 
        #320 
        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

        # causal Modeling
        self.num_num_causal_concepts = num_causal_concepts
        self.causal_latent_dim_N = 1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_mu = nn.Linear(block_out_channels[-1], self.causal_latent_dim_N*self.num_num_causal_concepts)
        self.fc_var = nn.Linear(block_out_channels[-1], self.causal_latent_dim_N*self.num_num_causal_concepts)
        self.scale = np.array([[0,1],[0,1],[0,1],[0,1]])
        self.causal_mask = causalnn.CausalModeling(latent_dim=self.causal_latent_dim_N*self.num_num_causal_concepts, num_var=self.num_num_causal_concepts, learn=False)
        self.causal_discover = causal_discover
        self.A = None    
        # if self.causal_discover:
        #     self.A = nn.Parameter(1-torch.eye((4)))
        self.up_emb = nn.Sequential(nn.Linear(self.causal_latent_dim_N*self.num_num_causal_concepts,block_out_channels[-1]),
                                     nn.SiLU())
    def reparameterize(self,m, v):
        """
        Reparameterization Trick.
        """
        sample = torch.randn(m.size()).to(m.device)
        z = m + (v**0.5)*sample

        return z
    
    def set_A_martix(self,A):
        self.A = A

    def prior(self, scale, label, dim):
        mean = torch.ones(label.size()[0],label.size()[1],dim)
        var = torch.ones(label.size()[0],label.size()[1],dim)
        for i in range(label.size()[0]):
            for j in range(label.size()[1]):
                mul = (float(label[i][j])-scale[j][0])/(scale[j][1]-0)
                mean[i][j] = mul
        return mean, var

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

    def matrix_poly(self, matrix, d,device):
        x = torch.eye(d).to(device)+ torch.div(matrix.to(device), d).to(device)
        return torch.matrix_power(x, d)

    def _h_A(self,A, m,device):
        expm_A = self.matrix_poly(A*A, m,device)
        h_A = torch.trace(expm_A) - m
        return h_A
        

    def kl_normal(self, qm, qv, pm, pv):
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
    
    def inference(self, conditioning, sample=False,intervention_indx=None,intervention_values=None,label=None):
        embedding = self.conv_in(conditioning)
        embedding = self.bn_in(embedding)
        embedding = F.leaky_relu(embedding)

        # Apply each block (conv + batch norm + activation)
        for i in range(0, len(self.blocks), 4):  # Step by 4 because each block has 4 layers (conv, bn, conv, bn)
            embedding = self.blocks[i](embedding)  # First conv
            embedding = self.blocks[i+1](embedding)  # First batch norm
            embedding = F.leaky_relu(embedding)
            embedding = self.blocks[i+2](embedding)  # Second conv (with stride 2)
            embedding = self.blocks[i+3](embedding)  # Second batch norm
            embedding = F.leaky_relu(embedding)
        
       
        bs, channel, height,weight = embedding.shape
        # causal modeling
        device = embedding.device
        embedding = self.adaptive_pool(embedding)
        embedding = embedding.view(embedding.size(0), -1)
        
        mu = self.fc_mu(embedding)
        log_var = self.fc_var(embedding)
        var = F.softplus(log_var) + 1e-8

        if label is not None:
            x_expanded = label.unsqueeze(-1).repeat(1, 1, self.causal_latent_dim_N)  # shape: (bs, 4, 16)
            x_repeated = x_expanded.reshape(bs, self.causal_latent_dim_N*self.num_num_causal_concepts)
            mu=x_repeated       # shape: (bs, 64)
        if sample==True:
            mu = torch.randn((1,mu.size()[1])).to(mu.device)
            mu = mu.repeat(bs,1)
        
        reason_v_indices, result_v_indices = self.get_result_variable_indices(self.A)
        if intervention_indx !=None:
            # 0-3
            
            if len(reason_v_indices)>0 and intervention_indx in reason_v_indices:
                start_truncate = intervention_indx*self.causal_latent_dim_N
                end_truncate = (intervention_indx*self.causal_latent_dim_N)+self.causal_latent_dim_N
                mu[:, start_truncate:end_truncate] = torch.ones((embedding.shape[0], self.causal_latent_dim_N)) * intervention_values
                #z=torch.randn(z.shape).to(device)
        ## edit on mu
        # if intervention_indx !=None:
            
        #     start_truncate = intervention_indx*16
        #     end_truncate = (intervention_indx*16)+16
        #     mu[:, start_truncate:end_truncate] = torch.ones((embedding.shape[0], 16)) * intervention_values
        #     #z=torch.randn(z.shape).to(device)

        z_pre = self.causal_mask.causal_masking(mu, self.A)
        
        z_post = self.causal_mask.nonlinearity_add_back_noise(mu, z_pre,result_v_indices=result_v_indices).to(device)
        
        z_post=z_post.reshape(-1,self.causal_latent_dim_N*self.num_num_causal_concepts)
       
        if intervention_indx!=None or sample==True:
            var = torch.ones(var.shape).to(device)
        
        if intervention_indx !=None:
            # 0-3
            if len(result_v_indices)>0 and intervention_indx in result_v_indices:
                start_truncate = intervention_indx*self.causal_latent_dim_N
                end_truncate = (intervention_indx*self.causal_latent_dim_N)+self.causal_latent_dim_N
                z_post[:, start_truncate:end_truncate] = torch.ones((embedding.shape[0], self.causal_latent_dim_N)) * intervention_values
                

        z = self.reparameterize(z_post, var * 0.001)
        
        z = self.up_emb(z)
        
        # Step 1: unsqueeze to add two new dimensions, turning (bs, 256) -> (bs, 256, 1, 1)
        z_expanded = z.unsqueeze(2).unsqueeze(3)

        # Step 2: expand to shape (bs, 256, 12, 12) by repeating the values
        z_reshaped = z_expanded.expand(-1, -1, height,weight)
                
        output = self.conv_out(z_reshaped)
        return output,None




    def forward(self, conditioning, label):
        embedding = self.conv_in(conditioning)
        embedding = self.bn_in(embedding)
        embedding = F.leaky_relu(embedding)

        # Apply each block (conv + batch norm + activation)
        for i in range(0, len(self.blocks), 4):  # Step by 4 because each block has 4 layers (conv, bn, conv, bn)
            embedding = self.blocks[i](embedding)  # First conv
            embedding = self.blocks[i+1](embedding)  # First batch norm
            embedding = F.leaky_relu(embedding)
            embedding = self.blocks[i+2](embedding)  # Second conv (with stride 2)
            embedding = self.blocks[i+3](embedding)  # Second batch norm
            embedding = F.leaky_relu(embedding)

        # if self.causal_discover is False:
        #     self.A = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],dtype=embedding.dtype).to(embedding.device)

        bs, channel, height,weight = embedding.shape
        # causal modeling
        device = embedding.device
        embedding = self.adaptive_pool(embedding)
        embedding = embedding.view(embedding.size(0), -1)
             
        mu = self.fc_mu(embedding)
        log_var = self.fc_var(embedding)
        var = F.softplus(log_var) + 1e-8
        

        z_pre = self.causal_mask.causal_masking(mu, self.A)

        reason_v_indices, result_v_indices = self.get_result_variable_indices(self.A)
        z_post = self.causal_mask.nonlinearity_add_back_noise(mu, z_pre,result_v_indices=result_v_indices).to(device)
        
        z_post=z_post.reshape(-1,self.causal_latent_dim_N*self.num_num_causal_concepts)

        z = self.reparameterize(z_post, var * 0.001)
        
        z = self.up_emb(z)
        
        # Step 1: unsqueeze to add two new dimensions, turning (bs, 256) -> (bs, 256, 1, 1)
        z_expanded = z.unsqueeze(2).unsqueeze(3)

        # Step 2: expand to shape (bs, 256, 12, 12) by repeating the values
        z_reshaped = z_expanded.expand(-1, -1, height,weight)
                

        # compute loss
        num_vars= self.num_num_causal_concepts
        zero_mean = torch.zeros(mu.shape).to(device)
        unit_var = torch.ones(var.shape).to(device)
        # [bs,4,1280]
        y_prior_mean, y_var = self.prior(self.scale, label, dim=mu.shape[1] // num_vars)

        kld = 0.0
        
        kld = self.kl_normal(mu, var, zero_mean, unit_var) # for standard Gaussian

        for i in range(num_vars):
            
            kld = kld + self.kl_normal(z_post.reshape(-1, num_vars, mu.shape[1] // num_vars)[:, i, :], 
                                    unit_var.reshape(-1, num_vars, mu.shape[1] // num_vars)[:, i, :], 
                                    y_prior_mean[:, i, :].to(device), 
                                    unit_var.reshape(-1, num_vars, mu.shape[1] // num_vars)[:, i, :])
        
        
        # For training the Causal Matrix A: 
        if self.causal_discover:  
            h_a = self._h_A(self.A, self.A.size()[0],device)
            kld = kld + 3*h_a + 0.5*h_a*h_a

        #embedding = self.conv_out(embedding)
        #return embedding  
        # bs 320
        output = self.conv_out(z_reshaped)
        return output,kld



class ControlNetConditioningEmbedding_Textcond(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
        causal_discover:bool=False,
        num_causal_concepts:int = 4,
        dataset = 'pendulum',
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(block_out_channels[0])  # Batch norm after first conv

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.BatchNorm2d(channel_in))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))
            self.blocks.append(nn.BatchNorm2d(channel_out)) 
        

        # causal Modeling
        self.causal_latent_dim_N = 1
        self.num_causal_concepts = num_causal_concepts
        if dataset == 'pendulum':
            # 1 for ce loss, 2 for mse loss
            self.num_causal_concepts = 4
            self.loss_types = [2,2,2,2]
        elif dataset == 'celeA':
            self.num_causal_concepts = 4
            self.loss_types = [1,1,1,1]
        elif dataset == 'chexpert':
            self.num_causal_concepts = 3
            self.loss_types = [1,2,1]
        elif dataset == 'skin_cancer':
            self.num_causal_concepts = 3
            self.loss_types = [2,2,2,2]

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_mu = nn.Linear(block_out_channels[-1], self.causal_latent_dim_N*self.num_causal_concepts)
        self.fc_var = nn.Linear(block_out_channels[-1], self.causal_latent_dim_N*self.num_causal_concepts)
        self.scale = np.array([[0,1],[0,1],[0,1],[0,1]])
        self.causal_mask = causalnn.CausalModeling(latent_dim=self.causal_latent_dim_N*self.num_causal_concepts, num_var=self.num_causal_concepts, learn=False)
        
        #self.A = None    

        # self.up_emb = nn.Sequential(LinearParallel(self.causal_latent_dim_N,128,num_causal_concepts),
        #                              nn.LeakyReLU(),)
        # #self.zero_linear = LinearParallel(128,768,num_causal_concepts)
        # self.zero_linear = zero_module(LinearParallel(128,768,num_causal_concepts))
        
                                     
    def reparameterize(self,m, v):
        """
        Reparameterization Trick.
        """
        sample = torch.randn(m.size()).to(m.device)
        z = m + (v**0.5)*sample

        return z
    
    def set_A_martix(self,A):
        self.A = A

    def prior(self, scale, label, dim,device):
        mean = torch.ones(label.size()[0],label.size()[1],dim).to(device)
        for i in range(label.size()[0]):
            for j in range(label.size()[1]):
                mean[i][j] = label[i][j]
        return mean

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


    def kl_normal(self, qm, qv, pm, pv):
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
    
    def inference(self, conditioning, sample=False,intervention_indx=None,intervention_values=None,label=None):
        embedding = self.conv_in(conditioning)
        embedding = self.bn_in(embedding)
        embedding = F.leaky_relu(embedding)

        # Apply each block (conv + batch norm + activation)
        for i in range(0, len(self.blocks), 4):  # Step by 4 because each block has 4 layers (conv, bn, conv, bn)
            embedding = self.blocks[i](embedding)  # First conv
            embedding = self.blocks[i+1](embedding)  # First batch norm
            embedding = F.leaky_relu(embedding)
            embedding = self.blocks[i+2](embedding)  # Second conv (with stride 2)
            embedding = self.blocks[i+3](embedding)  # Second batch norm
            embedding = F.leaky_relu(embedding)
        
       
        bs, channel, height,weight = embedding.shape
        # causal modeling
        device = embedding.device
        embedding = self.adaptive_pool(embedding)
        embedding = embedding.view(embedding.size(0), -1)
        
        mu = self.fc_mu(embedding)
        log_var = self.fc_var(embedding)
        var = F.softplus(log_var) + 1e-8

        if label is not None:
            x_expanded = label.unsqueeze(-1).repeat(1, 1, self.causal_latent_dim_N)  # shape: (bs, 4, 16)
            x_repeated = x_expanded.reshape(bs, self.causal_latent_dim_N*self.num_causal_concepts)
            mu=x_repeated
        
        if sample==True:
            mu = torch.randn((1,mu.size()[1])).to(mu.device)
            mu = mu.repeat(bs,1)

        if intervention_indx !=None:
            # 0-3
            reason_v_indices, result_v_indices = self.get_result_variable_indices(self.A)
            if len(reason_v_indices)>0 and intervention_indx in reason_v_indices:
                start_truncate = intervention_indx*self.causal_latent_dim_N
                end_truncate = (intervention_indx*self.causal_latent_dim_N)+self.causal_latent_dim_N
                mu[:, start_truncate:end_truncate] = torch.ones((embedding.shape[0], self.causal_latent_dim_N)) * intervention_values
                #z=torch.randn(z.shape).to(device)
        ## edit on mu
        # if intervention_indx !=None:
            
        #     start_truncate = intervention_indx*16
        #     end_truncate = (intervention_indx*16)+16
        #     mu[:, start_truncate:end_truncate] = torch.ones((embedding.shape[0], 16)) * intervention_values
        #     #z=torch.randn(z.shape).to(device)

        z_pre = self.causal_mask.causal_masking(mu, self.A)
        
        z_post = self.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(device)
        
        z=z_post.reshape(-1,self.causal_latent_dim_N*self.num_causal_concepts)
       
        if intervention_indx!=None or sample==True:
            var = torch.ones(var.shape).to(device)
        
        if intervention_indx !=None:
            # 0-3
            if len(result_v_indices)>0 and intervention_indx in result_v_indices:
                start_truncate = intervention_indx*self.causal_latent_dim_N
                end_truncate = (intervention_indx*self.causal_latent_dim_N)+self.causal_latent_dim_N
                z_post[:, start_truncate:end_truncate] = torch.ones((embedding.shape[0], self.causal_latent_dim_N)) * intervention_values
                
        
        # #z = self.reparameterize(z_post, var * 0.001)
        # z=z_post
        # z = self.up_emb(z)
        # output = self.zero_linear(z)
        output = z.reshape(z.shape[0], self.num_causal_concepts,self.causal_latent_dim_N)
        return output,None




    def forward(self, conditioning, label):
        embedding = self.conv_in(conditioning)
        embedding = self.bn_in(embedding)
        embedding = F.leaky_relu(embedding)

        # Apply each block (conv + batch norm + activation)
        for i in range(0, len(self.blocks), 4):  # Step by 4 because each block has 4 layers (conv, bn, conv, bn)
            embedding = self.blocks[i](embedding)  # First conv
            embedding = self.blocks[i+1](embedding)  # First batch norm
            embedding = F.leaky_relu(embedding)
            embedding = self.blocks[i+2](embedding)  # Second conv (with stride 2)
            embedding = self.blocks[i+3](embedding)  # Second batch norm
            embedding = F.leaky_relu(embedding)

        # if self.causal_discover is False:
        #     self.A = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],dtype=embedding.dtype).to(embedding.device)

        bs, channel, height,weight = embedding.shape
        # causal modeling
        device = embedding.device
        embedding = self.adaptive_pool(embedding)
        embedding = embedding.view(embedding.size(0), -1)
             
        mu = self.fc_mu(embedding)
        log_var = self.fc_var(embedding)
        var = F.softplus(log_var) + 1e-8
        

        z_pre = self.causal_mask.causal_masking(mu, self.A)

        
        z_post = self.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(device)
        

        # z=z_post
        # z = self.up_emb(z)
        # output = self.zero_linear(z)

        # compute loss
        
        num_vars= self.num_causal_concepts
        y_prior_mean = self.prior(self.scale, label.to(device), dim=mu.shape[1] // num_vars,device=device)
        loss = 0

        # #use kld to limit the mu value range 
        # zero_mean = torch.zeros(mu.shape).to(device)
        # unit_var = torch.ones(var.shape).to(device)
        # kld = self.kl_normal(mu, var, zero_mean, unit_var) # for standard Gaussian

        # loss = kld

        for i in range(num_vars):
            # if self.loss_types[i] == 1:
            #     loss = loss + F.binary_cross_entropy_with_logits(z_post.reshape(-1, num_vars, mu.shape[1] // num_vars)[:, i, :],label[:,i], dim=1)
            # elif self.loss_types[i] == 2:
            # (4,self.causal_latent_dim_N)
            l2_loss = F.mse_loss(z_post.reshape(-1, num_vars, mu.shape[1] // num_vars)[:, i, :],y_prior_mean[:, i],reduction='none')
            loss = loss+l2_loss.sum(-1)

        #embedding = self.conv_out(embedding)
        #return embedding  
        # bs 320
        #output = self.zero_linear(z)
        output = z_post
        return output,loss


class ControlNetConditioningEmbedding_V4(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
        causal_discover:bool=False,
        num_causal_concepts:int = 4,
        dataset = 'pendulum',
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(block_out_channels[0])  # Batch norm after first conv

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.BatchNorm2d(channel_in))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))
            self.blocks.append(nn.BatchNorm2d(channel_out)) 
        #320 
        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

        # causal Modeling
        self.causal_latent_dim_N = 1
        self.num_num_causal_concepts = num_causal_concepts
        # if dataset == 'pendulum':
        #     # 1 for ce loss, 2 for mse loss
        #     self.num_num_causal_concepts = 4
        #     self.loss_types = [2,2,2,2]
        # elif dataset == 'celeA':
        #     self.num_num_causal_concepts = 4
        #     self.loss_types = [1,1,1,1]
        # elif dataset == 'chexpert':
        #     self.num_num_causal_concepts = 3
        #     self.loss_types = [1,2,1]
        # elif dataset == 'skin_cancer':
        #     self.num_num_causal_concepts = 3
        #     self.loss_types = [2,2,2,2]

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_mu = nn.Linear(block_out_channels[-1], self.causal_latent_dim_N*self.num_num_causal_concepts)
        self.fc_var = nn.Linear(block_out_channels[-1], self.causal_latent_dim_N*self.num_num_causal_concepts)
        self.scale = np.array([[0,1],[0,1],[0,1],[0,1]])
        self.causal_mask = causalnn.CausalModeling(latent_dim=self.causal_latent_dim_N*self.num_num_causal_concepts, num_var=self.num_num_causal_concepts, learn=False)
        self.causal_discover = causal_discover
        self.A = None    
        # if self.causal_discover:
        #     self.A = nn.Parameter(1-torch.eye((4)))
        self.up_emb = nn.Sequential(nn.Linear(self.causal_latent_dim_N*self.num_num_causal_concepts,block_out_channels[-1]),
                                     nn.LeakyReLU())
    def reparameterize(self,m, v):
        """
        Reparameterization Trick.
        """
        sample = torch.randn(m.size()).to(m.device)
        z = m + (v**0.5)*sample

        return z
    
    def set_A_martix(self,A):
        self.A = A

    def prior(self, scale, label, dim,device):
        mean = torch.ones(label.size()[0],label.size()[1],dim).to(device)
        for i in range(label.size()[0]):
            for j in range(label.size()[1]):
                mean[i][j] = label[i][j]
        return mean

    def get_result_variable_indices(self,A):
        # Check each row to see if it has only zeros (does not cause any other variables)
        # and check each column to see if it has at least one non-zero (is influenced by other variables)
        result_variable_indices = []
        for i in range(A.size(0)):
            if torch.all(A[i] == 0) and torch.any(A[:, i] != 0):
                result_variable_indices.append(i)

        # Find reason variable indices
        reason_variable_indices = []
        for i in range(A.size(1)):
            if torch.all(A[:, i] == 0) and torch.any(A[i] != 0):
                reason_variable_indices.append(i)
        return reason_variable_indices,result_variable_indices

    def matrix_poly(self, matrix, d,device):
        x = torch.eye(d).to(device)+ torch.div(matrix.to(device), d).to(device)
        return torch.matrix_power(x, d)

    def _h_A(self,A, m,device):
        expm_A = self.matrix_poly(A*A, m,device)
        h_A = torch.trace(expm_A) - m
        return h_A
        

    def kl_normal(self, qm, qv, pm, pv):
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
    
    def inference(self, conditioning, sample=False,intervention_indx=None,intervention_values=None,label=None):
        embedding = self.conv_in(conditioning)
        embedding = self.bn_in(embedding)
        embedding = F.leaky_relu(embedding)

        # Apply each block (conv + batch norm + activation)
        for i in range(0, len(self.blocks), 4):  # Step by 4 because each block has 4 layers (conv, bn, conv, bn)
            embedding = self.blocks[i](embedding)  # First conv
            embedding = self.blocks[i+1](embedding)  # First batch norm
            embedding = F.leaky_relu(embedding)
            embedding = self.blocks[i+2](embedding)  # Second conv (with stride 2)
            embedding = self.blocks[i+3](embedding)  # Second batch norm
            embedding = F.leaky_relu(embedding)
        
       
        bs, channel, height,weight = embedding.shape
        # causal modeling
        device = embedding.device
        embedding = self.adaptive_pool(embedding)
        embedding = embedding.view(embedding.size(0), -1)
        
        mu = self.fc_mu(embedding)
        log_var = self.fc_var(embedding)
        var = F.softplus(log_var) + 1e-8
        # if label is not None:
        #     mu = label.to(mu.device)
        #     mu = mu.unsqueeze(0)
        
        if sample==True:
            mu = torch.randn((1,mu.size()[1])).to(mu.device)
            mu = mu.repeat(bs,1)

        if intervention_indx !=None:
            # 0-3
            reason_v_indices, result_v_indices = self.get_result_variable_indices(self.A)
            if len(reason_v_indices)>0 and intervention_indx in reason_v_indices:
                start_truncate = intervention_indx*self.causal_latent_dim_N
                end_truncate = (intervention_indx*self.causal_latent_dim_N)+self.causal_latent_dim_N
                mu[:, start_truncate:end_truncate] = torch.ones((embedding.shape[0], self.causal_latent_dim_N)) * intervention_values
                #z=torch.randn(z.shape).to(device)
        ## edit on mu
        # if intervention_indx !=None:
            
        #     start_truncate = intervention_indx*16
        #     end_truncate = (intervention_indx*16)+16
        #     mu[:, start_truncate:end_truncate] = torch.ones((embedding.shape[0], 16)) * intervention_values
        #     #z=torch.randn(z.shape).to(device)

        z_pre = self.causal_mask.causal_masking(mu, self.A)
        
        z_post = self.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(device)
        
        z=z_post.reshape(-1,self.causal_latent_dim_N*self.num_num_causal_concepts)
       
        if intervention_indx!=None or sample==True:
            var = torch.ones(var.shape).to(device)
        
        if intervention_indx !=None:
            # 0-3
            if len(result_v_indices)>0 and intervention_indx in result_v_indices:
                start_truncate = intervention_indx*self.causal_latent_dim_N
                end_truncate = (intervention_indx*self.causal_latent_dim_N)+self.causal_latent_dim_N
                z_post[:, start_truncate:end_truncate] = torch.ones((embedding.shape[0], self.causal_latent_dim_N)) * intervention_values
                
        
        #z = self.reparameterize(z_post, var * 0.001)
        
        z = self.up_emb(z)
        
        # Step 1: unsqueeze to add two new dimensions, turning (bs, 256) -> (bs, 256, 1, 1)
        z_expanded = z.unsqueeze(2).unsqueeze(3)

        # Step 2: expand to shape (bs, 256, 12, 12) by repeating the values
        z_reshaped = z_expanded.expand(-1, -1, height,weight)
                

        output = self.conv_out(z_reshaped)
        return output,None




    def forward(self, conditioning, label):
        embedding = self.conv_in(conditioning)
        embedding = self.bn_in(embedding)
        embedding = F.leaky_relu(embedding)

        # Apply each block (conv + batch norm + activation)
        for i in range(0, len(self.blocks), 4):  # Step by 4 because each block has 4 layers (conv, bn, conv, bn)
            embedding = self.blocks[i](embedding)  # First conv
            embedding = self.blocks[i+1](embedding)  # First batch norm
            embedding = F.leaky_relu(embedding)
            embedding = self.blocks[i+2](embedding)  # Second conv (with stride 2)
            embedding = self.blocks[i+3](embedding)  # Second batch norm
            embedding = F.leaky_relu(embedding)

        # if self.causal_discover is False:
        #     self.A = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],dtype=embedding.dtype).to(embedding.device)

        bs, channel, height,weight = embedding.shape
        # causal modeling
        device = embedding.device
        embedding = self.adaptive_pool(embedding)
        embedding = embedding.view(embedding.size(0), -1)
             
        mu = self.fc_mu(embedding)
        log_var = self.fc_var(embedding)
        var = F.softplus(log_var) + 1e-8
        

        z_pre = self.causal_mask.causal_masking(mu, self.A)

        
        z_post = self.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(device)
        

        z = z_post.reshape(-1,self.causal_latent_dim_N*self.num_num_causal_concepts)
        #z = self.reparameterize(z_post, var * 0.001)
        
        z = self.up_emb(z)
        
        # Step 1: unsqueeze to add two new dimensions, turning (bs, 256) -> (bs, 256, 1, 1)
        z_expanded = z.unsqueeze(2).unsqueeze(3)

        # Step 2: expand to shape (bs, 256, 32, 32) by repeating the values
        z_reshaped = z_expanded.expand(-1, -1, height,weight)
                

        # compute loss
        
        num_vars= self.num_num_causal_concepts
        y_prior_mean = self.prior(self.scale, label.to(device), dim=mu.shape[1] // num_vars,device=device)
        loss = 0.0

        # # use kld to limit the mu value range 
        # zero_mean = torch.zeros(mu.shape).to(device)
        # unit_var = torch.ones(var.shape).to(device)
        # kld = self.kl_normal(mu, var, zero_mean, unit_var) # for standard Gaussian

        # loss = kld
        
        for i in range(num_vars):
            # if self.loss_types[i] == 1:
            #     loss = loss + F.binary_cross_entropy_with_logits(z_post.reshape(-1, num_vars, mu.shape[1] // num_vars)[:, i, :],label[:,i], dim=1)
            # elif self.loss_types[i] == 2:
            # (4,self.causal_latent_dim_N)
            l2_loss = F.mse_loss(z_post.reshape(-1, num_vars, mu.shape[1] // num_vars)[:, i, :],y_prior_mean[:, i],reduction='none')
            loss = loss+l2_loss.sum(-1)
        
        # # For training the Causal Matrix A: 
        # if self.causal_discover:  
        #     h_a = self._h_A(self.A, self.A.size()[0],device)
        #     kld = kld + 3*h_a + 0.5*h_a*h_a

        #embedding = self.conv_out(embedding)
        #return embedding  
        # bs 320
        output = self.conv_out(z_reshaped)
        return output,loss




class Causal_ControlNetModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    """
    A ControlNet model.

    Args:
        in_channels (`int`, defaults to 4):
            The number of channels in the input sample.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, defaults to 0):
            The frequency shift to apply to the time embedding.
        down_block_types (`tuple[str]`, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use.
        only_cross_attention (`Union[bool, Tuple[bool]]`, defaults to `False`):
        block_out_channels (`tuple[int]`, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, defaults to 2):
            The number of layers per block.
        downsample_padding (`int`, defaults to 1):
            The padding to use for the downsampling convolution.
        mid_block_scale_factor (`float`, defaults to 1):
            The scale factor to use for the mid block.
        act_fn (`str`, defaults to "silu"):
            The activation function to use.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups to use for the normalization. If None, normalization and activation layers is skipped
            in post-processing.
        norm_eps (`float`, defaults to 1e-5):
            The epsilon to use for the normalization.
        cross_attention_dim (`int`, defaults to 1280):
            The dimension of the cross attention features.
        transformer_layers_per_block (`int` or `Tuple[int]`, *optional*, defaults to 1):
            The number of transformer blocks of type [`~models.attention.BasicTransformerBlock`]. Only relevant for
            [`~models.unet_2d_blocks.CrossAttnDownBlock2D`], [`~models.unet_2d_blocks.CrossAttnUpBlock2D`],
            [`~models.unet_2d_blocks.UNetMidBlock2DCrossAttn`].
        encoder_hid_dim (`int`, *optional*, defaults to None):
            If `encoder_hid_dim_type` is defined, `encoder_hidden_states` will be projected from `encoder_hid_dim`
            dimension to `cross_attention_dim`.
        encoder_hid_dim_type (`str`, *optional*, defaults to `None`):
            If given, the `encoder_hidden_states` and potentially other embeddings are down-projected to text
            embeddings of dimension `cross_attention` according to `encoder_hid_dim_type`.
        attention_head_dim (`Union[int, Tuple[int]]`, defaults to 8):
            The dimension of the attention heads.
        use_linear_projection (`bool`, defaults to `False`):
        class_embed_type (`str`, *optional*, defaults to `None`):
            The type of class embedding to use which is ultimately summed with the time embeddings. Choose from None,
            `"timestep"`, `"identity"`, `"projection"`, or `"simple_projection"`.
        addition_embed_type (`str`, *optional*, defaults to `None`):
            Configures an optional embedding which will be summed with the time embeddings. Choose from `None` or
            "text". "text" will use the `TextTimeEmbedding` layer.
        num_class_embeds (`int`, *optional*, defaults to 0):
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim`, when performing
            class conditioning with `class_embed_type` equal to `None`.
        upcast_attention (`bool`, defaults to `False`):
        resnet_time_scale_shift (`str`, defaults to `"default"`):
            Time scale shift config for ResNet blocks (see `ResnetBlock2D`). Choose from `default` or `scale_shift`.
        projection_class_embeddings_input_dim (`int`, *optional*, defaults to `None`):
            The dimension of the `class_labels` input when `class_embed_type="projection"`. Required when
            `class_embed_type="projection"`.
        controlnet_conditioning_channel_order (`str`, defaults to `"rgb"`):
            The channel order of conditional image. Will convert to `rgb` if it's `bgr`.
        conditioning_embedding_out_channels (`tuple[int]`, *optional*, defaults to `(16, 32, 96, 256)`):
            The tuple of output channel for each block in the `conditioning_embedding` layer.
        global_pool_conditions (`bool`, defaults to `False`):
            TODO(Patrick) - unused parameter.
        addition_embed_type_num_heads (`int`, defaults to 64):
            The number of heads to use for the `TextTimeEmbedding` layer.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        conditioning_channels: int = 3,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int, ...]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        projection_class_embeddings_input_dim: Optional[int] = None,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
        global_pool_conditions: bool = False,
        addition_embed_type_num_heads: int = 64,
        task_cond:str  ='generation_text_global_after',
        num_causal_concepts:int = 4,
        dataset = 'pendulum',
    ):
        super().__init__()

        # If `num_attention_heads` is not defined (which is the case for most models)
        # it will default to `attention_head_dim`. This looks weird upon first reading it and it is.
        # The reason for this behavior is to correct for incorrectly named variables that were introduced
        # when this library was created. The incorrect naming was only discovered much later in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131
        # Changing `attention_head_dim` to `num_attention_heads` for 40,000+ configurations is too backwards breaking
        # which is why we correct for the naming here.
        num_attention_heads = num_attention_heads or attention_head_dim

        # Check inputs
        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        # input
        conv_in_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        # time
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
        )

        if encoder_hid_dim_type is None and encoder_hid_dim is not None:
            encoder_hid_dim_type = "text_proj"
            self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)
            logger.info("encoder_hid_dim_type defaults to 'text_proj' as `encoder_hid_dim` is defined.")

        if encoder_hid_dim is None and encoder_hid_dim_type is not None:
            raise ValueError(
                f"`encoder_hid_dim` has to be defined when `encoder_hid_dim_type` is set to {encoder_hid_dim_type}."
            )

        if encoder_hid_dim_type == "text_proj":
            self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
        elif encoder_hid_dim_type == "text_image_proj":
            # image_embed_dim DOESN'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image_proj"` (Kandinsky 2.1)`
            self.encoder_hid_proj = TextImageProjection(
                text_embed_dim=encoder_hid_dim,
                image_embed_dim=cross_attention_dim,
                cross_attention_dim=cross_attention_dim,
            )

        elif encoder_hid_dim_type is not None:
            raise ValueError(
                f"encoder_hid_dim_type: {encoder_hid_dim_type} must be None, 'text_proj' or 'text_image_proj'."
            )
        else:
            self.encoder_hid_proj = None

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
            # The projection `class_embed_type` is the same as the timestep `class_embed_type` except
            # 1. the `class_labels` inputs are not first converted to sinusoidal embeddings
            # 2. it projects from an arbitrary input dimension.
            #
            # Note that `TimestepEmbedding` is quite general, being mainly linear layers and activations.
            # When used for embedding actual timesteps, the timesteps are first converted to sinusoidal embeddings.
            # As a result, `TimestepEmbedding` can be passed arbitrary vectors.
            self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        else:
            self.class_embedding = None

        if addition_embed_type == "text":
            if encoder_hid_dim is not None:
                text_time_embedding_from_dim = encoder_hid_dim
            else:
                text_time_embedding_from_dim = cross_attention_dim

            self.add_embedding = TextTimeEmbedding(
                text_time_embedding_from_dim, time_embed_dim, num_heads=addition_embed_type_num_heads
            )
        elif addition_embed_type == "text_image":
            # text_embed_dim and image_embed_dim DON'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image"` (Kandinsky 2.1)`
            self.add_embedding = TextImageTimeEmbedding(
                text_embed_dim=cross_attention_dim, image_embed_dim=cross_attention_dim, time_embed_dim=time_embed_dim
            )
        elif addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

        elif addition_embed_type is not None:
            raise ValueError(f"addition_embed_type: {addition_embed_type} must be None, 'text' or 'text_image'.")
        
        task_names= ['discovery_image','discovery_text_local','discovery_text_global_after',
                              'generation_image','generation_text_local','generation_text_global','generation_text_global_after','generation_text_local_after']
        assert task_cond in task_names,'{} should be in one of the {}'.format(task_cond,task_names)
        self.task_cond = task_cond
        self.dataset = dataset
        # control net conditioning embedding
        if 'discovery' in self.task_cond:
            if 'image' in self.task_cond:
                self.controlnet_cond_embedding = causal_modules.DAG_discovery5(
                    conditioning_embedding_channels=block_out_channels[0],
                    block_out_channels=conditioning_embedding_out_channels,
                    dims=[num_causal_concepts,1,1], 
                    bias = True
                )
            else:
                self.controlnet_cond_embedding = load_dataset_model(
                    num_causal_concepts,
                    dataset,
                    task_cond=self.task_cond
                )
        # else is counterfactual genetation task
        else:
            self.controlnet_cond_embedding = load_dataset_model(
                    num_causal_concepts,
                    dataset,
            )
            if 'image' in self.task_cond:
                if self.dataset == 'ADNI':
                    self.up_emb_blocks = nn.Sequential(nn.Linear(12,conditioning_embedding_out_channels[-1]),
                                        nn.SiLU())
                else:
                    self.up_emb_blocks = nn.Sequential(nn.Linear(num_causal_concepts,conditioning_embedding_out_channels[-1]),
                                        nn.SiLU())
                self.up_zero_block = zero_module(
                    nn.Conv2d(conditioning_embedding_out_channels[-1], block_out_channels[0], kernel_size=3, padding=1)
                )
            else:
                pass


            

        self.down_blocks = nn.ModuleList([])
        self.controlnet_down_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]

        controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=num_attention_heads[i],
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                downsample_padding=downsample_padding,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.down_blocks.append(down_block)

            for _ in range(layers_per_block):
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

            if not is_final_block:
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

        # mid
        mid_block_channel = block_out_channels[-1]

        controlnet_block = nn.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_mid_block = controlnet_block

        if mid_block_type == "UNetMidBlock2DCrossAttn":
            self.mid_block = UNetMidBlock2DCrossAttn(
                transformer_layers_per_block=transformer_layers_per_block[-1],
                in_channels=mid_block_channel,
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=num_attention_heads[-1],
                resnet_groups=norm_num_groups,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
            )
        elif mid_block_type == "UNetMidBlock2D":
            self.mid_block = UNetMidBlock2D(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                num_layers=0,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
                add_attention=False,
            )
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")
        
        


    @classmethod
    def from_unet(
        cls,
        unet: UNet2DConditionModel,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
        load_weights_from_unet: bool = True,
        conditioning_channels: int = 3,
        task_cond:str = "generation_text_global_after",
        num_causal_concepts:int = 4,
        dataset='pendulum'
    ):
        r"""
        Instantiate a [`ControlNetModel`] from [`UNet2DConditionModel`].

        Parameters:
            unet (`UNet2DConditionModel`):
                The UNet model weights to copy to the [`ControlNetModel`]. All configuration options are also copied
                where applicable.
        """
        transformer_layers_per_block = (
            unet.config.transformer_layers_per_block if "transformer_layers_per_block" in unet.config else 1
        )
        encoder_hid_dim = unet.config.encoder_hid_dim if "encoder_hid_dim" in unet.config else None
        encoder_hid_dim_type = unet.config.encoder_hid_dim_type if "encoder_hid_dim_type" in unet.config else None
        addition_embed_type = unet.config.addition_embed_type if "addition_embed_type" in unet.config else None
        addition_time_embed_dim = (
            unet.config.addition_time_embed_dim if "addition_time_embed_dim" in unet.config else None
        )

        controlnet = cls(
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=unet.config.in_channels,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
            down_block_types=unet.config.down_block_types,
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            downsample_padding=unet.config.downsample_padding,
            mid_block_scale_factor=unet.config.mid_block_scale_factor,
            act_fn=unet.config.act_fn,
            norm_num_groups=unet.config.norm_num_groups,
            norm_eps=unet.config.norm_eps,
            cross_attention_dim=unet.config.cross_attention_dim,
            attention_head_dim=unet.config.attention_head_dim,
            num_attention_heads=unet.config.num_attention_heads,
            use_linear_projection=unet.config.use_linear_projection,
            class_embed_type=unet.config.class_embed_type,
            num_class_embeds=unet.config.num_class_embeds,
            upcast_attention=unet.config.upcast_attention,
            resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
            projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,
            mid_block_type=unet.config.mid_block_type,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
            task_cond = task_cond,
            num_causal_concepts = num_causal_concepts,
            dataset = dataset
        )

        if load_weights_from_unet:
            controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
            controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
            controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())

            if controlnet.class_embedding:
                controlnet.class_embedding.load_state_dict(unet.class_embedding.state_dict())

            if hasattr(controlnet, "add_embedding"):
                controlnet.add_embedding.load_state_dict(unet.add_embedding.state_dict())

            controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
            controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())

        return controlnet

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attention_slice
    def set_attention_slice(self, slice_size: Union[str, int, List[int]]) -> None:
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module splits the input tensor in slices to compute attention in
        several steps. This is useful for saving some memory in exchange for a small decrease in speed.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, input to the attention heads is halved, so attention is computed in two steps. If
                `"max"`, maximum amount of memory is saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value: bool = False) -> None:
        if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D)):
            module.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
        training:bool=True,
        sampling: bool = False,
        text_encoder = None,
        do_classifier_free_guidance = False,
        disentangle = False,
        p2p_edit = False,
        **kwargs
    ) -> Union[ControlNetOutput, Tuple[Tuple[torch.Tensor, ...], torch.Tensor]]:
        """
        The [`ControlNetModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor.
            timestep (`Union[torch.Tensor, float, int]`):
                The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                token_ids
            controlnet_cond (`torch.Tensor`):
                The conditional input tensor of shape `(batch_size, sequence_length, hidden_size)`.
            conditioning_scale (`float`, defaults to `1.0`):
                The scale factor for ControlNet outputs.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond (`torch.Tensor`, *optional*, defaults to `None`):
                Additional conditional embeddings for timestep. If provided, the embeddings will be summed with the
                timestep_embedding passed through the `self.time_embedding` layer to obtain the final timestep
                embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            added_cond_kwargs (`dict`):
                Additional conditions for the Stable Diffusion XL UNet.
            cross_attention_kwargs (`dict[str]`, *optional*, defaults to `None`):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor`.
            guess_mode (`bool`, defaults to `False`):
                In this mode, the ControlNet encoder tries its best to recognize the input content of the input even if
                you remove all prompts. A `guidance_scale` between 3.0 and 5.0 is recommended.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~models.controlnet.ControlNetOutput`] instead of a plain tuple.

        Returns:
            [`~models.controlnet.ControlNetOutput`] **or** `tuple`:
                If `return_dict` is `True`, a [`~models.controlnet.ControlNetOutput`] is returned, otherwise a tuple is
                returned where the first element is the sample tensor.
        """
        # check channel order
        channel_order = self.config.controlnet_conditioning_channel_order

        # if channel_order == "rgb":
        #     # in rgb order by default
        #     ...
        # elif channel_order == "bgr":
        #     controlnet_cond = torch.flip(controlnet_cond, dims=[1])
        # else:
        #     raise ValueError(f"unknown `controlnet_conditioning_channel_order`: {channel_order}")

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        if self.config.addition_embed_type is not None:
            if self.config.addition_embed_type == "text":
                aug_emb = self.add_embedding(encoder_hidden_states)

            elif self.config.addition_embed_type == "text_time":
                if "text_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                    )
                text_embeds = added_cond_kwargs.get("text_embeds")
                if "time_ids" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                    )
                time_ids = added_cond_kwargs.get("time_ids")
                time_embeds = self.add_time_proj(time_ids.flatten())
                time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

                add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                add_embeds = add_embeds.to(emb.dtype)
                aug_emb = self.add_embedding(add_embeds)

        emb = emb + aug_emb if aug_emb is not None else emb

        # 2. pre-process
        sample = self.conv_in(sample)
        
    
        if 'image' in self.task_cond:
            if training:
                #pass
                controlnet_cond,causal_loss = self.controlnet_cond_embedding(kwargs['label'])
                encoder_hidden_states = text_encoder(encoder_hidden_states)[0].to(dtype=emb.dtype)
            else:
                #controlnet_cond,causal_loss = self.controlnet_cond_embedding(controlnet_cond,kwargs['label'])
                controlnet_cond,causal_loss = self.controlnet_cond_embedding.inference(kwargs['label'].squeeze(2),intervention_indx=kwargs['intervention_indx'],intervention_values=kwargs['intervention_values'],disentangle=disentangle)
            '''Project causal attrs outside'''
            if controlnet_cond.shape != kwargs['label'].shape:
                controlnet_cond = controlnet_cond.squeeze(3)
            sample_batch_size = sample.shape[0]

            #controlnet_cond = kwargs['label']
            if controlnet_cond.shape[0] != sample_batch_size:
                controlnet_cond = controlnet_cond.repeat(2, 1, 1)

            if self.dataset == 'ADNI':      
                controlnet_cond=controlnet_cond[:,4:,:]

            z = self.up_emb_blocks(controlnet_cond.squeeze(2))
            # Step 1: unsqueeze to add two new dimensions, turning (bs, 256) -> (bs, 256, 1, 1)
            z_expanded = z.unsqueeze(2).unsqueeze(3)

            # Step 2: expand to shape (bs, 256, 32, 32) by repeating the values
            z_reshaped = z_expanded.expand(-1, -1, 32,32)
            z_out = self.up_zero_block(z_reshaped)

            z_out_cfg = torch.cat([z_out] * 2) if do_classifier_free_guidance else z_out
            
            sample = sample + z_out_cfg
            causal_loss = 0
            
        # for discovery
        elif 'discovery' in self.task_cond and 'text' in self.task_cond:
            if training:
                #pass
                controlnet_cond,causal_loss = self.controlnet_cond_embedding(kwargs['label'])
            else:
                #controlnet_cond,causal_loss = self.controlnet_cond_embedding(controlnet_cond,kwargs['label'])
                controlnet_cond,causal_loss = self.controlnet_cond_embedding.inference(kwargs['label'],intervention_indx=kwargs['intervention_indx'],intervention_values=kwargs['intervention_values'],disentangle=disentangle)
            # controlnet_cond (bs, num_concepts, n) or 
            if 'after' in self.task_cond:
                # insert embedding after transformer
                def get_concept_ids(text_encoder):
                    model = text_encoder.module if hasattr(text_encoder, "module") else text_encoder
                    return model.text_model.embeddings.embed_control.control_concept_ids

                concept_ids = get_concept_ids(text_encoder)
                input_ids = encoder_hidden_states.clone()
                encoder_hidden_states = text_encoder(encoder_hidden_states)[0].to(dtype=emb.dtype)
                if self.dataset == 'ADNI':
                    controlnet_cond_clone = controlnet_cond.clone()
                    if len(concept_ids)==3:
                        #if controlnet_cond_clone.shape[1] == 16:
                        # only use brain_v, ven_v and slice 0-9 following benchmark
                        controlnet_cond_clone=controlnet_cond_clone[:,4:,:]
                    if controlnet_cond_clone.shape[1]==6:
                        for i,token_id in enumerate(concept_ids):
                            placeholder_idx = torch.where(input_ids == token_id)
                            encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond_clone[:,i,:]
                    elif controlnet_cond_clone.shape[1] == 12:
                        for i,token_id in enumerate(concept_ids):
                            placeholder_idx = torch.where(input_ids == token_id)
                            if i==len(concept_ids)-1:
                                encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond_clone[:,i:].reshape(-1,1)
                            else:
                                encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond_clone[:,i,:]
                    elif controlnet_cond_clone.shape[1] > 12:
                        for i,token_id in enumerate(concept_ids):
                            placeholder_idx = torch.where(input_ids == token_id)
                            if i==0:
                                encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond_clone[:,:2].reshape(-1,1)
                            elif i==len(concept_ids)-1:
                                encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond_clone[:,-10:].reshape(-1,1)
                            else:
                                encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond_clone[:,i+1,:]
                else:
                    for i,token_id in enumerate(concept_ids):
                        placeholder_idx = torch.where(input_ids == token_id)
                        encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond[:,i,:]
                    
                    # extract_embedds = torch.ones(encoder_hidden_states.shape[0],len(concept_ids),768)
                    # for i,token_id in enumerate(concept_ids):
                    #     placeholder_idx = torch.where(input_ids == token_id)
                    #     extract_embedds[:,i,:] = encoder_hidden_states[placeholder_idx]

                    # controlnet_cond,causal_loss = self.controlnet_cond_embedding(extract_embedds)
                    # for i,token_id in enumerate(concept_ids):
                    #     placeholder_idx = torch.where(input_ids == token_id)
                    #     encoder_hidden_states[placeholder_idx] = controlnet_cond[:,i,:]
            else:    
                encoder_hidden_states = text_encoder(encoder_hidden_states,attribute_cond = controlnet_cond)[0].to(dtype=emb.dtype)
        # for counterfactuals
        elif 'generation' in self.task_cond and 'text' in self.task_cond:
            if training:
                controlnet_cond,causal_loss = self.controlnet_cond_embedding(kwargs['label'])
                # controlnet_cond (bs, num_concepts, n) or 
                if 'after' in self.task_cond:
                    # insert embedding after transformer
                    def get_concept_ids(text_encoder):
                        model = text_encoder.module if hasattr(text_encoder, "module") else text_encoder
                        return model.text_model.embeddings.embed_control.control_concept_ids

                    concept_ids = get_concept_ids(text_encoder)
                    input_ids = encoder_hidden_states.clone()
                    encoder_hidden_states = text_encoder(encoder_hidden_states)[0].to(dtype=emb.dtype)
                    if self.dataset == 'ADNI':
                        controlnet_cond_clone = controlnet_cond.clone()
                        if len(concept_ids)==3:
                            #if controlnet_cond_clone.shape[1] == 16:
                            # only use brain_v, ven_v and slice 0-9 following benchmark
                            controlnet_cond_clone=controlnet_cond_clone[:,4:,:]
                        if controlnet_cond_clone.shape[1]==6:
                            for i,token_id in enumerate(concept_ids):
                                placeholder_idx = torch.where(input_ids == token_id)
                                encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond_clone[:,i,:]
                        elif controlnet_cond_clone.shape[1] == 12:
                            for i,token_id in enumerate(concept_ids):
                                placeholder_idx = torch.where(input_ids == token_id)
                                if i==len(concept_ids)-1:
                                    encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond_clone[:,i:].reshape(-1,1)
                                else:
                                    encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond_clone[:,i,:]
                        elif controlnet_cond_clone.shape[1] > 12:
                            for i,token_id in enumerate(concept_ids):
                                placeholder_idx = torch.where(input_ids == token_id)
                                if i==0:
                                    encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond_clone[:,:2].reshape(-1,1)
                                elif i==len(concept_ids)-1:
                                    encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond_clone[:,-10:].reshape(-1,1)
                                else:
                                    encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond_clone[:,i+1,:]
                    else:
                        for i,token_id in enumerate(concept_ids):
                            placeholder_idx = torch.where(input_ids == token_id)
                            encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond[:,i,:]
                        
                else:    
                    encoder_hidden_states = text_encoder(encoder_hidden_states,attribute_cond = controlnet_cond)[0].to(dtype=emb.dtype)
            else:
                controlnet_cond = 0
                causal_loss = 0
                #controlnet_cond,causal_loss = self.controlnet_cond_embedding(controlnet_cond,kwargs['label'])
                #controlnet_cond,causal_loss = self.controlnet_cond_embedding.inference(kwargs['label'],intervention_indx=kwargs['intervention_indx'],intervention_values=kwargs['intervention_values'],disentangle=disentangle)


        # if do_classifier_free_guidance:
        #     if kwargs['negtive_prompt_embedding'] is not None:
        #         encoder_hidden_states = torch.cat([kwargs['negtive_prompt_embedding'], encoder_hidden_states],dim=0)
        #     else:
        #         assert kwargs['negtive_prompt_embedding'] is not None, 'negative_prompt_embedding should be provided when do_classifier_free_guidance is True'


        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = self.mid_block(sample, emb)

        # 5. Control net blocks
        controlnet_down_block_res_samples = ()
        # pass the zero convolution
        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)
        # len() = 12
        # Shapes: [torch.Size([2, 320, 32, 32]), torch.Size([2, 320, 32, 32]), torch.Size([2, 320, 32, 32]), torch.Size([2, 320, 16, 16]), torch.Size([2, 640, 16, 16]), torch.Size([2, 640, 16, 16]), torch.Size([2, 640, 8, 8]), torch.Size([2, 1280, 8, 8]), torch.Size([2, 1280, 8, 8]), torch.Size([2, 1280, 4, 4]), torch.Size([2, 1280, 4, 4]), torch.Size([2, 1280, 4, 4])]
        down_block_res_samples = controlnet_down_block_res_samples
        
        mid_block_res_sample = self.controlnet_mid_block(sample)

        # 6. scaling
        if guess_mode and not self.config.global_pool_conditions:
            scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1, device=sample.device)  # 0.1 to 1.0
            scales = scales * conditioning_scale
            down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
            mid_block_res_sample = mid_block_res_sample * scales[-1]  # last one
        else:
            down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
            mid_block_res_sample = mid_block_res_sample * conditioning_scale

        if self.config.global_pool_conditions:
            down_block_res_samples = [
                torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples
            ]
            mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3), keepdim=True)

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample),causal_loss,encoder_hidden_states,controlnet_cond

        return ControlNetOutput(down_block_res_samples=down_block_res_samples, mid_block_res_sample=mid_block_res_sample),causal_loss,encoder_hidden_states,controlnet_cond


def zero_module(module):
    #return module 
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
