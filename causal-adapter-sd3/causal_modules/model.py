#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import torch
import numpy as np
from .codebase import utils as ut
from .codebase.models import nns
from torch import nn
from torch.nn import functional as F
#device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")


class Causal_SCM(nn.Module):
    def __init__(self, nn='mask', name='vae', z_dim=16, z1_dim=4, z2_dim=4, inference = False, alpha=0.3, beta=1, initial=True):
        super().__init__()

        self.name = name
        self.z_dim = z_dim # the full dimension
        self.z1_dim = z1_dim # number of concepts
        self.z2_dim = z2_dim # the dimension of each concepts
        self.channel = 4
        #self.scale = np.array([[0,44],[100,40],[6.5, 3.5],[10,5]])
        self.scale = np.array([[0,1],[0,1],[0,1],[0,1]])
        ## nn='mask'
        nn = getattr(nns, nn)

        self.rep_emb = nn.GaussianConvEncoder(in_channels=3, latent_dim=self.rep_dim, num_vars=4)
        
        #self.enc = nn.Encoder(self.z_dim, self.channel)
        #self.dec = nn.Decoder_DAG(self.z_dim, self.z1_dim, self.z2_dim)
        self.dag = nn.DagLayer(self.z1_dim, self.z1_dim, i = inference, initial=initial)
        self.attn = nn.Attention(self.z2_dim)
        self.mask_z = nn.MaskLayer(self.z_dim, z2_dim=self.z2_dim)
        self.mask_u = nn.MaskLayer(self.z1_dim, z2_dim=1)


        

    def negative_elbo_bound(self, x, label, mask = None, sample = False, adj = None, alpha=0.3, beta=1, lambdav=0.001,device= torch.device("cuda:0")):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        
        assert label.size()[1] == self.z1_dim
        
        # # mean , var = split(x_hidden,2) <---- (bs,2*z_dim=16)
        q_m, q_v = self.enc.encode(x.to(device))
        q_m, q_v = q_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]),torch.ones(q_m.size()[0], self.z1_dim,self.z2_dim).to(device)

        # '''directly input text embedding here'''
        # # q_V is covered by torch.ones
        q_m = torch.matmul(x_concepts, x_concepts.transpose(-2, -1))
        q_v = torch.ones(q_m.size()[0], self.z1_dim,self.z2_dim).to(device)
        #q_m, q_v = q_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]),torch.ones(q_m.size()[0], self.z1_dim,self.z2_dim).to(device)
        # decoder_v: ones(bs,4,4)
        decode_m, decode_v = self.dag.calculate_dag(q_m.to(device), torch.ones(q_m.size()[0], self.z1_dim,self.z2_dim).to(device))
        decode_m, decode_v = decode_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]),decode_v
        if sample == False:
          if mask != None and mask < 2:
              z_mask = torch.ones(q_m.size()[0], self.z1_dim,self.z2_dim).to(device)*adj
              decode_m[:, mask, :] = z_mask[:, mask, :]
              decode_v[:, mask, :] = z_mask[:, mask, :]
          # A_T * x
          m_zm, m_zv = self.dag.mask_z(decode_m.to(device)).reshape([q_m.size()[0], self.z1_dim,self.z2_dim]),decode_v.reshape([q_m.size()[0], self.z1_dim,self.z2_dim])
          m_u = self.dag.mask_u(label.to(device))
          
          f_z = self.mask_z.mix(m_zm).reshape([q_m.size()[0], self.z1_dim,self.z2_dim]).to(device)
          e_tilde = self.attn.attention(decode_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]).to(device),q_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]).to(device))[0]
          if mask != None and mask < 2:
              z_mask = torch.ones(q_m.size()[0],self.z1_dim,self.z2_dim).to(device)*adj
              e_tilde[:, mask, :] = z_mask[:, mask, :]
              
          f_z1 = f_z+e_tilde
          if mask!= None and mask == 2 :
              z_mask = torch.ones(q_m.size()[0],self.z1_dim,self.z2_dim).to(device)*adj
              f_z1[:, mask, :] = z_mask[:, mask, :]
              m_zv[:, mask, :] = z_mask[:, mask, :]
          if mask!= None and mask == 3 :
              z_mask = torch.ones(q_m.size()[0],self.z1_dim,self.z2_dim).to(device)*adj
              f_z1[:, mask, :] = z_mask[:, mask, :]
              m_zv[:, mask, :] = z_mask[:, mask, :]
          g_u = self.mask_u.mix(m_u).to(device)
          z_given_dag = ut.conditional_sample_gaussian(f_z1, m_zv*lambdav)
        
        # decoded_bernoulli_logits,x1,x2,x3,x4 = self.dec.decode_sep(z_given_dag.reshape([z_given_dag.size()[0], self.z_dim]), label.to(device))
        
        # rec = ut.log_bernoulli_with_logits(x, decoded_bernoulli_logits.reshape(x.size()))
        # rec = -torch.mean(rec)

        p_m, p_v = torch.zeros(q_m.size()), torch.ones(q_m.size())
        cp_m, cp_v = ut.condition_prior(self.scale, label, self.z2_dim)
        cp_v = torch.ones([q_m.size()[0],self.z1_dim,self.z2_dim]).to(device)
        cp_z = ut.conditional_sample_gaussian(cp_m.to(device), cp_v.to(device))
        kl = torch.zeros(1).to(device)
        ##CausalDiffAE add the following 
        kl = alpha*ut.kl_normal(q_m.view(-1,self.z_dim).to(device), q_v.view(-1,self.z_dim).to(device), p_m.view(-1,self.z_dim).to(device), p_v.view(-1,self.z_dim).to(device))
        
        for i in range(self.z1_dim):
            kl = kl + beta*ut.kl_normal(decode_m[:,i,:].to(device), cp_v[:,i,:].to(device),cp_m[:,i,:].to(device), cp_v[:,i,:].to(device))
        kl = torch.mean(kl)
        mask_kl = torch.zeros(1).to(device)
        mask_kl2 = torch.zeros(1).to(device)

        for i in range(4):
            mask_kl = mask_kl + 1*ut.kl_normal(f_z1[:,i,:].to(device), cp_v[:,i,:].to(device),cp_m[:,i,:].to(device), cp_v[:,i,:].to(device))
        
        
        u_loss = torch.nn.MSELoss()
        mask_l = torch.mean(mask_kl) + u_loss(g_u, label.float().to(device))
        #nelbo = rec + kl + mask_l
        nelbo = kl+mask_l
        return nelbo, kl, z_given_dag,self.dag.A

    def loss(self, x):
        nelbo, kl, rec, _, _ = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))

def matrix_poly(matrix, d,device):
    x = torch.eye(d).to(device)+ torch.div(matrix.to(device), d).to(device)
    return torch.matrix_power(x, d)

def _h_A(A, m,device):
    expm_A = matrix_poly(A*A, m,device)
    h_A = torch.trace(expm_A) - m
    return h_A


def linear_kl_weight_scheduler(step, total_steps, initial, final):
        """Linear scheduler"""

        if step >= total_steps:
            return final
        if step <= 0:
            return initial
        if total_steps <= 1:
            return final

        t = step / (total_steps - 1)
        return (1.0 - t) * initial + t * final

class Causal_SCM_v2(nn.Module):
    def __init__(self, nn='mask', name='vae', z_dim=16, z1_dim=4, z2_dim=4, inference = False, alpha=0.3, beta=1, initial=True):
        super().__init__()

        self.name = name
        self.z_dim = z_dim # the full dimension
        self.z1_dim = z1_dim # number of concepts
        self.z2_dim = z2_dim # the dimension of each concepts
        self.channel = 4
        #self.scale = np.array([[0,44],[100,40],[6.5, 3.5],[10,5]])
        self.scale = np.array([[0,1],[0,1],[0,1],[0,1]])
        ## nn='mask'
        nn = getattr(nns, nn)
        self.in_channels = 3
        self.rep_dim  = 64
        
        self.rep_emb = nn.GaussianConvEncoder(in_channels=self.in_channels, latent_dim=self.rep_dim, num_vars=4)
        
        self.causal_mask = nn.CausalModeling(latent_dim=64, num_var=4, learn=False)
        self.up_emb  = torch.nn.Linear(self.rep_dim,1280)
        


    def negative_elbo_bound(self, x, label, mask = None, sample = False, adj = None, alpha=0.3, beta=1, lambdav=0.001,device= torch.device("cuda:0"),intervention_indx=0,intervention_values=0):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        
        assert label.size()[1] == self.z1_dim
        # pick the 4 concepts embedding from the input [bs,77,1280]
        x,label = x.to(device),label.to(device)
        
        mu,var = self.rep_emb.encode(x)

        if intervention_indx !=0:
            start_truncate = intervention_indx*16-16
            end_truncate = intervention_indx*16
            mu[:, start_truncate:end_truncate] = torch.ones((x.shape[0], 16)) * intervention_values
            #z=torch.randn(z.shape).to(device)
        
        A = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.float32).to(device)

        z_pre = self.causal_mask.causal_masking(mu, A)

        
        z_post = self.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(device)
        # if intervention_indx !=0:
        #     start_truncate = intervention_indx*16-16
        #     end_truncate = intervention_indx*16
        #     z_post[:, start_truncate:end_truncate] = torch.ones((x.shape[0], 16)) * intervention_values

        
        # bs,64 -> bs,4,16
        z = self.reparameterize(z_post, var * 0.001)
        # if intervention_indx !=0:
        #     start_truncate = intervention_indx*16-16
        #     end_truncate = intervention_indx*16
        #     z[:, start_truncate:end_truncate] = torch.ones((x.shape[0], 16)) * intervention_values
        #     #z=torch.randn(z.shape).to(device)
        z = self.up_emb(z)
        

        #z = z+residual_x

        # if self.masking:
        #     context_mask = torch.bernoulli(torch.zeros(z.shape[0])+ (1-self.drop_prob)).to(z.device)
        #     base_context_mask = context_mask
        #     context_mask = context_mask[:, None]
        #     context_mask = context_mask.repeat(1, 512)
        #     # context_mask = context_mask.repeat(1, 64)

            
        #     z = z * context_mask
        #     z = z.float()
            
        #     z_post = (z_post * context_mask).float()
        #     mask = base_context_mask
        #     # var = (var * context_mask).float()
        # [bs,4,16] -> [bs,4,1280]
        

        # compute loss
        num_vars= 4
        zero_mean = torch.zeros(mu.shape).to(device)
        unit_var = torch.ones(var.shape).to(device)
        # [bs,4,1280]
        y_prior_mean, y_var = self.prior(self.scale, label, dim=mu.shape[1] // num_vars)

        kld = 0.0
        
        kld = self.kl_normal(mu, var, zero_mean, unit_var) # for standard Gaussian

        for i in range(4):
            
            kld = kld + self.kl_normal(z_post.reshape(-1, num_vars, mu.shape[1] // num_vars)[:, i, :], 
                                    unit_var.reshape(-1, num_vars, mu.shape[1] // num_vars)[:, i, :], 
                                    y_prior_mean[:, i, :].to(device), 
                                    unit_var.reshape(-1, num_vars, mu.shape[1] // num_vars)[:, i, :])

        # if mask is not None:
        #     masked_kl = kld * mask
        #     kld = torch.sum(masked_kl) / torch.sum(mask)


        return kld,z

    def reparameterize(self,m, v):
        """
        Reparameterization Trick.
        """
        sample = torch.randn(m.size()).to(m.device)
        z = m + (v**0.5)*sample
        
        return z

    def prior(self, scale, label, dim):
        mean = torch.ones(label.size()[0],label.size()[1],dim)
        var = torch.ones(label.size()[0],label.size()[1],dim)
        for i in range(label.size()[0]):
            for j in range(label.size()[1]):
                mul = (float(label[i][j])-scale[j][0])/(scale[j][1]-0)
                mean[i][j] = mul
        return mean, var

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

    def kl_normal_3d(self, qm, qv, pm, pv, eps=1e-6):
        """
        Computes the element-wise KL divergence between two normal distributions KL(q || p)
        for 3D tensors and sums over the last two dimensions (n_words, embeddings).

        Args:
            qm: tensor: (batch_size, n_words, embeddings): q mean
            qv: tensor: (batch_size, n_words, embeddings): q variance
            pm: tensor: (batch_size, n_words, embeddings): p mean
            pv: tensor: (batch_size, n_words, embeddings): p variance
            eps: small value to prevent log(0)

        Return:
            kl: tensor: (batch_size,): kl divergence for each sample
        """
        # Adding eps to prevent instability in log and division
        # qv = torch.clamp(qv, min=eps)
        # pv = torch.clamp(pv, min=eps)
        
        # Compute element-wise KL divergence
        element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
        
        # Sum over the last two dimensions (n_words and embeddings)
        kl = element_wise.sum(dim=(-1, -2))
        
        return kl



class Causal_SCM_v3(nn.Module):
    def __init__(self, nn='mask', name='vae', z_dim=16, z1_dim=4, z2_dim=4, inference = False, alpha=0.3, beta=1, initial=True):
        super().__init__()

        self.name = name
        self.z_dim = z_dim # the full dimension
        self.z1_dim = z1_dim # number of concepts
        self.z2_dim = z2_dim # the dimension of each concepts
        self.channel = 4
        #self.scale = np.array([[0,44],[100,40],[6.5, 3.5],[10,5]])
        self.scale = np.array([[0,1],[0,1],[0,1],[0,1]])
        ## nn='mask'
        nn = getattr(nns, nn)

        self.rep_emb = nn.GaussianConvEncoder(in_channels=3, latent_dim=z_dim, num_vars=4)
        
        #self.enc = nn.Encoder(self.z_dim, self.channel)
        #self.dec = nn.Decoder_DAG(self.z_dim, self.z1_dim, self.z2_dim)
        self.dag = nn.DagLayer(self.z1_dim, self.z1_dim, i = inference, initial=initial,causal_discover=True)
        self.attn = nn.Attention(self.z2_dim)
        self.mask_z = nn.MaskLayer(self.z_dim, z2_dim=self.z2_dim)
        self.mask_u = nn.MaskLayer(self.z1_dim, z2_dim=1)


        

    def forward(self, x, label, mask = None, sample = False, adj = None, alpha=0.3, beta=1, lambdav=0.001):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        
        assert label.size()[1] == self.z1_dim
        device = x.device
        # # mean , var = split(x_hidden,2) <---- (bs,2*z_dim=16)
        q_m, q_v = self.rep_emb.encode(x.to(device))
        q_m, q_v = q_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]),torch.ones(q_m.size()[0], self.z1_dim,self.z2_dim).to(device)

        # # q_V is covered by torch.ones
       
        # decoder_v: ones(bs,4,4)
        decode_m, decode_v = self.dag.calculate_dag(q_m.to(device), torch.ones(q_m.size()[0], self.z1_dim,self.z2_dim).to(device))
        decode_m, decode_v = decode_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]),decode_v
        if sample == False:
          if mask != None and mask < 2:
              z_mask = torch.ones(q_m.size()[0], self.z1_dim,self.z2_dim).to(device)*adj
              decode_m[:, mask, :] = z_mask[:, mask, :]
              #decode_v[:, mask, :] = z_mask[:, mask, :]
          # A_T * x
          m_zm, m_zv = self.dag.mask_z(decode_m.to(device)).reshape([q_m.size()[0], self.z1_dim,self.z2_dim]),decode_v.reshape([q_m.size()[0], self.z1_dim,self.z2_dim])
          m_u = self.dag.mask_u(label.to(device))
          
          f_z = self.mask_z.mix(m_zm).reshape([q_m.size()[0], self.z1_dim,self.z2_dim]).to(device)
          e_tilde = self.attn.attention(decode_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]).to(device),q_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]).to(device))[0]
          if mask != None and mask < 2:
              z_mask = torch.ones(q_m.size()[0],self.z1_dim,self.z2_dim).to(device)*adj
              e_tilde[:, mask, :] = z_mask[:, mask, :]
              
          f_z1 = f_z+e_tilde
          if mask!= None and mask == 2 :
              z_mask = torch.ones(q_m.size()[0],self.z1_dim,self.z2_dim).to(device)*adj
              f_z1[:, mask, :] = z_mask[:, mask, :]
              #m_zv[:, mask, :] = z_mask[:, mask, :]
          if mask!= None and mask == 3 :
              z_mask = torch.ones(q_m.size()[0],self.z1_dim,self.z2_dim).to(device)*adj
              f_z1[:, mask, :] = z_mask[:, mask, :]
              #m_zv[:, mask, :] = z_mask[:, mask, :]
          g_u = self.mask_u.mix(m_u).to(device)
          z_given_dag = ut.conditional_sample_gaussian(f_z1, m_zv*lambdav)
        
        # decoded_bernoulli_logits,x1,x2,x3,x4 = self.dec.decode_sep(z_given_dag.reshape([z_given_dag.size()[0], self.z_dim]), label.to(device))
        
        # rec = ut.log_bernoulli_with_logits(x, decoded_bernoulli_logits.reshape(x.size()))
        # rec = -torch.mean(rec)

        p_m, p_v = torch.zeros(q_m.size()), torch.ones(q_m.size())
        cp_m, cp_v = ut.condition_prior(self.scale, label, self.z2_dim)
        cp_v = torch.ones([q_m.size()[0],self.z1_dim,self.z2_dim]).to(device)
        cp_z = ut.conditional_sample_gaussian(cp_m.to(device), cp_v.to(device))
        kl = torch.zeros(1).to(device)
        ##CausalDiffAE add the following 
        kl = alpha*ut.kl_normal(q_m.view(-1,self.z_dim).to(device), q_v.view(-1,self.z_dim).to(device), p_m.view(-1,self.z_dim).to(device), p_v.view(-1,self.z_dim).to(device))
        
        for i in range(self.z1_dim):
            kl = kl + beta*ut.kl_normal(decode_m[:,i,:].to(device), cp_v[:,i,:].to(device),cp_m[:,i,:].to(device), cp_v[:,i,:].to(device))
        kl = torch.mean(kl)
        mask_kl = torch.zeros(1).to(device)

        for i in range(4):
            mask_kl = mask_kl + 1*ut.kl_normal(f_z1[:,i,:].to(device), cp_v[:,i,:].to(device),cp_m[:,i,:].to(device), cp_v[:,i,:].to(device))    
        
        u_loss = torch.nn.MSELoss()
        mask_l = torch.mean(mask_kl) + u_loss(g_u, label.float().to(device))
        #nelbo = rec + kl + mask_l
        dag_param = self.dag.A
        h_a = self._h_A(dag_param, dag_param.size()[0],device)
        L = kl + mask_l + 3*h_a + 0.5*h_a*h_a 
        #- torch.norm(dag_param) 

        return L, z_given_dag.view(z_given_dag.size(0),-1)

    def loss(self, x):
        nelbo, kl, rec, _, _ = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))

    def _h_A(self,A, m,device):
        expm_A = matrix_poly(A*A, m,device)
        h_A = torch.trace(expm_A) - m
        return h_A


