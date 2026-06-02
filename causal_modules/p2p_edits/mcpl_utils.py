# Copyright AstraZeneca UK Ltd. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
def prompt_contrastive_loss(controller,cond,args):

    # CJ : 1) cond [B,N=77,D=1280] is embedded_text, calculate CL over specified presudo_words: presudo_words_infonce
    # CJ : 1.1) select specified presudo_words (e.g. specify [*,@]) then n=2, thus cond [B,77,1280] -> cond_presudo_words [B,2,1280]
    B, N, D = cond.shape
    # mask out non-presudo words
    mask_infonce = controller.local_blend.alpha_infonce != 0
    cond_presudo_words = cond.masked_select(mask_infonce) 
    cond_presudo_words = cond_presudo_words.view(B,-1,D)
    # optional use adj as additional augmented positive examples to constrain ROI and correlation
    if controller.local_blend.adj_aug_infonce[0] != '':
        mask_adj_aug = controller.local_blend.alpha_adj_aug != 0
        # skip 1st iter in RELATE training
        if mask_adj_aug.sum(1)[0,0] != 0:
            cond_adj_aug = cond.masked_select(mask_adj_aug) 
            cond_adj_aug = cond_adj_aug.view(B,-1,D)
            # B, N, D -> 2*B
            cond_presudo_words = torch.cat([cond_presudo_words,cond_adj_aug],dim=0)
    # CJ : 1.2) calculate InfoNCE loss
    B, n_presudo_words, D = cond_presudo_words.shape
    # reshape to [B*2, 1280] here B equvelent to num of augmentation
    # thus each presudo word has B positive examples, and B * (n_presudo_words-1) negative examples
    feats = cond_presudo_words.view(-1,D)
    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    # Find positive example -> n_presudo_words away from the original example, @1 -> @2
    #consider the embeddings of the same concept as positive samples while the others as negative
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // n_presudo_words, dims=0)
    # InfoNCE loss
    cl_temperature = args.infonce_temperature
    cos_sim = cos_sim / cl_temperature
    # maximize simlarity => minimize distance, second part is for softmax, 
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()            
    nll *= args.infonce_scale
    # CJ : 2) add InfoNCE loss 
    return nll      
