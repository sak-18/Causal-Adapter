import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
import diffusers
import importlib
importlib.reload(diffusers)
from diffusers.utils import load_image
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
from torch.optim.adam import Adam
from typing import Optional, Union, Tuple, Dict
from torchvision import transforms
from transformers import CLIPTextModel
from diffusers.models.modeling_utils import load_state_dict
import torch.nn.functional as nnf
import abc
import torch.nn as nn
from copy import deepcopy
from causal_modules.p2p_edits import ptp_utils
from causal_modules.p2p_edits.scheduler_dev import DDIMSchedulerDev
from causal_modules.p2p_edits.inversion import DirectInversion
from causal_modules.p2p_edits.attention_control import EmptyControl, AttentionStore, make_controller
from causal_modules.p2p_edits.p2p_guidance_forward import direct_inversion_p2p_guidance_forward
device = torch.device("cuda")

MAX_NUM_WORDS = 77
LOW_RESOURCE = False
NUM_DDIM_STEPS=50


def align_batch_size(start_latents, prompt, label):
    """
    Aligns the batch size of start_latents, prompt, and label.
    If any of them has batch size > 1, repeat others with batch size 1 to match the maximum batch size.

    Args:
        start_latents (Tensor): A tensor with shape (B1, ...).
        prompt (str or List[str]): A single prompt string or a list of prompt strings.
        label (Tensor): A tensor with shape (B3, ...).

    Returns:
        Tuple[Tensor, List[str], Tensor]: Aligned start_latents, repeated prompt list, and label.
    """
    
    # Convert prompt to a list if it is a single string
    if isinstance(prompt, str):
        prompt = [prompt]

    batch_sizes = [start_latents.size(0), len(prompt), label.size(0)]
    max_batch_size = max(batch_sizes)

    def repeat_tensor(tensor, max_bs):
        if tensor.size(0) == 1:
            repeat_times = [max_bs] + [1] * (tensor.dim() - 1)
            return tensor.repeat(*repeat_times)
        return tensor

    def repeat_prompt_list(prompt_list, max_bs):
        if len(prompt_list) == 1:
            return prompt_list * max_bs
        return prompt_list

    start_latents = repeat_tensor(start_latents, max_batch_size)
    prompt = repeat_prompt_list(prompt, max_batch_size)
    label = repeat_tensor(label, max_batch_size)

    return start_latents, prompt, label

@torch.no_grad()
def prompt_aligned_injection(pipe,inputs_id,controlnet_cond):
    data_type=pipe.dtype
    text_encoder= pipe.text_encoder
    if 'after' in pipe.controlnet.task_cond:
        # insert embedding after transformer
        def get_concept_ids(text_encoder):
            model = text_encoder.module if hasattr(text_encoder, "module") else text_encoder
            return model.text_model.embeddings.embed_control.control_concept_ids

        concept_ids = get_concept_ids(text_encoder)
        input_ids_clone = inputs_id.clone()
        encoder_hidden_states = text_encoder(inputs_id)[0].to(dtype=data_type)
        if pipe.controlnet.dataset == 'ADNI':
            controlnet_cond_clone = controlnet_cond.clone()
            if len(concept_ids)==3:
                #if controlnet_cond_clone.shape[1] == 16:
                # only use brain_v, ven_v and slice 0-9 following benchmark
                controlnet_cond_clone=controlnet_cond_clone[:,4:,:]
            if controlnet_cond_clone.shape[1]==6:
                for i,token_id in enumerate(concept_ids):
                    placeholder_idx = torch.where(input_ids_clone == token_id)
                    encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond_clone[:,i,:]
            elif controlnet_cond_clone.shape[1] == 12:
                for i,token_id in enumerate(concept_ids):
                    placeholder_idx = torch.where(input_ids_clone == token_id)
                    if i==len(concept_ids)-1:
                        encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond_clone[:,i:].reshape(-1,1)
                    else:
                        encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond_clone[:,i,:]
            elif controlnet_cond_clone.shape[1] > 12:
                for i,token_id in enumerate(concept_ids):
                    placeholder_idx = torch.where(input_ids_clone == token_id)
                    if i==0:
                        encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond_clone[:,:2].reshape(-1,1)
                    elif i==len(concept_ids)-1:
                        encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond_clone[:,-10:].reshape(-1,1)
                    else:
                        encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond_clone[:,i+1,:]
        else:
            for i,token_id in enumerate(concept_ids):
                placeholder_idx = torch.where(input_ids_clone == token_id)
                encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond[:,i,:]
    else:    
        encoder_hidden_states = text_encoder(inputs_id,attribute_cond = controlnet_cond)[0].to(dtype=data_type)
    
    return encoder_hidden_states
    
    
@torch.no_grad()
def prepare_source_target_embedding(pipe,prompt,label,DSCM_labels=None,intervention_indx=None,intervention_values=None,disentangle=False,guidance_scale=1.0,device = torch.device("cuda")):
    # Get source embedding
    input_ids = pipe.tokenizer(prompt,
                        padding="max_length",
                        truncation=True,
                        max_length=pipe.tokenizer.model_max_length,
                        return_tensors="pt",
    ).input_ids.to(device)
    batch_size = input_ids.shape[0]
    task_cond = pipe.controlnet.task_cond
   
    if 'generation' in task_cond and 'text' in task_cond: 
        source_hidden_states = prompt_aligned_injection(pipe,input_ids.clone(),label.unsqueeze(2))
    else:
        source_hidden_states =  pipe.text_encoder(input_ids.clone())[0].to(dtype=pipe.dtype)
    # Get target embedding
    if DSCM_labels is None:
        causal_cond,_ = pipe.controlnet.controlnet_cond_embedding.inference(label,intervention_indx=intervention_indx,intervention_values=intervention_values,disentangle=disentangle)
    else:
        causal_cond = DSCM_labels
    if 'generation' in task_cond and 'text' in task_cond: 
        target_hidden_states = prompt_aligned_injection(pipe,input_ids.clone(),causal_cond)
    else:
        target_hidden_states =  pipe.text_encoder(input_ids.clone())[0].to(dtype=pipe.dtype)
    # Unconditional Embedding
    #uncond_prompt = ""
    uncond_prompt = "ugly, blurry, low res, unrealistic"
    uncond_ids = pipe.tokenizer(
        [uncond_prompt] * batch_size,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(device)
    uncond_embeddings = pipe.text_encoder(uncond_ids)[0]  # [B, 77, D]

    # 5) Make sure dtypes/devices align
    dtype = source_hidden_states.dtype
    uncond_embeddings = uncond_embeddings.to(device=device, dtype=dtype)
    target_hidden_states = target_hidden_states.to(device=device, dtype=dtype)

    # 6) Concatenate along batch dim: [B(uncond), B(source), B(target)] -> [3B, 77, D]
    text_embeddings = torch.cat(
        [uncond_embeddings,uncond_embeddings, source_hidden_states, target_hidden_states],
        dim=0
    )
    return text_embeddings,causal_cond

# Sample function (regular DDIM)
@torch.no_grad()
def sample(
    pipe,
    prompt,
    start_step=0,
    start_latents=None,
    guidance_scale=3.5,
    num_inference_steps=30,
    num_images_per_prompt=1,
    negative_prompt="",
    device=device,
    controlnet_image=None,
    intervention_indx=None,
    intervention_values=None,
    label=None,
    return_PIL = True,
    disentangle=False,
    uncond_embeddings = None,
    controller = None
    
):
    if guidance_scale>1:
        do_classifier_free_guidance = True
    else:
        do_classifier_free_guidance= False    
    if start_latents is None:
        start_latents = torch.randn(1, 4, 32, 32, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma
    if controller is not None:
        ptp_utils.register_attention_control_controlnet(pipe, controller)
    start_latents, prompt, label = align_batch_size(start_latents, prompt, label)
    
    _,negtive_prompt_embedding = pipe.encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )
    input_ids = pipe.tokenizer(prompt,
                        padding="max_length",
                        truncation=True,
                        max_length=pipe.tokenizer.model_max_length,
                        return_tensors="pt",
                    ).input_ids.to(device)
    if input_ids.dim() ==1:
        input_ids=input_ids.unsqueeze(0)


    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    
    
    start_latents, label = start_latents.to(device),label.to(device)

    latents = start_latents.clone()
    task_cond = pipe.controlnet.task_cond
    causal_cond,causal_loss = pipe.controlnet.controlnet_cond_embedding.inference(label,intervention_indx=intervention_indx,intervention_values=intervention_values,disentangle=disentangle)
    
    
    if 'generation' in task_cond and 'text' in task_cond: 
        cond_embeddings = prompt_aligned_injection(pipe,input_ids.clone(),causal_cond)
    else:
        cond_embeddings = pipe.text_encoder(input_ids)[0].to(dtype=pipe.dtype)
    
    for i in tqdm(range(start_step, num_inference_steps)):

        t = pipe.scheduler.timesteps[i]
        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        if do_classifier_free_guidance and uncond_embeddings is not None:
            negtive_prompt_embedding = uncond_embeddings[i].expand(*negtive_prompt_embedding.shape)
        if do_classifier_free_guidance:
            if negtive_prompt_embedding is not None:
                encoder_hidden_states = torch.cat([negtive_prompt_embedding, cond_embeddings],dim=0)
            else:
                assert negtive_prompt_embedding is not None, 'negative_prompt_embedding should be provided when do_classifier_free_guidance is True'
        else:
            encoder_hidden_states = cond_embeddings
        (down_block_res_samples, mid_block_res_sample),_,_,_ = pipe.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=None,
                    return_dict=False,
                    label = causal_cond.clone(),
                    training=False,
                    sampling=True,
                    intervention_indx=intervention_indx,
                    intervention_values=intervention_values

        )


        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False)[0]

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        prev_timestep = t.item() - pipe.scheduler.config.num_train_timesteps // num_inference_steps
        alpha_prod_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_prod_t_prev = pipe.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else pipe.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * noise_pred
        latents = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        # if controller is not None:
        #     latents = controller.step_callback(latents)

        
    # Post-processing
    images = pipe.vae.decode(latents/ pipe.vae.config.scaling_factor,return_dict=False)[0]
    if return_PIL:
        do_denormalize = [True] * images.shape[0]
        output = pipe.image_processor.postprocess(images,do_denormalize=do_denormalize)
    else:
        output = images
    return output,causal_cond



@torch.no_grad()
def sample_schedulers(
    pipe,
    prompt,
    start_step=0,
    start_latents=None,
    guidance_scale=3.5,
    num_inference_steps=30,
    num_images_per_prompt=1,
    negative_prompt="",
    device=device,
    controlnet_image=None,
    intervention_indx=None,
    intervention_values=None,
    label=None,
    return_PIL = True,
    disentangle=False,
    uncond_embeddings = None,
    controller = None
    
):
    if guidance_scale>1:
        do_classifier_free_guidance = True
    else:
        do_classifier_free_guidance= False    
    if start_latents is None:
        start_latents = torch.randn(1, 4, 32, 32, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma
    if controller is not None:
        ptp_utils.register_attention_control_controlnet(pipe, controller)
    start_latents, prompt, label = align_batch_size(start_latents, prompt, label)
    
    _,negtive_prompt_embedding = pipe.encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )
    input_ids = pipe.tokenizer(prompt,
                        padding="max_length",
                        truncation=True,
                        max_length=pipe.tokenizer.model_max_length,
                        return_tensors="pt",
                    ).input_ids.to(device)
    if input_ids.dim() ==1:
        input_ids=input_ids.unsqueeze(0)


    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    
    
    start_latents, label = start_latents.to(device),label.to(device)

    latents = start_latents.clone()
    task_cond = pipe.controlnet.task_cond
    causal_cond,causal_loss = pipe.controlnet.controlnet_cond_embedding.inference(label,intervention_indx=intervention_indx,intervention_values=intervention_values,disentangle=disentangle)
    
    
    if 'generation' in task_cond and 'text' in task_cond: 
        cond_embeddings = prompt_aligned_injection(pipe,input_ids.clone(),causal_cond)
    else:
        cond_embeddings = pipe.text_encoder(input_ids)[0].to(dtype=pipe.dtype)
    
    for i in tqdm(range(start_step, num_inference_steps)):

        t = pipe.scheduler.timesteps[i]
        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        if do_classifier_free_guidance and uncond_embeddings is not None:
            negtive_prompt_embedding = uncond_embeddings[i].expand(*negtive_prompt_embedding.shape)
        if do_classifier_free_guidance:
            if negtive_prompt_embedding is not None:
                encoder_hidden_states = torch.cat([negtive_prompt_embedding, cond_embeddings],dim=0)
            else:
                assert negtive_prompt_embedding is not None, 'negative_prompt_embedding should be provided when do_classifier_free_guidance is True'
        else:
            encoder_hidden_states = cond_embeddings
        (down_block_res_samples, mid_block_res_sample),_,_,_ = pipe.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=None,
                    return_dict=False,
                    label = causal_cond.clone(),
                    training=False,
                    sampling=True,
                    intervention_indx=intervention_indx,
                    intervention_values=intervention_values

        )

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False)[0]

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        # if controller is not None:
        #     latents = controller.step_callback(latents)

        
    # Post-processing
    images = pipe.vae.decode(latents/ pipe.vae.config.scaling_factor,return_dict=False)[0]
    if return_PIL:
        do_denormalize = [True] * images.shape[0]
        output = pipe.image_processor.postprocess(images,do_denormalize=do_denormalize)
    else:
        output = images
    return output,causal_cond

@torch.no_grad()
def find_update_token_index(label, causal_cond, prompt, presudo_list,dataset):
    """
    Args:
        label:        torch.Tensor [B, num_attrs]
        causal_cond:  torch.Tensor [B, num_attrs]
        prompt:       str, e.g. "a human of @ and * and & and !"
        presudo_list: list[str], e.g. ["@", "*", "&", "!"]

    Returns:
        List[List[int]]: for each batch item, a list of token indices (in the
                         space-split prompt) whose attributes changed.
    """
    assert label.shape == causal_cond.shape, "label and causal_cond must have same shape"
    eps = 1e-6  # tolerance for float comparisons

    if dataset in ['ADNI']:
        # use the last 12 attributes
        label_sub, causal_cond_sub = label[:, 4:], causal_cond[:, 4:]  # [B,12]
        B, num_attrs = label_sub.shape
        assert num_attrs == 12, f"Expected 12 attrs for ADNI subvector, got {num_attrs}"

        # tokenize prompt
        tokens = prompt.split(" ")

        # find positions of @, *, and all &
        at_pos  = tokens.index("@")  if "@" in tokens else -1
        star_pos = tokens.index("*") if "*" in tokens else -1
        amp_positions = [i for i, tok in enumerate(tokens) if tok == "&"]

        # diff mask with tolerance (safe for float)
        diff_mask = (label_sub - causal_cond_sub).abs() > eps  # [B,12]

        out_positions = []
        for b in range(B):
            pos_list = []

            # index 0 -> '@'
            if diff_mask[b, 0].item() and at_pos != -1:
                pos_list.append(at_pos + 1)  # +1 offset

            # index 1 -> '*'
            if diff_mask[b, 1].item() and star_pos != -1:
                pos_list.append(star_pos + 1)

            # indices 2..11 -> all '&'
            if diff_mask[b, 2:].any().item() and amp_positions:
                pos_list.extend([p + 1 for p in amp_positions])

            out_positions.append(pos_list)
        return out_positions
    else:
        B, num_attrs = label.shape

        # 1) Tokenize the prompt by spaces (no tokenizer; literal positions)
        tokens = prompt.split(" ")

        # 2) Build a map: attribute index -> token position in the prompt
        #    (We assume each pseudo placeholder appears exactly once.)
        pseudo_to_pos = {tok: i for i, tok in enumerate(tokens)}
        attr_to_pos = []
        for i in range(num_attrs):
            p = presudo_list[i]
            pos = pseudo_to_pos.get(p, -1)   # -1 if not found
            attr_to_pos.append(pos)

        # 3) Per-sample: find which attrs changed, map to token positions
        #diff_mask = (label != causal_cond)   # [B, num_attrs]
        diff_mask = (label - causal_cond).abs() > eps
        out_positions = []
        for b in range(B):
            idxs = torch.nonzero(diff_mask[b], as_tuple=True)[0].tolist()  # changed attr indices
            positions = [attr_to_pos[i]+1 for i in idxs if attr_to_pos[i] != -1]
            out_positions.append(positions)
        return out_positions

## Inversion
@torch.no_grad()
def invert(
    pipe,
    start_latents,
    prompt,
    guidance_scale=1,
    num_inference_steps=80,
    num_images_per_prompt=1,
    negative_prompt="",
    device=device,
    controlnet_image=None,
    intervention_indx=None,
    intervention_values=None,
    label=None,
):    
    if guidance_scale>1:
        do_classifier_free_guidance = True
    else:
        do_classifier_free_guidance= False    

    # # Encode prompt
    start_latents, prompt, label = align_batch_size(start_latents, prompt, label)
    # extra negative prompt embed
    _,negtive_prompt_embedding = pipe.encode_prompt(
        prompt, device, num_images_per_prompt, True, negative_prompt
    )
    input_ids = pipe.tokenizer(prompt,
                        padding="max_length",
                        truncation=True,
                        max_length=pipe.tokenizer.model_max_length,
                        return_tensors="pt",
                    ).input_ids.to(device)
    if input_ids.dim() ==1:
        input_ids=input_ids.unsqueeze(0)
    
    
    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 32, 32, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma
    # Latents are now the specified start latents
    start_latents, label = start_latents.to(device),label.to(device)
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = [latents]

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)
    task_cond = pipe.controlnet.task_cond
    
    causal_cond,causal_loss = pipe.controlnet.controlnet_cond_embedding.inference(label,intervention_indx=intervention_indx,intervention_values=intervention_values)
    if 'generation' in task_cond and 'text' in task_cond: 
            encoder_hidden_states = prompt_aligned_injection(pipe,input_ids.clone(),causal_cond)
    else:
        encoder_hidden_states = pipe.text_encoder(input_ids)[0].to(dtype=pipe.dtype)
    control_embeddings = encoder_hidden_states.clone()
    if do_classifier_free_guidance:
        if negtive_prompt_embedding is not None:
            encoder_hidden_states = torch.cat([negtive_prompt_embedding, encoder_hidden_states],dim=0)
        else:
            assert negtive_prompt_embedding is not None, 'negative_prompt_embedding should be provided when do_classifier_free_guidance is True'
    #for i, t in enumerate(timesteps):
    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

                
        (down_block_res_samples, mid_block_res_sample),_,_,_ = pipe.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                    label = label.unsqueeze(2),
                    training=False,
                    sampling=True,
                    intervention_indx=intervention_indx,
                    intervention_values=intervention_values

        )
        
        
        
        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False)[0]
        #noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        

        # Perform guidance
        if guidance_scale>1:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        

        timestep, next_timestep = min(t.item() - pipe.scheduler.config.num_train_timesteps // num_inference_steps, 999), t.item()
        alpha_prod_t = pipe.scheduler.alphas_cumprod[timestep] if timestep >= 0 else pipe.scheduler.final_alpha_cumprod
        alpha_prod_t_next = pipe.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * noise_pred
        latents = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction

        # Store
        intermediate_latents.append(latents)
    output_context= None
    if do_classifier_free_guidance==False:
        output_context =  torch.cat([negtive_prompt_embedding, encoder_hidden_states.clone()],dim=0)
    else:
        output_context = encoder_hidden_states.clone()
    return torch.stack(intermediate_latents, dim=0),output_context,input_ids.clone()



def save_images_grid(images_list, grid_size, save_path=None):
    """
    Save a list of lists of images in a grid format.

    Parameters:
    - images_list: List of lists containing numpy images (4x4).
    - grid_size: tuple (grid_rows, grid_cols) for arranging images in the grid.
    - save_path: file path where the grid image will be saved.
    """
    # Flatten the list of lists into a single list of images

    images = [np.asarray(img) for sublist in images_list for img in sublist]

    # Determine the dimensions of the first image to set up the grid
    H, W, C = images[0].shape
    grid_rows, grid_cols = grid_size

    # Check if the grid can fit all images
    assert grid_rows * grid_cols >= len(images), "Grid size is too small for the number of images"

    # Create an empty array for the grid image
    grid_image = np.zeros((grid_rows * H, grid_cols * W, C), dtype=images[0].dtype)

    # Fill the grid with images
    for idx, img in enumerate(images):
        row = idx // grid_cols
        col = idx % grid_cols
        grid_image[row * H:(row + 1) * H, col * W:(col + 1) * W, :] = img

    # Plot and save the grid image
    plt.figure(figsize=(grid_cols*2, grid_rows*2))
    plt.imshow(grid_image)
    plt.axis('off')  # Turn off axis labels




def load_mcpl_embeddings(base_model_path,tokenizer,embedding_path=None,presudo_token_ids=None,embed_control=True):
    text_encoder = CLIPTextModel.from_pretrained(
        base_model_path, subfolder="text_encoder"
    )
    if embedding_path is not None:
        state_dict = load_state_dict(embedding_path)
        embeddings = []
        tokens = []
        for key,embed in state_dict.items():
            tokens.append(key)
            embeddings.append(embed)
        token_ids = tokenizer.encode(tokens, add_special_tokens=False)
        # 7.4 Load token and embedding
        for token_id, embedding in zip(token_ids, embeddings):
            # add tokens and get ids
            # tokenizer.add_tokens(token)
            # token_id = tokenizer.convert_tokens_to_ids(token)
            text_encoder.get_input_embeddings().weight.data[token_id] = embedding
            print(f"Loaded textual inversion embedding for {token_id}.")



    text_encoder.eval()
    return text_encoder



from PIL import Image
import torch
import torchvision.transforms as transforms

def resize_tensor(input_tensors, pipe, dataset, return_tensor=True, mode='bicubic'):
    # Determine target size based on dataset name
    device = input_tensors[0].device
    if 'celeA' in dataset or 'celebahq' in dataset:
        size = (64, 64)
    elif 'MorphoMNIST' in dataset:
        size = (32, 32)
    elif 'ADNI' in dataset:
        size = (192, 192)
    elif 'pendulum' in dataset:
        size= (96,96)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Handle list input
    if isinstance(input_tensors, list):
        input_tensors = torch.stack(input_tensors)

    # Check input shape: 4D or 5D
    is_5d = (input_tensors.dim() == 5)
    if is_5d:
        n_cycles, batch_size = input_tensors.shape[:2]
        input_tensors = input_tensors.view(-1, *input_tensors.shape[2:])  # → (n_cycles * batch_size, C, H, W)

    # Prepare denormalization flags
    do_denormalize = [True] * input_tensors.shape[0]

    # Convert to list of PIL Images using pipeline's processor
    pil_images = pipe.image_processor.postprocess(input_tensors, do_denormalize=do_denormalize)

    # Choose interpolation mode
    interpolation = {
        'bilinear': Image.BILINEAR,
        'nearest': Image.NEAREST,
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS
    }.get(mode.lower(), Image.BICUBIC)

    # Grayscale flag
    is_grayscale = dataset in ['ADNI', 'MorphoMNIST']

    # Resize + convert to grayscale if needed
    resized_pil_images = [
        img.resize(size, interpolation).convert('L') if is_grayscale else img.resize(size, interpolation)
        for img in pil_images
    ]

    if return_tensor:
        to_tensor = transforms.ToTensor()
        tensor_images = [to_tensor(img) for img in resized_pil_images]  # (C, H, W)

        tensor_batch = torch.stack(tensor_images)  # (N, C, H, W)
        if is_grayscale and tensor_batch.shape[1] != 1:
            tensor_batch = tensor_batch[:, 0:1, :, :]  # force (N, 1, H, W)

        if is_5d:
            tensor_batch = tensor_batch.view(n_cycles, batch_size, *tensor_batch.shape[1:])  # (N, B, C, H, W)

        return tensor_batch.to(device)

    else:
        if is_5d:
            # Reshape back to nested list [n_cycles][batch_size]
            reshaped_pil = [
                resized_pil_images[i * batch_size : (i + 1) * batch_size]
                for i in range(n_cycles)
            ]
            return reshaped_pil
        else:
            return resized_pil_images



def get_dataset_attrs(dataset):
    if 'celeA' in dataset:
        if 'simple' in dataset:
            attr_keys = ['Smiling','Eyeglasses']
        
        elif 'complex' in dataset:
            attr_keys = ['Young','Male','No_Beard','Bald']
    if 'celebahq' in dataset:
        if 'simple' in dataset:
            attr_keys = ['Smiling','Eyeglasses','Mouth_Slightly_Open','Male','Bald','Wearing_Lipstick','Wearing_Hat']
        
        elif 'complex' in dataset:
            pass

    elif 'MorphoMNIST' in dataset:
        pass
    elif 'ADNI' in dataset:
        attr_keys = ['apoE', 'age', 'sex', 'brain_vol', 'vent_vol', 'slice']
    elif 'pendulum' in dataset:
        attr_keys = ["pendulum","light","shadow_length", "shadow_position"]
    return attr_keys


def ddim_editing(pipe, input_image,label,prompt,num_steps = 50,invert_guidance_scale=1.0,set_guidance_scale  = 1.0,intervention_indx=None,intervention_values=None,return_PIL = True,disentangle=False,DSCM_labels=None,null_optimization=False,controller=None,pnp_inversion=False):
    # global update the num_DDIM_steps
    global NUM_DDIM_STEPS
    NUM_DDIM_STEPS = num_steps
    generator = torch.manual_seed(0)
    with torch.no_grad():
        latent = pipe.vae.encode(input_image.to(device))
        img_latent =  0.18215 * latent.latent_dist.mean
        #img_latent = 0.18215 * latent.latent_dist.sample()
    # Keep inversion guidance scale to 1 will protect the identity
    inverted_latents,output_context,input_ids = invert(pipe,
        img_latent,
        prompt,
        guidance_scale=invert_guidance_scale,
        num_inference_steps=num_steps,
        num_images_per_prompt=1,
        negative_prompt=None,
        device=device,
        controlnet_image=None,
        intervention_indx=None,
        intervention_values=None,
        label=label.clone())
    num_inner_steps = 10
    early_stop_epsilon = 1e-5
    if null_optimization:
        null_inversion = NullInversion(pipe,num_steps,output_context,input_ids,label.clone(),GUIDANCE_SCALE=set_guidance_scale)
        uncond_embeddings = null_inversion.invert(ddim_latents=inverted_latents,num_inner_steps=num_inner_steps, early_stop_epsilon=early_stop_epsilon)
    else:
        uncond_embeddings = None
    s_step = 0

    if DSCM_labels is not None:
        # use the predicted labels from deepSCM for editing
        label = DSCM_labels
        intervention_indx=None
        intervention_values=None

    if pnp_inversion is True:
        'Direct inversion+PNP'
        pnp = PNP(deepcopy(pipe),num_steps,device=device)
        final_im,causal_cond = pnp.run_pnp(inverted_latents,
                    prompt,
                    start_step=s_step,
                    guidance_scale=set_guidance_scale,
                    num_inference_steps=num_steps-1,
                    num_images_per_prompt=1,
                    negative_prompt="",
                    intervention_indx=intervention_indx,
                    intervention_values=intervention_values,
                    label=label.clone(),
                    return_PIL = return_PIL,
                    disentangle= disentangle,
                    pnp_f_t = 0.8,
                    pnp_attn_t = 0.5,
                    )
        #release the pipe
        del pnp.pipe
    else:
        # normal DDIM
        final_im,causal_cond = sample(
                pipe,
                prompt,
                start_step=s_step,
                start_latents=inverted_latents[-(s_step + 1)].clone(),
                guidance_scale=set_guidance_scale,
                num_inference_steps=num_steps,
                num_images_per_prompt=1,
                negative_prompt=None,
                device=device,
                controlnet_image=None,
                intervention_indx=intervention_indx,
                intervention_values=intervention_values,
                label=label.clone(),
                return_PIL = return_PIL,
                disentangle= disentangle,
                uncond_embeddings=uncond_embeddings,
                controller=controller
            )
    return final_im,inverted_latents,causal_cond,uncond_embeddings

# Direct inversion + P2P
@torch.inference_mode()
def P2P_editing(pipe, input_image,label,prompt,presudo_list,num_steps = 50,invert_guidance_scale=1.0,set_guidance_scale  = 1.0,intervention_indx=None,intervention_values=None,return_PIL = True,disentangle=False,
                DSCM_labels=None,
                cross_replace_steps=0.4,
                self_replace_steps=0.6,
                blend_word=True,
                blend_params = {'start_blend':0.2,'th':(0.3,0.3)},
                eq_params=None,
                is_replace_controller=True,
                return_recons_counter= False):
    # global update the num_DDIM_steps
    scheduler = DDIMSchedulerDev(beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule="scaled_linear",
                                    clip_sample=False,
                                    set_alpha_to_one=False)
    pipe.scheduler = scheduler
    pipe.scheduler.set_timesteps(num_steps)
    generator = torch.manual_seed(0)
    label =label.to(device)

    with torch.no_grad():
        latent = pipe.vae.encode(input_image.to(device))
        img_latent =  0.18215 * latent.latent_dist.mean
        #img_latent = 0.18215 * latent.latent_dist.sample()
    # Keep inversion guidance scale to 1 will protect the identity

    start_latents, prompt, label = align_batch_size(img_latent, prompt, label)
    
    text_embeddings,causal_cond = prepare_source_target_embedding(pipe,prompt,label.clone(),DSCM_labels,intervention_indx=intervention_indx,intervention_values=intervention_values,disentangle=disentangle,guidance_scale=set_guidance_scale,device = device)
    
    need_to_update_token_index = find_update_token_index(label,causal_cond.squeeze(2),prompt[0],presudo_list,pipe.controlnet.dataset)
    
    substruct_token_index= None
    if pipe.controlnet.dataset in ['celeA_complex']:
        # presudo position [4,6,8,10] 
        # subtruct gender mask
        if intervention_indx in [0]:
            pass
            #substruct_token_index = []
            # for indexes in need_to_update_token_index:
            #     # allow indexes to be int or an iterable of ints
            #     if isinstance(indexes, (list, tuple, set)):
            #         exclude = set(indexes)
            #     else:
            #         exclude = {indexes}
            #     substruct_token_index.append([p for p in [8,10] if p not in exclude])
        elif intervention_indx in [1]:
            pass
            #substruct_token_index = []
            # for indexes in need_to_update_token_index:
            #     # allow indexes to be int or an iterable of ints
            #     if isinstance(indexes, (list, tuple, set)):
            #         exclude = set(indexes)
            #     else:
            #         exclude = {indexes}
            #     substruct_token_index.append([p for p in [8,10] if p not in exclude])
        elif intervention_indx in [2]:
            substruct_token_index = [[10]]*len(need_to_update_token_index)
            # for indexes in need_to_update_token_index:
            #     # allow indexes to be int or an iterable of ints
            #     if isinstance(indexes, (list, tuple, set)):
            #         exclude = set(indexes)
            #     else:
            #         exclude = {indexes}
            #     substruct_token_index.append([p for p in [10] if p not in exclude])
        elif intervention_indx in [3]:
            substruct_token_index = [[6,8]]*len(need_to_update_token_index)
            # for indexes in need_to_update_token_index:
            #     # allow indexes to be int or an iterable of ints
            #     if isinstance(indexes, (list, tuple, set)):
            #         exclude = set(indexes)
            #     else:
            #         exclude = {indexes}
            #     substruct_token_index.append([p for p in [6,8] if p not in exclude])
        


    null_inversion = DirectInversion(pipe=pipe,
                                    num_ddim_steps=num_steps,context = text_embeddings,
                                    label =label.unsqueeze(2),device=device)
    _, _, x_stars, noise_loss_list = null_inversion.invert(
        img_latent=img_latent,guidance_scale=set_guidance_scale)
    x_t = x_stars[-1]

    ########## edit ##########
    cross_replace_steps = {
        'default_': cross_replace_steps,
    }

    controller = make_controller(pipeline=pipe,
                                update_token_index=need_to_update_token_index,
                                is_replace_controller=is_replace_controller,
                                cross_replace_steps=cross_replace_steps,
                                self_replace_steps=self_replace_steps,
                                blend_words=blend_word,
                                blend_params = blend_params,
                                equilizer_params=eq_params,
                                num_ddim_steps=num_steps,
                                substruct_token_index=substruct_token_index,
                                device=device)
    
    latents, _ = direct_inversion_p2p_guidance_forward(pipe=pipe, 
                                       text_embeddings=text_embeddings, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=num_steps, 
                                       guidance_scale=set_guidance_scale,
                                       causal_cond = causal_cond,
                                       generator=None)
    # source_batch_size = len(prompt)
    # images = pipe.vae.decode(latents[source_batch_size:]/ pipe.vae.config.scaling_factor,return_dict=False)[0]
    if return_recons_counter:
        source_batch_size=0
    else:
        source_batch_size = len(prompt)
    images = pipe.vae.decode(latents[source_batch_size:]/ pipe.vae.config.scaling_factor,return_dict=False)[0]
    if return_PIL:
        do_denormalize = [True] * images.shape[0]
        output = pipe.image_processor.postprocess(images.detach(),do_denormalize=do_denormalize)
    else:
        output = images
    final_im = output
    uncond_embeddings=None
    
    controller.remove()

    return final_im,x_t,causal_cond,uncond_embeddings




class NullInversion:
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t,context):
        
        (down_block_res_samples, mid_block_res_sample),causal_loss,_,_ = self.model.controlnet(
                    latents,
                    t,
                    encoder_hidden_states=context,
                    controlnet_cond=None,
                    return_dict=False,
                    training=False,
                    
        )
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False)[0]
        
        return noise_pred

    def get_noise_pred(self, latents, t,is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else self.GUIDANCE_SCALE
        if guidance_scale > 1:
            do_classifier_free_guidance = True
        else:
            do_classifier_free_guidance = False
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        #latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)
        (down_block_res_samples, mid_block_res_sample),causal_loss,_,_ = self.model.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=context,
                    controlnet_cond=None,
                    return_dict=False,
                    training=False,
                    
        )
        noise_pred = self.model.unet(latent_model_input, t, encoder_hidden_states=context,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False)[0]
        if guidance_scale>1:            
            noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * self.NUM_DDIM_STEPS)
        for i in range(self.NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            #optimizer = Adam([uncond_embeddings], lr=1e-2 )
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t,cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t,uncond_embeddings)
                noise_pred = noise_pred_uncond + self.GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad(): 
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        bar.close()
        return uncond_embeddings_list
    
    def invert(self, ddim_latents: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=True):
        
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return uncond_embeddings
        
    
    def __init__(self, model,NUM_DDIM_STEPS,context,inputs_ids,label,GUIDANCE_SCALE):
        # scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
        #                           set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.context = context
        self.NUM_DDIM_STEPS = NUM_DDIM_STEPS
        self.input_ids = inputs_ids
        self.label = label
        self.GUIDANCE_SCALE = GUIDANCE_SCALE


class LocalBlend:
    
    def get_mask(self, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(self.xt_shape))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        return mask
    
    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:
           
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            #x_t = mask*x_t
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts,words,tokenizer,xt_shape,substruct_words=None, start_blend=0.2, th=(.3, .3)):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        
        # alpha_layers = torch.zeros(batch_size,  1, 1, 1, 1, MAX_NUM_WORDS)
        # for i, (prompt, words_) in enumerate(zip(prompts, words)):
        #     if type(words_) is str:
        #         words_ = [words_]
        #     for word in words_:
        #         ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
        #         alpha_layers[i, :, :, :, :, ind] = 1
        
        # if substruct_words is not None:
        #     substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        #     for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
        #         if type(words_) is str:
        #             words_ = [words_]
        #         for word in words_:
        #             ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
        #             substruct_layers[i, :, :, :, :, ind] = 1
        #     self.substruct_layers = substruct_layers.to(device)
        # else:
        self.xt_shape = xt_shape
        self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)
        self.counter = 0 
        self.th=th


        
        
class EmptyControl:
    
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

    
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class SpatialReplace(EmptyControl):
    
    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)
        

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


class AttentionControlEdit_blend(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    
    def __init__(self, local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit_blend, self).__init__()
        
        self.local_blend = local_blend










class PNP(nn.Module):
    def __init__(self, pipe,n_timesteps=NUM_DDIM_STEPS,device="cuda"):
        super().__init__()
        self.device = device

        # Create SD models
        #print('Loading SD model')

        self.pipe = pipe
        self.scheduler = self.pipe.scheduler
        self.scheduler.set_timesteps(n_timesteps, device=self.device)
        self.n_timesteps=NUM_DDIM_STEPS
        #print('SD model loaded')

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        # Tokenize text and get embeddings
        text_input = self.pipe.tokenizer(prompt, padding='max_length', max_length=self.pipe.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.pipe.tokenizer(negative_prompt, padding='max_length', max_length=self.pipe.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def decode_latent(self, latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img

    @torch.no_grad()
    def denoise_step(self, x, t,guidance_scale,noisy_latent):
        # register the time step and features in pnp injection modules
        latent_model_input = torch.cat(([noisy_latent]+[x] * 2))

        register_time(self, t.item())

        # compute text embeddings
        text_embed_input = torch.cat([self.pnp_guidance_embeds, self.text_embeds], dim=0)

        # apply the denoising network
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

        # perform guidance
        _,noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        return denoised_latent

    def init_pnp(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []

        # self.qk_injection_timesteps = self.scheduler.timesteps[-qk_injection_t:] if qk_injection_t >= 0 else []
        # self.conv_injection_timesteps = self.scheduler.timesteps[-conv_injection_t:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_control_efficient(self.pipe, self.qk_injection_timesteps)
        register_conv_control_efficient(self.pipe, self.conv_injection_timesteps)
    
    @torch.no_grad()
    def run_pnp(self,noisy_latent,prompt,
                start_step=0,
                guidance_scale=3.5,
                num_inference_steps=30,
                num_images_per_prompt=1,
                negative_prompt="ugly, blurry, black, low res, unrealistic",
                intervention_indx=None,
                intervention_values=None,
                label=None,
                return_PIL = True,
                disentangle=False,pnp_f_t=0.8,pnp_attn_t=0.5):

        if guidance_scale>1:
            do_classifier_free_guidance = True
        else:
            do_classifier_free_guidance= False
        start_latents = noisy_latent[-1]
        start_latents, prompt, label = align_batch_size(start_latents, prompt, label)
        start_latents, label = start_latents.to(device),label.to(device)
        # X_t[-1]
        #combine the pnp_guidance and negative prompt
        pnp_guidance_embeds = self.get_text_embeds(negative_prompt,"",batch_size=start_latents.size(0))
        # self.text_embeds = self.get_text_embeds(target_prompt, negative_prompt)
        # self.pnp_guidance_embeds = self.get_text_embeds("", "").chunk(2)[0]
        input_ids = self.pipe.tokenizer(prompt,
                        padding="max_length",
                        truncation=True,
                        max_length=self.pipe.tokenizer.model_max_length,
                        return_tensors="pt",
                    ).input_ids.to(self.device)
        if input_ids.dim() ==1:
            input_ids=input_ids.unsqueeze(0)

        pnp_f_t = int(self.n_timesteps * pnp_f_t)
        pnp_attn_t = int(self.n_timesteps * pnp_attn_t)
        self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        
        task_cond = self.pipe.controlnet.task_cond
        latents = start_latents.clone()
        causal_cond,causal_loss = self.pipe.controlnet.controlnet_cond_embedding.inference(label,intervention_indx=intervention_indx,intervention_values=intervention_values,disentangle=disentangle)
        if 'generation' in task_cond and 'text' in task_cond: 
            encoder_hidden_states = prompt_aligned_injection(self.pipe,input_ids.clone(),causal_cond)
        else:
            encoder_hidden_states = self.pipe.text_encoder(input_ids)[0].to(dtype=self.pipe.dtype)

        if do_classifier_free_guidance:
            if pnp_guidance_embeds is not None:
                encoder_hidden_states = torch.cat([pnp_guidance_embeds, encoder_hidden_states],dim=0)
            else:
                assert pnp_guidance_embeds is not None, 'pnp_guidance_embeds should be provided when do_classifier_free_guidance is True'
        
        for i in tqdm(range(start_step, num_inference_steps)):
            pnp_noisy_latents= noisy_latent[-1-i]
            t = self.scheduler.timesteps[i]
            register_time(self.pipe, t.item())
            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat(([pnp_noisy_latents]+[latents] * 2)) if do_classifier_free_guidance else latents
            #latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            (down_block_res_samples, mid_block_res_sample),_,_,_ = self.pipe.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=None,
                    return_dict=False,
                    label = label,
                    training=False,
                    sampling=True,

             )


            # Predict the noise residual
            noise_pred = self.pipe.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False)[0]

            # Perform guidance
            if do_classifier_free_guidance:
                _, noise_pred_uncond, noise_pred_text = noise_pred.chunk(3)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            prev_timestep = t.item() - self.scheduler.config.num_train_timesteps // num_inference_steps
            alpha_prod_t = self.scheduler.alphas_cumprod[t.item()]
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
            beta_prod_t = 1 - alpha_prod_t
            pred_original_sample = (latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
            pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * noise_pred
            latents = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction


        images = self.pipe.vae.decode(latents/ self.pipe.vae.config.scaling_factor,return_dict=False)[0]
        if return_PIL:
            do_denormalize = [True] * images.shape[0]
            output = self.pipe.image_processor.postprocess(images,do_denormalize=do_denormalize)
        else:
            output = images
        return output,causal_cond
        



def register_attention_control_efficient(model, injection_schedule):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads

            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            if not is_cross and self.injection_schedule is not None and (
                    self.t in self.injection_schedule or self.t == 1000):
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)

                source_batch_size = int(q.shape[0] // 3)
                # inject unconditional
                q[source_batch_size:2 * source_batch_size] = q[:source_batch_size]
                k[source_batch_size:2 * source_batch_size] = k[:source_batch_size]
                # inject conditional
                q[2 * source_batch_size:] = q[:source_batch_size]
                k[2 * source_batch_size:] = k[:source_batch_size]

                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)
            else:
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)
                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)

            v = self.to_v(encoder_hidden_states)
            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)


def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
            # for controlnet?
            module2 = model.controlnet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module2, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)
    module2 = model.controlnet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module2, 't', t)



def register_conv_control_efficient(model, injection_schedule):
    def conv_forward(self):
        def forward(input_tensor, temb):
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // 3)
                # inject unconditional
                hidden_states[source_batch_size:2 * source_batch_size] = hidden_states[:source_batch_size]
                # inject conditional
                hidden_states[2 * source_batch_size:] = hidden_states[:source_batch_size]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    conv_module = model.unet.up_blocks[1].resnets[1]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)





