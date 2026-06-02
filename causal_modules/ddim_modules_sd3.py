# Copyright AstraZeneca UK Ltd. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

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
import os
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

from diffusers.models.modeling_utils import load_state_dict
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast,CLIPTextModelWithProjection,T5EncoderModel
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps


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


def save_images_grid(images_list, grid_size, save_path=None,show_subtitle=True):
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



def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

def load_tokenizers_text_encoders(base_model_path,load_dtype):
    tokenizer_one = CLIPTokenizer.from_pretrained(
            base_model_path,
            subfolder="tokenizer",
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        base_model_path,
        subfolder="tokenizer_2",
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        base_model_path,
        subfolder="tokenizer_3",
    )

    text_encoder_one = CLIPTextModelWithProjection.from_pretrained(
            base_model_path, subfolder="text_encoder" ,torch_dtype=load_dtype
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        base_model_path, subfolder="text_encoder_2",torch_dtype=load_dtype
    )
    text_encoder_three = T5EncoderModel.from_pretrained(
        base_model_path, subfolder="text_encoder_3",torch_dtype=load_dtype
    )

    text_encoder_three.resize_token_embeddings(len(tokenizer_three))
    return [tokenizer_one,tokenizer_two,tokenizer_three],[text_encoder_one,text_encoder_two,text_encoder_three]

def load_mcpl_embeddings(                # kept for signature compatibility (unused here)
    embedding_path_list,             # [path_clip1, path_clip2, path_t5]
    tokenizers,                      # [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders,                   # [text_encoder_one, text_encoder_two, text_encoder_three]
    load_dtype=torch.bfloat16,
):
    """
    Load textual inversion embeddings into multiple encoders (two CLIP + one T5).
    Assumes each embedding file is a dict: {token_string: embedding_tensor}.
    Returns the same list of text_encoders after in-place modification.
    """

    for idx, (emb_path, tokenizer, text_encoder) in enumerate(
        zip(embedding_path_list, tokenizers, text_encoders)
    ):
        if emb_path is not None:
            # Load the saved embeddings (token -> tensor)
            state_dict = load_state_dict(emb_path, map_location="cpu")
            
            # Now load the vectors
            for token_str, embedding in state_dict.items():
                token_id = tokenizer.convert_tokens_to_ids(token_str)

                # Assign
                with torch.no_grad():
                    text_encoder.get_input_embeddings().weight.data[token_id] = embedding.to(dtype=load_dtype)
                print(f"[Encoder {idx+1}] Loaded textual inversion embedding for '{token_str}' (id={token_id}).")

        text_encoder.eval()

    return text_encoders

def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
    weight_dtype = None
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device))[0]

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=text_encoder.device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    # prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    # prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device), output_hidden_states=True)

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    # prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    # prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)


    return prompt_embeds, pooled_prompt_embeds



def encode_prompt_pai(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    text_input_ids_list=None,
    label=None,                   # [B, 7]
    presudo_token_lists=None,     # [presudo_tokens_one(7), presudo_tokens_two(7)]
    weight_dtype=None,
    is_negative_embeddings = False
):
    # prompt to list
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(label)

    # unpack ids (provided)
    string_tokens_one, string_tokens_two,string_tokens_three = text_input_ids_list
    if string_tokens_one is not None:
        if string_tokens_one.shape[0] == 1 and batch_size > 1:
            string_tokens_one = string_tokens_one.repeat(batch_size, 1)  # [B, L_clip]
            string_tokens_two = string_tokens_two.repeat(batch_size, 1)  # [B, L_t5]
            string_tokens_three = string_tokens_three.repeat(batch_size, 1)

    # dtype / device
    if hasattr(text_encoders[0], "module"):
        dtype = text_encoders[0].module.dtype
    else:
        dtype = text_encoders[0].dtype
    device = device if device is not None else text_encoders[1].device

    # --- encode CLIP (seq + pooled) ---
    clip_seq, pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        text_input_ids=string_tokens_one if text_input_ids_list is not None else None,
     
    )  # clip_seq: [B, L_clip, D_clip]; pooled: [B, D_clip]

    clip_seq_2, pooled_prompt_embeds_2 = _encode_prompt_with_clip(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        text_input_ids=string_tokens_two if text_input_ids_list is not None else None,
  
    )  # clip_seq: [B, L_clip, D_clip]; pooled: [B, D_clip]

    # --- encode T5 (seq only) ---
    t5_seq = _encode_prompt_with_t5(
        text_encoder=text_encoders[2],
        tokenizer=tokenizers[2],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=1,
        device=device,
        text_input_ids=string_tokens_three if text_input_ids_list is not None else None,

    )  # [B, L_t5, D_t5]

    # ensure dtype/device
    clip_seq = clip_seq.to(device=device, dtype=weight_dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=weight_dtype)
    clip_seq_2 = clip_seq_2.to(device=device, dtype=weight_dtype)
    pooled_prompt_embeds_2 = pooled_prompt_embeds_2.to(device=device, dtype=weight_dtype)
    t5_seq = t5_seq.to(device=device, dtype=weight_dtype)
    
    if is_negative_embeddings:
        # no injection for negative embeddings
        pass
    else:
        # --- prompt-aligned injection (token-level) ---
        presudo_tokens_one, presudo_tokens_two,presudo_tokens_three = presudo_token_lists  # len == 7
        # inject into CLIP token embeddings
        for i,token_id in enumerate(presudo_tokens_one):
            placeholder_idx = torch.where(string_tokens_one == token_id)
            clip_seq[placeholder_idx] = clip_seq[placeholder_idx]+ label[:,i,:]
        for i,token_id in enumerate(presudo_tokens_two):
            placeholder_idx = torch.where(string_tokens_two == token_id)
            clip_seq_2[placeholder_idx] = clip_seq_2[placeholder_idx]+ label[:,i,:]

        #if enable_T5_injection:
        for i,token_id in enumerate(presudo_tokens_three):
            placeholder_idx = torch.where(string_tokens_three == token_id)
            t5_seq[placeholder_idx] = t5_seq[placeholder_idx]+ label[:,i,:]

    clip_prompt_embeds = torch.cat([clip_seq,clip_seq_2], dim=-1)
    pooled_prompt_embeds = torch.cat([pooled_prompt_embeds,pooled_prompt_embeds_2], dim=-1).to(dtype=weight_dtype)

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_seq.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_seq], dim=-2).to(dtype=weight_dtype)


    return prompt_embeds, pooled_prompt_embeds






def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Foward process in flow-matching

    Args:
        sample (`torch.FloatTensor`):
            The input sample.
        timestep (`int`, *optional*):
            The current timestep in the diffusion chain.

    Returns:
        `torch.FloatTensor`:
            A scaled input sample.
    """
    # if scheduler.step_index is None:
    scheduler._init_step_index(timestep)

    sigma = scheduler.sigmas[scheduler.step_index]
    sample = sigma * noise + (1.0 - sigma) * sample

    return sample


def calc_v_sd3(pipe, src_tar_latent_model_input, src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t):
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(src_tar_latent_model_input.shape[0])
    # joint_attention_kwargs = {}
    # # add timestep to joint_attention_kwargs
    # joint_attention_kwargs["timestep"] = timestep[0]
    # joint_attention_kwargs["timestep_idx"] = i

    with torch.no_grad():

        control_block_samples = pipe.controlnet(
                hidden_states=src_tar_latent_model_input,
                timestep=timestep,
                encoder_hidden_states=src_tar_prompt_embeds,
                pooled_projections=src_tar_pooled_prompt_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

        # # predict the noise for the source prompt
        noise_pred_src_tar = pipe.transformer(
            hidden_states=src_tar_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tar_prompt_embeds,
            pooled_projections=src_tar_pooled_prompt_embeds,
            block_controlnet_hidden_states=control_block_samples,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        # perform guidance source
        #if pipe.do_classifier_free_guidance:
        src_noise_pred_uncond, src_noise_pred_text, tar_noise_pred_uncond, tar_noise_pred_text = noise_pred_src_tar.chunk(4)
        if src_guidance_scale>1.0:
            noise_pred_src = src_noise_pred_uncond + src_guidance_scale * (src_noise_pred_text - src_noise_pred_uncond)
        else:
            noise_pred_src = src_noise_pred_text
        if tar_guidance_scale>1.0:
            noise_pred_tar = tar_noise_pred_uncond + tar_guidance_scale * (tar_noise_pred_text - tar_noise_pred_uncond)
        else:
            noise_pred_tar = tar_noise_pred_text

    return noise_pred_src, noise_pred_tar


@torch.no_grad()
def prompt_aligned_injection(pipe,prompt,presudo_list,label,negative_prompt="",load_dtype=torch.float16):

    presudo_token_ids_one = pipe.tokenizer.encode(' '.join(presudo_list), add_special_tokens=False)
    presudo_token_ids_two = pipe.tokenizer_2.encode(' '.join(presudo_list), add_special_tokens=False)
    presudo_token_ids_three = pipe.tokenizer_3.encode(' '.join(presudo_list), add_special_tokens=False)

    string_tokens_one = tokenize_prompt(pipe.tokenizer, prompt, max_sequence_length=77)
    string_tokens_two = tokenize_prompt(pipe.tokenizer_2, prompt, max_sequence_length=77)
    string_tokens_three = tokenize_prompt(pipe.tokenizer_3, prompt, max_sequence_length=77)

    prompt_embeds, pooled_prompt_embeds = encode_prompt_pai(
                                text_encoders=[pipe.text_encoder, pipe.text_encoder_2,pipe.text_encoder_3],
                                tokenizers=[None, None,None],
                                text_input_ids_list=[string_tokens_one, string_tokens_two,string_tokens_three],
                                max_sequence_length=77,
                                prompt=prompt,
                                label = label.to(load_dtype).unsqueeze(2),
                                presudo_token_lists =[presudo_token_ids_one,presudo_token_ids_two,presudo_token_ids_three],
                                weight_dtype = load_dtype,
                                is_negative_embeddings = False
    )
    # null embeddings
    negative_prompt_embeds, negative_pooled_prompt_embeds = encode_prompt_pai(
                            text_encoders=[pipe.text_encoder, pipe.text_encoder_2,pipe.text_encoder_3],
                            tokenizers=[pipe.tokenizer, pipe.tokenizer_2,pipe.tokenizer_3],
                            text_input_ids_list=[None, None,None],
                            max_sequence_length=77,
                            prompt=negative_prompt,
                            label = label.to(dtype=load_dtype).unsqueeze(2),
                            presudo_token_lists =[presudo_token_ids_one,presudo_token_ids_two,presudo_token_ids_three],
                            weight_dtype = load_dtype,
                            is_negative_embeddings = True
    )

    return prompt_embeds,negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

@torch.no_grad()
def FlowEditSD3(pipe,
    scheduler,
    x_src,
    prompt,
    presudo_list,
    label,
    DSCM_label,
    negative_prompt="",
    load_dtype= torch.float16,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 1.0,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 15,):
    
    device = x_src.device

    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)

    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)
    pipe._guidance_scale = src_guidance_scale
    
    # src prompts
    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = prompt_aligned_injection(
        pipe,
        prompt,
        presudo_list,
        label=label,
        negative_prompt=negative_prompt,
        load_dtype = load_dtype
    )

    # tar prompts
    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = prompt_aligned_injection(
        pipe,
        prompt,
        presudo_list,
        label=DSCM_label,
        negative_prompt=negative_prompt,
        load_dtype = load_dtype
    )
 
    # CFG prep
    src_tar_prompt_embeds = torch.cat([src_negative_prompt_embeds, src_prompt_embeds, tar_negative_prompt_embeds, tar_prompt_embeds], dim=0)
    src_tar_pooled_prompt_embeds = torch.cat([src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0)
    
    # initialize our ODE Zt_edit_1=x_src
    zt_edit = x_src.clone()

    for i, t in tqdm(enumerate(timesteps)):
        
        if T_steps - i > n_max:
            continue
        
        t_i = t/1000
        if i+1 < len(timesteps): 
            t_im1 = (timesteps[i+1])/1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)
        
        if T_steps - i > n_min:

            # Calculate the average of the V predictions
            V_delta_avg = torch.zeros_like(x_src)
            for k in range(n_avg):

                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                
                zt_src = (1-t_i)*x_src + (t_i)*fwd_noise

                zt_tar = zt_edit + zt_src - x_src

                src_tar_latent_model_input = torch.cat([zt_src, zt_src, zt_tar, zt_tar]) if pipe.do_classifier_free_guidance else torch.cat([zt_src, zt_src, zt_tar, zt_tar])

                Vt_src, Vt_tar = calc_v_sd3(pipe, src_tar_latent_model_input,src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t)

                V_delta_avg += (1/n_avg) * (Vt_tar - Vt_src) # - (hfg-1)*( x_src))

            # propagate direct ODE
            zt_edit = zt_edit.to(torch.float32)

            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else: # i >= T_steps-n_min # regular sampling for last n_min steps

            if i == T_steps-n_min:
                # initialize SDEDIT-style generation phase
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src
                
            src_tar_latent_model_input = torch.cat([xt_tar, xt_tar, xt_tar, xt_tar]) if pipe.do_classifier_free_guidance else torch.cat([zt_src, zt_src, zt_tar, zt_tar])
            
            _, Vt_tar = calc_v_sd3(pipe, src_tar_latent_model_input,src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t)

            xt_tar = xt_tar.to(torch.float32)

            prev_sample = xt_tar + (t_im1 - t_i) * (Vt_tar)

            prev_sample = prev_sample.to(Vt_tar.dtype)

            xt_tar = prev_sample
        
    return zt_edit if n_min == 0 else xt_tar


@torch.no_grad()
def Flow_editing(pipe,
    scheduler,
    input_image,
    prompt,
    presudo_list,
    label,
    DSCM_label,
    negative_prompt="",
    load_dtype= torch.float16,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 1.0,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 15,
    return_PIL = False):
    device = pipe.device
    with torch.autocast("cuda"), torch.inference_mode():
        x0_src_denorm = pipe.vae.encode(input_image).latent_dist.mode()
    x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    # send to cuda
    x0_src = x0_src.to(device)
    x0_src,prompt,label= align_batch_size(x0_src,prompt,label)
    negative_prompts = [""]*len(prompt)
    x0_tar = FlowEditSD3(pipe,
                        pipe.scheduler,
                        x0_src.clone(),
                        prompt,
                        presudo_list,
                        label.clone().to(pipe.device),
                        DSCM_label.to(pipe.device),
                        negative_prompt=negative_prompts,
                        T_steps= T_steps,
                        n_avg= n_avg,
                        src_guidance_scale= src_guidance_scale,
                        tar_guidance_scale = tar_guidance_scale,
                        n_min= n_min,
                        n_max= n_max,)

    x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    with torch.autocast("cuda"), torch.inference_mode():
        image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
        if return_PIL:
            image_tar = pipe.image_processor.postprocess(image_tar)
    
    return image_tar



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