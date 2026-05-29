# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm



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

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img,


def view_images(images, out_path='../outputs/p2p/attention.png', num_rows=1, offset_ratio=0.02, return_img=False, save_img=False):
    """
    Display or save a grid of images.

    Args:
        images (list or np.ndarray): List of images or image array.
        out_path (str): Path to save the image. Default is '../outputs/p2p/attention.png'.
        num_rows (int): Number of rows in the grid. Default is 1.
        offset_ratio (float): Offset ratio between images. Default is 0.02.
        return_img (bool): Whether to return the image. Default is False.
        save_img (bool): Whether to save the image. Default is False.
    
    Returns:
        PIL.Image: Image if return_img is True.
    """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    elif images.ndim==5:
        images = images.squeeze()
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    
    if return_img:
        return pil_img
    elif save_img:
        pil_img.save(out_path)
    else:
        display(pil_img)


def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.config.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.config.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_ldm(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
):
    register_attention_control(model, controller)
    height = width = 256
    batch_size = len(prompt)
    
    uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
    uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]
    
    text_input = model.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
    text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    context = torch.cat([uncond_embeddings, text_embeddings])
    
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale)
    
    image = latent2image(model.vqvae, latents)
   
    return image, latent


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
    img_size: int = 512,
    
):
    register_attention_control(model, controller)
    height = width = img_size
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    
    # set timesteps
    extra_set_kwargs = {"offset": 1}
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)
    
    image = latent2image(model.vae, latents)
  
    return image, latent

@torch.no_grad()
def DDIM_sample_mcpl(
    model, # model is pipe
    prompt,
    controller,
    start_step=0,
    start_latents=None,
    guidance_scale=3.5,
    num_inference_steps=30,
    num_images_per_prompt=1,
    negative_prompt="",
    controlnet_image=None,
    intervention_indx=None,
    intervention_values=None,
    generator: Optional[torch.Generator] = None,
    img_size: int = 512,
    label=None,

):
    register_attention_control(model, controller)
    device = model.device
    height = width = img_size
    batch_size = len(prompt)

    if guidance_scale>1:
        do_classifier_free_guidance = True
    else:
        do_classifier_free_guidance= False  
    
    start_latents, prompt, label = align_batch_size(start_latents, prompt, label)
    _,negtive_prompt_embedding = model.encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )
    input_ids = model.tokenizer(prompt,
                        padding="max_length",
                        truncation=True,
                        max_length=model.tokenizer.model_max_length,
                        return_tensors="pt",
                    ).input_ids.to(device)
    if input_ids.dim() ==1:
        input_ids=input_ids.unsqueeze(0)
    
    # set timesteps
    extra_set_kwargs = {"offset": 1}
    model.scheduler.set_timesteps(num_inference_steps,device=device)
    if start_latents is None:
        start_latents = torch.randn(1, 4, 32, 32, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma
    
    start_latents, label = start_latents.to(device),label.to(device)

    latents = start_latents.clone()
    for i in tqdm(range(start_step, num_inference_steps)):

        t = model.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)

    
        encoder_hidden_states = model.text_encoder(input_ids)[0].to(dtype=latent_model_input.dtype)
        if do_classifier_free_guidance:
            if negtive_prompt_embedding is not None:
                encoder_hidden_states = torch.cat(negtive_prompt_embedding, encoder_hidden_states)
            else:
                assert encoder_hidden_states is not None, 'negative_prompt_embedding should be provided when do_classifier_free_guidance is True'

        # Predict the noise residual
        noise_pred = model.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states,
                    return_dict=False)[0]



        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Normally we'd rely on the scheduler to handle the update step:
        # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        #latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        # Instead, let's do it ourselves:
        # prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
        # alpha_t = model.scheduler.alphas_cumprod[t.item()]
        # alpha_t_prev = model.scheduler.alphas_cumprod[prev_t]
        # predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        # direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
        # latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt
        latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
        latents = controller.step_callback(latents)

    # Post-processing
    # images = model.vae.decode(latents/ model.vae.config.scaling_factor,return_dict=False)[0]
    # do_denormalize = [True] * images.shape[0]
    # image = model.image_processor.postprocess(images,do_denormalize=do_denormalize)
    image = latent2image(model.vae, latents)
  
    return image, latents



@torch.no_grad()
def DDIM_sample_textcond(
    model, # model is pipe
    prompt,
    presudo_token_ids,
    controller,
    start_step=0,
    start_latents=None,
    guidance_scale=3.5,
    num_inference_steps=30,
    num_images_per_prompt=1,
    negative_prompt="",
    controlnet_image=None,
    intervention_indx=None,
    intervention_values=None,
    generator: Optional[torch.Generator] = None,
    img_size: int = 512,
    label=None,

):
    register_attention_control_controlnet(model, controller)
    device = model.device
    height = width = img_size
    batch_size = len(prompt)

    if guidance_scale>1:
        do_classifier_free_guidance = True
    else:
        do_classifier_free_guidance= False  
    
    start_latents, prompt, label = align_batch_size(start_latents, prompt, label)
    _,negtive_prompt_embedding = model.encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )
    input_ids = model.tokenizer(prompt,
                        padding="max_length",
                        truncation=True,
                        max_length=model.tokenizer.model_max_length,
                        return_tensors="pt",
                    ).input_ids.to(device)
    if input_ids.dim() ==1:
        input_ids=input_ids.unsqueeze(0)
    
    # set timesteps
    extra_set_kwargs = {"offset": 1}
    model.scheduler.set_timesteps(num_inference_steps,device=device)
    if start_latents is None:
        start_latents = torch.randn(1, 4, 32, 32, device=device)
        start_latents *= model.scheduler.init_noise_sigma
    
    start_latents, label = start_latents.to(device),label.to(device)

    latents = start_latents.clone()
    task_cond = model.controlnet.task_cond
    causal_cond,causal_loss = model.controlnet.controlnet_cond_embedding.inference(label,intervention_indx=intervention_indx,intervention_values=intervention_values)
    # if do_classifier_free_guidance and uncond_embeddings is not None:
    #     negtive_prompt_embedding = uncond_embeddings[i].expand(*negtive_prompt_embedding.shape)
    if 'generation' in task_cond and 'text' in task_cond:
        from causal_modules.ddim_modules import prompt_aligned_injection
        encoder_hidden_states = prompt_aligned_injection(model,input_ids.clone(),causal_cond,presudo_token_ids)
    else:
        encoder_hidden_states = model.text_encoder(input_ids)[0].to(dtype=model.dtype)

    if do_classifier_free_guidance:
        if negtive_prompt_embedding is not None:
            encoder_hidden_states = torch.cat([negtive_prompt_embedding, encoder_hidden_states],dim=0)
        else:
            assert negtive_prompt_embedding is not None, 'negative_prompt_embedding should be provided when do_classifier_free_guidance is True'
    
    for i in tqdm(range(start_step, num_inference_steps)):

        t = model.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)

        (down_block_res_samples, mid_block_res_sample),_,_ = model.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                    label = causal_cond,
                    training=False,
                    sampling=True,

        )
        noise_pred = model.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False)[0]
        # if 'global' in task_cond:
        #     encoder_hidden_states = control_embeddings
        # # else is use the local embedding for unet
        # else:
        #     encoder_hidden_states = model.text_encoder(input_ids)[0].to(dtype=mid_block_res_sample.dtype)
        #     if do_classifier_free_guidance:
        #         if negtive_prompt_embedding is not None:
        #             encoder_hidden_states = torch.cat(negtive_prompt_embedding, encoder_hidden_states)
        #         else:
        #             assert encoder_hidden_states is not None, 'negative_prompt_embedding should be provided when do_classifier_free_guidance is True'

        # # Predict the noise residual
        # noise_pred = model.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states,
        #             down_block_additional_residuals=down_block_res_samples,
        #             mid_block_additional_residual=mid_block_res_sample,
        #             return_dict=False)[0]



        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Normally we'd rely on the scheduler to handle the update step:
        # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        #latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        # Instead, let's do it ourselves:
        # prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
        # alpha_t = model.scheduler.alphas_cumprod[t.item()]
        # alpha_t_prev = model.scheduler.alphas_cumprod[prev_t]
        # predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        # direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
        # latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt
        latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
        latents = controller.step_callback(latents)

    # Post-processing
    # images = model.vae.decode(latents/ model.vae.config.scaling_factor,return_dict=False)[0]
    # do_denormalize = [True] * images.shape[0]
    # image = model.image_processor.postprocess(images,do_denormalize=do_denormalize)
    image = latent2image(model.vae, latents)
  
    return image, latents

def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None,temb=None,):
            is_cross = encoder_hidden_states is not None
            
            residual = hidden_states

            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_probs = self.get_attention_scores(query, key, attention_mask)
            attention_probs = controller(attention_probs, is_cross, place_in_unet)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = to_out(hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states
        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        
        if net_.__class__.__name__ == 'Attention':
        ## new diffusers use Attention class
        #if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count

def register_attention_control_controlnet(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None,temb=None,):
            is_cross = encoder_hidden_states is not None
            
            residual = hidden_states

            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_probs = self.get_attention_scores(query, key, attention_mask)
            attention_probs = controller(attention_probs, is_cross, place_in_unet)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = to_out(hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states
        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        
        if net_.__class__.__name__ == 'Attention':
        ## new diffusers use Attention class
        #if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.controlnet.named_children()
    # for controlnet append the down and mid blocks
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    sub_nets = model.unet.named_children()
    # for unet append the up blocks here
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count

    
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words
