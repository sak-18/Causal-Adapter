from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionPipeline
import torch.nn.functional as nnf
import numpy as np
import abc
from . import ptp_utils,seq_aligner
from torch import nn
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import os
from PIL import Image
from tqdm import tqdm
import sys
sys.path.append('${PROJECT_ROOT}')
from causal_modules import ddim_modules
LOW_RESOURCE = False
img_size = 256


## Inversion
@torch.no_grad()
def invert_mcpl(
    pipe,
    start_latents,
    prompt,
    guidance_scale=1,
    num_inference_steps=80,
    num_images_per_prompt=1,
    negative_prompt="",
    device=None,
    controlnet_image=None,
    intervention_indx=None,
    intervention_values=None,
    label=None,
):    
    if guidance_scale>1:
        do_classifier_free_guidance = True
    else:
        do_classifier_free_guidance= False    
    if start_latents is None:
        start_latents = torch.randn(1, 4, 32, 32, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma
    # # Encode prompt
    start_latents, prompt, label = ddim_modules.align_batch_size(start_latents, prompt, label)
    
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
    
    
    # Create a random starting point if we don't have one already
    
    # Latents are now the specified start latents
    start_latents, label = start_latents.to(device),label.to(device)
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)
    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

    
        encoder_hidden_states = pipe.text_encoder(input_ids)[0].to(dtype=latent_model_input.dtype)
        if do_classifier_free_guidance:
            if negtive_prompt_embedding is not None:
                encoder_hidden_states = torch.cat([negtive_prompt_embedding, encoder_hidden_states])
            else:
                assert encoder_hidden_states is not None, 'negative_prompt_embedding should be provided when do_classifier_free_guidance is True'

        
        
        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states,
                    return_dict=False)[0]
        #noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        

        # Perform guidance
        if guidance_scale>1:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.stack(intermediate_latents, dim=0)

## Inversion
@torch.no_grad()
def invert_textcond(
    pipe,
    start_latents,
    input_ids,
    guidance_scale=1,
    num_inference_steps=80,
    num_images_per_prompt=1,
    negative_prompt="",
    device=None,
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
    # text_embeddings,negtive_prompt_embedding = pipe.encode_prompt(
    #     prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    # )
    # if do_classifier_free_guidance:
    #     text_embeddings = torch.cat([negtive_prompt_embedding, text_embeddings])
    
    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 32, 32, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma
    
    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)
    task_cond = pipe.controlnet.task_cond
    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        (down_block_res_samples, mid_block_res_sample),causal_loss,control_embeddings,_ = pipe.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=input_ids,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                    training = False,
                    sampling=False,
                    intervention_indx=intervention_indx,
                    intervention_values=intervention_values,
                    text_encoder = pipe.text_encoder,
                    label = label
                    
        )
        if 'global' in task_cond:
            encoder_hidden_states = control_embeddings
        # else is use the local embedding for unet
        else:
            encoder_hidden_states = pipe.text_encoder(input_ids)[0].to(dtype=mid_block_res_sample.dtype)
        # Predict the noise residual
        '''think should I use the control residual here?'''
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states,down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False)[0]
        #noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings, return_dict=False)[0]
        

        # Perform guidance
        if guidance_scale>1:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)


def print_global_var():
    print(f"Global variable: {img_size}")

def modify_global_var(value):
    global img_size
    img_size = value

class LocalBlend:
    
    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold


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

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
    
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

    def get_average_attention(self,average_att_time: bool = True):
        self.average_att_time = average_att_time
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, average_att_time: bool = True):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.average_att_time = average_att_time

        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
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
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
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
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer

from PIL import Image

def aggregate_attention(prompts: list, attention_store: AttentionStore, res: int, \
                        from_where: List[str], is_cross: bool, select: int, \
                        average_att_ch: bool = True, average_att_time: bool = True):
    """
    Aggregate attention maps from the attention store.

    Args:
        prompts (list): List of prompts.
        attention_store (AttentionStore): Attention store object.
        res (int): Resolution of attention maps.
        from_where (List[str]): List of attention map sources.
        is_cross (bool): Whether to aggregate cross-attention maps.
        select (int): Index of the prompt to select.
        average_att_ch (bool): Whether to average attention maps across channels.
        average_att_time (bool): Whether to average attention maps across time steps.

    Returns:
        torch.Tensor: Aggregated attention map.
    """
    out = []
    attention_maps = attention_store.get_average_attention(average_att_time=average_att_time)

    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
            if location == 'mid' and item.shape[1] != num_pixels:
                import torch.nn.functional as F

                # item: (B, H*W, T) → (B, H, W, T)
                B, HW, T = item.shape
                orig_res = int(HW ** 0.5)
                assert orig_res * orig_res == HW, "Original spatial size must be square"

                # reshape to (B, T, H, W) for interpolation
                item_reshaped = item.reshape(B, orig_res, orig_res, T).permute(0, 3, 1, 2)  # (B, T, H, W)

                # interpolate to new resolution (e.g., new_res = 8)
                new_res = res
                resized = F.interpolate(item_reshaped, size=(new_res, new_res), mode='bilinear', align_corners=False)

                # permute back to (B, new_res*new_res, T)
                resized_flat = resized.permute(0, 2, 3, 1).reshape(B, new_res, new_res, T)
                out.append(resized_flat)
    out = torch.cat(out, dim=0)
    if average_att_ch:
        out = out.sum(0) / out.shape[0]
    else: # max
        out, _ = out.max(0)
    return out.cpu()


def _remove_axes(ax):
    """
    Remove axes from a matplotlib Axes object.

    Args:
        ax (matplotlib.axes.Axes): Axes object to remove axes from.
    """
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xticks([])
    ax.set_yticks([])

def remove_axes(axes):
    """
    Remove axes from a list of matplotlib Axes objects.

    Args:
        axes (list): List of Axes objects to remove axes from.
    """
    if len(axes.shape) == 2:
        for ax1 in axes:
            for ax in ax1:
                _remove_axes(ax)
    else:
        for ax in axes:
            _remove_axes(ax)

def apply_mask(image, mask, select_c, white_bg=True):
    """
    Apply a binary mask to an image.
    Foreground is kept while the background is replaced with NaN.

    Args:
        image (PIL.Image): Input image.
        mask (np.array): Binary mask.
        select_c (int): Selected class for masking.
        white_bg (bool): Whether to use a white background.

    Returns:
        PIL.Image: Masked image.

    """
    mask_height, mask_width = mask.shape[:2]  # If mask is grayscale, shape will be (height, width)

    # Resize the image to match the mask's size
    resized_img = image.resize((mask_width, mask_height))
    # Convert mask to boolean array

    mask_bool =mask[:,:,select_c-1].numpy().astype(bool)
    #mask_bool = mask == select_c

    # Extend the mask to all channels (assuming a 3-channel RGB image)
    mask_bool_rgb = np.stack([mask_bool]*3, axis=-1)

    # Create a white background image
    white_background = np.ones_like(resized_img) * 255

    # Convert image to numpy array
    image_array = np.array(resized_img).astype(np.float64)

    if white_bg:
        # Copy over the masked region from the original image to the white background
        white_background[mask_bool_rgb] = image_array[mask_bool_rgb]
        result_image = Image.fromarray(white_background, 'RGB')
    else:
        # For non-white background, retain the original behavior
        image_array[~mask_bool_rgb] = np.nan
        result_image = Image.fromarray(np.uint8(image_array), 'RGB')

    return result_image

def plot_img_mask(stablediff,tokenizer, prompts, emb_path_list, exp_names, device, out_dir, out_name, \
    latent=None, GUIDANCE_SCALE=5.0, attn_threshold=0.5, select_clsses = ['*','&'], \
    show_text=True, mask_concepts=False, g_gpu=None):
    """
    Plot images and attention masks for given embeddings.

    Args:
        ldm: Latent diffusion model.
        prompts (list): List of prompts.
        emb_path_list (list): List of embedding paths.
        exp_names (list): List of experiment names.
        device (str): Device to run the model on.
        out_dir (str): Output directory for saving images.
        out_name (str): Output file name.
        config: Configuration object.
        latent: Latent tensor.
        array_latent (bool): Whether to use array latent.
        GUIDANCE_SCALE (float): Guidance scale for the diffusion process.
        attn_threshold (float): Threshold for attention masks.
        select_clsses (list): List of selected classes.
        show_text (bool): Whether to show text in the images.
        mask_concepts (bool): Whether to mask concepts.
        g_gpu (torch.Generator): Generator for random numbers.
    """
    n_rows = 1
    n_cols = len(emb_path_list)+1 # img, mask, attn
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols * 10, n_rows * 10))


    for i, embedding_path in enumerate(emb_path_list):
        

        if g_gpu is None:
            g_gpu = torch.Generator(device='cuda')
        else:
            g_gpu = g_gpu
        controller = AttentionStore()
        images, _ = run_and_display(stablediff, prompts, controller, latent=latent, \
                            run_baseline=False, generator=g_gpu, \
                            guidance_scale=GUIDANCE_SCALE, return_img=True)
        attn_img = show_cross_attention(tokenizer, prompts, controller, res=16, from_where=["up", "down"], return_img=True, show_text=show_text, select_clsses=select_clsses)

        # ave-channel + ave-time attention
        attn_mask = show_cross_attention_mask_merged(tokenizer, prompts, controller, res=16, \
            from_where=["up", "down"], select_clsses = select_clsses, \
                average_att_ch=True, average_att_time=True, threshold=attn_threshold, return_img=True)

        img = ptp_utils.view_images(images, return_img=True)
        if i == 0:
            ax[0].imshow(img)
        
        ax[i+1].imshow(attn_mask)
        # if show_text:
        #     ax[i+1].set_xlabel(exp_names[i], fontsize=40)
    remove_axes(ax)
    plt.tight_layout()
    if not os.path.exists(out_dir):
        os.mkdir(out_dir) 
    plt.savefig(os.path.join(out_dir, out_name))
    # plt.show()
    # plt.clf()

def plot_img_attn_mask(stablediff,tokenizer, prompts, emb_path_list, exp_names, device, out_dir, out_name,res=16,  \
    latent=None, array_latent=False, GUIDANCE_SCALE=5.0, attn_threshold=0.5, select_clsses = ['*','&'], \
    show_text=True, mask_concepts=False, g_gpu=None,num_steps=50):
    """
    Plot images, attention masks, and masked concepts for given embeddings.

    Args:
        ldm: Latent diffusion model.
        prompts (list): List of prompts.
        emb_path_list (list): List of embedding paths.
        exp_names (list): List of experiment names.
        device (str): Device to run the model on.
        out_dir (str): Output directory for saving images.
        out_name (str): Output file name.
        config: Configuration object.
        latent: Latent tensor.
        array_latent (bool): Whether to use array latent.
        GUIDANCE_SCALE (float): Guidance scale for the diffusion process.
        attn_threshold (float): Threshold for attention masks.
        select_clsses (list): List of selected classes.
        show_text (bool): Whether to show text in the images.
        mask_concepts (bool): Whether to mask concepts.
        g_gpu (torch.Generator): Generator for random numbers.
    """
    n_rows = len(emb_path_list)
    n_cols = 3 # img, mask, attn
    if mask_concepts:
        n_cols += len(select_clsses)
    if mask_concepts and not show_text:
        n_attns = len(select_clsses)
        for p in select_clsses:
            mask_path = os.path.join(out_dir, 'masked_'+p)
            if not os.path.exists(mask_path):
                os.mkdir(mask_path) 
    else:
        n_attns = len(prompts[0].split(' '))
    if show_text:
        n_attns -= 2
    if mask_concepts:
        width_ratios = [1,1]
        for _ in range(len(select_clsses)):
            width_ratios.append(1)
        width_ratios.append(n_attns)
        fig, ax = plt.subplots(n_rows, n_cols, figsize=((n_cols+n_attns-1) * 5, n_rows * 5), \
            gridspec_kw={'width_ratios': width_ratios})
    else:
        fig, ax = plt.subplots(n_rows, n_cols, figsize=((n_cols+n_attns-1) * 5, n_rows * 5), \
            gridspec_kw={'width_ratios': [1,1,n_attns]})

    if n_rows==1:
        ax = np.reshape(ax, (1, n_cols))
        print('ax shape:',ax.shape)
    for i, embedding_path in enumerate(emb_path_list):
        controller = AttentionStore()
        controller.reset()
        # ldm.embedding_manager.load(embedding_path)
        # ldm = ldm.to(device)
        # tokenizer = ldm.cond_stage_model.tknz_fn.tokenizer
        model_id,tokenizer = stablediff.name_or_path, stablediff.tokenizer
        text_encoder = load_mcpl_embeddings(model_id,tokenizer,embedding_path)
        stablediff.text_encoder=text_encoder
        stablediff.to(device)
        img_latent= latent.clone()
        inverted_latents = invert(stablediff,
            img_latent,
            prompts[0],
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=num_steps,
            num_images_per_prompt=1,
            negative_prompt="",
            device=device,
        )
        if g_gpu is None:
            g_gpu = torch.Generator(device='cuda')
        else:
            g_gpu = g_gpu
        
        images, _ = run_and_display(stablediff, prompts, controller, latent=inverted_latents[-(0+ 1)][None], \
                            run_baseline=False, generator=g_gpu, \
                            guidance_scale=GUIDANCE_SCALE, return_img=True,num_inference_steps = num_steps)
        if mask_concepts and not show_text:
            attn_img = show_cross_attention(tokenizer, prompts, controller, res=res, from_where=["up", "down"], return_img=True, show_text=show_text, select_clsses=select_clsses)
        else:
            attn_img = show_cross_attention(tokenizer, prompts, controller, res=res, from_where=["up", "down"], return_img=True, show_text=show_text)

        # ave-channel + ave-time attention
        attn_mask = show_cross_attention_mask_merged(tokenizer, prompts, controller, res=res, \
            from_where=["up", "down"], select_clsses = select_clsses, \
                average_att_ch=True, average_att_time=True, threshold=attn_threshold, return_img=True)

        img = ptp_utils.view_images(images, return_img=True)
        
        ax[i,0].imshow(img)
        if show_text:
            ax[i,0].set_xlabel(exp_names[i], fontsize=40)
        ax[i,1].imshow(attn_mask)
        if show_text:
            ax[i,1].set_xlabel("Attn Masks", fontsize=40)
        col = 2
        if mask_concepts:
            for c in range(1,len(select_clsses)+1):
                masked_img = apply_mask(img, attn_mask, c)
                ax[i,col].imshow(masked_img)
                if show_text:
                    ax[i,col].set_xlabel(select_clsses[c-1], fontsize=40)
                else:
                    mask_path = os.path.join(out_dir, 'masked_'+select_clsses[c-1])
                    masked_img.save(os.path.join(mask_path, exp_names[i]+'.png'), 'PNG')
                    masked_img_wbg = apply_mask(img, attn_mask, c, white_bg=True)
                    mask_path_wbg = os.path.join(out_dir, 'masked_'+select_clsses[c-1]+'white_bg')
                    if not os.path.exists(mask_path_wbg):
                        os.mkdir(mask_path_wbg) 
                    masked_img_wbg.save(os.path.join(mask_path_wbg, exp_names[i]+'.png'), 'PNG')
                col += 1
            img_path = os.path.join(out_dir, 'gen_img')
            if not os.path.exists(img_path):
                os.mkdir(img_path) 
            img.save(os.path.join(img_path, exp_names[i]+'.png'), 'PNG')

            mask_path = os.path.join(out_dir, 'attn_mask')
            if not os.path.exists(mask_path):
                os.mkdir(mask_path) 
            Image.fromarray(attn_mask.byte().numpy()).save(os.path.join(mask_path, exp_names[i]+'.png'), 'PNG')
        ax[i,col].imshow(attn_img)
    
    remove_axes(ax)
    plt.tight_layout()
    if not os.path.exists(out_dir):
        os.mkdir(out_dir) 
    #plt.savefig(os.path.join(out_dir, out_name))
    plt.show()
    plt.clf()

def plot_img_attn_mask_textcontrol(pipe, prompts, presudo_words,presudo_token_ids,condition_image, device, out_dir, out_name,res=16,  \
    latent=None, GUIDANCE_SCALE=5.0, attn_threshold=0.5,intervention_indx=None,intervention_values=None,only_sampling=False,  \
    show_text=True, mask_concepts=False,class_select=True, g_gpu=None,exp_names=None,num_steps=50,img_size=256,from_where=["up", "down"],label=None,dataset='celeba',save_masks=False):
    """
    Plot images, attention masks, and masked concepts for given embeddings.

    Args:
        ldm: Latent diffusion model.
        prompts (list): List of prompts.
        emb_path_list (list): List of embedding paths.
        exp_names (list): List of experiment names.
        device (str): Device to run the model on.
        out_dir (str): Output directory for saving images.
        out_name (str): Output file name.
        config: Configuration object.
        latent: Latent tensor.
        array_latent (bool): Whether to use array latent.
        GUIDANCE_SCALE (float): Guidance scale for the diffusion process.
        attn_threshold (float): Threshold for attention masks.
        select_clsses (list): List of selected classes.
        show_text (bool): Whether to show text in the images.
        mask_concepts (bool): Whether to mask concepts.
        g_gpu (torch.Generator): Generator for random numbers.
    """
    #n_rows = len(emb_path_list)
    n_rows = 1 
    n_cols = 3 # img, mask, attn
    select_clsses = presudo_words.split(',')
    if mask_concepts:
        n_cols += len(select_clsses)
    if mask_concepts and not show_text:
        n_attns = len(select_clsses)
        for p in select_clsses:
            mask_path = os.path.join(out_dir, 'masked_'+p)
            if not os.path.exists(mask_path):
                os.mkdir(mask_path) 
    else:
        n_attns = len(prompts[0].split(' '))
    if show_text:
        n_attns -= 2
    if mask_concepts:
        width_ratios = [1,1]
        for _ in range(len(select_clsses)):
            width_ratios.append(1)
        width_ratios.append(n_attns)
        fig, ax = plt.subplots(n_rows, n_cols, figsize=((n_cols+n_attns-1) * 5, n_rows * 5), \
            gridspec_kw={'width_ratios': width_ratios})
    else:
        fig, ax = plt.subplots(n_rows, n_cols, figsize=((n_cols+n_attns-1) * 5, n_rows * 5), \
            gridspec_kw={'width_ratios': [1,1,n_attns]})

    if n_rows==1:
        ax = np.reshape(ax, (1, n_cols))
        print('ax shape:',ax.shape)
    #for i, embedding_path in enumerate(emb_path_list):

    model_id,tokenizer = pipe.name_or_path, pipe.tokenizer
    # text_encoder = pipe.text_encoder
    # presudo_token_ids = tokenizer.encode(' '.join(select_clsses), add_special_tokens=False)
    input_ids = tokenizer(prompts[0],
                        padding="max_length",
                        truncation=True,
                        max_length=tokenizer.model_max_length,
                        return_tensors="pt",
                    ).input_ids[0].to(device)
    if input_ids.dim() ==1:
        input_ids=input_ids.unsqueeze(0)

    with torch.no_grad():
        latent = pipe.vae.encode(latent.clone().to(device))
        img_latent = 0.18215 * latent.latent_dist.sample()

    inverted_latents,_,_ = ddim_modules.invert(pipe,
        img_latent,
        prompts[0],
        presudo_token_ids,
        guidance_scale=1.0,
        num_inference_steps=num_steps,
        num_images_per_prompt=1,
        negative_prompt=None,
        device=device,
        controlnet_image=None,
        intervention_indx=None,
        intervention_values=None,
        label=label.clone())

    if g_gpu is None:
        g_gpu = torch.Generator(device='cuda')
    else:
        g_gpu = g_gpu

    s_step = 0
    if only_sampling:
        start_lant = None
    else:
        start_lant= inverted_latents[-(s_step + 1)].clone()
    controller = AttentionStore()
    controller.reset()

    images, _ = ptp_utils.DDIM_sample_textcond(pipe, prompts[0], presudo_token_ids,controller, start_step=s_step,
                                                start_latents=start_lant,
                                                guidance_scale=GUIDANCE_SCALE,
                                                num_inference_steps=num_steps,
                                                num_images_per_prompt=1,
                                                negative_prompt=None,
                                                controlnet_image=None,
                                                intervention_indx=intervention_indx,
                                                intervention_values=intervention_values,
                                                generator=g_gpu,
                                                img_size = img_size,
                                                label=label.clone())
    ptp_utils.view_images(images, out_path='../outputs/p2p/attention.png', return_img=True, save_img=False)
    # images, _ = run_and_display(pipe, prompts, controller, latent=inverted_latents[-(0+ 1)][None], \
    #                     run_baseline=False, generator=g_gpu, \
    #                     guidance_scale=GUIDANCE_SCALE, return_img=True,num_inference_steps = num_steps)
    if mask_concepts and not show_text:
        if save_masks is True:
            return_img_bool = False
        else:
            return_img_bool = True
        if class_select:
            attn_img = show_cross_attention(tokenizer, prompts, controller, res=res, from_where=from_where, return_img=return_img_bool, show_text=show_text, select_clsses=select_clsses,dataset=dataset,save_img=save_masks)
        else:
            attn_img = show_cross_attention(tokenizer, prompts, controller, res=res, from_where=from_where, return_img=return_img_bool, show_text=show_text,dataset=dataset,save_img=save_masks)
    else:
        attn_img = show_cross_attention(tokenizer, prompts, controller, res=res, from_where=from_where, return_img=True, show_text=show_text,dataset=dataset)

    # ave-channel + ave-time attention
    attn_mask,overlapped_mask = show_cross_attention_mask_merged(tokenizer, prompts, controller, res=res, \
        from_where=from_where, select_clsses = select_clsses, \
            average_att_ch=True, average_att_time=True, threshold=attn_threshold, return_img=True,dataset=dataset)

    img = ptp_utils.view_images(images, return_img=True)
    i=0
    ax[i,0].imshow(img)
    if show_text:
        pass
        #ax[i,0].set_xlabel('Image', fontsize=40)
        #ax[i,0].set_xlabel(exp_names[i], fontsize=40)
    ax[i,1].imshow(attn_mask)
    if show_text:
        pass
        #ax[i,1].set_xlabel("Attn Masks", fontsize=40)
    col = 2
    if mask_concepts:
        for c in range(1,len(select_clsses)+1):
            #masked_img = apply_mask(img, attn_mask, c)
            masked_img = apply_mask(img, overlapped_mask, c)
            ax[i,col].imshow(masked_img)
            if show_text:
                #ax[i,col].set_xlabel(select_clsses[c-1], fontsize=40)

                mask_path = os.path.join(out_dir, 'masked_'+select_clsses[c-1])
                #masked_img.save(os.path.join(mask_path, exp_names[i]+'.png'), 'PNG')
                #masked_img_wbg = apply_mask(img, attn_mask, c, white_bg=False)
                masked_img_wbg = apply_mask(img, overlapped_mask, c, white_bg=False)
                mask_path_wbg = os.path.join(out_dir, 'masked_'+select_clsses[c-1]+'white_bg')
                if not os.path.exists(mask_path_wbg):
                    os.mkdir(mask_path_wbg) 
                #masked_img_wbg.save(os.path.join(mask_path_wbg, exp_names[i]+'.png'), 'PNG')
            col += 1
        img_path = os.path.join(out_dir, 'gen_img')
        if not os.path.exists(img_path):
            os.mkdir(img_path) 
        img.save(os.path.join(img_path, exp_names[i]+'.png'), 'PNG')

        mask_path = os.path.join(out_dir, 'attn_mask')
        if not os.path.exists(mask_path):
            os.mkdir(mask_path) 
        Image.fromarray(attn_mask.byte().numpy()).save(os.path.join(mask_path, exp_names[i]+'.png'), 'PNG')
    ax[i,col].imshow(attn_img)
    controller.reset()
    remove_axes(ax)
    plt.tight_layout()
    if not os.path.exists(out_dir):
        os.mkdir(out_dir) 
    plt.savefig(os.path.join(out_dir, out_name))
    plt.show()
    plt.clf()

    return overlapped_mask,attn_img

def plot_img_attn_mask_mcpl(pipe, prompts, presudo_words,condition_image, device, out_dir, out_name,res=16,  \
    latent=None, GUIDANCE_SCALE=5.0, attn_threshold=0.5,intervention_indx=None,intervention_values=None,only_sampling=False,  \
    show_text=True, mask_concepts=False,class_select=True, g_gpu=None,exp_names=None,num_steps=50,img_size=256,from_where=["up", "down"],label=None,dataset='celeba'):
    """
    Plot images, attention masks, and masked concepts for given embeddings.

    Args:
        ldm: Latent diffusion model.
        prompts (list): List of prompts.
        emb_path_list (list): List of embedding paths.
        exp_names (list): List of experiment names.
        device (str): Device to run the model on.
        out_dir (str): Output directory for saving images.
        out_name (str): Output file name.
        config: Configuration object.
        latent: Latent tensor.
        array_latent (bool): Whether to use array latent.
        GUIDANCE_SCALE (float): Guidance scale for the diffusion process.
        attn_threshold (float): Threshold for attention masks.
        select_clsses (list): List of selected classes.
        show_text (bool): Whether to show text in the images.
        mask_concepts (bool): Whether to mask concepts.
        g_gpu (torch.Generator): Generator for random numbers.
    """
    #n_rows = len(emb_path_list)
    n_rows = 1 
    n_cols = 3 # img, mask, attn
    select_clsses = presudo_words.split(',')
    if mask_concepts:
        n_cols += len(select_clsses)
    if mask_concepts and not show_text:
        n_attns = len(select_clsses)
        for p in select_clsses:
            mask_path = os.path.join(out_dir, 'masked_'+p)
            if not os.path.exists(mask_path):
                os.mkdir(mask_path) 
    else:
        n_attns = len(prompts[0].split(' '))
    if show_text:
        n_attns -= 2
    if mask_concepts:
        width_ratios = [1,1]
        for _ in range(len(select_clsses)):
            width_ratios.append(1)
        width_ratios.append(n_attns)
        fig, ax = plt.subplots(n_rows, n_cols, figsize=((n_cols+n_attns-1) * 5, n_rows * 5), \
            gridspec_kw={'width_ratios': width_ratios})
    else:
        fig, ax = plt.subplots(n_rows, n_cols, figsize=((n_cols+n_attns-1) * 5, n_rows * 5), \
            gridspec_kw={'width_ratios': [1,1,n_attns]})

    if n_rows==1:
        ax = np.reshape(ax, (1, n_cols))
        print('ax shape:',ax.shape)
    #for i, embedding_path in enumerate(emb_path_list):

    model_id,tokenizer = pipe.name_or_path, pipe.tokenizer
    # text_encoder = pipe.text_encoder
    # presudo_token_ids = tokenizer.encode(' '.join(select_clsses), add_special_tokens=False)
    input_ids = tokenizer(prompts[0],
                        padding="max_length",
                        truncation=True,
                        max_length=tokenizer.model_max_length,
                        return_tensors="pt",
                    ).input_ids[0].to(device)
    if input_ids.dim() ==1:
        input_ids=input_ids.unsqueeze(0)

    with torch.no_grad():
        latent = pipe.vae.encode(latent.clone().to(device))
        img_latent = 0.18215 * latent.latent_dist.sample()

    inverted_latents = invert_mcpl(pipe,
        img_latent,
        prompts[0],
        guidance_scale=1.0,
        num_inference_steps=num_steps,
        num_images_per_prompt=1,
        negative_prompt=None,
        device=device,
        controlnet_image=None,
        intervention_indx=None,
        intervention_values=None,
        label=label.clone())

    if g_gpu is None:
        g_gpu = torch.Generator(device='cuda')
    else:
        g_gpu = g_gpu

    s_step = 0
    if only_sampling:
        start_lant = None
    else:
        start_lant= inverted_latents[-(s_step + 1)].clone()
    controller = AttentionStore()
    controller.reset()

    images, _ = ptp_utils.DDIM_sample_mcpl(pipe, prompts[0], controller, start_step=s_step,
                                                start_latents=start_lant,
                                                guidance_scale=GUIDANCE_SCALE,
                                                num_inference_steps=num_steps,
                                                num_images_per_prompt=1,
                                                negative_prompt=None,
                                                controlnet_image=None,
                                                intervention_indx=intervention_indx,
                                                intervention_values=intervention_values,
                                                generator=g_gpu,
                                                img_size = img_size,
                                                label=label.clone())
    ptp_utils.view_images(images, out_path='../outputs/p2p/attention.png', return_img=True, save_img=False)
    # images, _ = run_and_display(pipe, prompts, controller, latent=inverted_latents[-(0+ 1)][None], \
    #                     run_baseline=False, generator=g_gpu, \
    #                     guidance_scale=GUIDANCE_SCALE, return_img=True,num_inference_steps = num_steps)
    if mask_concepts and not show_text:
        if class_select:
            attn_img = show_cross_attention(tokenizer, prompts, controller, res=res, from_where=from_where, return_img=True, show_text=show_text, select_clsses=select_clsses,dataset=dataset)
        else:
            attn_img = show_cross_attention(tokenizer, prompts, controller, res=res, from_where=from_where, return_img=True, show_text=show_text,dataset=dataset)

    else:
        attn_img = show_cross_attention(tokenizer, prompts, controller, res=res, from_where=from_where, return_img=True, show_text=show_text,dataset=dataset)

    # ave-channel + ave-time attention
    attn_mask,overlapped_mask = show_cross_attention_mask_merged(tokenizer, prompts, controller, res=res, \
        from_where=from_where, select_clsses = select_clsses, \
            average_att_ch=True, average_att_time=True, threshold=attn_threshold, return_img=True,dataset=dataset)

    img = ptp_utils.view_images(images, return_img=True)
    i=0
    ax[i,0].imshow(img)
    if show_text:
        pass
        #ax[i,0].set_xlabel('Image', fontsize=40)
        #ax[i,0].set_xlabel(exp_names[i], fontsize=40)
    ax[i,1].imshow(attn_mask)
    if show_text:
        pass
        #ax[i,1].set_xlabel("Attn Masks", fontsize=40)
    col = 2
    if mask_concepts:
        for c in range(1,len(select_clsses)+1):
            #masked_img = apply_mask(img, attn_mask, c)
            masked_img = apply_mask(img, overlapped_mask, c)
            ax[i,col].imshow(masked_img)
            if show_text:
                #ax[i,col].set_xlabel(select_clsses[c-1], fontsize=40)

                mask_path = os.path.join(out_dir, 'masked_'+select_clsses[c-1])
                #masked_img.save(os.path.join(mask_path, exp_names[i]+'.png'), 'PNG')
                #masked_img_wbg = apply_mask(img, attn_mask, c, white_bg=False)
                masked_img_wbg = apply_mask(img, overlapped_mask, c, white_bg=False)
                mask_path_wbg = os.path.join(out_dir, 'masked_'+select_clsses[c-1]+'white_bg')
                if not os.path.exists(mask_path_wbg):
                    os.mkdir(mask_path_wbg) 
                #masked_img_wbg.save(os.path.join(mask_path_wbg, exp_names[i]+'.png'), 'PNG')
            col += 1
        img_path = os.path.join(out_dir, 'gen_img')
        if not os.path.exists(img_path):
            os.mkdir(img_path) 
        img.save(os.path.join(img_path, exp_names[i]+'.png'), 'PNG')

        mask_path = os.path.join(out_dir, 'attn_mask')
        if not os.path.exists(mask_path):
            os.mkdir(mask_path) 
        Image.fromarray(attn_mask.byte().numpy()).save(os.path.join(mask_path, exp_names[i]+'.png'), 'PNG')
    ax[i,col].imshow(attn_img)
    controller.reset()
    remove_axes(ax)
    plt.tight_layout()
    if not os.path.exists(out_dir):
        os.mkdir(out_dir) 
    plt.savefig(os.path.join(out_dir, out_name))
    plt.show()
    plt.clf()

    return overlapped_mask,attn_img

# def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
#     tokens = tokenizer.encode(prompts[select])
#     decoder = tokenizer.decode
#     attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
#     images = []
#     for i in range(len(tokens)):
#         image = attention_maps[:, :, i]
#         image = 255 * image / image.max()
#         image = image.unsqueeze(-1).expand(*image.shape, 3)
#         image = image.numpy().astype(np.uint8)
#         image = np.array(Image.fromarray(image).resize((256, 256)))
#         image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
#         images.append(image)
#     ptp_utils.view_images(np.stack(images, axis=0))
def show_cross_attention(tokenizer: nn.Module, prompts: list, attention_store: AttentionStore, \
        res: int, from_where: List[str], select: int = 0, return_img: bool = False, \
        out_path_img='./output/attn_maps/', save_img=False, show_text=True, select_clsses=[],dataset='celeba'):
    """
    Show cross-attention maps for the given prompts.

    Args:
        tokenizer (nn.Module): Tokenizer for processing prompts.
        prompts (list): List of prompts.
        attention_store (AttentionStore): Attention store object.
        res (int): Resolution of attention maps.
        from_where (List[str]): List of attention map sources.
        select (int): Index of the prompt to select.
        return_img (bool): Whether to return the image.
        out_path_img (str): Output path for saving the image.
        save_img (bool): Whether to save the image.
        show_text (bool): Whether to show text in the images.
        select_clsses (list): List of selected classes.
    """
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
    images = []
    threshold = 0.8
    for i in range(1,len(tokens)-1):
        if len(select_clsses) > 0 and decoder(int(tokens[i])) not in select_clsses:
            continue
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()

        # image_rgb = image.unsqueeze(-1).expand(*image.shape, 3).numpy().astype(np.uint8)

        # # Apply threshold mask: pixels below threshold → black
        # mask = (image >= (threshold*255))
        # image_rgb[~mask] = 0  # set those pixels to black

        # # Resize image
        # image = np.array(Image.fromarray(image_rgb).resize((256, 256)))

        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        if show_text:
            '''to delete at end'''
            decoder_int = decoder(int(tokens[i]))
            # if dataset=='celeba' and decoder_int=='of':
            #     decoder_int = 'is'
            if dataset == 'celeba':
                decoder_int = 'a human is @ and * and & and !'.split(' ')[i-1]
            image = ptp_utils.text_under_image(image, decoder_int)
        images.append(image)
    
    if return_img:
        pil_img = ptp_utils.view_images(np.stack(images, axis=0), 'cross_attention', return_img=return_img)
        return pil_img
    elif save_img:
        #ptp_utils.view_images(np.stack(images, axis=0), save_img=save_img, out_path=out_path_img)  
        tokens = 'a human is @ and * and & and !'.split(' ')
        arr = np.stack(images, axis=0)
        os.makedirs(out_path_img, exist_ok=True)
        # Save each image
        for i in range(10):
            img = Image.fromarray(arr[i])
            filename = os.path.join(out_path_img,f"{i}.png")
            img.save(filename)
            print(f"Saved {filename}")
    else:
        ptp_utils.view_images(np.stack(images, axis=0), 'cross_attention')

def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))

def show_cross_attention_with_mask(tokenizer: nn.Module, prompts: list, \
                                    attention_store: AttentionStore, res: int, from_where: List[str], \
                                    select: int = 0, threshold: float = 0.3, \
                                    average_att_ch: bool = True, average_att_time: bool = True, \
                                    segment_method: str = 'threshold'):
    """
    Show cross-attention maps with masks for the given prompts.

    Args:
        tokenizer (nn.Module): Tokenizer for processing prompts.
        prompts (list): List of prompts.
        attention_store (AttentionStore): Attention store object.
        res (int): Resolution of attention maps.
        from_where (List[str]): List of attention map sources.
        select (int): Index of the prompt to select.
        threshold (float): Threshold for masking.
        average_att_ch (bool): Whether to average attention maps across channels.
        average_att_time (bool): Whether to average attention maps across time steps.
        segment_method (str): Method for segmenting the masks ('threshold' or 'kmean').
    """
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select, average_att_ch=average_att_ch, average_att_time=average_att_time)
    images = []
    masks = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = image / image.max()
        image = F.resize(image.unsqueeze(0), 256).squeeze(0)
        if segment_method == 'threshold':
            mask = (image.gt(threshold)).int()
        elif segment_method == 'kmean':
            X = np.array(image.view(-1).unsqueeze(1).cpu())
            kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
            kmean_pred = kmeans.labels_.reshape(256, 256)
            mask = torch.tensor(kmean_pred.astype(int))
        mask *= (i+1)
        mask = mask.unsqueeze(-1).expand(*mask.shape, 3)
        mask = mask.numpy().astype(np.uint8)
        mask = np.array(Image.fromarray(mask).resize((256, 256), Image.NEAREST))
        mask = ptp_utils.text_under_image(mask, decoder(int(tokens[i])))
        masks.append(mask)
    ptp_utils.view_masks(np.stack(masks, axis=0)) 

def show_cross_attention_mask_merged(tokenizer: nn.Module, prompts: list, \
                                        attention_store: AttentionStore, res: int, from_where: List[str], \
                                        select: int = 0, threshold: float = 0.65, select_clsses: list = ['a','photo'], \
                                        average_att_ch: bool = True, average_att_time: bool = True, masked_scale: bool = False, \
                                        background_scale: float = 1.0, images = None, blend = 0.3, return_img: bool = False,dataset='celeba'):
    """
    Show merged cross-attention masks for the given prompts.

    Args:
        tokenizer (nn.Module): Tokenizer for processing prompts.
        prompts (list): List of prompts.
        attention_store (AttentionStore): Attention store object.
        res (int): Resolution of attention maps.
        from_where (List[str]): List of attention map sources.
        select (int): Index of the prompt to select.
        threshold (float): Threshold for masking.
        select_clsses (list): List of selected classes.
        average_att_ch (bool): Whether to average attention maps across channels.
        average_att_time (bool): Whether to average attention maps across time steps.
        masked_scale (bool): Whether to scale masked regions.
        background_scale (float): Scale for background regions.
        images (np.array): Array of images.
        blend (float): Blending factor for masks.
        return_img (bool): Whether to return the image.
    """
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    #attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select, average_att_ch=average_att_ch, average_att_time=average_att_time)
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
    # attention_maps = attention_maps[:, :, select_clsses]
    attention_maps_temp = None
    for select_c in range(len(tokens)):
        if decoder(int(tokens[select_c])) not in select_clsses:
            continue
        # select_c+1 indicates 
        attention_map = attention_maps[:, :, select_c].unsqueeze(-1)

        if attention_maps_temp is None:
            attention_maps_temp = attention_map
        else:
            attention_maps_temp = torch.cat([attention_maps_temp, attention_map], dim=-1)


    attention_maps_temp = attention_maps_temp.permute(2,0,1)
    attention_maps_temp = F.resize(attention_maps_temp, 256)
    attention_maps_temp = attention_maps_temp.permute(1,2,0)
    map_len = attention_maps_temp.shape[-1]
    attention_maps_max, _ = attention_maps_temp.view(-1, map_len).max(0)
    attention_maps_norm = attention_maps_temp / attention_maps_max
    # scale background class
    # attention_maps_norm[:,:,0] *= background_scale
    mask = torch.zeros(attention_maps_norm.shape)
    seg = torch.zeros(attention_maps_norm.shape[:-1]).long()
    for i in range(attention_maps_norm.shape[-1]):
        mask[:,:,i] = (attention_maps_norm[:,:,i].gt(threshold)).long()
        seg += (attention_maps_norm[:,:,i].gt(threshold)).long() * (i+1)
    
    if dataset == 'celeba':
        for i in range(attention_maps_norm.shape[-1]):
            current_mask = (attention_maps_norm[:, :, i] > threshold)
            seg[current_mask] = i + 1  # label starts from 1
    # intersect = mask.sum(-1) > 1
    # _, mask_merge = attention_maps_norm.view(-1, map_len).max(-1)
    # mask_merge = mask_merge.view(256,256)
    # mask_merge += 1
    # seg[intersect] = mask_merge[intersect]
        
    if images is not None:
        img = Image.fromarray(images[0].astype(np.uint8)).convert('RGBA')
        plt.imshow(Image.fromarray(np.array(seg).astype(np.uint8)))
        plt.axis('off')
        plt.savefig("temp.png", bbox_inches='tight', pad_inches=0.0)
        seg = Image.open("temp.png")
        seg = seg.resize((256,256), Image.NEAREST)
        seg = Image.blend(img, seg, blend)
        os.remove("temp.png")
    if return_img:
        return seg,mask
    else:
        plt.imshow(seg)
        plt.axis('off')
        # pil_img = Image.fromarray(mask_merge.numpy().astype(np.uint8))
        # display(pil_img)
        # ptp_utils.view_images(mask_merge.numpy().astype(np.uint8))



def run_and_display(ldm_stable,prompts, controller, latent=None, run_baseline=False, generator=None,num_inference_steps: int = 100,
                                                        guidance_scale: float = 7.5,
                                                        low_resource: bool = False,return_img=False, save_img=False, out_path_img='../outputs/p2p/attention.png'):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = ptp_utils.text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator, low_resource=low_resource,img_size=img_size)
    ptp_utils.view_images(images, out_path=out_path_img, return_img=return_img, save_img=save_img)
    return images, x_t