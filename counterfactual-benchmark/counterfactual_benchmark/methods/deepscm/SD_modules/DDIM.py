import sys
from pathlib import Path
# edit here
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(REPO_ROOT / "causal-adapter-sd15"))
import diffusers
from diffusers import StableDiffusionCausalControlNetPipeline, Causal_ControlNetModel, UniPCMultistepScheduler,StableDiffusionPipeline
import importlib
importlib.reload(diffusers)
from diffusers.utils import load_image
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from diffusers import DDIMInverseScheduler, DDIMScheduler
from tqdm import tqdm

device = torch.device("cuda")
# Sample function (regular DDIM)
@torch.no_grad()
def sample(
    pipe,
    input_ids,
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
    label=None
    
):
    if guidance_scale>1:
        do_classifier_free_guidance = True
    else:
        do_classifier_free_guidance= False    

    # Encode prompt
    # text_embeddings,negtive_prompt_embedding = pipe.encode_prompt(
    #     prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    # )
    # if do_classifier_free_guidance:
    #     text_embeddings = torch.cat([negtive_prompt_embedding, text_embeddings])
    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 32, 32, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()
    task_cond = pipe.controlnet.task_cond
    for i in tqdm(range(start_step, num_inference_steps)):

        t = pipe.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        (down_block_res_samples, mid_block_res_sample),causal_loss,control_embeddings = pipe.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=input_ids,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                    label = label,
                    training=False,
                    sampling=True,
                    intervention_indx=intervention_indx,
                    intervention_values=intervention_values,
                    text_encoder = pipe.text_encoder
                    
        )
        if 'global' in task_cond:
            encoder_hidden_states = control_embeddings
        # else is use the local embedding for unet
        else:
            encoder_hidden_states = pipe.text_encoder(input_ids)[0].to(dtype=mid_block_res_sample.dtype)


        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False)[0]

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Normally we'd rely on the scheduler to handle the update step:
        # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        #latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        # Instead, let's do it ourselves:
        prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
        latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt

    # Post-processing
    images = pipe.vae.decode(latents/ pipe.vae.config.scaling_factor,return_dict=False)[0]
    do_denormalize = [True] * images.shape[0]
    image = pipe.image_processor.postprocess(images,do_denormalize=do_denormalize)

    return image

## Inversion
@torch.no_grad()
def invert(
    pipe,
    start_latents,
    input_ids,
    guidance_scale=1,
    num_inference_steps=80,
    num_images_per_prompt=1,
    negative_prompt="",
    device=device,
    controlnet_image=None,
    intervention_indx=None,
    intervention_values=None,
    label=None
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

        (down_block_res_samples, mid_block_res_sample),causal_loss,control_embeddings = pipe.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=input_ids,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                    label = label,
                    training = False,
                    sampling=False,
                    intervention_indx=intervention_indx,
                    intervention_values=intervention_values,
                    text_encoder = pipe.text_encoder
                    
        )
        # Predict the noise residual
        '''think should I use the control residual here?'''
        if 'global' in task_cond:
            encoder_hidden_states = control_embeddings
        # else is use the local embedding for unet
        else:
            encoder_hidden_states = pipe.text_encoder(input_ids)[0].to(dtype=mid_block_res_sample.dtype)
        #encoder_hidden_states = pipe.text_encoder(input_ids)[0].to(dtype=mid_block_res_sample.dtype)
        
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



from torchvision import transforms
from edit_modules.clip import CLIPTextModel
from edit_modules.embed_manager import EmbeddingManager,Embed_control_manager
from diffusers.models.modeling_utils import load_state_dict
size = 256

image_transforms = transforms.Compose(
        [
            transforms.CenterCrop(150),
            transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
        )
original_transforms = transforms.Compose(
        [
            transforms.CenterCrop(150),
            transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR),
        ]
        )

conditioning_image_transforms = transforms.Compose(
            [
                transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                #transforms.Normalize([0.5], [0.5]),
            ]
        )


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


        embed_proj_path  = embedding_path.replace("learned_embeds", "embeds_proj")
            
        if os.path.exists(embed_proj_path):
            embedding_manager = EmbeddingManager(token_ids)
            text_encoder.text_model.embeddings.set_embedding_manager(embedding_manager)
            linear_state_dict = load_state_dict(embed_proj_path)
            embedding_manager.embed_proj.load_state_dict(linear_state_dict)
            embedding_manager.eval()
            print('extend projection')
    

    embed_control=embed_control
    if embed_control:
        embed_control_manager =Embed_control_manager(presudo_token_ids)
        text_encoder.text_model.embeddings.set_embed_control(embed_control_manager)
        print('load embedding control')

    text_encoder.eval()
    return text_encoder