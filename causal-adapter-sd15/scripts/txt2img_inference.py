from diffusers import StableDiffusionPipeline,DDIMScheduler,UniPCMultistepScheduler
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def save_images_grid(images, grid_size, save_path=None):
    """
    Save a list of images in a grid format.

    Parameters:
    - images: numpy array of shape (N, H, W, C) containing the images.
    - grid_size: tuple (grid_rows, grid_cols) for arranging images in the grid.
    - save_path: file path where the grid image will be saved.
    """
    N, H, W, C = images.shape
    grid_rows, grid_cols = grid_size

    # Check if the grid can fit all images
    assert grid_rows * grid_cols >= N, "Grid size is too small for the number of images"

    # Create an empty array for the grid image
    grid_image = np.zeros((grid_rows * H, grid_cols * W, C), dtype=images.dtype)

    # Fill the grid with images
    for idx, img in enumerate(images):
        row = idx // grid_cols
        col = idx % grid_cols
        grid_image[row * H:(row + 1) * H, col * W:(col + 1) * W, :] = img

    # Plot and save the grid image
    plt.figure(figsize=(grid_cols, grid_rows))
    plt.imshow(grid_image)
    plt.axis('off')  # Turn off axis labels
    plt.savefig(save_path)
    plt.close()


model_id = "/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser/.cache/huggingface/hub/models--sd-legacy--stable-diffusion-v1-5/snapshots/f03de327dd89b501a01da37fc5240cf4fdba85a1"
pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(
    pipe.scheduler.config
)
pipe.safety_checker = None
pipe.requires_safety_checker = False

repo_id_embeds = "/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser/logs/2024-10-11T10-58-52-mcpl-all/learned_embeds-steps-7000.safetensors"
pipe.load_mcpl_inversion(repo_id_embeds)


embed_name = repo_id_embeds.split('/')[-2]
output = "/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser/outputs/txt2img/{}".format(embed_name)
os.makedirs(output, exist_ok=True)
prompt = "a photo of orange @ and red * and black & and black !"
save_path = os.path.join(output,prompt+'.png')

images = pipe(prompt, num_inference_steps=50, guidance_scale=1,num_images_per_prompt=8,height=96,width=96).images
save_images_grid(np.stack([np.asarray(img) for img in images]),(1,8),save_path)