from diffusers import StableDiffusionPipeline,DDIMScheduler,UniPCMultistepScheduler
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

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


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


model_id = os.getenv("BASE_MODEL_PATH", "stable-diffusion-v1-5/stable-diffusion-v1-5")
pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(
    pipe.scheduler.config
)
pipe.safety_checker = None
pipe.requires_safety_checker = False

repo_id_embeds = _require_env("MCPL_EMBEDDING_PATH")
pipe.load_mcpl_inversion(repo_id_embeds)


embed_name = repo_id_embeds.split('/')[-2]
output_base = os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "outputs" / "txt2img"))
output = os.path.join(output_base, embed_name)
os.makedirs(output, exist_ok=True)
prompt = "a photo of orange @ and red * and black & and black !"
save_path = os.path.join(output,prompt+'.png')

images = pipe(prompt, num_inference_steps=50, guidance_scale=1,num_images_per_prompt=8,height=96,width=96).images
save_images_grid(np.stack([np.asarray(img) for img in images]),(1,8),save_path)