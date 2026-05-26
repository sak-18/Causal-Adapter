import sys
from pathlib import Path
import os

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.append(str(PROJECT_ROOT))
from diffusers import StableDiffusionCausalControlNetPipeline, Causal_ControlNetModel, UniPCMultistepScheduler,StableDiffusionPipeline
from diffusers.utils import load_image
import torch
import numpy as np
import matplotlib.pyplot as plt

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
    plt.figure(figsize=(grid_cols, grid_rows))
    plt.imshow(grid_image)
    plt.axis('off')  # Turn off axis labels
    plt.savefig(save_path)
    plt.close()

def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


base_model_path = os.getenv("BASE_MODEL_PATH", "stable-diffusion-v1-5/stable-diffusion-v1-5")
controlnet_path = _require_env("CONTROLNET_PATH")
mcpl_embedding_path = _require_env("MCPL_EMBEDDING_PATH")
controlnet = Causal_ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float32)
controlnet.eval()
pipe = StableDiffusionCausalControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float32
)
pipe.safety_checker = None
pipe.requires_safety_checker = False
pipe.load_mcpl_inversion(mcpl_embedding_path)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
# memory optimization.
pipe.enable_model_cpu_offload()

'''from test set'''
#img_path = "/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser/dataset/causal_data/pendulum/test/a_9_69_6_7.png"
#img_path = "/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser/dataset/causal_data/pendulum/test/a_-26_147_11_14.png"
#c2 = load_image(img_path)
'''from training set'''
img_path = _require_env("CONTROL_IMAGE_PATH")
#img_path = "/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser/dataset/causal_data/pendulum/test/a_-3_126_4_12.png"
#img_path = random_image_path()
control_image = load_image(img_path)
prompt = "orange @ and red * and black & and black !"

# generate image
generator = torch.manual_seed(0)

embed_name = mcpl_embedding_path.split('/')[-2]
output_base = os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "outputs" / "control_sampling"))
output = os.path.join(output_base, embed_name)
os.makedirs(output, exist_ok=True)
save_path = os.path.join(output,'recons.png')
# save recons with or without controlnet
# recons_image = pipe(
#     [prompt,prompt], num_inference_steps=50, generator=generator, image=[c2,control_image],height=96,width=96,guidance_scale=1,intervention_indx=0,intervention_values=0
# ).images[0]
# recons_image = pipe(
#     prompt, num_inference_steps=50, generator=generator, image=control_image,height=96,width=96,guidance_scale=3,intervention_indx=None,intervention_values=None
# ).images[0]

# Perform Sample, txt2img

# Do intervention
image_lists = []
range_len = 20
images = []

for i in range(range_len): 
    interved_image = pipe(
    prompt, num_inference_steps=50, generator=generator, image=control_image,height=96,width=96,guidance_scale=3,sampling=True,intervention_indx=None,intervention_values=None
    ).images[0]
    
    images.append(interved_image)


image_lists.append([np.asarray(img) for img in images])

save_path = os.path.join(output,'sampling.png')
save_images_grid(image_lists,(1,range_len),save_path)
print('save imgs in {}'.format(save_path))
