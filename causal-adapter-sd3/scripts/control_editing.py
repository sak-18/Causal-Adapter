import sys
sys.path.append('~/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser/')
from diffusers import StableDiffusionCausalControlNetPipeline, Causal_ControlNetModel, UniPCMultistepScheduler,StableDiffusionPipeline
from diffusers.utils import load_image
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from causal_modules import Causal_SCM_v3

def save_images_grid(images, grid_size, save_path=None):
    """
    Save a list of images in a grid format.

    Parameters:
    - images: numpy array of shape (N, H, W, C) containing the images.
    - grid_size: tuple (grid_rows, grid_cols) for arranging images in the grid.
    - save_path: file path where the grid image will be saved.
    """
    
    images = np.stack([np.asarray(img) for img in images])


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



base_model_path = "/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser/.cache/huggingface/hub/models--sd-legacy--stable-diffusion-v1-5/snapshots/f03de327dd89b501a01da37fc5240cf4fdba85a1"

controlnet_path = "/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser/logs/2024-10-31T16-43-31-mcpl-all_controlv2_initial_mcpl_sampling/controlnet-steps-10000.safetensors/"
mcpl_embedding_path = "/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser/logs/2024-10-31T16-43-31-mcpl-all_controlv2_initial_mcpl_sampling/learned_embeds-steps-10000.safetensors"
controlnet = Causal_ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float32)
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
#c2 = load_image(img_path)
'''from training set'''
#img_path = "/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser/dataset/causal_4_concepts/pendulum/3/a_10_128_3_13.png"
img_path = "/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser/dataset/causal_data/pendulum/test/a_-3_126_4_12.png"
control_image = load_image(img_path)
prompt = "orange @ and red * and black & and black !"

# generate image
generator = torch.manual_seed(0)

embed_name = mcpl_embedding_path.split('/')[-2]
output = "/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser/outputs/control_edit/{}".format(embed_name)
os.makedirs(output, exist_ok=True)
save_path = os.path.join(output,'recons.png')
# save recons with or without controlnet
# recons_image = pipe(
#     [prompt,prompt], num_inference_steps=50, generator=generator, image=[c2,control_image],height=96,width=96,guidance_scale=1,intervention_indx=0,intervention_values=0
# ).images[0]
recons_image = pipe(
    prompt, num_inference_steps=50, generator=generator, image=control_image,height=96,width=96,guidance_scale=1,intervention_indx=0,intervention_values=0
).images[0]
# without_controlnet
without_control_pipe = StableDiffusionPipeline.from_pretrained(base_model_path,torch_dtype=torch.float32).to("cuda")
without_control_pipe.safety_checker = None
without_control_pipe.requires_safety_checker = False
without_control_pipe.load_mcpl_inversion(mcpl_embedding_path)
without_control_images = without_control_pipe(prompt, generator=generator,num_inference_steps=50, guidance_scale=1,num_images_per_prompt=1,height=96,width=96).images[0]

save_images_grid([control_image,without_control_images,recons_image],(1,3),save_path)
print('save imgs in {}'.format(save_path))

# Do intervention
inter_value  = -1.0
inter_id = 2    
image_list = []
range_len = 10
for i in range(range_len):
    
    interved_image = pipe(
    prompt, num_inference_steps=50, generator=generator, image=control_image,height=96,width=96,guidance_scale=1,intervention_indx=inter_id,intervention_values=inter_value
    ).images[0]
    
    image_list.append(interved_image)
    inter_value+=0.15
    #value*=10

save_path = os.path.join(output,'intervention_variable{}.png'.format(inter_id))
save_images_grid(image_list,(1,range_len),save_path)
print('save imgs in {}'.format(save_path))