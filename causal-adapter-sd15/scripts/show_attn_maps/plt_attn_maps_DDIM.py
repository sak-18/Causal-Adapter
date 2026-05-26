import argparse, os

from diffusers import StableDiffusionPipeline,DDIMScheduler,UniPCMultistepScheduler
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import ptp_utils
import seq_aligner
from ptp_tools import *
import argparse, os
from torchvision import transforms
from tqdm import tqdm

# from notebook.services.config import ConfigManager
# cm = ConfigManager().update('notebook', {'limit_output': 10})

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        '--prompts_string', 
        type=str, 
        nargs='+', 
        default=[])
    parser.add_argument(
        '--emb_path_list', 
        type=str, 
        nargs='+', 
        default=[])
    parser.add_argument(
        '--exp_names', 
        type=str, 
        nargs='+', 
        default=[])
    parser.add_argument(
        '--select_clsses', 
        type=str, 
        nargs='+', 
        default=[])
    parser.add_argument(
        "--ckpt_path",
        type=str,
        const=True,
        default='${PROJECT_ROOT}/.cache/huggingface/hub/models--sd-legacy--stable-diffusion-v1-5/snapshots/f03de327dd89b501a01da37fc5240cf4fdba85a1',
        nargs="?",
        help="pretrained model path",
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        const=True,
        default='/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL/logs',
        nargs="?",
        help="embedding_path",
    )
    parser.add_argument(
        "--out_base",
        type=str,
        const=True,
        default='/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL/outputs/attention_maps/',
        nargs="?",
        help="out_base",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        const=True,
        default='lge_cmr_mix_simple_MC_v2_allPsudo',
        nargs="?",
        help="exp_name",
    )
    parser.add_argument(
        "--inf_config",
        type=str,
        const=True,
        default="/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL/configs/latent-diffusion/txt2img-1p4B-finetune.yaml",
        nargs="?",
        help="inference config",
    )
    parser.add_argument("--num_diff_steps",
        type=int,
        default=100,
        help="num_diff_steps",
    )
    parser.add_argument("--manual_seed",
        type=int,
        default=0,
        help="manual_seed",
    )
    parser.add_argument("--guidance_scale",
        type=float,
        default=5.0,
        help="guidance_scale",
    )
    parser.add_argument("--attn_threshold",
        type=float,
        default=0.5,
        help="attn_threshold",
    )
    parser.add_argument("--input_image",
        type=str,
        default='${PROJECT_ROOT}/dataset/causal_4_concepts/pendulum/3/a_-35_71_3_4.png',
        help="image path",
    )
    parser.add_argument("--img_size",
        type=int,
        default=512,
        help="img_size",
    )
    parser.add_argument(
        "--save_jpeg", 
        type=str2bool, 
        nargs="?", 
        const=False, 
        default=False, 
        help="Optional save as jpeg to save memory, default save as png")

    return parser


## Inversion
@torch.no_grad()
def invert(
    pipe,
    start_latents,
    prompt,
    guidance_scale=3.5,
    num_inference_steps=80,
    num_images_per_prompt=1,
    negative_prompt="",
    device=None,
):
    
    if guidance_scale>1:
        do_classifier_free_guidance = True
    else:
        do_classifier_free_guidance= False    
    # Encode prompt
    text_embeddings,negtive_prompt_embedding = pipe.encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )
    if do_classifier_free_guidance:
        text_embeddings = torch.cat([negtive_prompt_embedding, text_embeddings])
    # Latents are now the specified start latents
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

        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings, return_dict=False)[0]

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





if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    NUM_DIFFUSION_STEPS = opt.num_diff_steps
    GUIDANCE_SCALE = opt.guidance_scale
    MAX_NUM_WORDS = 77
    
    modify_global_var(value=opt.img_size)
    print_global_var()
    

    model_id = opt.ckpt_path
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config
    )

    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    embedding_path =opt.embedding_path
    pipe.load_mcpl_inversion(embedding_path)

    tokenizer = pipe.tokenizer

    if opt.manual_seed != 0:
        g_gpu = torch.Generator().manual_seed(opt.manual_seed)
    else:
        g_gpu = torch.Generator()
    
    prompts = opt.prompts_string
    controller = AttentionStore()

    if not opt.save_jpeg:
        suffix = '.png'
    else:
        suffix = '.jpg'

    if not os.path.exists(opt.out_base):
        os.mkdir(opt.out_base) 
    out_path_base = os.path.join(opt.out_base, opt.exp_name)
    if not os.path.exists(out_path_base):
        os.mkdir(out_path_base) 
    out_path_prompt = os.path.join(out_path_base, prompts[0])
    if not os.path.exists(out_path_prompt):
        os.mkdir(out_path_prompt) 
    if opt.img_size ==96:
        # (9,36,144) for 96 size heatmap
        res = 6
    else:
        res = 16
    

    # run ours (CL / AttnMask)
    # unconditional generation with ours
    out_dir = out_path_prompt
    out_name = 'ours_CL_mask_attn_unconditioned'+suffix
    emb_path_list = [emb_path for emb_path in opt.emb_path_list] 
    exp_names = opt.exp_names
    print(f'DEBUG: emb_path_list = {emb_path_list}')
    print(f'DEBUG: exp_names = {exp_names}')
    
    # plot_img_attn_mask(pipe,tokenizer, prompts, emb_path_list, exp_names, \
    #     device, out_dir, out_name, latent=None,res=res, \
    #      GUIDANCE_SCALE=GUIDANCE_SCALE, \
    #     attn_threshold=opt.attn_threshold, select_clsses = opt.select_clsses, mask_concepts=True, g_gpu=g_gpu)
    # print('Unconditional generation done!')
    
    # #conditional generation based on baseline
    # out_name = 'ours_CL_mask_attn_conditioned'+suffix
    # plot_img_attn_mask(ldm, prompts, emb_path_list, exp_names, \
    #     device, out_dir, out_name, config, latent=x_t, \
    #     array_latent=True, GUIDANCE_SCALE=GUIDANCE_SCALE, \
    #     attn_threshold=opt.attn_threshold, select_clsses = opt.select_clsses, g_gpu=g_gpu)
    # print('Conditional generation done!')

    #todo: implement bewlow apply mask for each selected prompt, save each masked image, 
    # and in the plot, add each masked concept and selected key word
    
    image = Image.open(opt.input_image)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    condition_image = image.copy()
    image_transforms = transforms.Compose(
        [
            transforms.Resize((opt.img_size,opt.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
        )
    image = image_transforms(image) 

    with torch.no_grad():
        latent = pipe.vae.encode(image.unsqueeze(0).to(device))
    img_latent = 0.18215 * latent.latent_dist.sample()
    num_steps = 50
    device = torch.device("cuda")
    inverted_latents = invert(pipe,
        img_latent,
        prompts,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=num_steps,
        num_images_per_prompt=1,
        negative_prompt="",
        device=device,
    )

    out_name = 'ours_CL_mask_attn_conditioned_masked_concepts'+suffix
    plot_img_attn_mask(pipe,tokenizer, prompts, emb_path_list, exp_names, \
        device, out_dir, out_name, latent=inverted_latents[-(0+ 1)][None],res=res, \
         GUIDANCE_SCALE=GUIDANCE_SCALE, \
        attn_threshold=opt.attn_threshold, select_clsses = opt.select_clsses, mask_concepts=True, g_gpu=g_gpu,num_steps=num_steps)
    print('Conditional generation mask key words done!')

    # out_name = 'simple_mask_attn_conditioned'+suffix
    # plot_img_mask(ldm, prompts, emb_path_list, exp_names, \
    #     device, out_dir, out_name, config, latent=x_t, \
    #     array_latent=True, GUIDANCE_SCALE=GUIDANCE_SCALE, \
    #     attn_threshold=opt.attn_threshold, select_clsses = opt.select_clsses, mask_concepts=True, g_gpu=g_gpu)
    # print('Conditional simple attention mask done!')

    # # conditional generation based on baseline without text
    # out_name = 'ours_CL_mask_attn_conditioned_no_text'+suffix
    # plot_img_attn_mask(ldm, prompts, emb_path_list, exp_names, \
    #     device, out_dir, out_name, config, latent=x_t, \
    #     array_latent=True, GUIDANCE_SCALE=GUIDANCE_SCALE, \
    #     attn_threshold=opt.attn_threshold, select_clsses = opt.select_clsses, show_text=False, g_gpu=g_gpu)
    # print('Conditional generation (no text) done!')

    # out_name = 'ours_CL_mask_attn_conditioned_masked_concepts_no_text'+suffix
    # plot_img_attn_mask(ldm, prompts, emb_path_list, exp_names, \
    #     device, out_dir, out_name, config, latent=x_t, \
    #     array_latent=True, GUIDANCE_SCALE=GUIDANCE_SCALE, \
    #     attn_threshold=opt.attn_threshold, select_clsses = opt.select_clsses, \
    #     show_text=False, mask_concepts=True, g_gpu=g_gpu)
    # print('Conditional generation mask key words (no text) done!')

    print('All inference done!')