import argparse, os
import sys
sys.path.append('/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser')
from diffusers import StableDiffusionCausalControlNetPipeline, Causal_ControlNetModel, UniPCMultistepScheduler,StableDiffusionPipeline
from diffusers.utils import load_image
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from diffusers import DDIMInverseScheduler, DDIMScheduler
from tqdm import tqdm
from torchvision import transforms
from edit_modules.clip import CLIPTextModel
from edit_modules.embed_manager import EmbeddingManager,Embed_control_manager
from diffusers.models.modeling_utils import load_state_dict
'''Load pipeline'''
from torchvision import transforms as tfms
from PIL import Image
from transformers import CLIPTokenizer
from ptp_tools import *
# from notebook.services.config import ConfigManager
# cm = ConfigManager().update('notebook', {'limit_output': 10})


def load_mcpl_embeddings(base_model_path,tokenizer,embedding_path=None,presudo_token_ids=None):
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
    text_encoder.eval()

    embed_control=True
    if embed_control:
        embed_control =Embed_control_manager(presudo_token_ids)
        text_encoder.text_model.embeddings.set_embed_control(embed_control)
    return text_encoder


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
        "--ckpt_path",
        type=str,
        const=True,
        default='/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser/.cache/huggingface/hub/models--sd-legacy--stable-diffusion-v1-5/snapshots/f03de327dd89b501a01da37fc5240cf4fdba85a1',
        nargs="?",
        help="pretrained model path",
    )
    parser.add_argument(
        "--controlnet_path",
        type=str,
        const=True,
        default=None,
        nargs="?",
        help="controlnet_path",
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        const=True,
        default=None,
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
    parser.add_argument("--presudo_words", 
        type=str, 
        help="MCPL: A list of presudo words corresponding to multiple concepts.")
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
        default='/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser/dataset/causal_4_concepts/pendulum/3/a_-35_71_3_4.png',
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



if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    NUM_DIFFUSION_STEPS = opt.num_diff_steps
    GUIDANCE_SCALE = opt.guidance_scale
    MAX_NUM_WORDS = 77
    
    modify_global_var(value=opt.img_size)
    print_global_var()
    
    from torchvision import transforms as tfms
    from PIL import Image
    from transformers import CLIPTokenizer

    model_id = opt.ckpt_path
    
    
    mcpl_embedding_path = opt.embedding_path
    tokenizer = CLIPTokenizer.from_pretrained(model_id,subfolder="tokenizer")
    controlnet = Causal_ControlNetModel.from_pretrained(opt.controlnet_path,torch_dtype=torch.float32)
    A_matrix = torch.tensor([[0, 0, 1,1], [0, 0, 1,1], [0, 0, 0,0], [0, 0, 0,0]],dtype=torch.float32).to(device)
    controlnet.controlnet_cond_embedding.set_A_martix(A_matrix)
    controlnet.eval()
    
    presudo_list = opt.presudo_words.split(',')
    presudo_token_ids = tokenizer.encode(' '.join(presudo_list), add_special_tokens=False)
    text_encoder = load_mcpl_embeddings(model_id,tokenizer,mcpl_embedding_path,presudo_token_ids)
    #cover the new text_encode into the pipeline, otherwise load the original weight
    pipe = StableDiffusionCausalControlNetPipeline.from_pretrained(
        model_id, controlnet=controlnet,text_encoder=text_encoder ,torch_dtype=torch.float32
    )
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config
    )
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe = pipe.to(device)

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
    
    
    image = Image.open(opt.input_image)
    if not image.mode == "RGB":
        image = image.convert("RGB")

    original_img = image.copy()
    original_img = transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR)(original_img)
    condition_image = image.copy()
    image_transforms = transforms.Compose(
        [
            transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
            [
                transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                #transforms.Normalize([0.5], [0.5]),
            ]
        )
    image = image_transforms(image) 
    condition_image = conditioning_image_transforms(condition_image)

    with torch.no_grad():
        latent = pipe.vae.encode(image.unsqueeze(0).to(device))
    img_latent = 0.18215 * latent.latent_dist.sample()
    num_steps = 50
    device = torch.device("cuda")


    out_dir = out_path_prompt
    out_name = 'causalnet'+suffix
    plot_img_attn_mask_textcontrol(pipe, prompts, opt.presudo_words,condition_image, \
        device, out_dir, out_name, latent=img_latent,res=res, \
        GUIDANCE_SCALE=1,attn_threshold=opt.attn_threshold, mask_concepts=True, g_gpu=g_gpu,num_steps=num_steps,img_size=opt.img_size)
    print('Conditional generation mask key words done!')


    print('All inference done!')