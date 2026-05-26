#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
from accelerate.utils import write_basic_config
write_basic_config()

import argparse
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
from torchvision.datasets import CelebA
import pytorch_lightning.loggers
import numpy as np
import PIL
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
import accelerate
import itertools
# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTokenizer
from edit_modules.clip import CLIPTextModel
from edit_modules.embed_manager import EmbeddingManager
from edit_modules.TI_datasets_discovery import TextualInversionDataset
import datetime
import diffusers
import pandas as pd
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    Causal_ControlNetModel,
    StableDiffusionCausalControlNetPipeline,
    UniPCMultistepScheduler
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
import matplotlib.pyplot as plt
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.models.modeling_utils import load_state_dict
from diffusers.utils import load_image

if is_wandb_available():
    import wandb


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

logger = get_logger(__name__)

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def save_images_grid(adj_matrix, save_path):
    np.savetxt(save_path, adj_matrix, delimiter=',')



def save_model_card(repo_id: str, images: list = None, base_model: str = None, repo_folder: str = None):
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"
    model_description = f"""
# Textual inversion text2image fine-tuning - {repo_id}
These are textual inversion adaption weights for {base_model}. You can find some example images in the following. \n
{img_str}
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "text-to-image",
        "diffusers",
        "textual_inversion",
        "diffusers-training",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(controlnet, text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, epoch):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    # create pipeline (note: unet and vae are loaded again in float32)
    controlnet.eval()
    if args.mcpl_training:
        text_encoder.eval()
    
    import torch.nn.parallel

    # Check if controlnet is using DistributedDataParallel
    if isinstance(controlnet, torch.nn.parallel.DistributedDataParallel):
        model = controlnet.module  # Get the actual model inside DDP
    else:
        model = controlnet  # If not DDP, use it directly

    # Ensure model is in evaluation mode
    model.controlnet_cond_embedding.eval()

    # Compute causal matrix
    causal_matrix = model.controlnet_cond_embedding.fc1_to_adj()

    
    return causal_matrix


def save_progress(text_encoder, presudo_token_ids, accelerator, tokenizer, save_path, safe_serialization=True):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[presudo_token_ids]
    )
    tokens = tokenizer.convert_ids_to_tokens(presudo_token_ids)
    learned_embeds_dict = {}
    for token, embed in zip(tokens,learned_embeds):
        learned_embeds_dict[token] = embed.detach().cpu()
        
    #learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}
    em_manager = accelerator.unwrap_model(text_encoder).text_model.embeddings.embedding_manager
    if em_manager is not None:
        embed_proj_weights = em_manager.embed_proj.state_dict()
        embed_proj_save_path = save_path.replace("learned_embeds", "embeds_proj")
        if safe_serialization:
            safetensors.torch.save_file(embed_proj_weights, embed_proj_save_path, metadata={"format": "pt"})
        else:
            torch.save(embed_proj_weights, embed_proj_save_path)


    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5000,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--save_as_full_pipeline",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=1,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='pendulum',
        help="dataset",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, required=True, help="A folder containing the training data."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=False,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default=None, required=False, help="A token to use as initializer word."
    )
    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_name",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./logs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_false", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and Nvidia Ampere GPU or Intel Gen 4 Xeon (and later) ."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=2500,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=None,
        help=(
            "Deprecated in favor of validation_steps. Run validation every X epochs. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
    )
    # Parameters from MCPL
    parser.add_argument("--placeholder_string", 
        type=str, 
        help="MCPL: Placeholder string which will be used to denote the string holding multiple concepts. Overwrites the config options.")

    parser.add_argument("--presudo_words", 
        type=str, 
        help="MCPL: A list of presudo words corresponding to multiple concepts.")
    
    parser.add_argument("--presudo_words_infonce", 
        type=str, 
        default="", 
        help="PromptCL: A list of presudo words (semantic mutual exclusive) to calculate additional CL (infoNCE) loss")

    parser.add_argument("--adj_aug_infonce", 
        type=str, 
        default="", 
        help="Bind adjective: A list of adj. words to be treated as additional agumented positive of presudo_words_infonce in CL loss")

    parser.add_argument("--infonce_temperature",
        type=float,
        default=0.2,
        help="PromptCL: infonce_temperature",
    )

    parser.add_argument("--infonce_scale",
        type=float,
        default=0.0005,
        help="PromptCL: infonce_scale",
    )
    
    parser.add_argument("--presudo_words_softmax", 
        type=str, 
        default="", 
        help="PromptCL: A list of presudo words to calculate additional softmax with, default means no additional softmax")

    parser.add_argument("--attn_words", 
        type=str, 
        help="Attention Mask: A list of keywords for attention masking.")

    parser.add_argument("--attn_mask_type", 
        type=str, 
        default="hard", 
        help="Attention Mask: Type of attention mask, choose from 'hard: apply threhold', 'soft: no threshold' or 'skip: no mask (cause we want to keep controller for CL)'")

    parser.add_argument("--mcpl_training", 
        type=str2bool,
        default=False,
        help="enable MCPL training?")
    parser.add_argument("--mcpl_embedding_path", 
        type=str,
        default=None,
        help="load MCPL embeddings")
    parser.add_argument("--causal_training", 
        type=str2bool,
        default=False,
        help="enable causal intervention training?")
    parser.add_argument("--causalnet_path", 
        type=str,
        default=None,
        help="load causalnet")
    parser.add_argument("--task_cond", 
        type=str,
        default='generation_text_global',
        help="'discovery_image','discovery_text_local','discovery_text_global','generation_image','generation_text_local','generation_text_global','generation_text_global_after','generation_text_local_after'")
    parser.add_argument("--random_prompt_template", 
        type=str2bool,
        default=False,
        help="random_prompt_template,a photo of?")


    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args





def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])
    label = torch.stack([example["label"] for example in examples])
    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
        "label":label
    }


def main():
    args = parse_args()
    args.push_to_hub = False
    args.validation_prompt = args.placeholder_string
    print(args)
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-")
    args.output_dir = os.path.join(args.output_dir,now+args.output_name)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        # not run
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )
    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = Causal_ControlNetModel().from_pretrained(args.controlnet_model_name_or_path)
        #controlnet.controlnet_cond_embedding.set_A_martix(torch.eye((4)))
    else:
        logger.info("Initializing controlnet weights from unet")
        if args.dataset in ['celeA_simple']:
            num_causal_concepts = 2
        elif args.dataset in ['chexpert','skin_cancer','MorphoMNIST']:
            num_causal_concepts = 3
        elif args.dataset in ['pendulum','celeA_complex']:
            num_causal_concepts = 4
        elif args.dataset in ['ADNI']:
            num_causal_concepts = 6


        args.num_causal_concepts = num_causal_concepts
        controlnet = Causal_ControlNetModel().from_unet(unet,task_cond= 'discovery_image',num_causal_concepts=num_causal_concepts,dataset=args.dataset)

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Add the placeholder token in tokenizer
    # use presudo words to create embeddings
    # take care the tokenizer with return different id like word and word(/w)
    presudo_list = args.presudo_words.split(',')
    presudo_token_ids = tokenizer.encode(' '.join(presudo_list), add_special_tokens=False)
    #presudo_token_ids = tokenizer.encode(presudo_list, add_special_tokens=False)
    # If some word were forcely convert to end token
    # initilizse it as randn
    if 49407 in presudo_token_ids:
        unknown_words = [presudo_list[i] for i, x in enumerate(presudo_token_ids) if x == 49407]

        num_added_tokens = tokenizer.add_tokens(unknown_words)

        # # Convert the initializer_token, placeholder_token to ids 
        # # if add_speicial_tokens = True, add two token in the head and end[49406,word_token,49407],   "endtoken</w>": 40497,
        # # Will not split word, if word is unseen, forcely convert it to 494907
        # token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
        # # Check if initializer_token is a single token or a sequence of tokens
        # if len(token_ids) > 1:
        #     raise ValueError("The initializer token must be a single token.")
        #initializer_token_id = token_ids[0]
        unknown_token_ids = tokenizer.convert_tokens_to_ids(unknown_words)

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        # here, the unknow words has been intiilized 
        text_encoder.resize_token_embeddings(len(tokenizer))
        presudo_token_ids = tokenizer.encode(presudo_list, add_special_tokens=False)
        assert 49407 not in presudo_token_ids
        # Initialise the newly added placeholder token with the embeddings of the initializer token
        # token_embeds = text_encoder.get_input_embeddings().weight.data
        # with torch.no_grad():
        #     for token_id in unknown_token_ids:
        #         token_embeds[token_id] = token_embeds[initializer_token_id].clone()
    # string or list[str] get different token id 
    presudo_token_ids = tokenizer.encode(' '.join(presudo_list), add_special_tokens=False)

    concept_id = []

    for c_word in args.concept_words.split(','):
        t_id = presudo_list.index(c_word)
        concept_id.append(presudo_token_ids[t_id])

    mcpl_linear=False
    if args.mcpl_embedding_path is not None:
        embedding_path = args.mcpl_embedding_path
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
            mcpl_linear=True


            
        

    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    # Freeze all parameters except for the token embeddings in text encoder (text_encoder.text_model.embeddings.token_embedding)
    if args.mcpl_training:
        print('enable MCPL embedding training')
        text_encoder.text_model.encoder.requires_grad_(False)
        text_encoder.text_model.final_layer_norm.requires_grad_(False)
        text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
        trainable_params  = list(text_encoder.get_input_embeddings().parameters()) + list(controlnet.parameters())
        if mcpl_linear:
            for param in embedding_manager.embed_proj.parameters():
                param.requires_grad = True
            trainable_params += list(embedding_manager.embed_proj.parameters())
    else:
        text_encoder.requires_grad_(False)
        trainable_params  = list(controlnet.parameters())

        if mcpl_linear:
            for param in embedding_manager.embed_proj.parameters():
                param.requires_grad = False

    

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    # trainable_params  = list(text_encoder.get_input_embeddings().parameters()) + list(controlnet.parameters())
    # Initialize the optimizer
    print("Trainable Param: ",sum(p.numel() for p in trainable_params if p.requires_grad))
    optimizer = torch.optim.AdamW(
        trainable_params,  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=args.placeholder_string,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=False,
        flip_p=0.0,
        random_article=False,
        set="train",
        dataset =args.dataset
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers,collate_fn=collate_fn
    )
    if args.validation_epochs is not None:
        warnings.warn(
            f"FutureWarning: You are doing logging with validation_epochs={args.validation_epochs}."
            " Deprecated validation_epochs in favor of `validation_steps`"
            f"Setting `args.validation_steps` to {args.validation_epochs * len(train_dataset)}",
            FutureWarning,
            stacklevel=2,
        )
        args.validation_steps = args.validation_epochs * len(train_dataset)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
    )



    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # if args.dataset in ['pendulum','celeA']:
    #     A_matrix = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],dtype=weight_dtype).to(accelerator.device)
    # elif args.dataset in ['chexpert','skin_cancer']:
    #     A_matrix = torch.tensor([[0, 0, 1], [0, 0, 1], [0, 0, 0]],dtype=weight_dtype).to(accelerator.device)
    # controlnet.controlnet_cond_embedding.set_A_martix(A_matrix)
    controlnet.train()
    if args.mcpl_training:
        text_encoder.train()
        # Prepare everything with our `accelerator`.
        controlnet,text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            controlnet,text_encoder, optimizer, train_dataloader, lr_scheduler
        )
        accelerate_models = [controlnet,text_encoder]
    else:
        text_encoder.eval()
        controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            controlnet, optimizer, train_dataloader, lr_scheduler
        )
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        accelerate_models = [controlnet]

    # Move vae and unet to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # keep original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
    
    gammas = np.linspace(0, 1, 10)
    gamma = 1
    gamma_index = 0

    lamba_init,lamba_factor = 1,0.1

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(accelerate_models):
                # Convert images to latent space
                #controlnet_image  = batch["pixel_values"].clone().to(dtype=weight_dtype)
                controlnet.train()
                if args.mcpl_training:
                    text_encoder.train()
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)
                encoded_token_ids = batch["input_ids"][0]
                # Find the index of each element in first_list within second_list
                embed_idx = torch.tensor([torch.where(encoded_token_ids == x)[0] for x in concept_id])
                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                # controlnet output
                (down_block_res_samples, mid_block_res_sample),causal_loss = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                    label = batch['label'].to(dtype=weight_dtype),
                    embed_idx = embed_idx,
                    training = True,
                )

                # Predict the noise residual #.sample just output of unet, not sampling
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]
                # early stop
                # if global_step>1000 and causal_loss<=1e-8:
                #     if isinstance(controlnet, torch.nn.parallel.DistributedDataParallel):
                #         model = controlnet.module  # Get the actual model inside DDP
                #     else:
                #         model = controlnet  # If not DDP, use it directly
                #     # Compute causal matrix
                #     causal_matrix = model.controlnet_cond_embedding.fc1_to_adj()
                #     image_path = os.path.join(args.output_dir, f"CM-{global_step}-early.csv")
                #     save_images_grid(causal_matrix,image_path)


                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = lamba_init*F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                mse_loss_log = loss.detach().item()
                causal_loss_log = causal_loss.detach().item()

                

                #gamma = gammas[gamma_index]  # Select gamma from predefined list

                # Apply gamma to causal loss
                if global_step >= args.net_warmup_steps:
                    loss += gamma * causal_loss.mean()
                    
                else:
                    causal_loss = 0

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # Update gamma dynamically after warmup and every 500 global steps
                    if global_step >= args.net_warmup_steps and global_step>0 and global_step % 500 == 0:
                        gamma_index = min(gamma_index + 1, len(gammas) - 1)  # Ensure index does not exceed list size
                    if global_step >= args.net_warmup_steps and global_step>0 and global_step % 3000 == 0:
                        pass
                        #lamba_init= lamba_init*lamba_factor
                    # params_to_clip = (
                    #     itertools.chain(controlnet.parameters(), text_encoder.get_input_embeddings().parameters())
                    #     if args.mcpl_training
                    #     else controlnet.parameters()
                    # )
                    # # if len(list(params_to_clip))>0:
                    # #     print('Clip grad: ',list(params_to_clip))
                    # accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    params_to_clip = list(controlnet.parameters())
                    if args.mcpl_training:
                        if isinstance(text_encoder, torch.nn.parallel.DistributedDataParallel):
                            text_encoder_module = text_encoder.module  # Get the actual model inside DDP
                        else:
                            text_encoder_module = text_encoder  # If not DDP, use it directly
                        params_to_clip+=list(text_encoder_module.get_input_embeddings().parameters())
                        if mcpl_linear:
                            params_to_clip+=list(embedding_manager.embed_proj.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Let's make sure we don't update any embedding weights besides the newly added token
                index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                index_no_updates[presudo_token_ids] = False

                with torch.no_grad():
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates]

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                images = []
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    
                    if global_step == args.net_warmup_steps:
                        weight_name = (
                            f"learned_embeds-steps-wp{global_step}.bin"
                            if args.no_safe_serialization
                            else f"learned_embeds-steps-wp{global_step}.safetensors"
                        )
                        save_path = os.path.join(args.output_dir, weight_name)
                        save_progress(
                            text_encoder,
                            presudo_token_ids,
                            accelerator,
                            tokenizer,
                            save_path,
                            safe_serialization=not args.no_safe_serialization,
                        )
                        

                        controlnet_name = (
                            f"controlnet-steps-wp{global_step}.bin"
                            if args.no_safe_serialization
                            else f"controlnet-steps-wp{global_step}.safetensors"
                        )
                        net_save_path = os.path.join(args.output_dir, controlnet_name)
                        controlnet_unwrap = unwrap_model(controlnet)
                        controlnet_unwrap.save_pretrained(net_save_path)


                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:

                        concept_embeddings = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[concept_id]
                        causal_matrix = log_validation(
                            controlnet,text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, epoch,concept_embeddings
                        )
                        image_path = os.path.join(args.output_dir, f"CM-{global_step}.csv")
                        save_images_grid(causal_matrix,image_path)

            logs = {"recons_loss": mse_loss_log, "causal_loss":causal_loss_log,"lr": lr_scheduler.get_last_lr()[0],"gamma":gamma,"lamba":lamba_init}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            # Open the file in append mode
            with open(os.path.join(args.output_dir,'logs.txt'), "a") as log_file:
                log_file.write(f"Step {global_step}: {logs}\n")

            if global_step >= args.max_train_steps:
                break
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.push_to_hub and not args.save_as_full_pipeline:
            logger.warning("Enabling full model saving because --push_to_hub=True was specified.")
            save_full_model = True
        else:
            save_full_model = args.save_as_full_pipeline
        if save_full_model:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
            )
            pipeline.save_pretrained(args.output_dir)
        # Save the newly trained embeddings
        weight_name = "learned_embeds.bin" if args.no_safe_serialization else "learned_embeds.safetensors"
        save_path = os.path.join(args.output_dir, weight_name)
        save_progress(
            text_encoder,
            presudo_token_ids,
            accelerator,
            tokenizer,
            save_path,
            safe_serialization=not args.no_safe_serialization,
        )

        controlnet_name = (
                        f"controlnet.bin"
                        if args.no_safe_serialization
                        else f"controlnet.safetensors"
                    )
        net_save_path = os.path.join(args.output_dir, controlnet_name)
        controlnet_unwrap = unwrap_model(controlnet)
        controlnet_unwrap.save_pretrained(net_save_path)

        

    accelerator.end_training()


if __name__ == "__main__":
    main()
