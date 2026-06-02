#!/usr/bin/env python
# coding=utf-8
# Copyright AstraZeneca UK Ltd. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Causal-Adapter (Stable Diffusion 3) training entry point.

Trains the SD3 causal ControlNet head together with MCPL pseudo-token
embeddings on top of a frozen Stable Diffusion 3 backbone. The script is
invoked through ``accelerate`` (see ``scripts/adapter_training/train_sd3_celebahq.sh``).

Public CLI surface (everything else has a sane default):

* ``--pretrained_model_name_or_path``: HF model id *or* local path to the
  Stable Diffusion 3 checkpoint to load.
* ``--train_data_dir``: dataset root, required.
* ``--dataset``: one of ``celebahq_simple``, ``celeA_complex``,
  ``pendulum``, ``ADNI``, ``MorphoMNIST``, ... (any key registered in
  ``causal_datasets/_adapters/__init__.py``).
* ``--placeholder_string`` / ``--presudo_words`` / ``--presudo_words_infonce``:
  MCPL pseudo-token configuration.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import copy
import datetime
import itertools
import logging
import math
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
import transformers
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast, CLIPTextModel, T5EncoderModel

import diffusers
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from diffusers.models.controlnets.controlnet_sd3_causal import Causal_SD3ControlNetModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines import StableDiffusion3InpaintPipeline_Adapter
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from causal_datasets import TextualInversionDataset
from causal_modules.ddim_modules_sd3 import tokenize_prompt, encode_prompt_pai

if is_wandb_available():
    import wandb  # noqa: F401  (imported for side effect / availability check)


if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }


# Errors out if the locally-installed diffusers is older than required.
check_min_version("0.31.0.dev0")

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers (text encoder loading is SD3-specific: CLIP1 + CLIP2 + T5)
# ---------------------------------------------------------------------------

# Copied from dreambooth sd3 example
def load_text_encoders(class_one, class_two, class_three, args):
    """Load the three SD3 text encoders (CLIP-1, CLIP-2, T5) at once."""
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


# Copied from dreambooth sd3 example
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    """Resolve the HF text-encoder class to instantiate from model config."""
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_images_grid(observed_img, images_list, grid_size, save_path):
    """
    Save a list of lists of images in a grid format with `observed_img` in the top-left corner.

    Parameters:
    - observed_img: The main image to be placed at the top-left corner of the grid.
    - images_list: List of lists containing numpy images (4x4).
    - grid_size: tuple (grid_rows, grid_cols) for arranging images in the grid.
    - save_path: file path where the grid image will be saved.
    """
    # Flatten the list of lists into a single list of images
    images = [img for sublist in images_list for img in sublist]

    # Determine the dimensions of the first image to set up the grid
    H, W, C = images[0].shape
    grid_rows, grid_cols = grid_size[0], grid_size[1] + 1  # Adjust grid size to include `observed_img`

    # Check if the grid can fit all images
    assert grid_rows * grid_cols >= len(images) + 1, "Grid size is too small for the number of images"

    # Create an empty array for the grid image
    grid_image = np.zeros((grid_rows * H, grid_cols * W, C), dtype=images[0].dtype)

    # Place the observed_img in the top-left corner
    grid_image[0:H, 0:W, :] = observed_img

    # Fill the rest of the grid with images
    image_idx = 0
    for row in range(grid_rows):
        for col in range(grid_cols):
            # Skip the first column for rows 2-4
            if col == 0:
                continue  # Skip the top-left for `observed_img`
            

            # Place image if there are images left in images_list
            if image_idx < len(images):
                grid_image[row * H:(row + 1) * H, col * W:(col + 1) * W, :] = images[image_idx]
                image_idx += 1

    # Plot and save the grid image
    plt.figure(figsize=(grid_cols, grid_rows))
    plt.imshow(grid_image)
    plt.axis('off')  # Turn off axis labels
    plt.savefig(save_path)
    plt.close()


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


def save_progress(
    text_encoders,
    presudo_token_ids,
    accelerator,
    tokenizers,
    save_path,
    safe_serialization=True,
    global_step=None,
):
    text_encoder_one, text_encoder_two, text_encoder_three = text_encoders
    presudo_token_one, presudo_token_two, presudo_token_three = presudo_token_ids
    tokenizer_one, tokenizer_two, tokenizer_three = tokenizers

    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Saving learned embeddings to {save_path}")

    # --- Extract embeddings ---
    learned_embeds_one = (
        accelerator.unwrap_model(text_encoder_one)
        .get_input_embeddings()
        .weight[presudo_token_one]
    )
    learned_embeds_two = (
        accelerator.unwrap_model(text_encoder_two)
        .get_input_embeddings()
        .weight[presudo_token_two]
    )
    learned_embeds_three = (
        accelerator.unwrap_model(text_encoder_three)
        .get_input_embeddings()
        .weight[presudo_token_three]
    )

    # --- Convert token ids to strings ---
    tokens_one = tokenizer_one.convert_ids_to_tokens(presudo_token_one)
    tokens_two = tokenizer_two.convert_ids_to_tokens(presudo_token_two)
    tokens_three = tokenizer_three.convert_ids_to_tokens(presudo_token_three)

    # --- Build dicts ---
    learned_embeds_dict_one = {t: e.detach().cpu() for t, e in zip(tokens_one, learned_embeds_one)}
    learned_embeds_dict_two = {t: e.detach().cpu() for t, e in zip(tokens_two, learned_embeds_two)}
    learned_embeds_dict_three = {t: e.detach().cpu() for t, e in zip(tokens_three, learned_embeds_three)}
    
    # --- Define output file paths ---
    file_one = os.path.join(save_path, f"learned_embeds_clip1_{global_step}.safetensors" if safe_serialization else f"learned_embeds_clip1_{global_step}.pt")
    file_two = os.path.join(save_path, f"learned_embeds_clip2_{global_step}.safetensors" if safe_serialization else f"learned_embeds_clip2_{global_step}.pt")
    file_three = os.path.join(save_path, f"learned_embeds_t5_{global_step}.safetensors" if safe_serialization else f"learned_embeds_t5_{global_step}.pt")

    # --- Save ---
    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict_one, file_one, metadata={"format": "pt"})
        safetensors.torch.save_file(learned_embeds_dict_two, file_two, metadata={"format": "pt"})
        safetensors.torch.save_file(learned_embeds_dict_three, file_three, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict_one, file_one)
        torch.save(learned_embeds_dict_two, file_two)
        torch.save(learned_embeds_dict_three, file_three)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50000,
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
        "--num_extra_conditioning_channels",
        type=int,
        default=0,
        help="Number of extra conditioning channels for controlnet.",
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
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
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
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
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
        "--precondition_outputs",
        type=int,
        default=1,
        help="Flag indicating if we are preconditioning the model outputs or not as done in EDM. This affects how "
        "model `target` is calculated.",
    )
    parser.add_argument(
        "--use_adafactor",
        action="store_true",
        help=(
            "Adafactor is a stochastic optimization method based on Adam that reduces memory usage while retaining"
            "the empirical benefits of adaptivity. This is achieved through maintaining a factored representation "
            "of the squared gradient accumulator across training steps."
        ),
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
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
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the guidance scale used for transformer.",
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
    parser.add_argument(
        "--enable_npu_flash_attention", action="store_true", help="Whether or not to use npu flash attention."
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
    parser.add_argument("--T5_injection", 
        type=str2bool,
        default=True,
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


# ---------------------------------------------------------------------------
# Dataloader collate
# ---------------------------------------------------------------------------

def collate_fn(examples):
    """Stack ``input_ids`` / ``pixel_values`` / ``label`` into a batched dict."""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    label = torch.stack([example["label"] for example in examples])
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "label": label,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    args.output_dir = os.path.join(args.output_dir,now+args.output_name+args.task_cond)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
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
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # Load tokenizer (load from pipeline)
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
    )

    # import correct text encoder class
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )
   
    
    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three,args
    )

     # check it 
    text_encoder_three.resize_token_embeddings(len(tokenizer_three))
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    

    # Add the placeholder token in tokenizer
    # use presudo words to create embeddings
    # take care the tokenizer with return different id like word and word(/w)
    presudo_list = args.presudo_words.split(',')
    presudo_token_ids_one = tokenizer_one.encode(' '.join(presudo_list), add_special_tokens=False)
    presudo_token_ids_two = tokenizer_two.encode(' '.join(presudo_list), add_special_tokens=False)
    presudo_token_ids_three = tokenizer_three.encode(' '.join(presudo_list), add_special_tokens=False)
    assert len(presudo_token_ids_one) == len(presudo_token_ids_two)
    
    string_tokens_one = tokenize_prompt(tokenizer_one, args.placeholder_string, max_sequence_length=77)
    string_tokens_two = tokenize_prompt(tokenizer_two, args.placeholder_string, max_sequence_length=77)
    string_tokens_three = tokenize_prompt(tokenizer_three, args.placeholder_string, max_sequence_length=args.max_sequence_length)
    
    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = Causal_SD3ControlNetModel().from_pretrained(args.controlnet_model_name_or_path)
        #controlnet.controlnet_cond_embedding.set_A_martix(torch.eye((4)))
    else:
        if args.dataset in ['pendulum','celeA_complex','human']:
            num_causal_concepts=4
        elif args.dataset in ['celeA_simple']:
            num_causal_concepts=2
        elif args.dataset in ['ADNI']:
            num_causal_concepts=6
        elif args.dataset in ['celebahq_simple']:
            num_causal_concepts=7
        args.num_causal_concepts = num_causal_concepts
        controlnet = Causal_SD3ControlNetModel().from_transformer(transformer, num_extra_conditioning_channels=args.num_extra_conditioning_channels)
    if args.causalnet_path is not None:
        print('load pretrained causalnet weights')
        controlnet.controlnet_cond_embedding.load_state_dict(torch.load(args.causalnet_path,weights_only=True))

    
    '''MCPL_regularization_initial(opt)'''
    opt = copy.deepcopy(args)

    
    # CL-InfoNCE
    # opt.presudo_words_infonce = args.presudo_words_infonce.split(',')
    # opt.adj_aug_infonce = args.adj_aug_infonce.split(',')

    # mcpl_controller = None
    # if opt.presudo_words_infonce[0] != '' and len(opt.presudo_words_infonce )>0:
    # #if opt.attn_words is not None and len(opt.attn_words) > 0:
    #     # fake prompts and keywords to initialise controller for register purpose
    #     print('do contrastive embedding training!')
    #     fake_prompts = args.train_batch_size*['a photo of ' + args.placeholder_string]
    #     fake_keywords = [opt.attn_words for _ in range(args.train_batch_size)]
    #     presudo_words_softmax = opt.presudo_words_softmax
    #     presudo_words_infonce = opt.presudo_words_infonce
    #     adj_aug_infonce = opt.adj_aug_infonce
    #     # lb = LocalMask(tokenizer, fake_prompts, fake_keywords, opt.attn_mask_type, \
    #     #     presudo_words_softmax, presudo_words_infonce, adj_aug_infonce, accelerator.device)
    #     lb=None
    #     controller = AttentionMask(local_blend=lb)
    #     mcpl_controller = controller
    #     #register_attention_control_t2i(unet, mcpl_controller)
    '''MCPL_regularization_initial(opt)'''


    # Move vae and unet to device and cast to weight_dtype
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)

    # Freeze vae and unet
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    # Always train controlnet by default
    controlnet.requires_grad_(True)

    if args.mcpl_training:
        print('Enable MCPL embedding training')

        # ---- Freeze encoders, enable only embeddings ----
        # CLIP
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        text_encoder_one.get_input_embeddings().requires_grad_(True)
        text_encoder_two.get_input_embeddings().requires_grad_(True)

        # T5
        text_encoder_three.requires_grad_(False)
        if args.T5_injection:
            text_encoder_three.get_input_embeddings().requires_grad_(True)

        # ---- Mixed precision casting (after requires_grad set) ----
        if accelerator.mixed_precision == "fp16":
            models = [controlnet, text_encoder_one, text_encoder_two]
            if args.T5_injection:
                models.append(text_encoder_three)
            cast_training_params(models)  # keep this before building optimizer

        # ---- Build optimizer param list (embeddings + controlnet trainables) ----
        trainable_params = []
        trainable_params += list(text_encoder_one.get_input_embeddings().parameters())
        trainable_params += list(text_encoder_two.get_input_embeddings().parameters())
        if args.T5_injection:
            trainable_params += list(text_encoder_three.get_input_embeddings().parameters())

        # Include ControlNet params that remain trainable
        trainable_params += [p for p in controlnet.parameters() if p.requires_grad]
    else:
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        text_encoder_three.requires_grad_(False)
        trainable_params += [p for p in controlnet.parameters() if p.requires_grad]

     # `accelerate` 0.16.0 will have better support for customized saving
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
                load_model = Causal_SD3ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            
            transformer.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        controlnet.enable_gradient_checkpointing()
        if args.mcpl_training:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()
            if args.T5_injection:
                text_encoder_three.gradient_checkpointing_enable()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
    )
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    # trainable_params  = list(text_encoder.get_input_embeddings().parameters()) + list(controlnet.parameters())
    # Initialize the optimizer
    print("Trainable Param: ",sum(p.numel() for p in trainable_params if p.requires_grad))
    # use adafactor optimizer to save gpu memory
    if args.use_adafactor:
        from transformers import Adafactor

        optimizer = Adafactor(
            trainable_params,
            lr=args.learning_rate,
            scale_parameter=False,
            relative_step=False,
            # warmup_init=True,
            weight_decay=args.adam_weight_decay,
        )
    else:
        optimizer = optimizer_class(
            trainable_params,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Dataset and DataLoaders creation:
    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer_one,
        size=args.resolution,
        placeholder_token=args.placeholder_string,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=False,
        flip_p=0.0,
        random_article=False,
        set="train",
        dataset =args.dataset,
        random_prompt_template = args.random_prompt_template
    )
    # balanced sampler
    if args.dataset in ['celebahq_simple']:
        imglabel = train_dataset.imglabel
        glass_label = imglabel[:, 1]
        class_counts = torch.bincount(glass_label.long())
        class_weights = 1.0 / class_counts.float()
        # tempered inverse-frequency weights
        alpha = 0.5   # <-- your "rate": 0 = uniform, 1 = full inverse-freq
        class_weights = (1.0 / class_counts.float()) ** alpha
        # per-sample weights
        sample_weights = class_weights[glass_label.long()]
        sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
        #sampler = BalancedAttributeSampler(train_dataset.imglabel,attr_indices, batch_size=args.train_batch_size)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch_size,sampler=sampler, num_workers=args.dataloader_num_workers,collate_fn=collate_fn,drop_last=False
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers,collate_fn=collate_fn,drop_last=False
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


    controlnet.train()
    if args.mcpl_training:
        text_encoder_one.train()
        text_encoder_two.train()
        
        print(f"Using device: {accelerator.device}")
        print(f"NCCL_IGNORE_DISABLED_P2P: {os.environ.get('NCCL_IGNORE_DISABLED_P2P')}")
    
        if args.T5_injection:
            text_encoder_three.train()
            controlnet, text_encoder_one, text_encoder_two, text_encoder_three, optimizer, train_dataloader, lr_scheduler = \
                accelerator.prepare(controlnet, text_encoder_one, text_encoder_two, text_encoder_three,
                                    optimizer, train_dataloader, lr_scheduler)
            accelerate_models = [controlnet, text_encoder_one, text_encoder_two, text_encoder_three]
            
        else:
            controlnet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler = \
                accelerator.prepare(controlnet, text_encoder_one, text_encoder_two,
                                    optimizer, train_dataloader, lr_scheduler)
            accelerate_models = [controlnet, text_encoder_one, text_encoder_two]
        
    else:
        controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            controlnet, optimizer, train_dataloader, lr_scheduler
        )
        accelerate_models = [controlnet]



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

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # keep original embeddings as reference
    orig_embeds_params_one = accelerator.unwrap_model(text_encoder_one).get_input_embeddings().weight.data.clone()
    orig_embeds_params_two = accelerator.unwrap_model(text_encoder_two).get_input_embeddings().weight.data.clone()
    if args.T5_injection:
        orig_embeds_params_three = accelerator.unwrap_model(text_encoder_three).get_input_embeddings().weight.data.clone()
    switch_dataloader = True
    for epoch in range(first_epoch, args.num_train_epochs):
        if args.dataset in ['celeA_complex','celebahq_simple'] and epoch > 12 and switch_dataloader:
            # no sampler 
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers,collate_fn=collate_fn,drop_last=False
            )
            train_dataloader = accelerator.prepare(train_dataloader)
            switch_dataloader=False
            accelerator.wait_for_everyone()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(accelerate_models):
                
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                 # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                prompt_embeds, pooled_prompt_embeds = encode_prompt_pai(
                            text_encoders=[text_encoder_one, text_encoder_two,text_encoder_three],
                            tokenizers=[None, None,None],
                            text_input_ids_list=[string_tokens_one, string_tokens_two,string_tokens_three],
                            max_sequence_length=args.max_sequence_length,
                            prompt=batch['input_ids'],
                            label = batch['label'].to(dtype=weight_dtype).unsqueeze(2),
                            presudo_token_lists =[presudo_token_ids_one,presudo_token_ids_two,presudo_token_ids_three],
                            weight_dtype = weight_dtype
                )

                control_block_res_samples = controlnet(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    controlnet_cond=None,
                    return_dict=False,
                )[0]
                control_block_res_samples = [sample.to(dtype=weight_dtype) for sample in control_block_res_samples]

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    block_controlnet_hidden_states=control_block_res_samples,
                    return_dict=False,
                )[0]

                # Follow: Section 5 of https://huggingface.co/papers/2206.00364.
                # Preconditioning of the model outputs.
                if args.precondition_outputs:
                    model_pred = model_pred * (-sigmas) + noisy_model_input

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                if args.precondition_outputs:
                    target = model_input
                else:
                    target = noise - model_input

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                accelerator.backward(loss)
                # Check if the gradient of each model parameter contains NaN
                mse_loss_log = loss.detach().item()
                causal_loss_log=None
                null_loss_log=None

                if accelerator.sync_gradients:
                    params_to_clip = trainable_params
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # if step == 0 and accelerator.is_main_process:
                #     torch.cuda.reset_peak_memory_stats()
                # # after optimizer.step()
                # if accelerator.is_main_process:
                #     print(f"step {step} | alloc={torch.cuda.memory_allocated()/1e6:.1f}MB "
                #         f"peak={torch.cuda.max_memory_allocated()/1e6:.1f}MB")
                
                index_no_updates_one = torch.ones((len(tokenizer_one),), dtype=torch.bool)
                index_no_updates_two = torch.ones((len(tokenizer_two),), dtype=torch.bool)
                index_no_updates_three = torch.ones((len(tokenizer_three),), dtype=torch.bool)
                if args.mcpl_training:
                    # Let's make sure we don't update any embedding weights besides the newly added token
                    index_no_updates_one[presudo_token_ids_one] = False
                    index_no_updates_two[presudo_token_ids_two] = False
                    index_no_updates_three[presudo_token_ids_three] = False
                with torch.no_grad():
                    accelerator.unwrap_model(text_encoder_one).get_input_embeddings().weight[
                        index_no_updates_one
                    ] = orig_embeds_params_one[index_no_updates_one]

                    accelerator.unwrap_model(text_encoder_two).get_input_embeddings().weight[
                        index_no_updates_two
                    ] = orig_embeds_params_two[index_no_updates_two]
                    if args.T5_injection:
                        accelerator.unwrap_model(text_encoder_three).get_input_embeddings().weight[
                            index_no_updates_three
                        ] = orig_embeds_params_three[index_no_updates_three]

                
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                images = []
                progress_bar.update(1)
                global_step += 1
                
                if accelerator.is_main_process:
                    if global_step % args.save_steps == 0:
                        

                        save_progress(
                            [text_encoder_one, text_encoder_two,text_encoder_three],
                            [presudo_token_ids_one,presudo_token_ids_two,presudo_token_ids_three],
                            accelerator,
                            [tokenizer_one,tokenizer_two,tokenizer_three],
                            args.output_dir,
                            safe_serialization=not args.no_safe_serialization,
                            global_step= global_step
                        )
                        controlnet_name = (
                            f"controlnet-steps-{global_step}.bin"
                            if args.no_safe_serialization
                            else f"controlnet-steps-{global_step}.safetensors"
                        )
                        net_save_path = os.path.join(args.output_dir, controlnet_name)
                        controlnet_to_save = unwrap_model(controlnet)
                        controlnet_to_save.save_pretrained(net_save_path)


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
                        pass
                        #accelerator.wait_for_everyone()
                        # control_img, images = log_validation(
                        #     controlnet,text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, epoch
                        # )
                        # image_path = os.path.join(args.output_dir, f"edit-{global_step}.jpg")
                        # save_images_grid(control_img, images,(4,4),image_path)

            logs = {"recons_loss": mse_loss_log, "causal_loss":causal_loss_log, "nll_loss":null_loss_log,"lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            # Open the file in append mode
            with open(os.path.join(args.output_dir,'logs.txt'), "a") as log_file:
                log_file.write(f"Step {global_step}: {logs}\n")

            if global_step >= args.max_train_steps:
                accelerator.wait_for_everyone()
                break
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.push_to_hub and not args.save_as_full_pipeline:
            logger.warning("Enabling full model saving because --push_to_hub=True was specified.")
            save_full_model = True
        else:
            save_full_model = args.save_as_full_pipeline
        # Save the newly trained embeddings

        save_progress(
                [text_encoder_one, text_encoder_two,text_encoder_three],
                [presudo_token_ids_one,presudo_token_ids_two,presudo_token_ids_three],
                accelerator,
                [tokenizer_one,tokenizer_two,tokenizer_three],
                args.output_dir,
                safe_serialization=not args.no_safe_serialization,
                global_step= "final"
            
        )

        controlnet_name = (
                        f"controlnet.bin"
                        if args.no_safe_serialization
                        else f"controlnet.safetensors"
                    )
        net_save_path = os.path.join(args.output_dir, controlnet_name)
        controlnet = unwrap_model(controlnet)
        controlnet.save_pretrained(net_save_path)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
