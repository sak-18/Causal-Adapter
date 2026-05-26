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

import argparse
import pandas as pd
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

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
from edit_modules.TI_datasets_discovery import TextualInversionDataset as T2IDataset
# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTokenizer
from edit_modules.clip import CLIPTextModel
from edit_modules.embed_manager import EmbeddingManager
import datetime
import diffusers
from edit_modules.load_datasets_morphominist import _get_paths,load_idx
from edit_modules.load_datasets_adni import load_data,load_extra_attributes
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
import matplotlib.pyplot as plt

from p2p.p2p_ldm_utils_origin import LocalMask, AttentionMask
from p2p.ptp_utils import register_attention_control_t2i,register_attention_control
from p2p import mcpl_utils
import copy
from torchvision.datasets import CelebA
if is_wandb_available():
    import wandb

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
# ------------------------------------------------------------------------------
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
def add_label_on_embeddings(concept_ids,input_ids,encoder_hidden_states,controlnet_cond):
    # insert embedding after transformer
   
    for i,token_id in enumerate(concept_ids):
        placeholder_idx = torch.where(input_ids == token_id)
        encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond[:,i,:]
    return encoder_hidden_states


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

logger = get_logger(__name__)

def save_images_grid(images, grid_size, save_path):
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


def log_validation(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, epoch):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=unet,
        vae=vae,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    # images = []
    # for _ in range(args.num_validation_images):
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)
    images = pipeline(args.validation_prompt, num_inference_steps=50, generator=generator,guidance_scale=7.5,num_images_per_prompt=args.num_validation_images,height=args.resolution,width=args.resolution).images
    #     with autocast_ctx:
    #         image = pipeline(args.validation_prompt, num_inference_steps=50, generator=generator,guidance_scale=7.5).images[0]
    #     images.append(image)

    # for tracker in accelerator.trackers:
    #     if tracker.name == "tensorboard":
    #         np_images = np.stack([np.asarray(img) for img in images])
    #         tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
    #     if tracker.name == "wandb":
    #         tracker.log(
    #             {
    #                 "validation": [
    #                     wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
    #                 ]
    #             }
    #         )

    del pipeline
    torch.cuda.empty_cache()
    return np.stack([np.asarray(img) for img in images])


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
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
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
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="text-inversion-model",
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
        "--dataset",
        type=str,
        default='pendulum',
        help="dataset",
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
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
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
        default=500,
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
        default=0.07,
        help="PromptCL: infonce_temperature",
    )
    
    parser.add_argument("--infonce_scale",
        type=float,
        default=1.0,
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

    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument("--embedding_extension", 
        type=str2bool, 
        default=False, 
        help="add linear layer after embedding")


    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args


imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.0,
        set="train",
        placeholder_token="*",
        center_crop=False,
        random_article=False,
        dataset = 'pendulum',
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token

        assert len(placeholder_token.split(' ')) == len(self.tokenizer.encode(placeholder_token,add_special_tokens=False)),"Unknown words is wrong tokened" 

        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
        self.random_article = random_article

        
        self.dataset = dataset
        if dataset == 'pendulum':
            self.imglabel = [list(map(int,k[:-4].split("/")[-1].split('_')[1:])) for k in self.image_paths]
            self.scale = np.array([[2,42],[104,44],[7.5, 4.5],[11,8]])
            self.image_transforms = transforms.Compose(
            [
                transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR),
                #transforms.ToTensor(),
                #transforms.Normalize([0.5], [0.5]),
            ]
            )
            #print(self.imglabel)
        elif 'celeA' in dataset:
            data_dir = "/home/jovyan/fcvm-data-volume/kzzr229/workspace/counterfactual-benchmark/datasets/"
            self.data = CelebA(root=data_dir, split='train', transform=None, download=False)
            self.num_images = len(self.data)
            self._length = self.num_images
            if 'simple' in dataset:
                selected_item = ['Smiling','Eyeglasses']
            
            elif 'complex' in dataset:
                selected_item = ['Young','Male','No_Beard','Bald']

            else:
                AssertionError('no such {} dataset'.format(dataset))
            attribute_ids = [self.data.attr_names.index(attr) for attr in selected_item]
            metrics = {attr: torch.as_tensor(self.data.attr[:, attr_id], dtype=torch.float32) for attr, attr_id in zip(selected_item, attribute_ids)}

            attrs = torch.cat([metrics[attr].unsqueeze(1)
                                    for attr in selected_item], dim=1)
            self.imglabel= attrs
            possible_values = {attr: torch.unique(values, dim=0) for attr, values in metrics.items()}
            self.image_transforms = transforms.Compose(
            [
                transforms.CenterCrop(150),
                transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR),
                #transforms.ToTensor(),
                #transforms.Normalize([0.5], [0.5]),
            ]
            )
            self.normalize_transforms = transforms.Compose(
                [
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

        elif dataset == 'MorphoMNIST':
            # MIN_MAX = {
            #     "thickness": [0.87598526, 6.255515],
            #     "intensity": [66.601204, 254.90317],
            # }
            if set == "train":
                train_bool = True
            images_path, labels_path, metrics_path = _get_paths(self.data_root,train=train_bool)
            images = load_idx(images_path)
            # digits 10 numbers
            labels = load_idx(labels_path)
            # for thickness and intensity
            metric = pd.read_csv(metrics_path, index_col='index')
            
            #metric[['thickness', 'intensity']] = (metric[['thickness', 'intensity']] - metric[['thickness', 'intensity']].mean()) / metric[['thickness', 'intensity']].std()

            # Concatenate normalized metrics with labels
            metric['label'] = labels
            # Convert thickness and intensity to tensor
            df_normalized = (metric - metric.min()) / (metric.max() - metric.min())
            # Step 2: Transform to [-1, 1]
            df_transformed = df_normalized * 2 - 1
            self.num_images = len(images)
            self._length = self.num_images
            self.data = [Image.fromarray(images[i]) for i in range(images.shape[0])]
            # thickness   intensity  label

            self.image_transforms = transforms.Compose(
            [
                transforms.Pad(padding=2),
                transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR),
            ]
            )

        elif dataset == 'ADNI':
            num_of_slices = 20
            keep_only_screening = False
            data_dir = os.path.join(self.data_root, 'preprocessed_data')
            self.image_paths, attribute_dict, subject_dates_dict = load_data(data_dir, num_of_slices=num_of_slices,
                                                                    split=set,
                                                                    keep_only_screening=keep_only_screening)
            

            self.num_images = len(self.image_paths)
            self._length = self.num_images
            
            
            self.image_transforms = transforms.Compose(
            [
                transforms.Pad(padding=6),
                transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR),
           
            ]
            )

        elif dataset == 'chexpert':
            # name like sampling_100 , sampling_500 dir
            select_columns = ['Sex', 'Age', 'Pleural Effusion']
            dataset_id = self.data_root.split('/')[-1].split('_')[-1]
            root_parent = os.path.dirname(self.data_root)
            csv_path = os.path.join(root_parent,'meta_'+dataset_id+'.csv')
            df = pd.read_csv(csv_path)
            img_names =np.asarray(df.Path)
            self.image_paths = [os.path.join(self.data_root, file_path) for file_path in img_names]
            label_list = np.asarray(df[select_columns])
            mapping = {'Female': 0, 'Male': 1}
            # Convert the first column to integers using the mapping
            label_list[:, 0] = np.vectorize(mapping.get)(label_list[:, 0])
            # Convert the entire array to integer type
            label_list = label_list.astype(int)
            self.imglabel =label_list
            # this scale for max-min normalization
            self.scale = np.array([[0,1],[20,100],[0, 1]])

    def randn_article_words(self,placeholder_string):
        # Original string and list of replacement words
        original_string = placeholder_string
        replacement_words = ['a', 'one', 'the']

        # Split the string into words
        words = original_string.split()

        for i,word in enumerate(words):
            if word == 'a':
                words[i] = random.choice(replacement_words)

        # Join the words back into a string
        modified_string = ' '.join(words)

        return modified_string


    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        if self.dataset in ['celeA_simple','celeA_complex']:
            image = self.data[i%self.num_images][0]
        elif self.dataset in ['MorphoMNIST']:
            image = self.data[i%self.num_images]
        elif self.dataset in ['pendulum']:
            image = Image.open(self.image_paths[i % self.num_images])
        elif self.dataset in ['ADNI']:
            # Tiff image
            image = Image.open(self.image_paths[i % self.num_images])
            image_array = np.array(image)
            # Normalize the image (scale from min-max to 0-255)
            image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255
            # Convert to uint8 (required for RGB conversion)
            image_array = image_array.astype(np.uint8)
            # Convert back to a PIL image in RGB mode
            image = Image.fromarray(image_array, mode="L")


        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        if self.random_article:
            placeholder_string = self.randn_article_words(placeholder_string)
            
        text = random.choice(self.templates).format(placeholder_string)

        # example["input_ids"] = self.tokenizer(
        #     text,
        #     padding="max_length",
        #     truncation=True,
        #     max_length=self.tokenizer.model_max_length,
        #     return_tensors="pt",
        # ).input_ids[0]
        example["input_ids"] = text
        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)


        image = Image.fromarray(img)
        
        #image = image.resize((self.size, self.size), resample=self.interpolation)
        image = self.image_transforms(image)
        
        #image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


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


    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    # Freeze all parameters except for the token embeddings in text encoder (text_encoder.text_model.embeddings.token_embedding)
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    # Add embedding manager
    if args.embedding_extension:
        embedding_manager = EmbeddingManager(presudo_token_ids)
        text_encoder.text_model.embeddings.set_embedding_manager(embedding_manager)
        for param in embedding_manager.embed_proj.parameters():
            param.requires_grad = True 

    for name, param in unet.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name} | Requires Grad: {param.requires_grad}")

    '''MCPL_regularization_initial(opt)'''
    opt = copy.deepcopy(args)

    if args.attn_words is not None:
            opt.attn_words = args.attn_words.split(',')
    opt.presudo_words_softmax = args.presudo_words_softmax.split(',')
    opt.attn_mask_type = args.attn_mask_type
    
    # CL-InfoNCE
    opt.presudo_words_infonce = args.presudo_words_infonce.split(',')
    opt.adj_aug_infonce = args.adj_aug_infonce.split(',')

    mcpl_controller = None
    if opt.attn_words is not None and len(opt.attn_words) > 0:
        # fake prompts and keywords to initialise controller for register purpose
        fake_prompts = args.train_batch_size*['a photo of ' + args.placeholder_string]
        fake_keywords = [opt.attn_words for _ in range(args.train_batch_size)]
        presudo_words_softmax = opt.presudo_words_softmax
        presudo_words_infonce = opt.presudo_words_infonce
        adj_aug_infonce = opt.adj_aug_infonce
        lb = LocalMask(tokenizer, fake_prompts, fake_keywords, opt.attn_mask_type, \
            presudo_words_softmax, presudo_words_infonce, adj_aug_infonce, accelerator.device)
        controller = AttentionMask(local_blend=lb)
        mcpl_controller = controller
        #register_attention_control_t2i(unet, mcpl_controller)
    '''MCPL_regularization_initial(opt)'''

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
    update_param = list(text_encoder.get_input_embeddings().parameters())
    if args.embedding_extension:
        update_param+=list(embedding_manager.embed_proj.parameters())

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        update_param,  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = T2IDataset(
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
        dataset = args.dataset
    )
    # train_dataset = TextualInversionDataset(
    #     data_root=args.train_data_dir,
    #     tokenizer=tokenizer,
    #     size=args.resolution,
    #     placeholder_token=args.placeholder_string,
    #     repeats=args.repeats,
    #     learnable_property=args.learnable_property,
    #     center_crop=False,
    #     flip_p=0.0,
    #     random_article=False,
    #     set="train",
    #     dataset = args.dataset
    # )
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers,pin_memory=True
    # )
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

    text_encoder.train()
    # Prepare everything with our `accelerator`.
    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

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
    #torch.autograd.set_detect_anomaly(True)
    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder):
                prompt = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                prompt_token_id = tokenizer(
                        prompt,
                        padding="max_length",
                        truncation=True,
                        max_length=tokenizer.model_max_length,
                        return_tensors="pt",
                    ).input_ids.to(accelerator.device)
                # AttentionMask forward-re-register: update attn layer counts each forward ...
                if mcpl_controller is not None:
                    keywords = [opt.attn_words for _ in range(len(prompt))]
                    lb = LocalMask(tokenizer, prompt, keywords, \
                            opt.attn_mask_type, \
                            opt.presudo_words_softmax, \
                            opt.presudo_words_infonce, \
                            opt.adj_aug_infonce, \
                            accelerator.device)
                    # update local_blend at each step
                    mcpl_controller.local_blend = lb
                    register_attention_control(unet, mcpl_controller)
                    mcpl_controller.reset()
                
                # Convert images to latent space
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
                encoder_hidden_states = text_encoder(prompt_token_id)[0].to(dtype=weight_dtype)
                '''add the label here'''
                #encoder_hidden_states =add_label_on_embeddings(presudo_token_ids,prompt_token_id,encoder_hidden_states,batch['label'].to(dtype=weight_dtype).to(noisy_latents.device).unsqueeze(2))
                # Predict the noise residual #.sample just output of unet, not sampling
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                if mcpl_controller is not None and mcpl_controller.local_blend.presudo_words_infonce[0] != '':
                    nll_loss = mcpl_utils.prompt_contrastive_loss(mcpl_controller, encoder_hidden_states,opt)
                    loss+=nll_loss   
                accelerator.backward(loss)

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
                if global_step % args.save_steps == 0:
                    weight_name = (
                        f"learned_embeds-steps-{global_step}.bin"
                        if args.no_safe_serialization
                        else f"learned_embeds-steps-{global_step}.safetensors"
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

                if accelerator.is_main_process:
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
                        
                        if mcpl_controller is not None:
                            print('AttentionMask-DDIM-re-register and reset controller before 1st sample_log ...')
                            register_attention_control_t2i(unet, mcpl_controller)
                            mcpl_controller.reset()

                        images = log_validation(
                            text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, epoch
                        )
                        image_path = os.path.join(args.output_dir, f"txt2img-{global_step}.jpg")
                        save_images_grid(images,(1,args.num_validation_images),image_path)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

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
