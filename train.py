#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Causal-Adapter (SD1.5) training entry point.

Trains the causal ControlNet head together with the MCPL pseudo-token
embeddings on top of a frozen Stable Diffusion v1.5 / miniSD backbone. The
script is invoked through ``accelerate`` (see ``test_commands.md`` and the
README for example commands).

Public CLI surface (everything else has a sane default):

* ``--pretrained_model_name_or_path``: HF model id *or* local path to the
  Stable Diffusion checkpoint to load. Local paths work behind firewalls;
  open-source users can pass e.g. ``runwayml/stable-diffusion-v1-5``.
* ``--train_data_dir``: dataset root, required.
* ``--scm_path``: optional path to a pretrained SCM head produced by
  ``SCM_modeling``. Required when fine-tuning on top of a pretrained SCM.
* ``--dataset``: one of ``pendulum``, ``ADNI``, ``celeA_complex``,
  ``celeA_simple``, ``celebahq_simple``, ``MorphoMNIST``, ``human``,
  ``chexpert``.
* ``--placeholder_string`` / ``--presudo_words`` / ``--presudo_words_infonce``:
  MCPL pseudo-token configuration.
"""

import argparse
import copy
import datetime
import logging
import math
import os
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, CLIPTextModel

import diffusers
from diffusers import (
    AutoencoderKL,
    Causal_ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

import safetensors

from causal_datasets import (
    TextualInversionDataset,
    build_balanced_sampler,
    needs_uniform_switch,
)
from causal_modules.ddim_modules import prompt_aligned_injection_diff
from causal_modules.p2p_edits import mcpl_utils
from causal_modules.p2p_edits.p2p_ldm_utils import AttentionMask, LocalMask

if is_wandb_available():
    import wandb  # noqa: F401  (imported for side effect / availability check)

# Errors out if the locally-installed diffusers is older than required.
check_min_version("0.31.0.dev0")

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-dataset metadata
# ---------------------------------------------------------------------------

# Number of causal concepts per dataset (drives the SCM input dim and the
# size of the pseudo-token list expected in ``--presudo_words``).
NUM_CAUSAL_CONCEPTS = {
    "pendulum": 4,
    "celeA_complex": 4,
    "human": 4,
    "celeA_simple": 2,
    "ADNI": 6,
    "celebahq_simple": 7,
}

# Ground-truth adjacency masks consumed by ``ControlNetConditioningEmbedding``.
# Keys match ``--dataset``; missing keys fall back to a zero matrix of size
# ``num_causal_concepts``.
def _build_a_matrix(dataset: str, num_concepts: int, dtype, device):
    matrices = {
        "pendulum": [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        "celeA_complex": [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        "human": [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        "celeA_simple": [[0, 0], [0, 0]],
        "ADNI": [
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        "chexpert": [[0, 0, 1], [0, 0, 1], [0, 0, 0]],
        "skin_cancer": [[0, 0, 1], [0, 0, 1], [0, 0, 0]],
    }
    if dataset in matrices:
        return torch.tensor(matrices[dataset], dtype=dtype, device=device)
    return torch.zeros((num_concepts, num_concepts), dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_progress(text_encoder, presudo_token_ids, accelerator, tokenizer, save_path,
                  safe_serialization=True):
    """Persist the learned MCPL token embeddings as ``{token: tensor}``."""
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[presudo_token_ids]
    )
    tokens = tokenizer.convert_ids_to_tokens(presudo_token_ids)
    learned_embeds_dict = {token: embed.detach().cpu() for token, embed in zip(tokens, learned_embeds)}

    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def check_elements_exist(needles, haystack):
    """Assert every id in ``needles`` appears at least once in ``haystack``."""
    assert all(elem in haystack for elem in needles), (
        "Error: Not all pseudo words exist in placeholder_string."
    )


def collate_fn(examples):
    """Stack ``input_ids`` / ``pixel_values`` / ``label`` into a batched dict."""
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([ex["input_ids"] for ex in examples])
    label = torch.stack([ex["label"] for ex in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids, "label": label}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Causal-Adapter (SD1.5) training script.")

    # --- Required paths --------------------------------------------------
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help=(
            "HuggingFace model id (e.g. 'runwayml/stable-diffusion-v1-5') OR a "
            "local path to a Stable Diffusion / miniSD checkpoint folder. "
            "Local paths are recommended for users behind a firewall."
        ),
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Path to the training dataset root (layout depends on --dataset).",
    )
    parser.add_argument(
        "--scm_path",
        type=str,
        default=None,
        help=(
            "Optional path to a pretrained SCM checkpoint produced by "
            "SCM_modeling. Required when --causal_training is False and you "
            "want to load a frozen SCM head."
        ),
    )

    # --- Diffusion model loading ----------------------------------------
    parser.add_argument(
        "--revision", type=str, default=None,
        help="Revision of the pretrained model id (HuggingFace only).",
    )
    parser.add_argument(
        "--variant", type=str, default=None,
        help="Variant of the pretrained model files, e.g. 'fp16'.",
    )
    parser.add_argument(
        "--tokenizer_name", type=str, default=None,
        help="Pretrained tokenizer name or path (defaults to the diffusion model's).",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path", type=str, default=None,
        help=(
            "Optional pretrained Causal_ControlNet checkpoint. If unset, the "
            "ControlNet is initialized from the UNet."
        ),
    )

    # --- Dataset / output ------------------------------------------------
    parser.add_argument(
        "--dataset", type=str, default="pendulum",
        choices=list(NUM_CAUSAL_CONCEPTS.keys()) + ["MorphoMNIST", "chexpert"],
        help="Dataset adapter to use (controls labels, prompts, transforms).",
    )
    parser.add_argument(
        "--learnable_property", type=str, default="object",
        choices=["object", "style"],
        help="Textual-inversion learnable property (kept for upstream compat).",
    )
    parser.add_argument(
        "--output_name", type=str, default="causal-adapter",
        help="Run name; appended to --output_dir along with a timestamp.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./logs",
        help="Parent directory for run artefacts (logs, embeddings, controlnet).",
    )
    parser.add_argument(
        "--logging_dir", type=str, default="logs",
        help="TensorBoard subdirectory under --output_dir.",
    )

    # --- Training schedule ----------------------------------------------
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Image resolution; all inputs are resized to this.")
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Per-device training batch size.")
    parser.add_argument("--num_train_epochs", type=int, default=100,
                        help="Max epochs (overridden by --max_train_steps when set).")
    parser.add_argument("--max_train_steps", type=int, default=20000,
                        help="Total optimizer steps; overrides --num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Steps to accumulate before each optimizer update.")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Trade compute for memory via UNet/text-encoder checkpointing.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                        help="DataLoader worker processes (0 = main process).")

    # --- Optimizer / LR --------------------------------------------------
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Initial learning rate after the warmup period.")
    parser.add_argument("--scale_lr", action="store_true",
                        help="Scale LR by num_gpus * grad_accum * batch_size.")
    parser.add_argument(
        "--lr_scheduler", type=str, default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial",
                 "constant", "constant_with_warmup"],
        help="LR schedule type.",
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=0,
                        help="Number of warmup steps for the LR schedule.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
                        help="Number of resets for cosine_with_restarts.")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # --- Precision / acceleration ---------------------------------------
    parser.add_argument(
        "--mixed_precision", type=str, default="no",
        choices=["no", "fp16", "bf16"],
        help="Mixed-precision policy passed to accelerate.",
    )
    parser.add_argument("--allow_tf32", action="store_true",
                        help="Allow TF32 matmul on Ampere+ GPUs.")

    # --- Logging / checkpointing ----------------------------------------
    parser.add_argument(
        "--report_to", type=str, default="tensorboard",
        choices=["tensorboard", "wandb", "comet_ml", "all"],
        help="Logging integration for accelerate.",
    )
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save the embeddings + controlnet every N steps.")
    parser.add_argument("--checkpointing_steps", type=int, default=999999,
                        help="Save a full accelerate state every N steps.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None,
                        help="Cap the number of accelerate state checkpoints kept.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a checkpoint, or 'latest'.")
    parser.add_argument("--no_safe_serialization", action="store_true",
                        help="Save .bin instead of .safetensors.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training.")

    # --- MCPL / contrastive (PromptCL) ----------------------------------
    parser.add_argument(
        "--placeholder_string", type=str, required=True,
        help="MCPL placeholder string holding all pseudo tokens, e.g. 'a image of @ * & !'.",
    )
    parser.add_argument(
        "--presudo_words", type=str, required=True,
        help="Comma-separated pseudo tokens used in --placeholder_string, e.g. '@,*,&,!'.",
    )
    parser.add_argument(
        "--presudo_words_infonce", type=str, default="",
        help="Subset of --presudo_words used in the contrastive (InfoNCE) loss.",
    )
    parser.add_argument("--presudo_words_softmax", type=str, default="",
                        help="Pseudo words used for the auxiliary attention softmax.")
    parser.add_argument("--adj_aug_infonce", type=str, default="",
                        help="Adjective augmentations treated as positives in InfoNCE.")
    parser.add_argument("--infonce_temperature", type=float, default=0.2)
    parser.add_argument("--infonce_scale", type=float, default=0.0005)
    parser.add_argument(
        "--attn_mask_type", type=str, default="hard",
        choices=["hard", "soft", "skip"],
        help="Attention-mask policy for LocalMask in PromptCL.",
    )
    parser.add_argument(
        "--mcpl_training", type=str2bool, default=True,
        help="Train the MCPL pseudo-token embeddings.",
    )
    parser.add_argument(
        "--causal_training", type=str2bool, default=False,
        help="Also train the SCM head inside controlnet_cond_embedding.",
    )
    parser.add_argument(
        "--task_cond", type=str, default="generation_text_global_after",
        help=(
            "Where the causal embedding is injected. Options: "
            "'generation_image', 'generation_text_local', 'generation_text_global', "
            "'generation_text_global_after', 'generation_text_local_after'."
        ),
    )
    parser.add_argument("--random_prompt_template", type=str2bool, default=False,
                        help="Sample a random ImageNet-style template each step.")

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    print(args)

    # Append a timestamp + run name + task_cond so repeated launches don't
    # clobber each other.
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-")
    args.output_dir = os.path.join(args.output_dir, now + args.output_name + args.task_cond)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # MPS doesn't support AMP — disable it explicitly when running on Mac.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb" and not is_wandb_available():
        raise ImportError("Install wandb to use --report_to=wandb.")

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

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Tokenizer + diffusion backbone
    # ------------------------------------------------------------------
    tokenizer = CLIPTokenizer.from_pretrained(
        args.tokenizer_name or args.pretrained_model_name_or_path,
        subfolder=None if args.tokenizer_name else "tokenizer",
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        revision=args.revision, variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet",
        revision=args.revision, variant=args.variant,
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        return model._orig_mod if is_compiled_module(model) else model

    # ------------------------------------------------------------------
    # MCPL pseudo-token book-keeping
    # ------------------------------------------------------------------
    presudo_list = args.presudo_words.split(",")
    # Encode the joined string so the tokenizer collapses the BOS/EOS
    # and we get one id per pseudo token (verified later via
    # ``check_elements_exist``).
    presudo_token_ids = tokenizer.encode(" ".join(presudo_list), add_special_tokens=False)

    # ------------------------------------------------------------------
    # ControlNet (Causal_ControlNet)
    # ------------------------------------------------------------------
    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = Causal_ControlNetModel().from_pretrained(args.controlnet_model_name_or_path)
        # ``num_causal_concepts`` is normally inferred from the checkpoint
        # config; fall back to the dataset table when missing.
        args.num_causal_concepts = getattr(
            controlnet, "num_causal_concepts",
            NUM_CAUSAL_CONCEPTS.get(args.dataset, len(presudo_list)),
        )
    else:
        if args.dataset not in NUM_CAUSAL_CONCEPTS:
            raise ValueError(
                f"--dataset={args.dataset!r} has no entry in NUM_CAUSAL_CONCEPTS."
            )
        args.num_causal_concepts = NUM_CAUSAL_CONCEPTS[args.dataset]
        controlnet = Causal_ControlNetModel().from_unet(
            unet,
            task_cond=args.task_cond,
            num_causal_concepts=args.num_causal_concepts,
            dataset=args.dataset,
        )

    if args.scm_path is not None:
        logger.info(f"Loading pretrained SCM weights from {args.scm_path}")
        controlnet.controlnet_cond_embedding.load_state_dict(
            torch.load(args.scm_path, weights_only=True)
        )

    # ------------------------------------------------------------------
    # PromptCL controller (only built when InfoNCE is enabled)
    # ------------------------------------------------------------------
    opt = copy.deepcopy(args)
    opt.presudo_words_infonce = args.presudo_words_infonce.split(",")
    opt.adj_aug_infonce = args.adj_aug_infonce.split(",")

    mcpl_controller = None
    if opt.presudo_words_infonce[0] != "":
        # ``LocalMask`` is built per-step inside the training loop using
        # real prompts; here we only need a controller stub so the rest of
        # the code can dispatch on ``mcpl_controller is not None``.
        mcpl_controller = AttentionMask(local_blend=None)

    # ------------------------------------------------------------------
    # Trainable parameter selection
    # ------------------------------------------------------------------
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(True)

    if not args.causal_training:
        logger.info("Freezing controlnet.controlnet_cond_embedding (SCM head)")
        for p in controlnet.controlnet_cond_embedding.parameters():
            p.requires_grad = False
    else:
        logger.info("Training controlnet.controlnet_cond_embedding (SCM head)")

    if args.mcpl_training:
        logger.info("Training MCPL pseudo-token embeddings")
        # Only the embedding lookup is trainable; freeze the rest of CLIP.
        text_encoder.text_model.encoder.requires_grad_(False)
        text_encoder.text_model.final_layer_norm.requires_grad_(False)
        text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
        trainable_params = (
            list(text_encoder.get_input_embeddings().parameters())
            + list(filter(lambda p: p.requires_grad, controlnet.parameters()))
        )
    else:
        text_encoder.requires_grad_(False)
        trainable_params = list(filter(lambda p: p.requires_grad, controlnet.parameters()))

    if args.gradient_checkpointing:
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    logger.info(
        f"Trainable params: {sum(p.numel() for p in trainable_params if p.requires_grad)}"
    )
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ------------------------------------------------------------------
    # Dataset / dataloader
    # ------------------------------------------------------------------
    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=args.placeholder_string,
        learnable_property=args.learnable_property,
        center_crop=False,
        flip_p=0.0,
        random_article=False,
        set="train",
        dataset=args.dataset,
        random_prompt_template=args.random_prompt_template,
    )

    sampler = build_balanced_sampler(args.dataset, train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=sampler,
        # DataLoader rejects ``shuffle=True`` together with a sampler.
        shuffle=sampler is None,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # LR scheduler & precision
    # ------------------------------------------------------------------
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

    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
        accelerator.mixed_precision, torch.float32
    )

    # SCM adjacency mask for the causal head.
    A_matrix = _build_a_matrix(
        args.dataset, args.num_causal_concepts, weight_dtype, accelerator.device
    )
    controlnet.controlnet_cond_embedding.update_mask(A_matrix)

    # ------------------------------------------------------------------
    # Accelerate prepare()
    # ------------------------------------------------------------------
    controlnet.train()
    if args.mcpl_training:
        text_encoder.train()
        controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            controlnet, optimizer, train_dataloader, lr_scheduler
        )
        text_encoder = accelerator.prepare(text_encoder)
        accelerate_models = [controlnet, text_encoder]
    else:
        text_encoder.eval()
        controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            controlnet, optimizer, train_dataloader, lr_scheduler
        )
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        accelerate_models = [controlnet]

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # Recompute step counts now that the dataloader has been wrapped.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion", config=vars(args))

    total_batch_size = (
        args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total batch size (parallel + accum) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # ------------------------------------------------------------------
    # Resume-from-checkpoint
    # ------------------------------------------------------------------
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting fresh."
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
        disable=not accelerator.is_local_main_process,
    )

    # Snapshot of original token-embedding weights — we'll reset every
    # non-pseudo-token row each step so MCPL only touches its own slots.
    orig_embeds_params = (
        accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
    )
    switched_to_uniform = False

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(first_epoch, args.num_train_epochs):
        # CelebA-style schedule: warm up with a balanced sampler, then swap
        # to uniform shuffling once the rare-class loss has settled.
        if (
            needs_uniform_switch(args.dataset)
            and epoch > 12
            and not switched_to_uniform
        ):
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.train_batch_size,
                shuffle=True,
                num_workers=args.dataloader_num_workers,
                collate_fn=collate_fn,
                drop_last=False,
            )
            train_dataloader = accelerator.prepare(train_dataloader)
            switched_to_uniform = True
            accelerator.wait_for_everyone()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(accelerate_models):
                controlnet.train()
                if args.mcpl_training:
                    text_encoder.train()

                # 1) image → latent
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor

                # 2) sample noise + timestep
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Sanity-check that all pseudo tokens really appear in the
                # batch's input_ids before we try to inject embeddings.
                check_elements_exist(presudo_token_ids, batch["input_ids"][0])

                # 3) build the per-step LocalMask for PromptCL
                if mcpl_controller is not None:
                    prompt = batch["input_ids"].clone().detach()
                    mcpl_controller.local_blend = LocalMask(
                        tokenizer,
                        prompt,
                        presudo_token_ids,
                        opt.attn_mask_type,
                        [""],
                        presudo_token_ids,
                        [""],
                        accelerator.device,
                    )

                batch_input_ids = batch["input_ids"].clone()
                cn_module = controlnet.module if hasattr(controlnet, "module") else controlnet

                # 4) SCM head produces the causal conditioning vector +
                #    auxiliary causal_loss (DAG / reconstruction terms).
                controlnet_cond, causal_loss = cn_module.controlnet_cond_embedding(
                    batch["label"].to(dtype=weight_dtype)
                )

                # 5) inject the causal vector into the text-encoder hidden
                #    states at the pseudo-token positions.
                cond_embeddings = prompt_aligned_injection_diff(
                    text_encoder=text_encoder,
                    inputs_id=batch["input_ids"],
                    presudo_token_ids=presudo_token_ids,
                    controlnet_cond=controlnet_cond,
                    task_cond=cn_module.task_cond,
                    dataset=cn_module.dataset,
                    dtype=weight_dtype,
                )

                # 6) forward through the (causal) ControlNet
                (down_block_res_samples, mid_block_res_sample), _, _ = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=cond_embeddings,
                    controlnet_cond=None,
                    return_dict=False,
                )

                # 7) UNet text-conditioning: 'global' uses the injected
                #    embeddings; 'local' falls back to the raw text encoder.
                if "global" in args.task_cond:
                    encoder_hidden_states = cond_embeddings
                else:
                    encoder_hidden_states = text_encoder(batch_input_ids)[0].to(dtype=weight_dtype)

                # 8) UNet noise prediction with ControlNet residuals
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        s.to(dtype=weight_dtype) for s in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

                # 9) loss
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                mse_loss_log = loss.detach().item()
                causal_loss_log = None
                nll_loss_log = None

                if args.causal_training:
                    loss = loss + causal_loss.sum(-1).mean()
                    causal_loss_log = causal_loss.detach().mean(0).cpu().numpy()

                if (
                    mcpl_controller is not None
                    and mcpl_controller.local_blend.presudo_words_infonce[0] != ""
                ):
                    nll_loss = mcpl_utils.prompt_contrastive_loss(
                        mcpl_controller, encoder_hidden_states, opt
                    )
                    nll_loss_log = nll_loss.detach().item()
                    loss = loss + nll_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Pin every non-pseudo-token row back to its original value
                # so MCPL can only move the embeddings it owns.
                index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                if args.mcpl_training:
                    index_no_updates[presudo_token_ids] = False
                with torch.no_grad():
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates]

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.save_steps == 0:
                        ext = "bin" if args.no_safe_serialization else "safetensors"
                        save_progress(
                            text_encoder, presudo_token_ids, accelerator, tokenizer,
                            os.path.join(args.output_dir, f"learned_embeds-steps-{global_step}.{ext}"),
                            safe_serialization=not args.no_safe_serialization,
                        )
                        net_save_path = os.path.join(
                            args.output_dir, f"controlnet-steps-{global_step}.{ext}"
                        )
                        unwrap_model(controlnet).save_pretrained(net_save_path)

                    if global_step % args.checkpointing_steps == 0:
                        # Trim old accelerate state checkpoints first.
                        if args.checkpoints_total_limit is not None:
                            checkpoints = [
                                d for d in os.listdir(args.output_dir)
                                if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                for ckpt in checkpoints[:num_to_remove]:
                                    shutil.rmtree(os.path.join(args.output_dir, ckpt))
                                logger.info(f"Pruned {num_to_remove} old checkpoint(s)")
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "recons_loss": mse_loss_log,
                "causal_loss": causal_loss_log,
                "nll_loss": nll_loss_log,
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            with open(os.path.join(args.output_dir, "logs.txt"), "a") as log_file:
                log_file.write(f"Step {global_step}: {logs}\n")

            if global_step >= args.max_train_steps:
                accelerator.wait_for_everyone()
                break

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        ext = "bin" if args.no_safe_serialization else "safetensors"
        save_progress(
            text_encoder, presudo_token_ids, accelerator, tokenizer,
            os.path.join(args.output_dir, f"learned_embeds.{ext}"),
            safe_serialization=not args.no_safe_serialization,
        )
        unwrap_model(controlnet).save_pretrained(
            os.path.join(args.output_dir, f"controlnet.{ext}")
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
