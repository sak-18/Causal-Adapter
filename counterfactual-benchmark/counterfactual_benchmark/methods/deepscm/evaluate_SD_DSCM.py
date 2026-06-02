# Copyright AstraZeneca UK Ltd. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified Stable-Diffusion / DeepSCM counterfactual evaluation script.

Evaluates the Causal-Adapter SD generator with the benchmark metrics
(Effectiveness, Composition, Reverse, FID / minimality). Behaviour is selected
through CLI flags along two orthogonal axes:

    * editing backend  --  ``--editing standard`` (DDIM editing) vs
                           ``--editing p2p``      (Prompt-to-Prompt editing)
    * reverse metrics  --  ``--reverse`` adds the backward edit (cf -> factual)
                           and the IDP / reverse / composition LPIPS+L1 metrics.

Other run knobs:
    --max-batches      debug cap on the number of evaluated batches (default: all)
    --output-root      root folder for saved metrics/tensors (default: saved_benchmark)
    --anti-classifier  use the anti-causal celeba predictor + *_anti.json configs
    --controlnet_path  trained Causal-ControlNet checkpoint
    --mcpl_embedding_path  learned MCPL embeddings

See the counterfactual-benchmark README for full usage and example commands.
"""
import os
from pathlib import Path

# --- path bootstrap (must happen before any first-party import) -------------
from _paths import bootstrap, REPO_ROOT, DEEPSCM_CONFIG_DIR
bootstrap()

os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / "counterfactual-benchmark" / ".cache" / "torch"))

import sys
import argparse
import random
import pickle
from typing import Dict, List
from json import load
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms as tfms

from accelerate import Accelerator
from transformers import CLIPTokenizer
from diffusers import StableDiffusionCausalControlNetPipeline, Causal_ControlNetModel, DDIMScheduler

from causal_modules import ddim_modules
from model import SCM
from models.classifiers.classifier import Classifier
from models.classifiers.celeba_classifier import CelebaClassifier, Celeba_anticausal_Classifier
from models.classifiers.celeba_complex_classifier import CelebaComplexClassifier
from models.classifiers.pendulum_classifier import PendClassifier
from models.classifiers.adni_classifier import ADNIClassifier
from ctf_datasets.morphomnist.dataset import MorphoMNISTLike
from ctf_datasets.celeba_hq.dataset_SD import Celebahq
from ctf_datasets.celeba.dataset_SD import Celeba
from ctf_datasets.adni.dataset_SD import ADNI, bin_array, ordinal_array
from ctf_datasets.pendulum.dataset_SD import PendulumLike
from ctf_datasets.transforms import ReturnDictTransform, get_attribute_ids
from evaluation.metrics.composition_SD import composition, lipips_identity
from evaluation.metrics.minimality_SD import minimality
from evaluation.embeddings.embeddings import get_embedding_model, get_embedding_fn
from evaluation.metrics.fid_SD import fid
from evaluation.metrics.effectiveness_SD import effectiveness
from ctf_datasets.morphomnist.dataset import unnormalize as unnormalize_morphomnist
from ctf_datasets.celeba.dataset_SD import unnormalize as unnormalize_celeba
from ctf_datasets.adni.dataset import unnormalize as unnormalize_adni
from ctf_datasets.pendulum.dataset_SD import unnormalize as unnormalize_pend

torch.multiprocessing.set_sharing_strategy('file_system')

rng = np.random.default_rng()
device = torch.device("cuda")

# dataset name (as stored in the config "dataset" field) -> (Dataset class, unnormalize fn)
dataclass_mapping = {
    "morphomnist": (MorphoMNISTLike, unnormalize_morphomnist),
    "celeba": (Celeba, unnormalize_celeba),
    "adni": (ADNI, unnormalize_adni),
    "pendulum": (PendulumLike, unnormalize_pend),
    "celebahq": (Celebahq, unnormalize_celeba),
}


def unnormalize_0_1(tensor):
    """Unnormalize a tensor normalized with Normalize([0.5], [0.5]): [-1, 1] -> [0, 1]."""
    return tensor * 0.5 + 0.5


# ---------------------------------------------------------------------------
# Qualitative samples
# ---------------------------------------------------------------------------
def produce_qualitative_samples(pipe, prompt, dataset, scm, parents, intervention_source, unnormalize_fn,
                                num=20, show_difference=False, args=None):
    from evaluation.metrics.utils import save_plots
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.dataloader_num_workers)

    fig_idx = 0
    for i, (batch, idx) in tqdm(enumerate(data_loader)):
        if args.max_batches is not None and i >= args.max_batches:
            break
        first_img = batch.copy()
        first_img["image"] = ddim_modules.resize_tensor(first_img["image"], pipe, args.dataset, return_tensor=True, mode='bicubic')
        res = [first_img]
        # `parents` are the attributes; intervene on each in turn for the grid
        for do_parent in parents:
            cf = produce_counterfactuals(pipe, prompt, batch, scm, do_parent, intervention_source,
                                         force_change=True, possible_values=dataset.possible_values, args=args)
            # produce_counterfactuals may return a tuple when --reverse is on; keep only the cf batch
            res.append(cf[0] if isinstance(cf, tuple) else cf)
        save_plots(res, fig_idx, parents, unnormalize_fn, show_difference=show_difference)
        fig_idx += 1
    return


# ---------------------------------------------------------------------------
# Composition metric
# ---------------------------------------------------------------------------
def evaluate_composition(pipe, prompt, accelerator, test_set: Dataset, unnormalize_fn, batch_size: int,
                         cycles: List[int], scm: nn.Module, save_dir: str = "composition_samples",
                         embedding=None, embedding_fn=None, args=None):

    test_data_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=args.dataloader_num_workers, drop_last=True
    )
    test_data_loader = accelerator.prepare(test_data_loader)

    composition_scores_all = []   # List[Dict[int, np.ndarray]]
    l1_scores_int8_all = []
    l1_scores_uint8_all = []

    for i, (factual_batch, idx) in enumerate(tqdm(test_data_loader)):
        if args.max_batches is not None and i >= args.max_batches:
            break

        return_dict, image_batch = composition(
            pipe, prompt, args.dataset, factual_batch, unnormalize_fn,
            method=scm, cycles=cycles, embedding=embedding, embedding_fn=embedding_fn, args=args
        )
        composition_scores_all.append(return_dict['lpips'])
        l1_scores_uint8_all.append(return_dict['l1_uint8'])
        l1_scores_int8_all.append(return_dict['l1_int8'])

    # gather metric scores across processes, keyed per composition cycle
    def gather_nested_dict_list(dict_list):
        out = {cycle: [] for cycle in cycles}
        for entry in dict_list:
            for cycle in cycles:
                out[cycle].append(torch.tensor(entry[cycle]))
        for cycle in out:
            out[cycle] = torch.cat(out[cycle], dim=0).to(accelerator.device)
            out[cycle] = accelerator.gather(out[cycle]).cpu().numpy()
        return out

    composition_scores = gather_nested_dict_list(composition_scores_all)
    l1_scores_uint8 = gather_nested_dict_list(l1_scores_uint8_all)
    l1_scores_int8 = gather_nested_dict_list(l1_scores_int8_all)

    if accelerator.is_main_process:
        output_dir = os.path.join(experiment_output_dir(args), 'composition')
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, 'compositon.txt')
        with open(log_path, 'w') as f:
            f.write('')  # clear / create

        for cycle in cycles:
            comp_msg = f"Average lpips composition score for {cycle} cycles: mean {round(np.mean(composition_scores[cycle]), 3):.3f} std {round(np.std(composition_scores[cycle]), 3):.3f}"
            l1_uint8_msg = f"Average l1 uint8 score for {cycle} cycles: mean {round(np.mean(l1_scores_uint8[cycle]), 3):.3f} std {round(np.std(l1_scores_uint8[cycle]), 3):.3f}"
            l1_int8_msg = f"Average l1 int8 score for {cycle} cycles: mean {round(np.mean(l1_scores_int8[cycle]), 3):.3f} std {round(np.std(l1_scores_int8[cycle]), 3):.3f}"
            print(comp_msg)
            print(l1_uint8_msg)
            print(l1_int8_msg)
            with open(log_path, 'a') as f:
                f.write(comp_msg + "\n")
                f.write(l1_uint8_msg + "\n")
                f.write(l1_int8_msg + "\n")
    accelerator.wait_for_everyone()
    return


def different_value(possible_values, value, bins, attribute):
    """Return a boolean mask over `possible_values` selecting values that differ from `value`.

    For binned attributes, "different" means falling in a different bin.
    """
    if bins is not None and attribute in bins:
        return np.digitize(possible_values, bins[attribute]) != np.searchsorted(bins[attribute], value)
    else:
        return possible_values != value


# ---------------------------------------------------------------------------
# Counterfactual generation (the heart of the pipeline)
# ---------------------------------------------------------------------------
def _sample_interventions(do_parent, factual_batch, possible_values, bins, intervention_source,
                          force_change, device):
    """Sample intervention values for `do_parent`.

    Shared between all editing backends. Returns the `interventions` dict.
    """
    if force_change:
        possible_values = possible_values[do_parent]
        values = factual_batch[do_parent].cpu()
        if do_parent in ["pendulum", "light", "shadow_length", "shadow_position"]:
            # follow the causaldiff sampling ranges/scales
            pend_scale = {
                "pendulum": [2, 42], "light": [104, 44],
                "shadow_length": [7.5, 4.5], "shadow_position": [11, 8],
            }
            sampled_tensor = torch.tensor([
                (np.random.uniform(-40, 44) - pend_scale[do_parent][0]) / pend_scale[do_parent][1]
                if do_parent == "pendulum" else
                (np.random.uniform(60, 148) - pend_scale[do_parent][0]) / pend_scale[do_parent][1]
                if do_parent == "light" else
                (np.random.uniform(3, 9) - pend_scale[do_parent][0]) / pend_scale[do_parent][1]
                if do_parent == "shadow_length" else
                (np.random.uniform(3, 15) - pend_scale[do_parent][0]) / pend_scale[do_parent][1]
                for _ in values
            ], dtype=torch.float32).unsqueeze(1).to(device)
            return {do_parent: sampled_tensor}
        elif do_parent not in ["digit", "apoE", "slice"]:
            return {do_parent: torch.cat([torch.tensor(np.random.choice(possible_values[different_value(possible_values, value, bins, do_parent)])).unsqueeze(0)
                                          for value in values]).view(-1).unsqueeze(1).to(device)}
        else:
            return {do_parent: torch.cat([torch.tensor(rng.choice(possible_values[torch.where((different_value(possible_values, value, bins, do_parent)).any(dim=1))], axis=0)).unsqueeze(0)
                                          for value in values]).to(device)}
    else:
        batch_size = factual_batch["image"].shape[0]
        idxs = torch.randperm(len(intervention_source))[:batch_size]  # random interventions from train set
        return {do_parent: torch.cat([intervention_source[id][do_parent] for id in idxs]).view(-1).unsqueeze(1).to(device)
                if do_parent not in ["digit", "apoE", "slice"]
                else torch.cat([intervention_source[id][do_parent].unsqueeze(0).to(device) for id in idxs])}


def _p2p_blend_params(dataset, intervention_indx):
    """Per-intervention Prompt-to-Prompt blend / replace schedule (celeba-complex tuned)."""
    if dataset in ['celeA_complex']:
        if intervention_indx in [0, 1]:
            return {'start_blend': 0.0, 'th': (0.3, 0.6)}, 0.1, 0.1
        elif intervention_indx in [2]:
            return {'start_blend': 0.0, 'th': (0.1, 0.7)}, 0.1, 0.1
        elif intervention_indx in [3]:
            return {'start_blend': 0.0, 'th': (0.45, 0.6)}, 0.1, 0.1
    return {'start_blend': 0.0, 'th': (0.1, 0.0)}, 0.0, 0.0


def produce_counterfactuals(pipe, prompt, factual_batch: torch.Tensor, scm: nn.Module, do_parent: str,
                            intervention_source: Dataset, force_change: bool = False, possible_values=None,
                            device: str = 'cuda', bins=None, args=None):
    """Abduct -> intervene -> generate one counterfactual batch.

    Returns:
        * if ``args.reverse`` is False:  ``counterfactual_batch``
        * if ``args.reverse`` is True:   ``(counterfactual_batch, recons_or_None, recovered_from_cf)``
          where ``recons_or_None`` is the forward reconstruction (only the P2P
          backend produces it here; the DDIM backend computes it later) and
          ``recovered_from_cf`` is the backward edit cf -> factual.
    """
    factual_batch = {k: v.to(device) for k, v in factual_batch.items()}
    batch_size = factual_batch["image"].shape[0]

    interventions = _sample_interventions(do_parent, factual_batch, possible_values, bins,
                                           intervention_source, force_change, device)

    attr_keys = ddim_modules.get_dataset_attrs(args.dataset)
    label = torch.cat([factual_batch[k] for k in attr_keys], dim=1)
    counterfactual_batch = factual_batch.copy()
    intervention_indx = attr_keys.index(list(interventions.keys())[0])
    intervention_values = interventions[list(interventions.keys())[0]].squeeze()

    # infer exogenous noise / build the DSCM counterfactual labels
    if args.dataset == 'celebahq_simple':
        # celeba-HQ uses a hand-specified label edit instead of the SCM abduction
        dscm_labels = label.clone()
        dscm_labels[:, intervention_indx] = intervention_values
        if intervention_indx == 0:
            # enforce attr2 follows the (complemented) value of attr0, per sample
            dscm_labels[:, 2] = intervention_values
    else:
        abducted_noise = scm.encode(**factual_batch)
        scm_counterfactual_batch = scm.decode(interventions, **abducted_noise)
        dscm_labels = torch.cat([scm_counterfactual_batch[k] for k in attr_keys], dim=1)

    if args.dataset in ['celeA_complex'] and int(intervention_indx) == 2:
        row_index = torch.where(dscm_labels[:, 1] == 0)[0]
        dscm_labels[row_index, intervention_indx] = -1

    # ---- forward edit: factual -> counterfactual ----
    if args.editing == 'p2p':
        blend_params, cross_replace_steps, self_replace_steps = _p2p_blend_params(args.dataset, intervention_indx)
        normalized_tensor, _, causal_cond, _ = ddim_modules.P2P_editing(
            pipe, factual_batch["image"], label.clone(), prompt, args.presudo_list,
            num_steps=int(args.num_steps), invert_guidance_scale=1.0,
            set_guidance_scale=args.guidance_scale, intervention_indx=intervention_indx,
            intervention_values=intervention_values, return_PIL=False,
            blend_word=args.NTI, blend_params=blend_params, disentangle=True,
            cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps,
            DSCM_labels=dscm_labels.unsqueeze(2), return_recons_counter=True)
    else:
        normalized_tensor, _, causal_cond, _ = ddim_modules.ddim_editing(
            pipe, factual_batch["image"], label.clone(), prompt, num_steps=int(args.num_steps),
            invert_guidance_scale=1.0, set_guidance_scale=args.guidance_scale,
            intervention_indx=intervention_indx, intervention_values=intervention_values,
            return_PIL=False, DSCM_labels=dscm_labels, pnp_inversion=args.NTI)

    resized_input = ddim_modules.resize_tensor(normalized_tensor, pipe, args.dataset, return_tensor=True)
    if args.dataset in ['celeA_complex']:
        # correct OOD conditioning values to {0,1}
        causal_cond = (causal_cond > 0.5).float()

    # P2P_editing returns concat[recons, counterfactual]; DDIM returns just the counterfactual.
    if args.editing == 'p2p':
        counterfactual_batch["image"] = resized_input[batch_size:]
        recons_input = resized_input[:batch_size]
    else:
        counterfactual_batch["image"] = resized_input
        recons_input = None

    # write the (new) attribute values into the counterfactual batch
    if args.dataset in ['ADNI', 'MorphoMNIST']:
        start = 0
        attr_dims = args.attribute_size
        for attr in attr_keys:
            dim = attr_dims[attr]
            value = causal_cond[:, start:start + dim]
            value = value.squeeze(1) if dim == 1 else value.squeeze(2)
            counterfactual_batch[attr] = value
            start += dim
    else:
        for j, attr in enumerate(attr_keys):
            counterfactual_batch[attr] = causal_cond[:, j]

    if not args.reverse:
        return counterfactual_batch

    # ---- backward edit: counterfactual -> factual (reversibility) ----
    do_denormalize = [True] * normalized_tensor.shape[0]
    output_img = pipe.image_processor.postprocess(normalized_tensor.detach(), do_denormalize=do_denormalize)
    tensor_image_transforms = tfms.Compose([tfms.ToTensor(), tfms.Normalize([0.5], [0.5])])
    cf_tensor = torch.stack([tensor_image_transforms(img) for img in output_img], dim=0)

    if args.editing == 'p2p':
        out_tensor_back, _, _, _ = ddim_modules.P2P_editing(
            pipe, cf_tensor[batch_size:], dscm_labels.clone(), prompt, args.presudo_list,
            num_steps=int(args.num_steps), invert_guidance_scale=1.0,
            set_guidance_scale=args.guidance_scale, intervention_indx=intervention_indx,
            intervention_values=intervention_values, return_PIL=False,
            blend_word=args.NTI, blend_params=blend_params, disentangle=True,
            cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps,
            DSCM_labels=label.clone().unsqueeze(2).to(pipe.device), return_recons_counter=True)
        recovered_from_cf = ddim_modules.resize_tensor(out_tensor_back, pipe, args.dataset, return_tensor=True)[batch_size:]
    else:
        out_tensor_back, _, _, _ = ddim_modules.ddim_editing(
            pipe, cf_tensor, dscm_labels.clone(), prompt, num_steps=int(args.num_steps),
            invert_guidance_scale=1.0, set_guidance_scale=args.guidance_scale,
            intervention_indx=intervention_indx, intervention_values=intervention_values,
            return_PIL=False, DSCM_labels=label.clone(), pnp_inversion=args.NTI)
        recovered_from_cf = ddim_modules.resize_tensor(out_tensor_back, pipe, args.dataset, return_tensor=True)

    return counterfactual_batch, recons_input, recovered_from_cf


# ---------------------------------------------------------------------------
# Effectiveness metric (+ optional reversibility metrics)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_effectiveness(pipe, prompt, test_set: Dataset, unnormalize_fn, batch_size: int, accelerator,
                           scm: nn.Module, attributes: List[str], do_parent: str, intervention_source: Dataset,
                           predictors: Dict[str, Classifier], dataset: str, embedding_fn=None, args=None):
    predictors = {k: accelerator.prepare(v) for k, v in predictors.items()}

    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=args.dataloader_num_workers)
    test_data_loader = accelerator.prepare(test_data_loader)
    effectiveness_scores = {attr_key: [] for attr_key in attributes}
    all_results = {
        'predictions': {attr: [] for attr in attributes},
        'targets': {attr: [] for attr in attributes},
    }
    all_indices = []
    lpips_list = []                                              # non-reverse LPIPS identity
    IDP_l1_list, IDP_lpips_list = [], []                         # reverse-mode accumulators
    reverse_l1_list, reverse_lpips_list = [], []
    compos_l1_list, compos_lpips_list = [], []

    for i, (factual_batch, idx) in enumerate(tqdm(test_data_loader)):
        if args.max_batches is not None and i >= args.max_batches:
            break

        cf_out = produce_counterfactuals(pipe, prompt, factual_batch, scm, do_parent, intervention_source,
                                         force_change=True, possible_values=test_set.possible_values,
                                         bins=test_set.bins, args=args)
        if args.reverse:
            counterfactuals, recons_factual, recovered_from_cf = cf_out
        else:
            counterfactuals = cf_out

        e_score, results_dict = effectiveness(counterfactuals, unnormalize_fn, predictors, dataset)

        if args.embeddings == 'lpips':
            if args.reverse:
                factual_image = ddim_modules.resize_tensor(factual_batch['image'], pipe, args.dataset, return_tensor=True)
                # the DDIM backend doesn't return a forward reconstruction; compute it here
                if recons_factual is None:
                    if args.dataset == 'celebahq_simple':
                        _, recons_factual = composition(
                            pipe, prompt, args.dataset, factual_batch, unnormalize_fn,
                            method=scm, cycles=[1], embedding=args.embeddings, embedding_fn=embedding_fn, args=args)
                    else:
                        recons_factual = ddim_modules.resize_tensor(factual_batch["image"], pipe, args.dataset, return_tensor=True)
                IDP_dic = lipips_identity(counterfactuals['image'], recons_factual, embedding_fn)         # identity preservation
                reverse_dic = lipips_identity(factual_image, recovered_from_cf, embedding_fn)             # reversibility
                compos_dic = lipips_identity(factual_image, recons_factual, embedding_fn)                 # reconstruction
                IDP_l1_list.extend(IDP_dic['L1']);         IDP_lpips_list.extend(IDP_dic['LPIPS'])
                reverse_l1_list.extend(reverse_dic['L1']); reverse_lpips_list.extend(reverse_dic['LPIPS'])
                compos_l1_list.extend(compos_dic['L1']);   compos_lpips_list.extend(compos_dic['LPIPS'])
            else:
                # reconstruct factual to compute LPIPS identity-preservation
                if args.dataset == 'celebahq_simple':
                    _, resized_factual_image = composition(
                        pipe, prompt, args.dataset, factual_batch, unnormalize_fn,
                        method=scm, cycles=[1], embedding=args.embeddings, embedding_fn=embedding_fn, args=args)
                else:
                    resized_factual_image = ddim_modules.resize_tensor(factual_batch["image"], pipe, args.dataset, return_tensor=True)
                lipips_distance = lipips_identity(counterfactuals['image'], resized_factual_image, embedding_fn)
                lpips_list.extend(lipips_distance['LPIPS'])

        all_indices.extend(idx.cpu().numpy())
        for attr in attributes:
            effectiveness_scores[attr].append(e_score[attr])
            all_results['predictions'][attr].append(results_dict['predictions'][attr])
            all_results['targets'][attr].append(results_dict['targets'][attr])

    # gather predictions/targets across processes, sort + dedup by sample index
    for attr in attributes:
        preds = np.concatenate(all_results['predictions'][attr], axis=0)
        targets = np.concatenate(all_results['targets'][attr], axis=0)
        indices = np.array(all_indices)

        gathered_preds = accelerator.gather(torch.tensor(preds).to(accelerator.device)).cpu().numpy()
        gathered_targets = accelerator.gather(torch.tensor(targets).to(accelerator.device)).cpu().numpy()
        gathered_indices = accelerator.gather(torch.tensor(indices).to(accelerator.device)).cpu().numpy()

        if accelerator.is_main_process:
            sort_idx = np.argsort(gathered_indices)
            gathered_preds = gathered_preds[sort_idx]
            gathered_targets = gathered_targets[sort_idx]
            gathered_indices = gathered_indices[sort_idx]
            unique_indices, unique_pos = np.unique(gathered_indices, return_index=True)
            all_results['predictions'][attr] = gathered_preds[unique_pos]
            all_results['targets'][attr] = gathered_targets[unique_pos]

    def _gather_np(lst):
        return accelerator.gather(torch.tensor(np.array(lst)).to(accelerator.device)).cpu().numpy()

    if args.embeddings == 'lpips':
        if args.reverse:
            gathered = {
                'IDP_l1': _gather_np(IDP_l1_list), 'IDP_lpips': _gather_np(IDP_lpips_list),
                'reverse_l1': _gather_np(reverse_l1_list), 'reverse_lpips': _gather_np(reverse_lpips_list),
                'compos_l1': _gather_np(compos_l1_list), 'compos_lpips': _gather_np(compos_lpips_list),
            }
            if accelerator.is_main_process:
                all_results.update(gathered)
        else:
            gathered_lpips = _gather_np(lpips_list)
            if accelerator.is_main_process:
                all_results['lpips'] = gathered_lpips

    if accelerator.is_main_process:
        all_results['indices'] = unique_indices
        output_dir = os.path.join(experiment_output_dir(args), do_parent)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'final_results.pkl'), "wb") as f:
            pickle.dump(all_results, f)
        print(f"Effectiveness score do({do_parent}): {effectiveness_scores}")

    accelerator.wait_for_everyone()
    return


# ---------------------------------------------------------------------------
# FID (legacy helper) -- kept for reference; FID is normally computed via
# compute_FID.py. NOTE: this function references undefined data loaders and is
# not wired to run; preserved verbatim from the original scripts.
# ---------------------------------------------------------------------------
def evaluate_fid(real_set: Dataset, test_set: Dataset, batch_size: int, scm: nn.Module, attributes: List[str]):
    counterfactual_images = []
    for factual_batch in tqdm(test_data_loader):  # noqa: F821  (legacy: undefined, see note above)
        do_parent = random.choice(attributes)
        counterfactual_batch = produce_counterfactuals(factual_batch, scm, do_parent, intervention_source=real_set,
                                                        force_change=True, possible_values=test_set.possible_values, bins=real_set.bins)
        counterfactual_images.append(counterfactual_batch['image'])
    fid_score = fid(real_data_loader, counterfactual_images)  # noqa: F821
    print(f"FID: mean {round(np.mean(fid_score), 3):.3f} std {round(np.std(fid_score), 3):.3f}")
    return fid_score


# ---------------------------------------------------------------------------
# Minimality (also saves the counterfactual tensors used for FID)
# ---------------------------------------------------------------------------
def evaluate_minimality(pipe, prompt, accelerator, real_set: Dataset, test_set: Dataset, batch_size: int,
                        scm: nn.Module, attributes: List[str], embedding: str = None, embedding_fn=None, args=None):
    real_data_loader = torch.utils.data.DataLoader(real_set, batch_size=batch_size, shuffle=False, num_workers=args.dataloader_num_workers)
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=args.dataloader_num_workers)
    test_data_loader = accelerator.prepare(test_data_loader)

    factual_feats, factual_labels = [], []
    counterfactual_feats, counterfactual_labels = [], []
    intervention_list, counterfactual_images = [], []

    for i, (factual_batch, idx) in enumerate(tqdm(test_data_loader)):
        if args.max_batches is not None and i >= args.max_batches:
            break

        do_parent = random.choice(attributes)
        cf_out = produce_counterfactuals(pipe, prompt, factual_batch, scm, do_parent, intervention_source=real_set,
                                         force_change=True, possible_values=test_set.possible_values,
                                         bins=real_set.bins, args=args)
        counterfactual_batch = cf_out[0] if isinstance(cf_out, tuple) else cf_out

        resized_factual_image = ddim_modules.resize_tensor(factual_batch["image"], pipe, args.dataset, return_tensor=True)
        counterfactual_images.append(counterfactual_batch["image"])

        if embedding == "vae":
            factual_cond = torch.cat([factual_batch[att] for att in attributes], dim=1).to('cuda')
            counterfactual_cond = torch.cat([counterfactual_batch[att] for att in attributes], dim=1)
        elif "intensity" in factual_batch:
            factual_cond = factual_batch["intensity"].to('cuda')
            counterfactual_cond = counterfactual_batch["intensity"]
        else:
            factual_cond = None
            counterfactual_cond = None

        factual_features = embedding_fn(resized_factual_image.to('cuda'), factual_cond)
        counterfactual_features = embedding_fn(counterfactual_batch["image"], counterfactual_cond)
        factual_feats.append(torch.tensor(factual_features))
        counterfactual_feats.append(torch.tensor(counterfactual_features))

        if do_parent in ["digit", "apoE", "slice"]:
            if do_parent == "apoE":
                fac_labels = bin_array(factual_batch[do_parent], reverse=True).long().unsqueeze(1)
                counter_labels = bin_array(counterfactual_batch[do_parent], reverse=True).long().unsqueeze(1)
            elif do_parent == "slice":
                fac_labels = ordinal_array(factual_batch[do_parent], reverse=True).long().unsqueeze(1)
                counter_labels = ordinal_array(counterfactual_batch[do_parent], reverse=True).long().unsqueeze(1)
            factual_labels.append(fac_labels.to(torch.int))
            counterfactual_labels.append(counter_labels.to(torch.int))
        else:
            factual_labels.append(factual_batch[do_parent].to(torch.int))
            counterfactual_labels.append(counterfactual_batch[do_parent].to(torch.int))

        intervention_list.append([do_parent] * len(factual_batch["image"]))

    factual_feats = torch.cat(factual_feats, dim=0).to(accelerator.device)
    factual_labels = torch.cat(factual_labels, dim=0).to(accelerator.device)
    counterfactual_feats = torch.cat(counterfactual_feats, dim=0).to(accelerator.device)
    counterfactual_labels = torch.cat(counterfactual_labels, dim=0).to(accelerator.device)
    counterfactual_images = torch.stack(counterfactual_images, dim=0).to(accelerator.device)

    unique_attrs = {att: idx for idx, att in enumerate(attributes)}
    intervention_tensor = torch.tensor(
        [unique_attrs[attr] for sublist in intervention_list for attr in sublist], dtype=torch.int
    ).to(accelerator.device)

    factual_feats = accelerator.gather(factual_feats)
    factual_labels = accelerator.gather(factual_labels)
    counterfactual_feats = accelerator.gather(counterfactual_feats)
    counterfactual_labels = accelerator.gather(counterfactual_labels)
    counterfactual_images = accelerator.gather(counterfactual_images)
    intervention_tensor = accelerator.gather(intervention_tensor)

    if accelerator.is_main_process:
        factuals = list(zip(factual_feats.cpu().numpy(), factual_labels.cpu().numpy()))
        counterfactuals = list(zip(counterfactual_feats.cpu().numpy(), counterfactual_labels.cpu().numpy()))
        interventions = [attributes[idx] for idx in intervention_tensor.cpu().numpy()]

        output_dir = os.path.join(experiment_output_dir(args), 'FID')
        os.makedirs(output_dir, exist_ok=True)
        torch.save(counterfactual_images, os.path.join(output_dir, "counterfactual_tensors.pt"))
        torch.save({"factuals": factuals, "counterfactuals": counterfactuals, "interventions": interventions},
                   os.path.join(output_dir, "minimality.pt"))

        print('start Minimality calculation')
        minimality_scores, prob1s, prob2s = minimality(
            real=factuals, generated=counterfactuals, interventions=interventions,
            bins=real_set.bins, embedding=embedding)
        minimality_msg = f"Minimality score: mean {round(np.mean(minimality_scores), 3)}, std {round(np.std(minimality_scores), 3)}"
        prob_msg = f"Prob 1: {np.mean(prob1s)}, Prob 2: {np.mean(prob2s)}"
        print(minimality_msg)
        print(prob_msg)

        log_path = './SD_fid_results.txt'
        with open(log_path, 'w') as f:
            f.write('')
        with open(log_path, 'a') as f:
            f.write(minimality_msg + "\n")
            f.write(prob_msg + "\n")

    accelerator.wait_for_everyone()
    return


# ---------------------------------------------------------------------------
# Config / argument helpers
# ---------------------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def datasets_configs(dataset, anti_classifier=False):
    """Return (scm/vae config path, classifier config path) for a dataset.

    For celeba-HQ, ``anti_classifier`` selects the anti-causal predictor configs
    (``vae_anti.json`` / ``classifier_anti.json``), reproducing the original
    non-reverse celeba-HQ script. The reverse experiments used the plain configs.
    """
    config = classifier_config = None
    if 'celeA' in dataset:
        if 'complex' in dataset:
            config = str(DEEPSCM_CONFIG_DIR / "celeba" / "complex" / "vae.json")
            classifier_config = str(DEEPSCM_CONFIG_DIR / "celeba" / "complex" / "classifier.json")
    if 'celebahq' in dataset:
        if 'simple' in dataset:
            suffix = "_anti" if anti_classifier else ""
            config = str(DEEPSCM_CONFIG_DIR / "celebahq" / "simple" / f"vae{suffix}.json")
            classifier_config = str(DEEPSCM_CONFIG_DIR / "celebahq" / "simple" / f"classifier{suffix}.json")
    elif 'ADNI' in dataset:
        config = str(DEEPSCM_CONFIG_DIR / "adni" / "vae.json")
        classifier_config = str(DEEPSCM_CONFIG_DIR / "adni" / "classifier.json")
    elif 'pendulum' in dataset:
        config = str(DEEPSCM_CONFIG_DIR / "pendulum" / "vae.json")
        classifier_config = str(DEEPSCM_CONFIG_DIR / "pendulum" / "classifier.json")
    return config, classifier_config


def experiment_output_dir(args):
    """Build the per-run output directory (under ``--output-root``).

    Mirrors the original naming scheme. The tag encodes the editing backend and
    sampling settings so different runs do not clobber each other.
    """
    edit_tag = 'Blend' if args.editing == 'p2p' else 'NTI'
    run_name = 'DSCM_effectiveness_step{}_scale{}_{}{}_seed{}'.format(
        args.num_steps, args.guidance_scale, edit_tag, args.NTI, args.seed)
    return os.path.join(args.output_root, args.dataset, run_name)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default="celeA_complex", help="Experiment dataset key (celeA_complex, celebahq_simple, ADNI, pendulum).")
    parser.add_argument("--metrics", '-m', nargs="+", type=str,
                        choices=["composition", "effectiveness", "fid", "minimality"],
                        default=["composition", "effectiveness", "fid", "minimality"],
                        help="Metrics to calculate.")
    parser.add_argument("--editing", choices=["standard", "p2p"], default="standard",
                        help="Editing backend: 'standard' DDIM editing or 'p2p' Prompt-to-Prompt editing.")
    parser.add_argument("--reverse", action="store_true",
                        help="Also run the backward edit (cf->factual) and report IDP/reverse/composition LPIPS+L1 metrics.")
    parser.add_argument("--anti-classifier", action="store_true",
                        help="Use the anti-causal celeba-HQ predictor and *_anti.json configs.")
    parser.add_argument("--output-root", type=str, default="saved_benchmark",
                        help="Root folder for saved metrics/tensors (e.g. saved_benchmark, saved_ablation).")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Debug cap on the number of evaluated batches per metric. Default: evaluate the full set.")
    parser.add_argument("--seed", type=int, default=40, help="Random seed.")
    parser.add_argument("--bs", type=int, default=6, help="Batch size.")
    parser.add_argument("--cycles", '-cc', nargs="+", type=int, default=[1, 10], help="Composition cycles.")
    parser.add_argument("--qualitative", '-qn', type=int, default=0, help="Number of qualitative results to produce.")
    parser.add_argument("--show-difference", '-sd', action='store_true', help="Show counterfactual-factual difference on qualitative results.")
    parser.add_argument("--embeddings", type=str, choices=["vgg", "clfs", "vae", "lpips", "clip"],
                        help="Embeddings for the composition/identity metric. If unset, distance is computed in image space.")
    parser.add_argument("--sampling-temperature", '-temp', type=float, default=0.1, help="Sampling temperature, used for VAE, HVAE.")
    parser.add_argument("--num_steps", type=float, default=50, help="Number of diffusion sampling steps.")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Classifier-free guidance scale.")
    parser.add_argument("--NTI", type=str2bool, default=False, help="Enable null-text inversion / blend word (P2P).")
    parser.add_argument("--controlnet_path", type=str, default=None, help="Path to the trained Causal-ControlNet checkpoint.")
    parser.add_argument("--mcpl_embedding_path", type=str, default=None, help="Path to the learned MCPL embeddings.")
    parser.add_argument("--resolution", type=int, default=256, help="Image resolution (default 256).")
    parser.add_argument("--dataloader_num_workers", type=int, default=8,
                        help="Number of subprocesses for data loading. 0 = load in the main process.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset-specific prompt / causal-graph adjacency setup
# ---------------------------------------------------------------------------
def dataset_prompt_and_graph(dataset):
    """Return (prompt, pseudo-words csv, adjacency A_matrix, optional cond_path) per dataset."""
    cond_path = None
    if dataset in ['celeA_complex']:
        cond_path = str(REPO_ROOT / "logs" / "logs_celeA_complex_all" /
                        "2025-04-23T21-11-19-causalnet_pretrain" / "best_model.pt")
        A_matrix = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.float32).to(device)
        prompt = 'a human of @ and * and & and !'
        presudo_words = '@,*,&,!'
    elif dataset in ['celebahq_simple']:
        prompt = 'a human of @ and * and & and ! and % and $ and #'
        presudo_words = '@,*,&,!,%,$,#'
        A_matrix = torch.zeros((7, 7), dtype=torch.float32).to(device)
    elif dataset == 'ADNI':
        A_matrix = torch.tensor([[0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=torch.float32).to(device)
        prompt = 'a mri image of @ and * and &'
        prompt += (' ' + prompt[-1]) * (10 - 1)
        presudo_words = '@,*,&'
    elif dataset == 'pendulum':
        A_matrix = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.float32).to(device)
        prompt = 'a image of @ and * and & and !'
        presudo_words = '@,*,&,!'
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return prompt, presudo_words, A_matrix, cond_path


def build_predictors(dataset, attribute_size, intervention_attrs, config_cls, args):
    """Instantiate the anti-causal predictors and load their checkpoints."""
    if dataset == "morphomnist":
        predictors = {atr: Classifier(attr=atr, num_outputs=attribute_size[atr],
                                      context_dim=len(list(config_cls["anticausal_graph"][atr])))
                      for atr in attribute_size.keys()}
    elif dataset in ["celeba", 'celebahq']:
        if sum(attribute_size.values()) == 4:
            predictors = {atr: CelebaComplexClassifier(attr=atr, context_dim=len(list(config_cls["anticausal_graph"][atr])),
                                                       num_outputs=config_cls["attribute_size"][atr],
                                                       lr=config_cls["lr"], version=config_cls["version"]) for atr in attribute_size.keys()}
        elif args.anti_classifier:
            predictors = {atr: Celeba_anticausal_Classifier(attr=atr, num_outputs=config_cls["attribute_size"][atr],
                                                            lr=config_cls["lr"], pretrain=False) for atr in intervention_attrs}
        else:
            predictors = {atr: CelebaClassifier(attr=atr, num_outputs=config_cls["attribute_size"][atr],
                                                lr=config_cls["lr"], pretrain=False) for atr in intervention_attrs}
    elif dataset == "pendulum":
        predictors = {atr: PendClassifier(attr=atr, num_outputs=attribute_size[atr],
                                          context_dim=len(list(config_cls["anticausal_graph"][atr])))
                      for atr in attribute_size.keys()}
    else:
        attribute_ids = get_attribute_ids(attribute_size)
        predictors = {atr: ADNIClassifier(attr=atr, num_outputs=config_cls["attribute_size"][atr],
                                           children=config_cls["anticausal_graph"][atr],
                                           num_slices=config_cls["attribute_size"]['slice'],
                                           attribute_ids=attribute_ids, arch=config_cls['arch']) for atr in attribute_size.keys()}

    for key, cls in predictors.items():
        file_name = next((file for file in os.listdir(config_cls["ckpt_path"]) if file.startswith(key)), None)
        print(key, file_name)
        cls.load_state_dict(torch.load(config_cls["ckpt_path"] + file_name, map_location=torch.device('cuda'))["state_dict"])
        cls.to('cuda')
        cls.eval()
    return predictors


if __name__ == "__main__":
    args = parse_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(args)

    base_model_path = os.environ.get("CAUSAL_ADAPTER_SD15_BASE_MODEL", "lambdalabs/miniSD-diffusers")
    accelerator = Accelerator()

    # ---- build the diffusion pipeline ----
    controlnet = Causal_ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=torch.float32)
    prompt, presudo_words, A_matrix, cond_path = dataset_prompt_and_graph(args.dataset)
    if cond_path is not None and os.path.exists(cond_path):
        print('load pretrained controlnet cond-embedding weights')
        controlnet.controlnet_cond_embedding.load_state_dict(torch.load(cond_path, weights_only=True))
    print(prompt)
    controlnet.controlnet_cond_embedding.update_mask(A_matrix)
    controlnet.eval()

    presudo_list = presudo_words.split(',')
    args.presudo_list = presudo_list
    tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
    presudo_token_ids = tokenizer.encode(' '.join(presudo_list), add_special_tokens=False)
    embed_control_manager_bool = 'image' not in controlnet.task_cond
    text_encoder = ddim_modules.load_mcpl_embeddings(base_model_path, tokenizer, args.mcpl_embedding_path,
                                                     presudo_token_ids, embed_control=embed_control_manager_bool)

    pipe = StableDiffusionCausalControlNetPipeline.from_pretrained(
        base_model_path, controlnet=controlnet, text_encoder=text_encoder, torch_dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe = pipe.to(accelerator.device)

    # ---- load configs / SCM / datasets ----
    config_path, classifier_config = datasets_configs(args.dataset, anti_classifier=args.anti_classifier)
    with open(classifier_config, 'r') as f:
        config_cls = load(f)
    with open(config_path, 'r') as f:
        config = load(f)

    dataset = config["dataset"]
    attribute_size = config["attribute_size"]
    args.attribute_size = attribute_size

    models = {}
    for variable in config["causal_graph"].keys():
        if variable == 'image' or variable not in config["mechanism_models"]:
            continue
        model_config = config["mechanism_models"][variable]
        module = import_module(model_config["module"])
        model_class = getattr(module, model_config["model_class"])
        model = model_class(params=model_config["params"], attr_size=attribute_size)
        models[variable] = model
        if model_config["params"].get("finetune") == 1:
            model.name += '_finetuned'

    batch_size = args.bs
    scm_graph = config["causal_graph"]
    del scm_graph["image"]
    scm = SCM(checkpoint_dir=config["checkpoint_dir"], graph_structure=scm_graph, temperature=0.1, **models)

    data_class, unnormalize_fn = dataclass_mapping[dataset]
    transform = ReturnDictTransform(attribute_size)
    if dataset in ['celebahq']:
        train_set = data_class(attribute_size, split='train', transform=transform, resolution=args.resolution)
        test_set = data_class(attribute_size, split='test', transform=transform, resolution=args.resolution)
    else:
        train_set = data_class(attribute_size, split='train', transform=transform)
        test_set = data_class(attribute_size, split='test', transform=transform)

    if args.qualitative > 0:
        produce_qualitative_samples(pipe, prompt, dataset=test_set, scm=scm, parents=list(attribute_size.keys()),
                                    intervention_source=train_set, unnormalize_fn=unnormalize_fn, num=args.qualitative,
                                    show_difference=args.show_difference, args=args)

    embedding_model = get_embedding_model(args.embeddings, pretrained_vgg=True, classifier_config=classifier_config)
    embedding_fn = get_embedding_fn(args.embeddings, unnormalize_fn, accelerator.prepare(embedding_model))

    if "composition" in args.metrics:
        evaluate_composition(pipe, prompt, accelerator, test_set, unnormalize_fn, batch_size, cycles=args.cycles,
                             scm=scm, embedding=args.embeddings, embedding_fn=embedding_fn, args=args)

    intervention_attrs = ["Smiling", "Eyeglasses"] if dataset in ['celebahq'] else attribute_size.keys()

    if "effectiveness" in args.metrics:
        predictors = build_predictors(dataset, attribute_size, intervention_attrs, config_cls, args)
        for pa in intervention_attrs:
            evaluate_effectiveness(pipe, prompt, test_set, unnormalize_fn, batch_size, accelerator, scm=scm,
                                   attributes=intervention_attrs, do_parent=pa, intervention_source=train_set,
                                   predictors=predictors, dataset=dataset, embedding_fn=embedding_fn, args=args)

    if "fid" in args.metrics:
        evaluate_fid(real_set=train_set, test_set=test_set, batch_size=batch_size, scm=scm,
                     attributes=list(attribute_size.keys()))

    if "minimality" in args.metrics:
        evaluate_minimality(pipe, prompt, accelerator, real_set=train_set, test_set=test_set, batch_size=batch_size,
                            scm=scm, attributes=list(attribute_size.keys()), embedding=args.embeddings,
                            embedding_fn=embedding_fn, args=args)
