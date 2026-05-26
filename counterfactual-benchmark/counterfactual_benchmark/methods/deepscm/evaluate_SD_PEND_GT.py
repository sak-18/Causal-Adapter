import torch
import numpy as np
from typing import Dict, List
from json import load
from importlib import import_module
from model import SCM
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset
import os
os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["TORCH_HOME"] = "/home/jovyan/fcvm-data-volume/kzzr229/workspace/counterfactual-benchmark/.cache/torch"
import numpy as np
import argparse
import random
import sys
sys.path.append("../../")
sys.path.append('/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser')

from causal_modules import ddim_modules
import pickle
from models.classifiers.classifier import Classifier
from models.classifiers.celeba_classifier import CelebaClassifier
from models.classifiers.celeba_complex_classifier import CelebaComplexClassifier
from models.classifiers.pendulum_classifier import PendClassifier
from models.classifiers.adni_classifier import ADNIClassifier
from ctf_datasets.morphomnist.dataset import MorphoMNISTLike
from ctf_datasets.celeba.dataset_SD import Celeba
from ctf_datasets.adni.dataset_SD import ADNI
from ctf_datasets.pendulum.dataset_SD import PendulumLike
from ctf_datasets.transforms import ReturnDictTransform, get_attribute_ids
from accelerate import Accelerator
from evaluation.metrics.composition_SD import composition,lipips_identity
from evaluation.metrics.minimality_SD import minimality
from evaluation.embeddings.embeddings import get_embedding_model, get_embedding_fn
from evaluation.metrics.fid_SD import fid
from evaluation.metrics.effectiveness_SD import effectiveness
from evaluation.metrics.utils import save_selected_images, save_plots
from ctf_datasets.morphomnist.dataset import unnormalize as unnormalize_morphomnist
from ctf_datasets.celeba.dataset_SD import unnormalize as unnormalize_celeba
from ctf_datasets.adni.dataset import unnormalize as unnormalize_adni
from ctf_datasets.pendulum.dataset_SD import unnormalize as unnormalize_pend

import diffusers
from diffusers import StableDiffusionCausalControlNetPipeline, Causal_ControlNetModel, UniPCMultistepScheduler,StableDiffusionPipeline
from diffusers.utils import load_image
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from diffusers import DDIMInverseScheduler, DDIMScheduler
from tqdm import tqdm
from torchvision import transforms as tfms
from PIL import Image
from transformers import CLIPTokenizer
from ctf_datasets.adni.dataset_SD import bin_array,ordinal_array
import math
torch.multiprocessing.set_sharing_strategy('file_system')

rng = np.random.default_rng()

print("torch:", torch.__version__)
print("cuda.is_available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available():
    print("device_count:", torch.cuda.device_count())
    x = torch.zeros(1, device="cuda")
    print("ok:", x.device)


device = torch.device("cuda")
dataclass_mapping = {
    "morphomnist": (MorphoMNISTLike, unnormalize_morphomnist),
    "celeba": (Celeba, unnormalize_celeba),
    "adni": (ADNI, unnormalize_adni),
    "pendulum": (PendulumLike, unnormalize_pend)
}




def unnormalize_0_1(tensor):
    """
    Unnormalize a tensor that was normalized using Normalize([0.5], [0.5]).
    Converts from [-1, 1] → [0, 1].
    """
    return tensor * 0.5 + 0.5

def calculate_PEND_GT(label: torch.Tensor, intervention_idx: int, intervention_values):
    """
    label: 归一化后的标签，形状 [B,4] 或 [4]
           4个维度依次对应: [theta_src, phi_src, shadow_len, shadow_mid]
    intervention_idx: 需要干预的维度索引(0..3)
    intervention_values: 标量或形状 [B] 的张量，位于“归一化空间”
    return: 干预后的 label（仍在归一化空间，张量，保持原 device/dtype/shape）
    """

    # 统一为 [B,4]
    squeeze_back = (label.dim() == 1)
    if squeeze_back:
        label = label.unsqueeze(0)  # [1,4]

    # 克隆，避免原地改动破坏梯度
    z = label.clone()

    # 广播干预值
    if torch.is_tensor(intervention_values):
        v = intervention_values.to(device=z.device, dtype=z.dtype)
        if v.dim() == 0:
            v = v.view(1).expand(z.size(0))
    else:
        v = torch.as_tensor(intervention_values, device=z.device, dtype=z.dtype).view(1).expand(z.size(0))
    z[:, intervention_idx] = v
    if intervention_idx in [0,1]:
        # 去归一化（按列：x = z*sigma + mu），全程用 torch，避免来回 numpy
        mu    = torch.tensor([  2.0, 104.0,  7.5, 11.0], device=z.device, dtype=z.dtype)   # 均值
        sigma = torch.tensor([ 42.0,  44.0,  4.5,  8.0], device=z.device, dtype=z.dtype)   # 标准差
        denorm = z * sigma + mu                       # [B,4]

        # 角度（由第0/1列决定）
        theta = denorm[:, 0] * (torch.pi / 200.0)     # [B]
        phi   = denorm[:, 1] * (torch.pi / 200.0)     # [B]

        # 圆周位置
        x  = 10.0 + 8.0  * torch.sin(theta)           # [B]
        y  = 10.5 - 8.0  * torch.cos(theta)           # [B]
        bx = 10.0 + 9.5  * torch.sin(theta)           # [B]
        by = 10.5 - 9.5  * torch.cos(theta)           # [B]

        # 投影函数（向地面 y=base 的投影交点横坐标）
        def project(phi, x, y, base=-0.5):
            b = y - x * torch.tan(phi)
            return (torch.as_tensor(base, device=phi.device, dtype=phi.dtype) - b) / torch.tan(phi)

        p0 = project(phi, torch.tensor(10.0, device=z.device, dtype=z.dtype),
                        torch.tensor(10.5, device=z.device, dtype=z.dtype))             # 基准点投影
        p1 = project(phi, bx, by)                                                         # 小球位置投影

        mid   = (p0 + p1) / 2.0
        shade = torch.abs(p0 - p1)
        shade = torch.maximum(shade, torch.tensor(3.0, device=z.device, dtype=z.dtype))   # 最小长度3

        denorm[:, 2] = shade
        denorm[:, 3] = mid
        # ...then renormalize everything to keep z consistent
        z = (denorm - mu) / sigma
        # 返回到原始形状，仍为“归一化空间”的标签

    return z.squeeze(0) if squeeze_back else z


def produce_qualitative_samples(pipe,prompt,dataset, scm, parents, intervention_source, unnormalize_fn, num=20, show_difference=False,args=None):
    # test set 
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    produce_every = len(dataset) // num
    fig_idx = 0
    for i , (batch,idx) in tqdm(enumerate(data_loader)):
        # test 10 samples
        if i>=3:
            break
        #if i % produce_every == 0:
        else:
            #res = [batch]
            first_img = batch.copy()

            
            first_img["image"] = ddim_modules.resize_tensor(first_img["image"], pipe, args.dataset, return_tensor=True,mode='bicubic')
            res = [first_img]
            # patens = attrs, intervention_source: dataset class, 
            for do_parent in parents:
                counterfactual = produce_counterfactuals(pipe,prompt,batch, scm, do_parent, intervention_source,
                                                         force_change=True, possible_values=dataset.possible_values,args=args)
                res.append(counterfactual)

            save_plots(res, fig_idx, parents, unnormalize_fn, show_difference=show_difference)
            fig_idx += 1
    return


def evaluate_composition(pipe, prompt, accelerator, test_set: Dataset, unnormalize_fn, batch_size: int,
                         cycles: List[int], scm: nn.Module, save_dir: str = "composition_samples",
                         embedding=None, embedding_fn=None, args=None):
    
    test_data_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0,drop_last=True
    )
    test_data_loader = accelerator.prepare(test_data_loader)

    composition_scores_all = []   # List[Dict[int, np.ndarray]]
    l1_scores_int8_all = []       # List[Dict[int, np.ndarray]]
    l1_scores_uint8_all = []      # List[Dict[int, np.ndarray]]
    images_all = []               # List[np.ndarray]

    for i, (factual_batch, idx) in enumerate(tqdm(test_data_loader)):

        return_dict, image_batch = composition(
            pipe, prompt, args.dataset, factual_batch, unnormalize_fn,
            method=scm, cycles=cycles, embedding=embedding, embedding_fn=embedding_fn
        )

        composition_scores_all.append(return_dict['lpips'])
        l1_scores_uint8_all.append(return_dict['l1_uint8'])
        l1_scores_int8_all.append(return_dict['l1_int8'])

    #     # image_batch: numpy array, convert to tensor before gather
    #     images_all.append(torch.tensor(image_batch).to('cuda'))

    # # images_all is list[tensor], stack first
    # images_all = torch.cat(images_all, dim=0).to(accelerator.device)  # shape (B, H, W, C)
    # images_all = accelerator.gather(images_all)
    # images_all = images_all.cpu().numpy()

    # gather metric scores manually
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
        #os.makedirs(save_dir, exist_ok=True)
        # save_selected_images(images_all, composition_scores[cycles[-1]], save_dir=save_dir, lower_better=True)

        
        log_path = './saved/{}/composition/result.txt'.format(args.dataset)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Open the file in write mode, which will create it if it does not exist and clear it if it does
        with open(log_path, 'w') as f:
            f.write('')  # Clear the file content

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
    # value is the factual result
    if bins is not None and attribute in bins:
        # np.searchsorted  find the value required insertion points.
        # np.searchsorted([11,12,13,14,15], [-10, 20, 12, 13])
        # array([0, 5, 1, 2]), brief, output the same bins of possible values as false, choose other bins
        return np.digitize(possible_values, bins[attribute]) != np.searchsorted(bins[attribute], value)
    else:
        return possible_values != value

def produce_counterfactuals(pipe,prompt,factual_batch: torch.Tensor, scm: nn.Module, do_parent:str, intervention_source: Dataset,
                            force_change: bool = False, possible_values = None, device: str = 'cuda', bins = None,args=None):
    factual_batch = {k: v.to(device) for k, v in factual_batch.items()}

    #update with the counterfactual parent
    if force_change:
        possible_values = possible_values[do_parent]
        #current sample value 
        values = factual_batch[do_parent].cpu()
        if do_parent in ["pendulum","light","shadow_length", "shadow_position"]:
            'following causaldiff settings'
            pend_scale = {
                "pendulum": [2, 42],
                "light": [104, 44],
                "shadow_length": [7.5, 4.5],
                "shadow_position": [11, 8]
            }

            # sample a different intervention value per `value`
            sampled_tensor = torch.tensor([
                (np.random.uniform(-40, 44) - pend_scale[do_parent][0]) / pend_scale[do_parent][1]
                if do_parent == "pendulum" else
                (np.random.uniform(60, 148) - pend_scale[do_parent][0]) / pend_scale[do_parent][1]
                if do_parent == "light" else
                (np.random.uniform(3, 9) - pend_scale[do_parent][0]) / pend_scale[do_parent][1]
                if do_parent == "shadow_length" else
                (np.random.uniform(3, 15) - pend_scale[do_parent][0]) / pend_scale[do_parent][1]
                for _ in values  # generate one for each sample
            ], dtype=torch.float32).unsqueeze(1).to(device)

            interventions = {do_parent: sampled_tensor}
        elif do_parent not in ["digit", "apoE", "slice"]:
            interventions = {do_parent: torch.cat([torch.tensor(np.random.choice(possible_values[different_value(possible_values, value, bins, do_parent)])).unsqueeze(0)
                                                for value in values]).view(-1).unsqueeze(1).to(device)}
        else:
            interventions = {do_parent: torch.cat([torch.tensor(rng.choice(possible_values[torch.where((different_value(possible_values, value, bins, do_parent)).any(dim=1))], axis=0)).unsqueeze(0)
                                                for value in values]).to(device)}
    else:
        batch_size, _ , _ , _ = factual_batch["image"].shape
        idxs = torch.randperm(len(intervention_source))[:batch_size] # select random indices from train set to perform interventions

        interventions = {do_parent: torch.cat([intervention_source[id][do_parent] for id in idxs]).view(-1).unsqueeze(1).to(device)
                        if do_parent not in ["digit", "apoE", "slice"] else torch.cat([intervention_source[id][do_parent].unsqueeze(0).to(device) for id in idxs])}
    

    attr_keys = ddim_modules.get_dataset_attrs(args.dataset)
    label = torch.cat([factual_batch[k] for k in attr_keys], dim=1)  
    counterfactual_batch = factual_batch.copy()
    intervention_indx = attr_keys.index(list(interventions.keys())[0])
    intervention_values = interventions[list(interventions.keys())[0]].squeeze()
    # GT caculated
    causal_cond =  calculate_PEND_GT(label, intervention_indx,intervention_values)
    # for disentangle label 
    if args.caus_mechanims == 0:
        dscm_labels = label.clone()
        dscm_labels[:,intervention_indx] = intervention_values
    # for inferenced
    elif args.caus_mechanims == 1:
        pred_cond,_ = pipe.controlnet.controlnet_cond_embedding.inference(label.clone(),intervention_indx=intervention_indx,intervention_values=intervention_values,disentangle=False)
        dscm_labels = pred_cond.squeeze(2)
    # for GT
    elif args.caus_mechanims == 2:
        dscm_labels = causal_cond.clone()
    causal_cond = causal_cond.unsqueeze(2)

    blend_params = {'start_blend':0.0,'th':(0.3,0.3)}
    cross_replace_steps,self_replace_steps = 0.0,0.0
    if args.NTI:
        normalized_tensor,_ ,_,_= ddim_modules.P2P_editing(pipe, factual_batch["image"],
                                                    label.clone(),prompt,args.presudo_list,
                                                    num_steps = int(args.num_steps),invert_guidance_scale=1.0,
                                                    set_guidance_scale  = args.guidance_scale,intervention_indx=intervention_indx,
                                                    intervention_values=intervention_values,return_PIL=False,
                                                    blend_word=True,blend_params=blend_params,
                                                    disentangle=False,
                                                    cross_replace_steps=cross_replace_steps,self_replace_steps=self_replace_steps)        
    else:
        normalized_tensor,_ ,_,_= ddim_modules.ddim_editing(pipe, factual_batch["image"]
                                                ,label.clone(),prompt,num_steps =int(args.num_steps)
                                                ,invert_guidance_scale=1.0
                                                ,set_guidance_scale  = args.guidance_scale
                                                ,intervention_indx=intervention_indx
                                                ,intervention_values=intervention_values
                                                ,disentangle=True,DSCM_labels=dscm_labels
                                                ,return_PIL=False,pnp_inversion= False)
    resized_input = ddim_modules.resize_tensor(normalized_tensor, pipe, args.dataset, return_tensor=True)
    # correch the OOD conds in celeba
    if args.dataset in ['celeA_complex']:
        # correcy it if have negative values
        causal_cond = (causal_cond > 0.5).float() 
    
    counterfactual_batch["image"] = resized_input
    if args.dataset in ['ADNI','MorphoMNIST']:
        start = 0
        attr_dims = args.attribute_size
        for attr in attr_keys:
            dim = attr_dims[attr]
            value = causal_cond[:, start:start+dim]
            
            # Maintain shape convention
            if dim == 1:
                value = value.squeeze(1)  # (bs,)
            else:
                value = value.squeeze(2)
            counterfactual_batch[attr] = value
            start += dim
    else:
        for i, attr in enumerate(attr_keys):
            # Causal_cond[:, i] is the new attribute for attr,causal_cond is [bs,attrs,1]
            counterfactual_batch[attr] = causal_cond[:, i]
    if args.dataset in ['ADNI']:
        counterfactual_batch["sex"] = counterfactual_batch["sex"]>0.5
    return counterfactual_batch

@torch.no_grad()
def evaluate_effectiveness(pipe, prompt, test_set: Dataset, unnormalize_fn, batch_size:int,accelerator , scm: nn.Module, attributes: List[str], do_parent:str,
                           intervention_source: Dataset, predictors: Dict[str, Classifier], dataset: str,embedding_fn=None,args=None):

    # for attr in predictors:
    #     if torch.cuda.device_count() > 1:
    #         print('do parallel testing')
    #         predictors[attr] = nn.DataParallel(predictors[attr])
    #     predictors[attr] = predictors[attr].cuda()
    predictors = {k: accelerator.prepare(v) for k, v in predictors.items()}
        
    
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_data_loader= accelerator.prepare(test_data_loader)
    effectiveness_scores = {attr_key: [] for attr_key in attributes}
    all_results = {
        'predictions': {attr: [] for attr in attributes},
        'targets': {attr: [] for attr in attributes}
    }
    all_indices = []
    lpips_list = []
    for i, (factual_batch,idx) in enumerate(tqdm(test_data_loader)):
        # if i >3:
        #     break
  
        counterfactuals = produce_counterfactuals(pipe, prompt,factual_batch, scm, do_parent, intervention_source,
                                                  force_change=True, possible_values=test_set.possible_values, bins=test_set.bins,args=args)
        e_score,results_dict = effectiveness(counterfactuals, unnormalize_fn, predictors, dataset)
        if args.embeddings == 'lpips':
            resized_factual_image = ddim_modules.resize_tensor(factual_batch["image"], pipe, args.dataset, return_tensor=True)
            lipips_distance = lipips_identity(counterfactuals['image'],resized_factual_image,embedding_fn)
            lpips_list.extend(lipips_distance)
        all_indices.extend(idx.cpu().numpy())
        for attr in attributes:
            effectiveness_scores[attr].append(e_score[attr])
            all_results['predictions'][attr].append(results_dict['predictions'][attr])
            all_results['targets'][attr].append(results_dict['targets'][attr])

    
    # gather from all processes
    for attr in attributes:
        preds = np.concatenate(all_results['predictions'][attr], axis=0)
        targets = np.concatenate(all_results['targets'][attr], axis=0)
        indices = np.array(all_indices)  # shape (n_samples_per_process,)

        preds_tensor = torch.tensor(preds).to(accelerator.device)
        targets_tensor = torch.tensor(targets).to(accelerator.device)
        indices_tensor = torch.tensor(indices).to(accelerator.device)

        gathered_preds = accelerator.gather(preds_tensor).cpu().numpy()
        gathered_targets = accelerator.gather(targets_tensor).cpu().numpy()
        gathered_indices = accelerator.gather(indices_tensor).cpu().numpy()

        if accelerator.is_main_process:
            # 排序
            sort_idx = np.argsort(gathered_indices)
            gathered_preds = gathered_preds[sort_idx]
            gathered_targets = gathered_targets[sort_idx]
            gathered_indices = gathered_indices[sort_idx]

            # 去重（确保没有重复 index）
            unique_indices, unique_pos = np.unique(gathered_indices, return_index=True)

            # 应用唯一索引到结果
            all_results['predictions'][attr] = gathered_preds[unique_pos]
            all_results['targets'][attr] = gathered_targets[unique_pos]
    if args.embeddings == 'lpips':
        lpips_array = np.array(lpips_list)  # (n_samples,)
        lpips_tensor = torch.tensor(lpips_array).to(accelerator.device)
        gathered_lpips = accelerator.gather(lpips_tensor).cpu().numpy()
        if accelerator.is_main_process:
            # Save or process gathered_lpips as needed
            all_results['lpips'] = gathered_lpips

    # ✅ Outside the loop: save only in main process
    if accelerator.is_main_process:
        all_results['indices'] = unique_indices
        final_results = all_results
        output_dir = './saved_benchmark/{}/effectiveness_step{}_scale{}_blend{}_caumechanism{}/{}'.format(args.dataset, args.num_steps,args.guidance_scale,args.NTI,args.caus_mechanims,do_parent)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'final_results.pkl')

        with open(output_path, "wb") as f:
            pickle.dump(final_results, f)

        print(f"Effectiveness score do({do_parent}): {effectiveness_scores}")


    # === ✅ Sync all processes here ===
    accelerator.wait_for_everyone()

    return 


def evaluate_fid(real_set: Dataset, test_set: Dataset, batch_size: int, scm: nn.Module, attributes: List[str]):


    counterfactual_images = []
    for factual_batch in tqdm(test_data_loader):
        do_parent = random.choice(attributes)
        counterfactual_batch = produce_counterfactuals(factual_batch, scm, do_parent, intervention_source=real_set,
                                                        force_change=True, possible_values=test_set.possible_values, bins=real_set.bins)
        counterfactual_images.append(counterfactual_batch['image'])

    fid_score = fid(real_data_loader, counterfactual_images)

    print(f"FID: mean {round(np.mean(fid_score), 3):.3f} std {round(np.std(fid_score), 3):.3f}")

    return fid_score

#     return
def evaluate_minimality(pipe, prompt, accelerator, real_set: Dataset, test_set: Dataset, batch_size: int,
                        scm: nn.Module, attributes: List[str], embedding: str = None,
                        embedding_fn=None, args=None):
    real_data_loader = torch.utils.data.DataLoader(real_set, batch_size=batch_size, shuffle=False)
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    test_data_loader = accelerator.prepare(test_data_loader)

    factual_feats = []
    factual_labels = []
    counterfactual_feats = []
    counterfactual_labels = []
    intervention_list = []
    counterfactual_images = []

    for i, (factual_batch, idx) in enumerate(tqdm(test_data_loader)):
        do_parent = random.choice(attributes)
        counterfactual_batch = produce_counterfactuals(
            pipe, prompt, factual_batch, scm, do_parent,
            intervention_source=real_set,
            force_change=True,
            possible_values=test_set.possible_values,
            bins=real_set.bins,
            args=args
        )

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

        
        # Convert numpy to tensor before further processing
        factual_feats.append(torch.tensor(factual_features))
        counterfactual_feats.append(torch.tensor(counterfactual_features))
        if do_parent in ["digit", "apoE", "slice"]:
            if do_parent == "apoE":
                fac_labels = bin_array(factual_batch[do_parent], reverse=True).long().unsqueeze(1)
                counter_labels =  bin_array(counterfactual_batch[do_parent], reverse=True).long().unsqueeze(1)
            elif do_parent == "slice":
                fac_labels = ordinal_array(factual_batch[do_parent], reverse=True).long().unsqueeze(1)
                counter_labels =  ordinal_array(counterfactual_batch[do_parent], reverse=True).long().unsqueeze(1)
            else:
                pass
            factual_labels.append(fac_labels.to(torch.int))
            counterfactual_labels.append(counter_labels.to(torch.int))

        else:
            factual_labels.append(factual_batch[do_parent].to(torch.int))
            counterfactual_labels.append(counterfactual_batch[do_parent].to(torch.int))
        
        intervention_list.append([do_parent] * len(factual_batch["image"]))  # Use numpy shape here

    # Convert to tensors
    
    factual_feats = torch.cat(factual_feats, dim=0).to(accelerator.device)
    factual_labels = torch.cat(factual_labels, dim=0).to(accelerator.device)
    counterfactual_feats = torch.cat(counterfactual_feats, dim=0).to(accelerator.device)
    counterfactual_labels = torch.cat(counterfactual_labels, dim=0).to(accelerator.device)
    #5-D 
    counterfactual_images = torch.stack(counterfactual_images, dim=0).to(accelerator.device)

    # Convert interventions to string -> int -> tensor
    unique_attrs = {att: idx for idx, att in enumerate(attributes)}
    intervention_tensor = torch.tensor(
        [unique_attrs[attr] for sublist in intervention_list for attr in sublist],
        dtype=torch.int
    ).to(accelerator.device)

    # Gather
    factual_feats = accelerator.gather(factual_feats)
    factual_labels = accelerator.gather(factual_labels)
    counterfactual_feats = accelerator.gather(counterfactual_feats)
    counterfactual_labels = accelerator.gather(counterfactual_labels)
    counterfactual_images = accelerator.gather(counterfactual_images)
    intervention_tensor = accelerator.gather(intervention_tensor)

    # Convert back to numpy for minimality
    if accelerator.is_main_process:
        
        factuals = list(zip(factual_feats.cpu().numpy(), factual_labels.cpu().numpy()))
        counterfactuals = list(zip(counterfactual_feats.cpu().numpy(), counterfactual_labels.cpu().numpy()))
        interventions = [attributes[idx] for idx in intervention_tensor.cpu().numpy()]
        
        output_dir = './saved/{}/FID_step{}_scale{}'.format(args.dataset, args.num_steps,args.guidance_scale)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(counterfactual_images, os.path.join(output_dir, "counterfactual_tensors.pt"))

        # ✅ 统一保存其他数据为 dict
        torch.save({
            "factuals": factuals,
            "counterfactuals": counterfactuals,
            "interventions": interventions,
        }, os.path.join(output_dir, "minimality.pt"))
        

        
        print('start Minimality calculation')
        minimality_scores, prob1s, prob2s = minimality(
            real=factuals,
            generated=counterfactuals,
            interventions=interventions,
            bins=real_set.bins,
            embedding=embedding
        )
        minimality_msg = f"Minimality score: mean {round(np.mean(minimality_scores), 3)}, std {round(np.std(minimality_scores), 3)}"
        prob_msg = f"Prob 1: {np.mean(prob1s)}, Prob 2: {np.mean(prob2s)}"

        print(minimality_msg)
        print(prob_msg)    

        log_path = './SD_fid_results.txt'
        if os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write('')  # Clear the file content

        with open(log_path, 'a') as f:
            f.write(minimality_msg + "\n")
            f.write(prob_msg + "\n")

        
        print('start FID calculation')
        
        # fid_score = fid(real_data_loader, counterfactual_images, pipe, args)
        # fid_msg = f"FID: mean {round(np.mean(fid_score), 3):.3f} std {round(np.std(fid_score), 3):.3f}"
        # print(fid_msg)
        

    accelerator.wait_for_everyone()

    return


def datasets_configs(dataset):
    if 'celeA' in dataset:
        if 'simple' in dataset:
            pass
        elif 'complex' in dataset:
            config = '/home/jovyan/fcvm-data-volume/kzzr229/workspace/counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/configs/celeba/complex/vae.json'
            classifier_config = '/home/jovyan/fcvm-data-volume/kzzr229/workspace/counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/configs/celeba/complex/classifier.json'
    elif 'ADNI' in dataset:
        config = '/home/jovyan/fcvm-data-volume/kzzr229/workspace/counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/configs/adni/vae.json'
        classifier_config = '/home/jovyan/fcvm-data-volume/kzzr229/workspace/counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/configs/adni/classifier.json'
    elif 'pendulum' in dataset:
        config = '/home/jovyan/fcvm-data-volume/kzzr229/workspace/counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/configs/pendulum/vae.json'
        classifier_config = '/home/jovyan/fcvm-data-volume/kzzr229/workspace/counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/configs/pendulum/classifier.json'

    return config,classifier_config


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Config file for experiment.", default="celeA_complex")
    parser.add_argument("--metrics", '-m',
                        nargs="+", type=str,
                        help="Metrics to calculate. "
                        "Choose one or more of [composition, effectiveness, fid, minimality]. If not set, all metrics are calculated.",
                        choices=["composition", "effectiveness", "fid", "minimality"],
                        default=["composition", "effectiveness", "fid", "minimality"])
    parser.add_argument("--bs", type=int, help="Composition cycles.", default=6)                    
    parser.add_argument("--cycles", '-cc', nargs="+", type=int, help="Composition cycles.", default=[1, 10])
    parser.add_argument("--qualitative", '-qn', type=int, help="Number of qualitative results to produce", default=0)
    parser.add_argument("--show-difference", '-sd', action='store_true', help="Show counterfactual-factual difference on qualitative results")
    parser.add_argument("--embeddings", type=str, choices=["vgg", "clfs", "vae", "lpips", "clip"], help="What embeddings to use for composition metric. "
                        "Supported: [vgg, clfs, vae, lpips, clip]. If not set, will compute distance on image space")
    parser.add_argument("--sampling-temperature", '-temp', type=float, default=0.1, help="Sampling temperature, used for VAE, HVAE.")
    parser.add_argument("--NTI", 
        type=str2bool,
        default=False,
        help="enable null textual inversion?")
    parser.add_argument("--caus_mechanims", type=int, help="Number of qualitative results to produce", default=1)
    parser.add_argument("--num_steps", type=float, default=50, help="Sampling temperature, used for VAE, HVAE.")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Sampling temperature, used for VAE, HVAE.")
    parser.add_argument("--causalnet_path", 
        type=str,
        default=None,
        help="load causalnet")
    parser.add_argument("--mcpl_embedding_path", 
        type=str,
        default=None,
        help="load mcpl") 
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    seed_int = 42
    # torch.manual_seed(42)
    # random.seed(seed_int)
    # np.random.seed(seed_int)
    # torch.manual_seed(seed_int)

    print(args)
    base_model_path = "/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser/.cache/huggingface/hub/models--lambdalabs--miniSD-diffusers/snapshots/26ed8a9bfbf76f46a6cf60517dde321f900c44ce"
    controlnet_path = args.causalnet_path
    mcpl_embedding_path = args.mcpl_embedding_path
    accelerator = Accelerator()
    controlnet = Causal_ControlNetModel.from_pretrained(controlnet_path,torch_dtype=torch.float32)
    if args.dataset in ['celeA_complex']:
        cond_path = '/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser/logs/logs_celeA_complex_all/2025-04-23T21-11-19-causalnet_pretrain/best_model.pt'
        A_matrix = torch.tensor([[0, 0, 1,1], [0, 0, 1,1], [0, 0, 0,0], [0, 0, 0,0]],dtype=torch.float32).to(device)
        prompt = 'a human of @ and * and & and !'
        presudo_words= '@,*,&,!'
        if cond_path is not None:
            print('load pretrained causalnet weights')
            controlnet.controlnet_cond_embedding.load_state_dict(torch.load(cond_path,weights_only=True))
    elif args.dataset == 'ADNI':
        #A_matrix = torch.tensor([[0, 0,0, 1, 0,0], [0, 0,0, 1, 1,0], [0, 0,0,1,0, 0], [0, 0, 0, 0,1,0],[0, 0, 0, 0,0,0],[0, 0, 0, 0,0,0]],dtype=torch.float32).to(device)
        prompt = 'a mri image of @ and * and &'
        append_text = (' '+prompt[-1])*(10-1)
        prompt+=append_text
        presudo_words= '@,*,&'

        '''test with causal discovered matrix'''
        cond_path = "/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser/logs/logs_ADNI_all/2025-08-14T15-33-26-causal_discovered_matrix/best_model.pt"
        if cond_path is not None:
            print('load pretrained causalnet weights')
            controlnet.controlnet_cond_embedding.load_state_dict(torch.load(cond_path,weights_only=True))
        A_matrix=torch.tensor([[0, 0,0, 1, 0,0], [0, 0,1, 0, 1,0], [0, 0,0,1,0, 0], [0, 0, 0, 0,1,0],[0, 0, 0, 0,0,0],[0, 0, 0, 0,0,0]],dtype=torch.float32).to(device)
        # prompt = 'a mri image of @ and * and & and ! and $ and %'
        # presudo_word='@,*,&,!,$,%'
    elif args.dataset == 'pendulum':
        #A_matrix = torch.tensor([[0, 0, 1,1], [0, 0, 1,1], [0, 0, 0,0], [0, 0, 0,0]],dtype=torch.float32).to(device)
        prompt = 'a image of @ and * and & and !'
        presudo_words= '@,*,&,!'
        '''test with causal discovered matrix'''    
        cond_path = "/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser/logs/logs_pendulum_all/2025-08-14T14-58-49-causal_discovered_matrix/best_model.pt"
        if cond_path is not None:
            print('load pretrained causalnet weights')
            controlnet.controlnet_cond_embedding.load_state_dict(torch.load(cond_path,weights_only=True))
        A_matrix=torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 1, 0]],dtype=torch.float32).to(device)
    
    controlnet.controlnet_cond_embedding.update_mask(A_matrix)
    controlnet.eval()
    presudo_list = presudo_words.split(',')
    args.presudo_list = presudo_list
    tokenizer = CLIPTokenizer.from_pretrained(base_model_path,subfolder="tokenizer")
    presudo_token_ids = tokenizer.encode(' '.join(presudo_list), add_special_tokens=False)
    embed_control_manager_bool = True
    if 'image' in controlnet.task_cond:
        embed_control_manager_bool = False
    text_encoder = ddim_modules.load_mcpl_embeddings(base_model_path,tokenizer,mcpl_embedding_path,presudo_token_ids,embed_control=embed_control_manager_bool)

    pipe = StableDiffusionCausalControlNetPipeline.from_pretrained(
        base_model_path, controlnet=controlnet,text_encoder=text_encoder ,torch_dtype=torch.float32
    )
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config
    )
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    
    pipe = pipe.to(accelerator.device)
    config,classifier_config=  datasets_configs(args.dataset)
    with open(classifier_config, 'r') as f:
        config_cls = load(f)

    
    with open(config, 'r') as f:
        config = load(f)

    dataset = config["dataset"]
    attribute_size = config["attribute_size"]

    models = {}
    for variable in config["causal_graph"].keys():
        if variable not in config["mechanism_models"]:
            continue
        model_config = config["mechanism_models"][variable]

        module = import_module(model_config["module"])
        model_class = getattr(module, model_config["model_class"])
        model = model_class(params=model_config["params"], attr_size=attribute_size)

        models[variable] = model
        if "finetune" in model_config["params"] and model_config["params"]["finetune"] == 1:
            model.name += '_finetuned'

    batch_size = args.bs
    scm = None
    # scm = SCM(checkpoint_dir=config["checkpoint_dir"],
    #           graph_structure=config["causal_graph"],
    #           temperature=args.sampling_temperature,
    #           **models)

    dataset = config["dataset"]
    data_class, unnormalize_fn = dataclass_mapping[dataset]

    transform = ReturnDictTransform(attribute_size)

    train_set = data_class(attribute_size, split='train', transform=transform)
    # ablation with validation set
    #test_set = data_class(attribute_size, split='valid', transform=transform)
    test_set = data_class(attribute_size, split='test', transform=transform)
    args.attribute_size =attribute_size
    if args.qualitative > 0:
        produce_qualitative_samples(pipe,prompt, dataset=test_set, scm=scm, parents=list(attribute_size.keys()),
                                    intervention_source=train_set, unnormalize_fn=unnormalize_fn, num=args.qualitative,
                                    show_difference=args.show_difference,args=args)

    embedding_model = get_embedding_model(args.embeddings, pretrained_vgg=True, classifier_config=classifier_config)
    
    embedding_fn = get_embedding_fn(args.embeddings, unnormalize_fn, accelerator.prepare(embedding_model))

    if "composition" in args.metrics:
        evaluate_composition(pipe,prompt,accelerator,test_set, unnormalize_fn, batch_size, cycles=args.cycles, scm=scm, embedding=args.embeddings, embedding_fn=embedding_fn,args=args)


    if "effectiveness" in args.metrics:
        if dataset == "morphomnist":
            predictors = {atr: Classifier(attr=atr, num_outputs=attribute_size[atr], context_dim=len(list(config_cls["anticausal_graph"][atr])))
                          for atr in attribute_size.keys()}
        elif dataset == "celeba":
            if sum(attribute_size.values()) == 4:
                # predictors = {atr: CelebaComplexClassifier(attr=atr, context_dim=len(list(config_cls["anticausal_graph"][atr])),
                #                                   num_outputs=config_cls[atr +"_num_out"],
                #                                   lr=config_cls["lr"], version=config_cls["version"]) for atr in attribute_size.keys()}
                predictors = {atr: CelebaComplexClassifier(attr=atr, context_dim=len(list(config_cls["anticausal_graph"][atr])),
                                                  num_outputs=config_cls["attribute_size"][atr],
                                                  lr=config_cls["lr"], version=config_cls["version"]) for atr in attribute_size.keys()}
            else:
                #predictors = {atr: CelebaClassifier(attr=atr, num_outputs=config_cls[atr +"_num_out"], lr=config_cls["lr"]) for atr in attribute_size.keys()}
                predictors = {atr: CelebaClassifier(attr=atr, num_outputs=config_cls["attribute_size"][atr], lr=config_cls["lr"]) for atr in attribute_size.keys()}
        elif dataset == "pendulum":
            predictors = {atr: PendClassifier(attr=atr, num_outputs=attribute_size[atr], context_dim=len(list(config_cls["anticausal_graph"][atr])))
                          for atr in attribute_size.keys()}
        else:
            attribute_ids = get_attribute_ids(attribute_size)
            predictors = {atr: ADNIClassifier(attr=atr, num_outputs=config_cls["attribute_size"][atr], children=config_cls["anticausal_graph"][atr],
                                              num_slices=config_cls["attribute_size"]['slice'], attribute_ids=attribute_ids, arch=config_cls['arch']) for atr in attribute_size.keys()}

        # load checkpoints of the predictors
        for key , cls in predictors.items():
            print(key)
            file_name = next((file for file in os.listdir(config_cls["ckpt_path"]) if file.startswith(key)), None)
            print(file_name)
            cls.load_state_dict(torch.load(config_cls["ckpt_path"] + file_name , map_location=torch.device('cuda'))["state_dict"])
            cls.to('cuda')

        #for pa in attribute_size.keys():
        for pa in ['shadow_length','shadow_position']:
            evaluate_effectiveness(pipe, prompt,test_set, unnormalize_fn, batch_size,accelerator, scm=scm, attributes=list(attribute_size.keys()), do_parent=pa,
                            intervention_source=train_set, predictors=predictors, dataset=dataset,embedding_fn=embedding_fn,args=args)

    if "fid" in args.metrics:
        feat_dict = evaluate_fid(real_set=train_set, test_set=test_set, batch_size=batch_size, scm=scm, attributes=list(attribute_size.keys()),args=args)

    if "minimality" in args.metrics:
        evaluate_minimality(pipe, prompt,accelerator,real_set=train_set, test_set=test_set, batch_size=batch_size, scm=scm, attributes=list(attribute_size.keys()),
                            embedding=args.embeddings, embedding_fn=embedding_fn,args=args)
