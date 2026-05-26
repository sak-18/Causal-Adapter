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
sys.path.append('/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser-flux')

from causal_modules import ddim_modules_sd3 as ddim_modules
from causal_modules.ddim_modules_sd3 import Flow_editing
import pickle
from models.classifiers.classifier import Classifier
from models.classifiers.celeba_classifier import CelebaClassifier,Celeba_anticausal_Classifier
from models.classifiers.celeba_complex_classifier import CelebaComplexClassifier
from models.classifiers.pendulum_classifier import PendClassifier
from models.classifiers.adni_classifier import ADNIClassifier
from ctf_datasets.morphomnist.dataset import MorphoMNISTLike
from ctf_datasets.celeba_hq.dataset_SD import Celebahq
from ctf_datasets.celeba.dataset_SD import Celeba
from ctf_datasets.adni.dataset_SD import ADNI
from ctf_datasets.pendulum.dataset_SD import PendulumLike
from ctf_datasets.transforms import ReturnDictTransform, get_attribute_ids
from accelerate import Accelerator
from evaluation.metrics.composition_SD3 import composition,lipips_identity
from evaluation.metrics.minimality_SD import minimality
from evaluation.embeddings.embeddings import get_embedding_model, get_embedding_fn
from evaluation.metrics.fid_SD import fid
from evaluation.metrics.effectiveness_SD import effectiveness
from evaluation.metrics.utils import save_selected_images, save_plots
from ctf_datasets.morphomnist.dataset import unnormalize as unnormalize_morphomnist
from ctf_datasets.celeba.dataset_SD import unnormalize as unnormalize_celeba
from ctf_datasets.adni.dataset import unnormalize as unnormalize_adni
from ctf_datasets.pendulum.dataset_SD import unnormalize as unnormalize_pend

from diffusers.models.controlnets.controlnet_sd3_causal import Causal_SD3ControlNetModel
from causal_modules.ddim_modules_sd3 import load_tokenizers_text_encoders,load_mcpl_embeddings,tokenize_prompt
from diffusers.pipelines import StableDiffusion3InpaintPipeline_Adapter
from diffusers.utils import load_image
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision import transforms as tfms
from PIL import Image
from ctf_datasets.adni.dataset_SD import bin_array,ordinal_array

torch.multiprocessing.set_sharing_strategy('file_system')

rng = np.random.default_rng()

device = torch.device("cuda")
dataclass_mapping = {
    "morphomnist": (MorphoMNISTLike, unnormalize_morphomnist),
    "celeba": (Celeba, unnormalize_celeba),
    "adni": (ADNI, unnormalize_adni),
    "pendulum": (PendulumLike, unnormalize_pend),
    "celebahq": (Celebahq,unnormalize_celeba)
}
def unnormalize_0_1(tensor):
    """
    Unnormalize a tensor that was normalized using Normalize([0.5], [0.5]).
    Converts from [-1, 1] → [0, 1].
    """
    return tensor * 0.5 + 0.5


def produce_qualitative_samples(pipe,prompt,dataset, scm, parents, intervention_source, unnormalize_fn, num=20, show_difference=False,args=None):
    # test set 
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.dataloader_num_workers)

    produce_every = len(dataset) // num
    fig_idx = 0
    for i , (batch,idx) in tqdm(enumerate(data_loader)):
        # test 10 samples
        if i>=30:
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
        test_set, batch_size=batch_size, shuffle=False, num_workers=args.dataloader_num_workers,drop_last=True
    )
    test_data_loader = accelerator.prepare(test_data_loader)

    composition_scores_all = []   # List[Dict[int, np.ndarray]]
    l1_scores_int8_all = []       # List[Dict[int, np.ndarray]]
    l1_scores_uint8_all = []      # List[Dict[int, np.ndarray]]
    images_all = []               # List[np.ndarray]

    for i, (factual_batch, idx) in enumerate(tqdm(test_data_loader)):
        if i>=4:
            break

        return_dict, image_batch = composition(
            pipe, prompt, args.dataset, factual_batch, unnormalize_fn,
            method=scm, cycles=cycles, embedding=embedding, embedding_fn=embedding_fn,args=args
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

        if args.dataset == 'celeA_complex' and i<5:
            root_folder='saved_ablation'
        else:
            root_folder='saved_benchmark'
        model_name = args.causalnet_path.split('/')[-2]
        output_dir = './{}/{}/DSCM_effectiveness_step{}_scale{}_NTI{}_{}/composition/'.format(root_folder,args.dataset, args.num_steps,args.guidance_scale,args.NTI,model_name)
        os.makedirs(output_dir, exist_ok=True)

        log_path = os.path.join(output_dir, 'compositon.txt')
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
        
@torch.no_grad()
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
    #infer exgonous noise
    if args.dataset == 'celebahq_simple':
        dscm_labels = label.clone()

        # Set attribute `inter_id` to its complemented value, per sample
        dscm_labels[:, intervention_indx] = intervention_values

        # If we're intervening on attribute 0, enforce a dependency:
        # attr2 follows the (complemented) value of attr0, per sample.
        if intervention_indx == 0:
            dscm_labels[:, 2] = intervention_values
    else:
        abducted_noise = scm.encode(**factual_batch)
        scm_counterfactual_batch = scm.decode(interventions, **abducted_noise)
        dscm_labels = torch.cat([scm_counterfactual_batch[k] for k in attr_keys], dim=1)  

    if args.dataset in ['celeA_complex'] and int(intervention_indx)==2:
        row_index = torch.where(dscm_labels[:, 1] == 0)[0]
        dscm_labels[row_index, intervention_indx] = -1

    T_steps = int(args.num_steps)
    n_avg = 1
    src_guidance_scale = 1.0
    tar_guidance_scale = args.guidance_scale
    n_min = 0
    n_max = 50
    normalized_tensor = Flow_editing(pipe,
                        pipe.scheduler,
                        factual_batch["image"],
                        prompt,
                        args.presudo_list,
                        label.clone().to(pipe.device),
                        dscm_labels.clone().to(pipe.device),
                        negative_prompt="",
                        T_steps= T_steps,
                        n_avg= n_avg,
                        src_guidance_scale= src_guidance_scale,
                        tar_guidance_scale = tar_guidance_scale,
                        n_min= n_min,
                        n_max= n_max,)
    
    resized_input = ddim_modules.resize_tensor(normalized_tensor, pipe, args.dataset, return_tensor=True)
    causal_cond = dscm_labels
    if causal_cond.dim() != 3:
        causal_cond = causal_cond.unsqueeze(2)
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
    
    do_denormalize = [True] * normalized_tensor.shape[0]
    output_img = pipe.image_processor.postprocess(normalized_tensor.detach(),do_denormalize=do_denormalize)
    tensor_image_transforms = tfms.Compose(
            [
                tfms.ToTensor(),
                tfms.Normalize([0.5], [0.5]),
            ]
        )
    
    cf_tensor = torch.stack(
        [tensor_image_transforms(img) for img in output_img],
        dim=0
    )

    out_tensor_back= Flow_editing(pipe,
                        pipe.scheduler,
                        cf_tensor.to(device=pipe.device,dtype=pipe.dtype),
                        prompt,
                        args.presudo_list,
                        dscm_labels.clone().to(pipe.device),
                        label.clone().to(pipe.device),
                        negative_prompt="",
                        T_steps= T_steps,
                        n_avg= n_avg,
                        src_guidance_scale= src_guidance_scale,
                        tar_guidance_scale = tar_guidance_scale,
                        n_min= n_min,
                        n_max= n_max,)
    resized_back = ddim_modules.resize_tensor(out_tensor_back, pipe, args.dataset, return_tensor=True)

    return counterfactual_batch,resized_back

@torch.no_grad()
def evaluate_effectiveness(pipe, prompt, test_set: Dataset, unnormalize_fn, batch_size:int,accelerator , scm: nn.Module, attributes: List[str], do_parent:str,
                           intervention_source: Dataset, predictors: Dict[str, Classifier], dataset: str,embedding_fn=None,args=None):

    # for attr in predictors:
    #     if torch.cuda.device_count() > 1:
    #         print('do parallel testing')
    #         predictors[attr] = nn.DataParallel(predictors[attr])
    #     predictors[attr] = predictors[attr].cuda()
    predictors = {k: accelerator.prepare(v) for k, v in predictors.items()}
        
    
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=args.dataloader_num_workers)
    test_data_loader= accelerator.prepare(test_data_loader)
    effectiveness_scores = {attr_key: [] for attr_key in attributes}
    all_results = {
        'predictions': {attr: [] for attr in attributes},
        'targets': {attr: [] for attr in attributes}
    }
    all_indices = []
    IDP_l1_list, IDP_lpips_list, reverse_l1_list,reverse_lpips_list, compos_l1_list,compos_lpips_list = [],[],[],[],[],[]
    for i, (factual_batch,idx) in enumerate(tqdm(test_data_loader)):
        if i >=15:
            break

        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
  
        counterfactuals,recovered_from_cf = produce_counterfactuals(pipe, prompt,factual_batch, scm, do_parent, intervention_source,
                                                  force_change=True, possible_values=test_set.possible_values, bins=test_set.bins,args=args)
        e_score,results_dict = effectiveness(counterfactuals, unnormalize_fn, predictors, dataset)
        factual_image = ddim_modules.resize_tensor(factual_batch['image'], pipe, args.dataset, return_tensor=True)
        _,recons_factual = composition(
                pipe, prompt, args.dataset, factual_batch, unnormalize_fn,
                method=scm, cycles=[1], embedding=args.embeddings, embedding_fn=embedding_fn,args=args
            )
        if args.embeddings == 'lpips':
            # IDP
            IDP_dic = lipips_identity(counterfactuals['image'],recons_factual,embedding_fn)
            # reversibility
            reverse_dic =  lipips_identity(factual_image,recovered_from_cf,embedding_fn)
            # reconstruction
            compos_dic = lipips_identity(factual_image,recons_factual,embedding_fn)

            IDP_l1_list.extend(IDP_dic['L1'])
            IDP_lpips_list.extend(IDP_dic['LPIPS'])

            reverse_l1_list.extend(reverse_dic['L1'])
            reverse_lpips_list.extend(reverse_dic['LPIPS'])

            compos_l1_list.extend(compos_dic['L1'])
            compos_lpips_list.extend(compos_dic['LPIPS'])
        

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

        gathered_IDP_l1 = accelerator.gather(torch.tensor(np.array(IDP_l1_list)).to(accelerator.device)).cpu().numpy()
        gathered_IDP_lpips = accelerator.gather(torch.tensor(np.array(IDP_lpips_list)).to(accelerator.device)).cpu().numpy()

        gathered_reverse_l1 = accelerator.gather(torch.tensor(np.array(reverse_l1_list)).to(accelerator.device)).cpu().numpy()
        gathered_reverse_lpips = accelerator.gather(torch.tensor(np.array(reverse_lpips_list)).to(accelerator.device)).cpu().numpy()

        gathered_compos_l1 = accelerator.gather(torch.tensor(np.array(compos_l1_list)).to(accelerator.device)).cpu().numpy()
        gathered_compos_lpips = accelerator.gather(torch.tensor(np.array(compos_lpips_list)).to(accelerator.device)).cpu().numpy()
        if accelerator.is_main_process:
            # Save or process gathered_lpips as needed
            all_results['IDP_l1'] = gathered_IDP_l1
            all_results['IDP_lpips'] = gathered_IDP_lpips

            all_results['reverse_l1'] = gathered_reverse_l1
            all_results['reverse_lpips'] = gathered_reverse_lpips

            all_results['compos_l1'] = gathered_compos_l1
            all_results['compos_lpips'] = gathered_compos_lpips

    # ✅ Outside the loop: save only in main process
    if accelerator.is_main_process:
        all_results['indices'] = unique_indices
        final_results = all_results
        
        root_folder='saved_benchmark_full'
        output_dir = './{}/{}/200000_SD3_DSCM_effectiveness_step{}_scale{}_Blend{}_reverse/{}'.format(root_folder,args.dataset, args.num_steps,args.guidance_scale,args.NTI,do_parent)
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
    real_data_loader = torch.utils.data.DataLoader(real_set, batch_size=batch_size, shuffle=False,num_workers = args.dataloader_num_workers)
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,num_workers = args.dataloader_num_workers)
    test_data_loader = accelerator.prepare(test_data_loader)

    factual_feats = []
    factual_labels = []
    counterfactual_feats = []
    counterfactual_labels = []
    intervention_list = []
    counterfactual_images = []

    for i, (factual_batch, idx) in enumerate(tqdm(test_data_loader)):
        # if i>=4:
        #     break

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
        
        if args.dataset == 'celeA_complex' and i<12:
            root_folder='saved_ablation'
        else:
            root_folder='saved_benchmark'
        model_name = args.causalnet_path.split('/')[-2]
        output_dir = './{}/{}/DSCM_effectiveness_step{}_scale{}_NTI{}_{}/FID/'.format(root_folder,args.dataset, args.num_steps,args.guidance_scale,args.NTI,model_name)
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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def datasets_configs(dataset):
    if 'celeA' in dataset:
        if 'simple' in dataset:
            pass
        elif 'complex' in dataset:
            config = '/home/jovyan/fcvm-data-volume/kzzr229/workspace/counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/configs/celeba/complex/vae.json'
            classifier_config = '/home/jovyan/fcvm-data-volume/kzzr229/workspace/counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/configs/celeba/complex/classifier.json'
    if 'celebahq' in dataset:
        if 'simple' in dataset:
            config = '/home/jovyan/fcvm-data-volume/kzzr229/workspace/counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/configs/celebahq/simple/vae_anti.json'
            classifier_config = '/home/jovyan/fcvm-data-volume/kzzr229/workspace/counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/configs/celebahq/simple/classifier_anti.json'
        elif 'complex' in dataset:
            pass 
    
    elif 'ADNI' in dataset:
        config = '/home/jovyan/fcvm-data-volume/kzzr229/workspace/counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/configs/adni/vae.json'
        classifier_config = '/home/jovyan/fcvm-data-volume/kzzr229/workspace/counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/configs/adni/classifier.json'
    elif 'pendulum' in dataset:
        config = '/home/jovyan/fcvm-data-volume/kzzr229/workspace/counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/configs/pendulum/vae.json'
        classifier_config = '/home/jovyan/fcvm-data-volume/kzzr229/workspace/counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/configs/pendulum/classifier.json'

    return config,classifier_config

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
    parser.add_argument("--num_steps", type=float, default=50, help="Sampling temperature, used for VAE, HVAE.")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Sampling temperature, used for VAE, HVAE.")
    parser.add_argument("--NTI", 
        type=str2bool,
        default=False,
        help="enable null textual inversion?")
    parser.add_argument("--causalnet_path", 
        type=str,
        default=None,
        help="load causalnet")
    parser.add_argument("--mcpl_embedding_path", 
        type=str,
        default=None,
        help="load mcpl")
    parser.add_argument("--resolution", 
        type=int,
        default=256,
        help="image resoultion, default is 256")  
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    # torch.manual_seed(42)
    load_dtype = torch.float16
    print(args)
    base_model_path ="/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser-flux/.cache/huggingface/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/5fe80140eec27f0a4e1d02ea2b0b31d71ac38f75"
    model_dir_name = "2025-10-28T20-32-57-controlnet_textcond_constrastive_7attrsgeneration_text_global_after"
    #model_dir_name = "2025-10-28T20-32-57-controlnet_textcond_constrastive_7attrsgeneration_text_global_after"
    train_steps = 200000
    # currently not train the T5 embeddings
    controlnet_path = f"/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser-flux/logs/logs_celebahq_simple_all/{model_dir_name}/controlnet-steps-{train_steps}.safetensors"
    embedding_1_path = f"/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser-flux/logs/logs_celebahq_simple_all/{model_dir_name}/learned_embeds_clip1_{train_steps}.safetensors"
    embedding_2_path = f"/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser-flux/logs/logs_celebahq_simple_all/{model_dir_name}/learned_embeds_clip2_{train_steps}.safetensors"
    embedding_3_path = f"/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser-flux/logs/logs_celebahq_simple_all/{model_dir_name}/learned_embeds_t5_{train_steps}.safetensors"

    args.causalnet_path = controlnet_path
    device = torch.device("cuda")
    controlnet = Causal_SD3ControlNetModel.from_pretrained(controlnet_path,torch_dtype=load_dtype)
    controlnet.eval()

    # Load mcpl embedding

    prompt = 'a human of @ and * and mouth and gender and ** and $ and #'
    presudo_words= '@,*,mouth,gender,**,$,#'
    #presudo_words = 'young,female,beard,bald'
    presudo_list = presudo_words.split(',')
    args.presudo_list = presudo_list
    accelerator = Accelerator()
    # mcpl embeddings
    tokenizers, text_encoders = load_tokenizers_text_encoders(base_model_path,load_dtype)
    text_encoders = load_mcpl_embeddings([embedding_1_path,embedding_2_path,embedding_3_path], tokenizers, text_encoders,load_dtype)

    pipe = StableDiffusion3InpaintPipeline_Adapter.from_pretrained(
        base_model_path, controlnet=controlnet,
        tokenizer = tokenizers[0],tokenizer_2 = tokenizers[1],tokenizer_3 = tokenizers[2],
        text_encoder=text_encoders[0],text_encoder_2=text_encoders[1],text_encoder_3=text_encoders[2],
        torch_dtype=load_dtype
    )
    
    # pipe.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    # memory optimization.
    pipe = pipe.to(device)
    config,classifier_config=  datasets_configs(args.dataset)
    with open(classifier_config, 'r') as f:
        config_cls = load(f)

    
    with open(config, 'r') as f:
        config = load(f)

    dataset = config["dataset"]
    attribute_size = config["attribute_size"]

    models = {}
    for variable in config["causal_graph"].keys():
        # Only load causal inference module
        if variable != 'image':
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
    scm_graph = config["causal_graph"]
    del scm_graph["image"]
    # if 'celebahq' in args.dataset:
    #     scm = None
    # else:
    scm = SCM(checkpoint_dir=config["checkpoint_dir"],
            graph_structure=scm_graph,
            temperature=0.1,
            **models)

    dataset = config["dataset"]
    data_class, unnormalize_fn = dataclass_mapping[dataset]

    transform = ReturnDictTransform(attribute_size)
    if dataset in ['celebahq']:
        train_set = data_class(attribute_size, split='train', transform=transform,resolution=args.resolution)
        test_set = data_class(attribute_size, split='test', transform=transform,resolution=args.resolution)
    else:
        train_set = data_class(attribute_size, split='train', transform=transform)
        test_set = data_class(attribute_size, split='test', transform=transform)
    #test_set = data_class(attribute_size, split='test', transform=transform)
    args.attribute_size =attribute_size
    if args.qualitative > 0:
        produce_qualitative_samples(pipe,prompt, dataset=test_set, scm=scm, parents=list(attribute_size.keys()),
                                    intervention_source=train_set, unnormalize_fn=unnormalize_fn, num=args.qualitative,
                                    show_difference=args.show_difference,args=args)

    embedding_model = get_embedding_model(args.embeddings, pretrained_vgg=True, classifier_config=classifier_config)
    
    embedding_fn = get_embedding_fn(args.embeddings, unnormalize_fn, accelerator.prepare(embedding_model))

    if "composition" in args.metrics:
        evaluate_composition(pipe,prompt,accelerator,test_set, unnormalize_fn, batch_size, cycles=args.cycles, scm=scm, embedding=args.embeddings, embedding_fn=embedding_fn,args=args)


    if dataset in ['celebahq']:
        intervention_attrs = ["Smiling","Eyeglasses"]
    else:
        intervention_attrs = attribute_size.keys()
    if "effectiveness" in args.metrics:
        

        if dataset == "morphomnist":
            predictors = {atr: Classifier(attr=atr, num_outputs=attribute_size[atr], context_dim=len(list(config_cls["anticausal_graph"][atr])))
                          for atr in attribute_size.keys()}
        elif dataset in ["celeba",'celebahq']:
            if sum(attribute_size.values()) == 4:
                # predictors = {atr: CelebaComplexClassifier(attr=atr, context_dim=len(list(config_cls["anticausal_graph"][atr])),
                #                                   num_outputs=config_cls[atr +"_num_out"],
                #                                   lr=config_cls["lr"], version=config_cls["version"]) for atr in attribute_size.keys()}
                predictors = {atr: CelebaComplexClassifier(attr=atr, context_dim=len(list(config_cls["anticausal_graph"][atr])),
                                                  num_outputs=config_cls["attribute_size"][atr],
                                                  lr=config_cls["lr"], version=config_cls["version"]) for atr in attribute_size.keys()}
            else:
                predictors = {atr: Celeba_anticausal_Classifier(attr=atr, num_outputs=config_cls["attribute_size"][atr], lr=config_cls["lr"],pretrain=False) for atr in intervention_attrs}
                #predictors = {atr: CelebaClassifier(attr=atr, num_outputs=config_cls[atr +"_num_out"], lr=config_cls["lr"]) for atr in attribute_size.keys()}
                #predictors = {atr: CelebaClassifier(attr=atr, num_outputs=config_cls["attribute_size"][atr], lr=config_cls["lr"]) for atr in intervention_attrs}
                #predictors = {atr: CelebaClassifier(attr=atr, num_outputs=config_cls["attribute_size"][atr], lr=config_cls["lr"],pretrain=False) for atr in intervention_attrs}
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
            # try the last classifier
            # files = [f for f in os.listdir(config_cls["ckpt_path"]) if f.startswith(key)]
            # file_name = sorted(files)[-1] if files else None
            print(file_name)
            cls.load_state_dict(torch.load(config_cls["ckpt_path"] + file_name , map_location=torch.device('cuda'))["state_dict"])
            cls.to('cuda')
            cls.eval()
        

        
        for pa in intervention_attrs:
        #for pa in attribute_size.keys():
            evaluate_effectiveness(pipe, prompt,test_set, unnormalize_fn, batch_size,accelerator, scm=scm, attributes=intervention_attrs, do_parent=pa,
                            intervention_source=train_set, predictors=predictors, dataset=dataset,embedding_fn=embedding_fn,args=args)

    if "fid" in args.metrics:
        feat_dict = evaluate_fid(real_set=train_set, test_set=test_set, batch_size=batch_size, scm=scm, attributes=list(attribute_size.keys()),args=args)

    if "minimality" in args.metrics:
        evaluate_minimality(pipe, prompt,accelerator,real_set=train_set, test_set=test_set, batch_size=batch_size, scm=scm, attributes=list(attribute_size.keys()),
                            embedding=args.embeddings, embedding_fn=embedding_fn,args=args)
