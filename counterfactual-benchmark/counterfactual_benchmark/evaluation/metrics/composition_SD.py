import numpy as np
from torchvision import transforms as tfms
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(REPO_ROOT / "causal-adapter-sd15"))
sys.path.append(str(REPO_ROOT / "counterfactual-benchmark" / "counterfactual_benchmark" / "evaluation"))
from causal_modules import ddim_modules
import torch
from embeddings.embeddings import get_embedding_model, get_embedding_fn

def composition(pipe,prompt,dataset,factual_batch, unnormalize_fn, method, cycles=[1, 10], device='cuda', embedding=None, embedding_fn=None,args=None):
    factual_batch = {k: v.to(device) for k, v in factual_batch.items()}
    # TODO change this so classifier embeddings for other datasets are supported
    cond = factual_batch["intensity"] if "intensity" in factual_batch else None

    first_img = factual_batch.copy()
    # convert to 64*64 for saving
    attr_keys = ddim_modules.get_dataset_attrs(dataset)
    
    images = [first_img["image"]]
    label = torch.cat([factual_batch[k] for k in attr_keys], dim=1)  # shape: [1, 4]
    

    for _ in range(max(cycles)):

        normalized_tensor,_,_,_ = ddim_modules.ddim_editing(pipe, factual_batch["image"],label.clone(),prompt,num_steps = int(args.num_steps),set_guidance_scale  = 1.0,intervention_indx=None,intervention_values=None,return_PIL=False)

        images.append(normalized_tensor)
        factual_batch["image"] = normalized_tensor.clone()
    # resize images
    images=ddim_modules.resize_tensor(images, pipe, dataset, return_tensor=True,mode='bicubic')
    
    # compute lipips 
    composition_scores = l1_distance(images, cond, steps=cycles, embedding=embedding, embedding_fn=embedding_fn)
    # compute l1 
    l1_fn = get_embedding_fn(None, unnormalize_fn, None)
    # torch.uint8 will get wrong results for the l1 loss
    l1_scores = l1_distance(images, cond, steps=cycles, embedding='l1', embedding_fn=l1_fn,dtype=None)
    # int8 results 
    l1_scores_int8 = l1_distance(images, cond, steps=cycles, embedding='l1', embedding_fn=l1_fn,dtype=torch.int8)
    # stack images for all cycleso
    #all_images = np.concatenate([unnormalize_fn(image, "image").cpu().numpy() for image in images], axis=3)

    return_dict = {"lpips":composition_scores, "l1_uint8":l1_scores, "l1_int8":l1_scores_int8}
    return return_dict, images[-1]

def l1_distance(images, cond, steps, embedding, embedding_fn,dtype=None):

    distances = {}
    for step in steps:
        if embedding == "lpips":
            # input N,3,H,W, ouput: 1
            distances[step] = np.array([embedding_fn(images[step], images[0])])
        else:
            # input N,3,H,W, ouput: N
            if dtype is not None:
                # int8 dtype
                distances[step] = np.mean(np.abs(embedding_fn(images[step], cond).astype(int) - embedding_fn(images[0], cond).astype(int)), axis=(1,2,3))
            else:
                #original implement uint8
                distances[step] = np.mean(np.abs(embedding_fn(images[step], cond,torch.uint8) - embedding_fn(images[0], cond,torch.uint8)), axis=(1,2,3))
            #distances[step] = np.mean(np.abs(embedding_fn(images[step], cond) - embedding_fn(images[0], cond)), axis=(1,2,3) if embedding is None else 1)
    return distances


def lipips_identity(counter_images, factual_images, embedding_fn):    
    distance = np.array([embedding_fn(counter_images, factual_images)])
    l1_distance = np.array([torch.mean(torch.abs(counter_images-factual_images)).cpu().detach()])
  
    return {'L1':l1_distance,'LPIPS':distance}