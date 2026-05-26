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
import numpy as np
import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DEEPSCM_CONFIG_DIR = REPO_ROOT / "counterfactual-benchmark" / "counterfactual_benchmark" / "methods" / "deepscm" / "configs"
SD15_ROOT = REPO_ROOT / "causal-adapter-sd15"
sys.path.append("../../")
sys.path.append(str(REPO_ROOT / "causal-adapter-sd15"))

from causal_modules import ddim_modules
import pickle
from models.classifiers.classifier import Classifier
from models.classifiers.celeba_classifier import CelebaClassifier
from models.classifiers.celeba_complex_classifier import CelebaComplexClassifier
from models.classifiers.adni_classifier import ADNIClassifier
from ctf_datasets.morphomnist.dataset import MorphoMNISTLike
from ctf_datasets.celeba.dataset_SD import Celeba
from ctf_datasets.adni.dataset_SD import ADNI
from ctf_datasets.transforms import ReturnDictTransform, get_attribute_ids
from accelerate import Accelerator
from evaluation.metrics.composition_SD import composition
from evaluation.metrics.minimality_SD import minimality
from evaluation.embeddings.embeddings import get_embedding_model, get_embedding_fn
from evaluation.metrics.fid_SD import fid
from evaluation.metrics.effectiveness_SD import effectiveness
from evaluation.metrics.utils import save_selected_images, save_plots
from ctf_datasets.morphomnist.dataset import unnormalize as unnormalize_morphomnist
from ctf_datasets.celeba.dataset_SD import unnormalize as unnormalize_celeba
from ctf_datasets.adni.dataset import unnormalize as unnormalize_adni

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

torch.multiprocessing.set_sharing_strategy('file_system')

rng = np.random.default_rng()

device = torch.device("cuda")
dataclass_mapping = {
    "morphomnist": (MorphoMNISTLike, unnormalize_morphomnist),
    "celeba": (Celeba, unnormalize_celeba),
    "adni": (ADNI, unnormalize_adni)
}
def unnormalize_0_1(tensor):
    """
    Unnormalize a tensor that was normalized using Normalize([0.5], [0.5]).
    Converts from [-1, 1] → [0, 1].
    """
    return tensor * 0.5 + 0.5



def evaluate_minimality(pipe, prompt, file_saved_path,real_set: Dataset , batch_size: int, embedding: str = None,
                        embedding_fn=None, args=None):
    real_data_loader = torch.utils.data.DataLoader(real_set, batch_size=batch_size, shuffle=False)
    output_dir = file_saved_path
    
    counterfactual_images = torch.load(os.path.join(file_saved_path, "counterfactual_tensors.pt"))
    minimality_arrays  = torch.load(os.path.join(file_saved_path, "minimality.pt"))
    factuals = minimality_arrays["factuals"]
    counterfactuals = minimality_arrays["counterfactuals"]
    interventions = minimality_arrays["interventions"]
    
    
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

    log_path = os.path.join(output_dir,'SD_minimality_results.txt')
    if os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('')  # Clear the file content

    with open(log_path, 'a') as f:
        f.write(minimality_msg + "\n")
        f.write(prob_msg + "\n") 

    print('start FID calculation')
    fid_score = fid(real_data_loader, counterfactual_images, pipe, args)
    fid_msg = f"FID: mean {round(np.mean(fid_score), 3):.3f} std {round(np.std(fid_score), 3):.3f}"
    print(fid_msg)

    log_path = os.path.join(output_dir,'SD_fid_results.txt')
    if os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('')  # Clear the file content

    with open(log_path, 'a') as f:
        f.write(fid_msg + "\n")

    return


def datasets_configs(dataset):
    if 'celeA' in dataset:
        if 'simple' in dataset:
            pass
        elif 'complex' in dataset:
            config = str(DEEPSCM_CONFIG_DIR / "celeba" / "complex" / "vae.json")
            classifier_config = str(DEEPSCM_CONFIG_DIR / "celeba" / "complex" / "classifier.json")
    elif 'ADNI' in dataset:
        config = str(DEEPSCM_CONFIG_DIR / "adni" / "vae.json")
        classifier_config = str(DEEPSCM_CONFIG_DIR / "adni" / "classifier.json")

    return config,classifier_config

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Config file for experiment.", default="celeA_complex")
    parser.add_argument("--metrics", '-m',
                        nargs="+", type=str,
                        help="Metrics to calculate. "
                        "Choose one or more of [composition, effectiveness, fid, minimality]. If not set, all metrics are calculated.",
                        choices=["composition", "effectiveness", "fid", "minimality"],
                        default=["minimality"])
    parser.add_argument("--bs", type=int, help="Composition cycles.", default=6)                    
    parser.add_argument("--cycles", '-cc', nargs="+", type=int, help="Composition cycles.", default=[1, 10])
    parser.add_argument("--qualitative", '-qn', type=int, help="Number of qualitative results to produce", default=0)
    parser.add_argument("--show-difference", '-sd', action='store_true', help="Show counterfactual-factual difference on qualitative results")
    parser.add_argument("--embeddings", type=str, choices=["vgg", "clfs", "vae", "lpips", "clip"],default=["clip"], help="What embeddings to use for composition metric. "
                        "Supported: [vgg, clfs, vae, lpips, clip]. If not set, will compute distance on image space")
    parser.add_argument("--sampling-temperature", '-temp', type=float, default=0.1, help="Sampling temperature, used for VAE, HVAE.")
    parser.add_argument("--causalnet_path", 
        type=str,
        default=None,
        help="load causalnet")
    parser.add_argument("--mcpl_embedding_path", 
        type=str,
        default=None,
        help="load mcpl")
    parser.add_argument("--file_saved_path", 
        type=str,
        default=None,
        help="load mcpl") 
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    # torch.manual_seed(42)

    print(args)
    base_model_path = os.environ.get("CAUSAL_ADAPTER_SD15_BASE_MODEL", "lambdalabs/miniSD-diffusers")
    controlnet_path = args.causalnet_path
    mcpl_embedding_path = args.mcpl_embedding_path
    accelerator = Accelerator()
    controlnet = Causal_ControlNetModel.from_pretrained(controlnet_path,torch_dtype=torch.float32)
    if args.dataset in ['celeA_complex']:
        cond_path = str(SD15_ROOT / "logs" / "logs_celeA_complex_all" / "2025-04-23T21-11-19-causalnet_pretrain" / "best_model.pt")
        A_matrix = torch.tensor([[0, 0, 1,1], [0, 0, 1,1], [0, 0, 0,0], [0, 0, 0,0]],dtype=torch.float32).to(device)
        prompt = 'a human of @ and * and & and !'
        presudo_words= '@,*,&,!'
        if cond_path is not None:
            print('load pretrained causalnet weights')
            controlnet.controlnet_cond_embedding.load_state_dict(torch.load(cond_path,weights_only=True))
    elif args.dataset == 'ADNI':
        A_matrix = torch.tensor([[0, 0,0, 1, 0,0], [0, 0,0, 1, 1,0], [0, 0,0,1,0, 0], [0, 0, 0, 0,1,0],[0, 0, 0, 0,0,0],[0, 0, 0, 0,0,0]],dtype=torch.float32).to(device)
        prompt = 'a mri image of @ and * and &'
        append_text = (' '+prompt[-1])*(10-1)
        prompt+=append_text
        presudo_words= '@,*,&'
        '''text six attr'''
        # prompt = 'a mri image of @ and * and & and ! and $ and %'
        # presudo_word='@,*,&,!,$,%'
    
    controlnet.controlnet_cond_embedding.update_mask(A_matrix)
    controlnet.eval()
    presudo_list = presudo_words.split(',')
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

    #batch_size = config["mechanism_models"]["image"]["params"]["batch_size_val"]
    batch_size = args.bs
    scm = None


    dataset = config["dataset"]
    data_class, unnormalize_fn = dataclass_mapping[dataset]

    transform = ReturnDictTransform(attribute_size)

    train_set = data_class(attribute_size, split='train', transform=transform)
    test_set = data_class(attribute_size, split='test', transform=transform)

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

        for pa in attribute_size.keys():
            evaluate_effectiveness(pipe, prompt,test_set, unnormalize_fn, batch_size,accelerator, scm=scm, attributes=list(attribute_size.keys()), do_parent=pa,
                            intervention_source=train_set, predictors=predictors, dataset=dataset,args=args)

    if "fid" in args.metrics:
        feat_dict = evaluate_fid(real_set=train_set, test_set=test_set, batch_size=batch_size, scm=scm, attributes=list(attribute_size.keys()),args=args)

    if "minimality" in args.metrics:
        evaluate_minimality(pipe, prompt,args.file_saved_path,real_set=train_set,batch_size=batch_size,
                            embedding=args.embeddings, embedding_fn=embedding_fn,args=args)
