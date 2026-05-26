import torch
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.image.fid import NoTrainInceptionV3
from tqdm import tqdm
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.append("../../")
sys.path.append(str(REPO_ROOT / "causal-adapter-sd15"))
from models.utils import rgbify
from causal_modules import ddim_modules
def fid(real_images, generated_images,pipe,args):
    
    metric = FID(normalize=True, reset_real_features=False).set_dtype(torch.float32).to('cuda')
    for i,(real_batch,_) in enumerate(tqdm(real_images)):
        resized_factual_image = ddim_modules.resize_tensor(real_batch["image"], pipe, args.dataset, return_tensor=True)
        metric.update(rgbify(resized_factual_image).to('cuda'), real=True)

    for generated_batch in tqdm(generated_images):

        metric.update(rgbify(generated_batch).to('cuda'), real=False)

    fid_score = metric.compute().cpu().numpy()

    return fid_score
