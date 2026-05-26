import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PENDULUM_ROOT = REPO_ROOT / "causal-adapter-sd15" / "dataset" / "causal_data2" / "pendulum"
sys.path.append("../../")
from ctf_datasets.morphomnist.io import load_idx


def load_pendulum_like(image_paths):
    images,labels=[],[]
    image_transforms = transforms.Compose(
        [
            transforms.Resize((96,96), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            #transforms.Normalize([0.5], [0.5]),
        ]
    )


    for img_path in image_paths:
        img_label = list(map(float,img_path[:-4].split("/")[-1].split('_')[1:]))
        img = Image.open(img_path)
        if not img.mode == "RGB":
            img = img.convert("RGB")
        img = image_transforms(img)
        images.append(img)
        labels.append(img_label)
    return images, labels

def normalize_label_gaussian(label):
    scale = np.array([[2,42],[104,44],[7.5, 4.5],[11,8]])
    norm_label = np.zeros(label.shape)
    for i in range(label.shape[0]):
        norm_label[i] = (label[i] - scale[i][0]) / scale[i][1]
    return norm_label.astype(np.float32)

def unnormalize(value, name):
    # [-1,1] -> [0,1]
    value = (value + 1) / 2
    # [0,1] -> [min,max]
    value = (value * (MIN_MAX[name][1] - MIN_MAX[name][0])) +  MIN_MAX[name][0]
    return value

class PendulumLike(Dataset):
    def __init__(self, attribute_size, split='train', normalize_=True, transform=None, data_dir=None):
        self.has_valid_set = False
        self.root_dir = data_dir
        self.train = True if split == 'train' else False
        if self.train:
            self.root_dir = str(PENDULUM_ROOT / "train")
        else:
            self.root_dir = str(PENDULUM_ROOT / "test")
        self.transform = transform

        # digit is loaded from labels
        image_paths = [os.path.join(self.root_dir, file_path) for file_path in os.listdir(self.root_dir)]
        images,labels = load_pendulum_like(image_paths)
        
        self.images = images
        if normalize_:
            labels = torch.from_numpy(np.apply_along_axis(normalize_label_gaussian, 1, labels))
        
        self.metrics = {}
        attr_keys = ["pendulum","light","shadow_length","shadow_position"]
        for i, attr in enumerate(attr_keys):
            # Causal_cond[:, i] is the new attribute for attr,causal_cond is [bs,attrs,1]
            self.metrics[attr] = labels[:, i]

        self.attrs = torch.cat([self.metrics[attr].unsqueeze(1) for attr in attribute_size.keys()], dim=1)

        self.possible_values = {attr: torch.unique(values, dim=0) for attr, values in self.metrics.items()}
        # bins = array([-1.        , -0.77777778, -0.55555556, -0.33333333, -0.11111111,
        #0.11111111,  0.33333333,  0.55555556,  0.77777778,  1.        ])
        bins = np.linspace(-1, 1, 10)
        self.bins = {}
        for attr, values in self.metrics.items():
            if attr != "digit":
                data = values.numpy()
                # Return the indices of the bins to which each value in input array belongs.
                # data[:5] array([-0.43379235, -0.28877765, -0.3385411 , -0.26513082, -0.67472506])
                # digitized[:5] array([3, 4, 3, 4, 2])

                digitized = np.digitize(data, bins)
                # every bins average values
                # thickness: [-0.8286383, -0.6450405, -0.4468974, -0.23708051, -0.023331264, 0.19399273, 0.41256616, 0.61590433, 0.8744294]
                self.bins[attr] = [data[digitized == i].mean() for i in range(1, len(bins))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {col: values[idx] for col, values in self.metrics.items()}
        item['image'] = self.images[idx]
        item['attrs'] = self.attrs[idx]
        if self.transform:
            return self.transform(item["image"], item['attrs'])
        return item['image'], item['attrs']
