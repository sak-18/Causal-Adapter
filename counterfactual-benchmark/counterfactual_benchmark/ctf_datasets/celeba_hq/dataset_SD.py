from torch.utils.data import Dataset
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_DIR = str(REPO_ROOT / "counterfactual-benchmark" / "datasets")
sys.path.append(str(REPO_ROOT / "causal-adapter-sd15"))
from edit_modules.load_celebahq import CelebAHQ

from torchvision.transforms import Resize, ToTensor, CenterCrop, Compose, ConvertImageDtype
from torchvision import transforms
import torch

MIN_MAX = {
    'image': [0.0, 255.0]
}

def load_data(data_dir, split,resolution=256):
    
    image_transforms = transforms.Compose(
            [
                transforms.Resize((resolution,resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )
    
    #transforms = Compose([CenterCrop(150), Resize((64, 64)), ToTensor(), ConvertImageDtype(dtype=torch.float32),])
    data = CelebAHQ(root=data_dir, split=split, transform=image_transforms, download=False)
    return data

def unnormalize(value, name,dtype=torch.uint8):
    # [0,1] -> [min,max]
    #uint8 has problem for negative value subtraction
    value = (value * (MIN_MAX[name][1] - MIN_MAX[name][0])) +  MIN_MAX[name][0]
    return value.to(dtype)



class Celebahq(Dataset):
    def __init__(self, attribute_size, split='train', normalize_=True,
                 transform=None, transform_cls=None, data_dir=DEFAULT_DATA_DIR,resolution=256):
        super().__init__()
        self.has_valid_set = True
        self.transform = transform
        self.transform_cls = transform_cls
        resolution= 256
        self.data = load_data(data_dir, split,resolution)

        attribute_ids = [self.data.attr_names.index(attr) for attr in attribute_size.keys()]
        self.metrics = {attr: torch.as_tensor(self.data.attr[:, attr_id], dtype=torch.float32) for attr, attr_id in zip(attribute_size.keys(), attribute_ids)}

        self.attrs = torch.cat([self.metrics[attr].unsqueeze(1)
                                for attr in attribute_size.keys()], dim=1)
        self.possible_values = {attr: torch.unique(values, dim=0) for attr, values in self.metrics.items()}
        self.bins = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.data[idx][0], self.attrs[idx]),idx

        if self.transform_cls:
            return self.transform_cls(self.data[idx][0]), self.attrs[idx],idx

        return self.data[idx][0], self.attrs[idx],idx
