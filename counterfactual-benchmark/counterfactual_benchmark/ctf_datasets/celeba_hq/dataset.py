from torch.utils.data import Dataset
import sys
sys.path.append('/home/jovyan/fcvm-data-volume/kzzr229/workspace/MCPL-diffuser')
from edit_modules.load_celebahq import CelebAHQ
from torchvision.transforms import Resize, ToTensor, CenterCrop, Compose, ConvertImageDtype
from torch.utils.data import ConcatDataset
import torch

MIN_MAX = {
    'image': [0.0, 255.0]
}

def load_data(data_dir, split):
    transforms = Compose([Resize((64, 64)), ToTensor(), ConvertImageDtype(dtype=torch.float32),])
    # if split == 'train':
    #     data_train = CelebAHQ(root=data_dir, split='train', transform=transforms, download=False)
    #     data_test = CelebAHQ(root=data_dir, split='test', transform=transforms, download=False)
    #     data = ConcatDataset([data_train, data_test])

    #     # 🔸 Manually attach attributes from data_train
    #     data.attr_names = data_train.attr_names
    #     data.attr = torch.cat([data_train.attr, data_test.attr], dim=0)
    # else:
    #     data = CelebAHQ(root=data_dir, split=split, transform=transforms, download=False)
    data = CelebAHQ(root=data_dir, split=split, transform=transforms, download=False)
    return data


def unnormalize(value, name):
    # [0,1] -> [min,max]
    value = (value * (MIN_MAX[name][1] - MIN_MAX[name][0])) +  MIN_MAX[name][0]
    return value.to(torch.uint8)

class Celebahq(Dataset):
    def __init__(self, attribute_size, split='train', normalize_=True,
                 transform=None, transform_cls=None, data_dir='/home/jovyan/fcvm-data-volume/kzzr229/workspace/counterfactual-benchmark/datasets/'):
        super().__init__()
        self.has_valid_set = True
        self.transform = transform
        self.transform_cls = transform_cls
        self.data = load_data(data_dir, split)

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
            return self.transform(self.data[idx][0], self.attrs[idx])

        if self.transform_cls:
            return self.transform_cls(self.data[idx][0]), self.attrs[idx]

        return self.data[idx][0], self.attrs[idx]
