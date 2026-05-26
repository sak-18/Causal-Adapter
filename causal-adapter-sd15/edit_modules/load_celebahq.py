import csv
import os
import re

import PIL
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CelebA

from torchvision.datasets.utils import check_integrity, download_file_from_google_drive, extract_archive, verify_str_arg
from tqdm import tqdm

class CelebAHQ(CelebA):
    base_folder = 'celebahq'
    file_list = [
        # File ID                                      MD5 Hash                            Filename
        ('1badu11NqxGf6qM3PTTooQDJvQbejgbTv', 'b08032b342a8e0cf84c273db2b52eef3', 'CelebAMask-HQ.zip'),
        ('0B7EVK8r0v71pY0NSMzRuSXJEVkk', 'd32c9cbf5e040fd4025c592c306e6668', 'list_eval_partition.txt'),
    ]

    def __init__(
        self,
        root,
        split='train',
        target_type='attr',
        attributes=None,
        transform=None,
        target_transform=None,
        download=False,
    ):
        super(CelebA, self).__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        split_map = {
            'train': 0,
            'valid': 1,
            'test': 2,
            'all': None,
        }
        split_ = split_map[verify_str_arg(split.lower(), 'split', ('train', 'valid', 'test', 'all'))]
        splits = self._load_csv('list_eval_partition.txt')
        id_map = self._load_id_map('CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt', header=0)
        attr = self._load_csv('CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt', header=1)
        self.id_map_file_name = 'CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt'
        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

        if mask == slice(None):  # if split == "all"
            self.filename = splits.index
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        self.filename = [id_map[f] for f in self.filename if f in id_map.keys()]

        self.attr = torch.zeros((len(self.filename), attr.data.shape[1]), dtype=torch.int64)
        if split_ is not None:
            for i, f in enumerate(self.filename):
                num = int(re.sub(r'[^0-9]', '', f))
                self.attr[i] = attr.data[num]
        # map from {-1, 1} to {0, 1}
        self.attr = torch.div(self.attr + 1, 2, rounding_mode='floor')
        self.attr_names = attr.header

        if attributes is not None:
            self.attr_idxs = [self.attr_names.index(name) for name in attributes]

    def _check_integrity(self):
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in ['.zip', '.7z'] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, 'CelebAMask-HQ/CelebA-HQ-img'))

    def download(self):
        return
        # if self._check_integrity():
        #     print('Files already downloaded and verified')
        #     return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, None) #md5)

        extract_archive(os.path.join(self.root, self.base_folder, 'CelebAMask-HQ.zip'))

    def _load_id_map(self, filename, header=None):
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=' ', skipinitialspace=True))

        if header is not None:
            data = data[header + 1:]

        indices = [row[0] for row in data]
        data = [row[2] for row in data]  # orig_file

        id_map = {}
        for idx, orig_file in zip(indices, data):
            assert isinstance(orig_file, str)
            id_map[orig_file] = f'{idx}.jpg'

        return id_map

    def __getitem__(self, index):
        x = PIL.Image.open(
            os.path.join(self.root, self.base_folder, 'CelebAMask-HQ/CelebA-HQ-img', self.filename[index])
        )

        target = []
        for t in self.target_type:
            if t == 'attr':
                target.append(self.attr[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(f'Target type "{t}" is not recognized.')

        if self.transform is not None:
            x = self.transform(x)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        if hasattr(self, "attr_idxs"):
            target = target[self.attr_idxs]

        return x, target.float()