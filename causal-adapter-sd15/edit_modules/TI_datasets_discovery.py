import argparse
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import CelebA
from .load_celebahq import CelebAHQ
import pytorch_lightning.loggers
import numpy as np
import PIL
from PIL import Image
import safetensors
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from .load_datasets_morphominist import _get_paths,load_idx
from .load_datasets_adni import load_data,load_extra_attributes,ordinal_array
from .load_datasets_adni import normalize as adni_normalize
from .load_datasets_adni import unnormalize as adni_unormalize
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from packaging import version

ADNI_MIN_MAX = {
    
    'image': [0.0, 255.0],
    'age': [55.1, 89.3],
    'brain_vol': [669364.0, 1350180.0],
    'vent_vol': [5834.0, 145115.0]
}
MorphoMnist_MIN_MAX = {
    "thickness": [0.87598526, 6.255515],
    "intensity": [66.601204, 254.90317],
}


if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# -------

imagenet_templates_smallest = [
    "a photo of {}",
]

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

def bin_array(num: torch.Tensor, m: int = None, reverse: bool = False):

    if reverse:
        if num.dim() == 1:
            num = num.unsqueeze(dim=0)
        # num: shape (bs, m), binary vectors like [0, 1, 1, 0]
        # Output: shape (bs,), scalar representation
        bs, width = num.shape
        weights = 2 ** torch.arange(width - 1, -1, -1, dtype=torch.float32, device=num.device)
        return torch.sum(num * weights, dim=1)
    else:
        # num: shape (bs,), each element is an integer to be encoded to binary
        if m is None:
            m = int(torch.ceil(torch.log2(num.max().float() + 1)).item())
        bs = num.shape[0]
        powers = 2 ** torch.arange(m - 1, -1, -1, device=num.device)
        num = num.unsqueeze(1).long()
        return ((num & powers) > 0).float()

class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=1,
        interpolation="bicubic",
        flip_p=0.0,
        set="train",
        placeholder_token="*",
        center_crop=False,
        random_article=False,
        dataset='pendulum',
        random_prompt_template=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token

        #assert len(placeholder_token.split(' ')) == len(self.tokenizer.encode(placeholder_token,add_special_tokens=False)),"Unknown words is wrong tokened" 

        self.center_crop = center_crop
        self.flip_p = flip_p


        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]


        # new add normaliza here
        # self.conditioning_image_transforms = transforms.Compose(
        #     [
        #         transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR),
        #         transforms.ToTensor(),
        #         # transforms.Normalize([0.5], [0.5]),
        #     ]
        # )

        self.templates = imagenet_templates_small if learnable_property == "style" else imagenet_templates_small
        
        self.random_prompt_template = random_prompt_template
        self.random_article = random_article

        
        self.dataset = dataset
        if dataset == 'pendulum':
            self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]
            self.image_names = [file_path.split('.')[0] for file_path in os.listdir(self.data_root)]

            self.num_images = len(self.image_paths)
            self._length = self.num_images

            
            self._length = self.num_images
            self.imglabel = [list(map(float,k[:-4].split("/")[-1].split('_')[1:])) for k in self.image_paths]
            # process as causalvae but with min-max
            self.scale = np.array([[-40,43],[60,147],[3, 12],[2,19]])
            self.imglabel = torch.from_numpy(np.apply_along_axis(self.normalize_label_minmax, 1, self.imglabel))

            # process as DEAR
            # self.scale = torch.Tensor([[0.0000, 48.0000, 2.0000, 2.0178], [40.5000, 88.5000, 14.8639, 14.4211]])
            # self.imglabel = (torch.tensor(self.imglabel)-self.scale[0])/self.scale[1]
            
            #guaussian normalize (June 20th)
            #self.imglabel = torch.from_numpy(np.apply_along_axis(self.normalize_label_gaussian, 1, self.imglabel))
            
            #scale = torch.Tensor([[0.0000, 48.0000, 2.0000, 2.0178], [40.5000, 88.5000, 14.8639, 14.4211]])
            #scale = np.array([[2,42],[104,44],[7.5, 4.5],[11,8]])
            #mm = torch.Tensor([20.2500, 68.2500, 6.9928, 8.7982]), ss = torch.Tensor([11.8357, 11.8357, 2.8422, 2.1776])
            self.image_transforms = transforms.Compose(
            [
                transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                #transforms.Normalize([0.5], [0.5]),
            ]
            )
            self.normalize_transforms = transforms.Compose(
                    [
                        transforms.Normalize([0.5], [0.5]),
                    ]
            )
            #print(self.imglabel)
        elif 'celeA' in dataset:
            data_dir = self.data_root
            #self.data = CelebA(root=data_dir, split='train', transform=None, download=False)
            self.data = CelebA(root=data_dir, split='train', transform=None, download=False)
            self.num_images = len(self.data)
            self._length = self.num_images
            if 'simple' in dataset:
                selected_item = ['Smiling','Eyeglasses']
            elif 'complex' in dataset:
                selected_item = ['Young','Male','No_Beard','Bald']
            else:
                AssertionError('no such {} dataset'.format(dataset))
            attribute_ids = [self.data.attr_names.index(attr) for attr in selected_item]
            metrics = {attr: torch.as_tensor(self.data.attr[:, attr_id], dtype=torch.float32) for attr, attr_id in zip(selected_item, attribute_ids)}

            attrs = torch.cat([metrics[attr].unsqueeze(1)
                                    for attr in selected_item], dim=1)


            self.imglabel= attrs

            possible_values = {attr: torch.unique(values, dim=0) for attr, values in metrics.items()}
            self.image_transforms = transforms.Compose(
                [
                transforms.CenterCrop(150),
                transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize([0.5], [0.5]),
                ]
            )
            self.normalize_transforms = transforms.Compose(
                [
                    transforms.Normalize([0.5], [0.5]),
                ]
            )


            # dataset_id = self.data_root.split('/')[-1].split('_')[-1]
            # root_parent = os.path.dirname(self.data_root)
            # csv_path = os.path.join(root_parent,'meta_'+dataset_id+'.csv')
            # df = pd.read_csv(csv_path)
            # img_names =np.asarray(df.image_id)
            # self.image_paths = [os.path.join(self.data_root, file_path) for file_path in img_names]
            # label_list = np.asarray(df[selected_item])
            
            # # Convert the entire array to integer type
            # label_list = label_list.astype(int)
            # label_list[label_list == -1] = 0
            # self.imglabel =torch.from_numpy(label_list)
            
            # # this scale for max-min normalization
            # self.num_images = len(self.image_paths)
            # self._length = self.num_images
            
            # self.image_transforms = transforms.Compose(
            # [
            #     transforms.CenterCrop(150),
            #     transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR),
            #     transforms.ToTensor(),
            #     #transforms.Normalize([0.5], [0.5]),
            # ]
            # )
            # self.normalize_transforms = transforms.Compose(
            #     [
            #         transforms.Normalize([0.5], [0.5]),
            #     ]
            # )  
        elif 'celebahq' in dataset:
            data_dir = self.data_root
            
            pre_transforms = transforms.Compose(
                [
                transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR),
                ]
            )
            self.data = CelebAHQ(root=data_dir, split='train', transform=pre_transforms, download=False)
            self.num_images = len(self.data)
            self._length = self.num_images
            if 'simple' in dataset:
                selected_item = ['Smiling','Eyeglasses','Mouth_Slightly_Open','Male','Bald','Wearing_Lipstick','Wearing_Hat']
            elif 'complex' in dataset:
                pass
            else:
                AssertionError('no such {} dataset'.format(dataset))
            attribute_ids = [self.data.attr_names.index(attr) for attr in selected_item]
            metrics = {attr: torch.as_tensor(self.data.attr[:, attr_id], dtype=torch.float32) for attr, attr_id in zip(selected_item, attribute_ids)}

            attrs = torch.cat([metrics[attr].unsqueeze(1)
                                    for attr in selected_item], dim=1)


            self.imglabel= attrs

            possible_values = {attr: torch.unique(values, dim=0) for attr, values in metrics.items()}
            self.image_transforms = transforms.Compose(
                [
                #transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize([0.5], [0.5]),
                ]
            )
            self.normalize_transforms = transforms.Compose(
                [
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
        elif dataset == 'MorphoMNIST':
            # MIN_MAX = {
            #     "thickness": [0.87598526, 6.255515],
            #     "intensity": [66.601204, 254.90317],
            # }
            if set == "train":
                train_bool = True
            images_path, labels_path, metrics_path = _get_paths(self.data_root,train=train_bool)
            images = load_idx(images_path)
            # digits 10 numbers
            labels = load_idx(labels_path)
            # for thickness and intensity
            metric = pd.read_csv(metrics_path, index_col='index')
            
            #metric[['thickness', 'intensity']] = (metric[['thickness', 'intensity']] - metric[['thickness', 'intensity']].mean()) / metric[['thickness', 'intensity']].std()

            # Concatenate normalized metrics with labels
            metric['label'] = labels
            # # Convert thickness and intensity to tensor
            # df_normalized = (metric - metric.min()) / (metric.max() - metric.min())
            # # Step 2: Transform to [-1, 1]
            # df_transformed = df_normalized * 2 - 1
            # Select columns for Z-score normalization and Min-Max normalization
            z_score_columns = ['thickness', 'intensity']
            min_max_columns = ['label']

            df_transformed = self.zscore_continu_minmax_categorical(metric,z_score_columns,min_max_columns)


            self.num_images = len(images)
            self._length = self.num_images
            self.data = [Image.fromarray(images[i]) for i in range(images.shape[0])]
            # thickness   intensity  label
            self.imglabel = torch.from_numpy(df_transformed.values)

            self.image_transforms = transforms.Compose(
            [
                transforms.Pad(padding=2),
                transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                #transforms.Normalize([0.5], [0.5]),
            ]
            )
            self.normalize_transforms = transforms.Compose(
                [
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

        elif dataset == 'ADNI':
            num_of_slices = 10
            keep_only_screening = False
            data_dir = os.path.join(self.data_root, 'preprocessed_data')
            self.image_paths, attribute_dict, subject_dates_dict = load_data(data_dir, num_of_slices=num_of_slices,
                                                                    split=set,
                                                                    keep_only_screening=keep_only_screening)
            csv_path = list(Path(self.data_root).glob('ADNIMERGE*.csv'))[0]
            assert csv_path.is_file(), "Provide ADNIMERGE csv path"
            attribute_size={
                    "apoE": 2,
                    "age": 1,
                    "sex": 1,
                    "brain_vol": 1,
                    "vent_vol": 1,
                    "slice": num_of_slices
                }
            self.num_of_slices=num_of_slices
            attributes, indices_to_remove = load_extra_attributes(csv_path, attributes=attribute_size.keys(),
                                                                    attribute_dict=attribute_dict, subject_dates_dict=subject_dates_dict,
                                                                    keep_only_screening=keep_only_screening)
                                                                   
            attributes['slice'] = np.delete(attributes['slice'], indices_to_remove, axis=0)
            '''min-max normalize for [,sex]'''
            attributes = {attr: adni_normalize(torch.tensor(np.array(values), dtype=torch.float32), attr) for attr, values in attributes.items()}
            '''standard normalize follow the SDCD discovery'''
            # attributes['slice'] = ordinal_array(torch.from_numpy(attributes['slice']),reverse=True)
            # attributes['apoE'] = bin_array(torch.from_numpy(np.array(attributes['apoE'])),reverse=True)
            #scaler = StandardScaler()
            # attributes = {
            #     attr: torch.tensor(
            #         scaler.fit_transform(np.array(values).reshape(-1, 1)).flatten(),  # 标准化 + 降维
            #         dtype=torch.float32
            #     )
            #     for attr, values in attributes.items()
            # }
            "not do normalize for categorical feature?"
            # attributes = {
            #     attr: (
            #         torch.tensor(
            #             scaler.fit_transform(np.array(values).reshape(-1, 1)).flatten(),
            #             dtype=torch.float32
            #         ) if attr not in ['apoE', 'slice','sex'] else
            #         torch.tensor(np.array(values), dtype=torch.float32)
            #     )
            #     for attr, values in attributes.items()
            # }
            
            
            
            # [attr1,attr2,age,sex,b_v,ven_v,slide 0-9]
            attrs = torch.cat([attributes[attr].unsqueeze(1) if len(attributes[attr].shape) == 1 else attributes[attr]
                                for attr in attribute_size.keys()], dim=1)
            
            self.image_paths = [item for i, item in enumerate(self.image_paths) if i not in indices_to_remove]
            self.num_images = len(self.image_paths)
            self._length = self.num_images
            
            self.imglabel = attrs
            self.image_transforms = transforms.Compose(
            [
                transforms.Pad(padding=6),
                transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize([0.5], [0.5]),
            ]
            )
            self.normalize_transforms = transforms.Compose(
                [
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
        elif dataset=='human':
            
            img_names =os.listdir(self.data_root)
            self.image_paths = [os.path.join(self.data_root, file_path) for file_path in img_names]
            label_list = np.asarray([[1,1,1,1]])
            
            self.imglabel =torch.from_numpy(label_list)
            
            # this scale for max-min normalization
            self.num_images = len(self.image_paths)
            self._length = self.num_images
            
            self.image_transforms = transforms.Compose(
            [
                transforms.CenterCrop(150),
                transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                #transforms.Normalize([0.5], [0.5]),
            ]
            )
            self.normalize_transforms = transforms.Compose(
                [
                    transforms.Normalize([0.5], [0.5]),
                ]
            )  

        elif dataset == 'chexpert':
            # name like sampling_100 , sampling_500 dir
            select_columns = ['Sex', 'Age', 'Pleural Effusion']
            dataset_id = self.data_root.split('/')[-1].split('_')[-1]
            root_parent = os.path.dirname(self.data_root)
            csv_path = os.path.join(root_parent,'meta_'+dataset_id+'.csv')
            df = pd.read_csv(csv_path)
            img_names =np.asarray(df.Path)
            self.image_paths = [os.path.join(self.data_root, file_path) for file_path in img_names]
            label_list = np.asarray(df[select_columns])
            mapping = {'Female': 0, 'Male': 1}
            # Convert the first column to integers using the mapping
            label_list[:, 0] = np.vectorize(mapping.get)(label_list[:, 0])
            # Convert the entire array to integer type
            label_list = label_list.astype(int)
            self.imglabel =label_list
            # this scale for max-min normalization
            col_means = self.imglabel.mean(axis=0)
            col_stds = self.imglabel.std(axis=0)  # ddof=1 for sample std; use ddof=0 for population std
            self.scale = np.column_stack((col_means, col_stds))
            #self.scale = np.array([[0,1],[20,100],[0, 1]])

            self.image_transforms = transforms.Compose(
            [
                transforms.RandomApply(
                [transforms.RandomRotation(degrees=10)],  # Random rotation up to ±15 degrees
                p=0.3  # Probability of 0.3 to apply rotation
                ),  
                transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                #transforms.Normalize([0.5], [0.5]),
            ]
            )
            self.normalize_transforms = transforms.Compose(
                [
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
        else:
            AssertionError('no such {} dataset'.format(dataset))


    def randn_article_words(self,placeholder_string):
        # Original string and list of replacement words
        original_string = placeholder_string
        replacement_words = ['a', 'one', 'the']

        # Split the string into words
        words = original_string.split()

        for i,word in enumerate(words):
            if word == 'a':
                words[i] = random.choice(replacement_words)

        # Join the words back into a string
        modified_string = ' '.join(words)

        return modified_string

    def normalize_label_gaussian(self,label):
        scale = np.array([[2,42],[104,44],[7.5, 4.5],[11,8]])
        norm_label = np.zeros(label.shape)
        for i in range(label.shape[0]):
            norm_label[i] = (label[i] - scale[i][0]) / scale[i][1]
        return norm_label.astype(np.float32)

    def normalize_label_minmax(self,label):
        norm_label = torch.zeros(label.shape)
        for i in range(label.shape[0]):
            nor_v = (label[i] - self.scale[i][0]) / (self.scale[i][1] - self.scale[i][0])
            # further limit them to (-1,1)
            #norm_label[i] = nor_v*2-1
            norm_label[i] = nor_v
        return norm_label
    
    def zscore_continu_minmax_categorical(self,attr_df,z_score_columns,min_max_columns):
        # Apply Z-score normalization
        scaler_z = StandardScaler()
        attr_df[z_score_columns] = scaler_z.fit_transform(attr_df[z_score_columns])

        # Apply Min-Max normalization (scaled to range [0,1])
        scaler_minmax = MinMaxScaler()
        attr_df[min_max_columns] = scaler_minmax.fit_transform(attr_df[min_max_columns])

        # Transform Min-Max normalized data to range [-1,1]
        #attr_df[min_max_columns] = attr_df[min_max_columns] * 2 - 1
        return attr_df


    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        if self.dataset in ['celeA_simple','celeA_complex','celebahq_simple']:
            image = self.data[i%self.num_images][0]
            #image = Image.open(self.image_paths[i % self.num_images])
        elif self.dataset in ['MorphoMNIST']:
            image = self.data[i%self.num_images]
        elif self.dataset in ['pendulum','human']:
            image = Image.open(self.image_paths[i % self.num_images])
        elif self.dataset in ['ADNI']:
            # Tiff image
            #in ADNI, image_path is a list of [180,180] [0-1] images
            image = self.image_paths[i % self.num_images]    
            # Normalize the image (scale from min-max to 0-255)
            image_array = (image * 255).clip(0, 255).astype(np.uint8)
            image = Image.fromarray(image_array, mode="L")

        if not image.mode == "RGB":
            image = image.convert("RGB")

        
        # if self.random_article:
        #     placeholder_string = self.randn_article_words(placeholder_string)
        if self.random_prompt_template:
            text = random.choice(self.templates).format(self.placeholder_token)
        else:
            text = self.placeholder_token
        #for editing?
        if self.dataset == 'ADNI':
            if self.num_of_slices>1:
                # extend the slide as it encoder into one-hot encoder
                append_text = (' '+text[-1])*(self.num_of_slices-1)
                text+=append_text

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        image = self.image_transforms(image) 
        #condition_image = image.clone()
        image = self.normalize_transforms(image)

        # condition_image = image.copy()
        # image = self.image_transforms(image) 
        # condition_image = self.conditioning_image_transforms(condition_image)
        example["pixel_values"] = image
        #example["conditioning_pixel_values"] = condition_image
        

        if self.imglabel is not None:
            label = self.imglabel[i% self.num_images]
            # if self.dataset in ['celeA_simple','celeA_complex','MorphoMNIST','ADNI','human']:
            #   label = self.imglabel[i% self.num_images]    
            # else:
            #     label = torch.from_numpy(np.asarray(self.imglabel[i% self.num_images]))
            #    array1 = np.asarray(label).astype(np.float32)
            #    label = torch.from_numpy(array1)
            #    label = self.normalize_label_minmax(label)
            # Gaussian_normlaize for all
            
            # if self.dataset in ['pendulum']:
            #     label = self.normalize_label_gaussian(label)
            # elif self.dataset in ['celeA','skin_cancer','chexpert']:
            #     label = self.normalize_label_minmax(label)
            example["label"] = label
        return example