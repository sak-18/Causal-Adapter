"""Textual-inversion dataset dispatcher.

The :class:`TextualInversionDataset` here is a thin wrapper that picks the
appropriate :mod:`._adapters` adapter based on the ``dataset`` argument and
wires its loader/transforms/labels into the standard ``TextualInversion``
training contract:

- ``__getitem__`` returns a dict with ``input_ids``, ``pixel_values`` and
  (optionally) ``label``.
- ``__len__`` returns the dataset length.
- ``imglabel`` exposes the per-sample attribute tensor for samplers built on
  top of the dataset.

Per-dataset details (image source, attribute construction, transforms, prompt
extension) live in the adapters under :mod:`._adapters`.
"""

from __future__ import annotations

import random

import PIL
from packaging import version
from torch.utils.data import Dataset

from ._adapters import ADAPTERS


# Pillow 9.1+ moved the resampling enums into ``Image.Resampling``.
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


# Standard textual-inversion prompt templates. The "style" variant is unused
# in this codebase (kept for API symmetry with upstream textual_inversion).
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


class TextualInversionDataset(Dataset):
    """Dispatcher dataset for textual-inversion training across causal datasets."""

    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",
        size=512,
        repeats=1,
        interpolation="bicubic",
        flip_p=0.0,
        set="train",
        placeholder_token="*",
        center_crop=False,
        random_article=False,
        dataset="pendulum",
        random_prompt_template=False,
    ):
        if dataset not in ADAPTERS:
            raise ValueError(f"Unknown dataset: {dataset!r}")

        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        # NB: the original code unconditionally chose the "object" templates
        # regardless of ``learnable_property``; preserved for backward compat.
        self.templates = imagenet_templates_small
        self.random_prompt_template = random_prompt_template
        self.random_article = random_article
        self.dataset = dataset

        self.adapter = ADAPTERS[dataset](
            data_root=data_root,
            size=size,
            set_=set,
            dataset=dataset,
        )
        self.image_transforms = self.adapter.image_transforms
        self.normalize_transforms = self.adapter.normalize_transforms
        self.imglabel = self.adapter.imglabel
        self.num_images = self.adapter.num_images
        self._length = self.num_images

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        wrapped = i % self.num_images
        image = self.adapter.load_image(wrapped)

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.random_prompt_template:
            text = random.choice(self.templates).format(self.placeholder_token)
        else:
            text = self.placeholder_token
        text = self.adapter.extend_text(text)

        example = {}
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        image = self.image_transforms(image)
        image = self.normalize_transforms(image)
        example["pixel_values"] = image

        if self.imglabel is not None:
            example["label"] = self.imglabel[wrapped]
        return example
