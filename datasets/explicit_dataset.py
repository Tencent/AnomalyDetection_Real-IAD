from __future__ import division
from typing import List, Dict, Union

import json
import logging
import os.path as osp

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from datasets.base_dataset import BaseDataset, TestBaseTransform, TrainBaseTransform
from datasets.image_reader import build_image_reader
from datasets.transforms import RandomColorJitter

logger = logging.getLogger("global_logger")


def build_explicit_dataloader(cfg, training, distributed=True):

    image_reader = build_image_reader(cfg.image_reader)

    normalize_fn = transforms.Normalize(mean=cfg["pixel_mean"], std=cfg["pixel_std"])
    if training:
        transform_fn = TrainBaseTransform(
            cfg["input_size"], cfg["hflip"], cfg["vflip"], cfg["rotate"]
        )
    else:
        transform_fn = TestBaseTransform(cfg["input_size"])

    colorjitter_fn = None
    if cfg.get("colorjitter", None) and training:
        colorjitter_fn = RandomColorJitter.from_params(cfg["colorjitter"])

    logger.info("building ExplicitDataset from: {}".format(cfg["meta_file"]))

    dataset = ExplicitDataset(
        image_reader,
        cfg["meta_file"],
        training,
        transform_fn=transform_fn,
        normalize_fn=normalize_fn,
        colorjitter_fn=colorjitter_fn,
    )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        sampler=sampler,
        persistent_workers=True,
    )

    return data_loader


class ExplicitDataset(BaseDataset):
    def __init__(
        self,
        image_reader,
        meta_file,
        training,
        transform_fn,
        normalize_fn,
        colorjitter_fn=None,
    ):
        self.image_reader = image_reader
        self.meta_file = meta_file
        self.training = training
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        self.colorjitter_fn = colorjitter_fn

        if isinstance(self.meta_file, str):
            self.meta_file = [meta_file]

        # construct metas
        self.metas = sum((self.load_explicit(path, self.training)
                          for path in self.meta_file), [])

    @staticmethod
    def load_explicit(path: str, is_training: bool) -> List[Dict[str, Union[str, int]]]:
        SAMPLE_KEYS = {'category', 'anomaly_class', 'image_path', 'mask_path'}

        with open(path, 'r') as fp:
            info = json.load(fp)
            assert isinstance(info, dict) and all(
                key in info for key in ('meta', 'train', 'test')
            )
            meta = info['meta']
            train = info['train']
            test = info['test']
            raw_samples = train if is_training else test

        assert isinstance(raw_samples, list) and all(
            isinstance(sample, dict) and set(sample.keys()) == SAMPLE_KEYS
            for sample in raw_samples
        )
        assert isinstance(meta, dict)
        prefix = meta['prefix']
        normal_class = meta['normal_class']

        if is_training:
            return [dict(filename=osp.join(prefix, sample['image_path']),
                         label_name=normal_class, label=0,
                         clsname=sample['category'])
                    for sample in raw_samples]
        else:
            def as_normal(sample):
                return (sample['mask_path'] is None or
                        sample['anomaly_class'] == normal_class)

            return [dict(
                filename=osp.join(prefix, sample['image_path']),
                maskname=None if as_normal(sample)
                else osp.join(prefix, sample['mask_path']),
                label=0 if as_normal(sample) else 1,
                label_name=sample['anomaly_class'],
                clsname=sample['category']
            ) for sample in raw_samples]

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        input = {}
        meta = self.metas[index]

        # read image
        filename = meta["filename"]
        label = meta["label"]
        image = self.image_reader(meta["filename"])
        input.update(
            {
                "filename": filename,
                "height": image.shape[0],
                "width": image.shape[1],
                "label": label,
            }
        )

        if meta.get("clsname", None):
            input["clsname"] = meta["clsname"]
        else:
            input["clsname"] = filename.split("/")[-4]

        image = Image.fromarray(image, "RGB")

        # read / generate mask
        if meta.get("maskname", None):
            input['maskname'] = meta['maskname']
            mask = self.image_reader(meta["maskname"], is_mask=True)
        else:
            input['maskname'] = ''
            if label == 0:  # good
                mask = np.zeros((image.height, image.width)).astype(np.uint8)
            elif label == 1:  # defective
                mask = (np.ones((image.height, image.width)) * 255).astype(np.uint8)
            else:
                raise ValueError("Labels must be [None, 0, 1]!")

        mask = Image.fromarray(mask, "L")

        if self.transform_fn:
            image, mask = self.transform_fn(image, mask)
        if self.colorjitter_fn:
            image = self.colorjitter_fn(image)
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        if self.normalize_fn:
            image = self.normalize_fn(image)
        input.update({"image": image, "mask": mask})
        return input
