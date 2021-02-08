import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

from datagen import DataGen
from utils import *


class PytorchDataset(Dataset):
    def __init__(self, input_path, input_type,
                 augmentor_params=None,
                 class_mapping=None,
                 separable_class=False,
                 epoch_size=100, scene_samples=4,
                 img_transform=None, mask_transform=None):
        self.epoch_size = epoch_size
        self.scene_samples = scene_samples
        self.datagen = DataGen(
            input_path=input_path, input_type=input_type, augmentor_params=augmentor_params,
            class_mapping=class_mapping, separable_class=separable_class
        )
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        # return len(self.img_list)
        return self.epoch_size

    def __getitem__(self, idx):
        result, classes = self.datagen.get_scene(self.scene_samples)
        scene = result['scene']
        mask = result['masks']['class']  # Only class mask for now

        mask = human2machine_mask(mask, self.datagen.num2class)

        if self.img_transform:
            scene = self.img_transform(scene)

        if self.img_transform:
            mask = self.mask_transform(mask)

        return scene, mask, classes
