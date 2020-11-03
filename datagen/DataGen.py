import os
import random

from utils import *
from augment.SingleAugmentor import SingleAugmentor
from augment.MultiPartAugmentor import MultiPartAugmentor
from augment.SemanticAugmentor import SemanticAugmentor


class DataGen:
    def __init__(self, input_path, input_type,
                 img_prefix='rgb', mask_prefix='label',
                 augmentor_params=None, class_mapping=None,
                 balance=True, scene_samples=1):

        self.input_path = input_path
        self.input_type = input_type
        self.augmentor_params = augmentor_params
        self.img_prefix = img_prefix
        self.mask_prefix = mask_prefix
        self.class_mapping = class_mapping
        self.balance = balance
        self.scene_samples = scene_samples

        self._get_classes()
        self.set_class_mappings()
        self._get_pairs_list()
        self._get_stats()
        self._set_augmentor()

    def _set_augmentor(self):
        augmentor_types = {
            'single': SingleAugmentor,
            'multi-part': MultiPartAugmentor,
            'semantic': SemanticAugmentor
        }

        if self.input_type not in list(augmentor_types.keys()):
            raise UserWarning('Unrecognized input type: {}'.format(self.input_type))
        else:
            self.augmentor = augmentor_types[self.input_type](self.augmentor_params)

    def _get_classes(self):
        self.classes = next(os.walk(self.input_path))[1]

    def set_class_mappings(self):
        if self.class_mapping is None:
            class2num = {}

            for i, c in enumerate(self.classes):
                class2num[c] = i

            self.class2num = class2num

        else:
            self.class2num = self.class_mapping

        num2class = {}
        for k, v in self.class2num.items():
            num2class[v] = k

        self.num2class = num2class

    def _get_pairs_list(self):
        input_pairs = {}

        for class_name in self.classes:
            class_dir = os.path.join(self.input_path, class_name)
            subdirs = set(next(os.walk(class_dir))[1])

            if len(subdirs) == 0:
                images, masks = get_img_mask_list(class_dir, mask_path=None,
                                              img_prefix=self.img_prefix, mask_prefix=self.mask_prefix)

            elif subdirs == {self.img_prefix, self.mask_prefix}:
                img_class_dir = os.path.join(class_dir, self.img_prefix)
                mask_class_dir = os.path.join(class_dir, self.mask_prefix)

                images, masks = get_img_mask_list(img_class_dir, mask_path=mask_class_dir,
                                                  img_prefix=self.img_prefix, mask_prefix=self.mask_prefix)

            else:
                raise UserWarning('Wrong input files structure')

            input_pairs[class_name] = {
                'images': images,
                'masks': masks
            }

        self.input_pairs = input_pairs

    def _get_stats(self):
        self.classes = list(self.input_pairs.keys())
        self.num_classes = len(self.classes)

        self.sample_per_class = [len(self.input_pairs[c]['images']) for c in self.input_pairs]
        self.total_samples = sum(self.sample_per_class)

        if self.balance:
            self.class_weights = [i / self.total_samples for i in self.sample_per_class]
        else:
            self.class_weights = [1 / self.num_classes] * self.num_classes

    def _get_next_input_pair(self, class_name=None):
        if class_name is None:
            class_name = random.choices(self.classes, weights=self.class_weights, k=1)[0]

        sample_num = random.randint(0, len(self.input_pairs[class_name]['images']) - 1)
        img = self.input_pairs[class_name]['images'][sample_num]
        mask = self.input_pairs[class_name]['masks'][sample_num]

        return read(img), read(mask), self.class2num[class_name]

    def get_scene(self):
        images = []
        masks = []
        classes = []

        for i in range(self.scene_samples):
            img, msk, cl = self._get_next_input_pair()
            images.append(img)
            masks.append(msk)
            classes.append(cl)

        transformed_scene = self.augmentor.transform(images, masks)

        return transformed_scene, classes
