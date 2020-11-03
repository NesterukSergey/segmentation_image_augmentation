from utils import *
from augment.SingleAugmentor import SingleAugmentor
from augment.MultiPartAugmentor import MultiPartAugmentor
from augment.SemanticAugmentor import SemanticAugmentor


class DataGen:
    def __init__(self, input_path, input_type, augmentor_params={}):
        self.input_path = input_path
        self.input_type = input_type
        self.augmentor_params =augmentor_params
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

    def get_next(self):
        pass
