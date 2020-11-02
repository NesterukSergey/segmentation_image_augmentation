from augment.Augmentor import Augmentor
from utils import *


class SemanticAugmentor(Augmentor):

    def __init__(self, params):
        super().__init__(params)

    def get_input_type(self):
        return 'semantic'

    def _transform_masks(self):
        self._call_buffer['small_masks'] = []
        object_colors = generate_colors(len(self._call_buffer['mask_list']))

        if 'class' in self.output_type_list:
            class_colors = generate_colors(self.num_classes + 1)

        for i, mask in enumerate(self._call_buffer['mask_list']):
            m = {}

            if 'semantic' in self.output_type_list:
                m['semantic'] = mask

            if 'single' in self.output_type_list:
                m['single'] = single2multi(semantic2binary(mask))

            if 'multi-object' in self.output_type_list:
                m['multi-object'] = color_mask(mask, object_colors[i])

            if 'class' in self.output_type_list:
                m['class'] = color_mask(mask, class_colors[self._call_buffer['class_list'][i]])

            self._call_buffer['small_masks'].append(m)
