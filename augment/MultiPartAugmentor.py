from augment.Augmentor import Augmentor
from utils import *


class MultiPartAugmentor(Augmentor):

    def __init__(self, params):
        super().__init__(params)

    def get_input_type(self):
        return 'multi-part'

    def _transform_masks(self):
        self._call_buffer['small_masks'] = []
        object_colors = generate_colors(len(self._call_buffer['mask_list']))

        if 'class' in self.output_type_list:
            class_colors = generate_colors(self.num_classes + 1)

        if 'multi-part' in self.output_type_list:
            objects = []
            total_parts = 0

            for i, mask in enumerate(self._call_buffer['mask_list']):
                obj = semantic2binary_list(mask)
                total_parts += len(obj)
                objects.append(obj)

            parts_colors = generate_colors(total_parts)

        for i, mask in enumerate(self._call_buffer['mask_list']):
            m = {}

            if 'single' in self.output_type_list:
                m['single'] = single2multi(semantic2binary(mask))

            if 'multi-object' in self.output_type_list:
                m['multi-object'] = color_mask(mask, object_colors[i])

            if 'multi-part' in self.output_type_list:
                obj_colors = [parts_colors.pop(0) for i in range(len(objects[i]))]
                m['multi-part'] = binary_list2semantic(objects[i], obj_colors)

            if 'class' in self.output_type_list:
                m['class'] = color_mask(mask, class_colors[self._call_buffer['class_list'][i]])

            self._call_buffer['small_masks'].append(m)
