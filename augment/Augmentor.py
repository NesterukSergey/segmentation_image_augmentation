from abc import ABC, abstractmethod

from utils import *
from transformations import *


class Augmentor(ABC):

    def __init__(self, params):

        self.output_type_list = params.get('output_type_list', ['single', 'multi-object'])
        self.overlap_ratio = params.get('overlap_ratio', 0)
        self.persp_trans = params.get('persp_trans', 0)
        self.background = params.get('background', 'none')
        self.background_image_list = params.get('background_image_list', None)
        self.flip_prob = params.get('flip_prob', 0.5)
        self.max_rotate_degree = params.get('max_rotate_degree', 30)
        self.salt = params.get('salt', 0)
        self.pepper = params.get('pepper', 0)
        self.gauss_var = params.get('gauss_var', 0)
        self.smooth_kernel_size = params.get('smooth_kernel_size', 1)
        self.bboxes = params.get('bboxes', False)
        self.num_classes = params.get('num_classes', 0)
        self.adjust_mask = params.get('adjust_mask', True)
        self.pad_mask = params.get('pad_mask', 25)

        self.max_background_images = 20

        """
        single       - single-channel mask, shows object presence
        multi-object - multi-channel mask, separate color for each object (for each plant)
        multi-part   - multi-channel mask, separate color for each object part (for each leaf)
        semantic     - multi-channel mask, separate color for each type of object (leaf, root, flower)
        class        - multi-channel mask, separate color for each class (plant variety)
        """
        self.possible_inp2out = {
            'single': ['single', 'multi-object', 'class'],
            'multi-part': ['single', 'multi-object', 'multi-part', 'class'],
            'semantic': ['single', 'multi-object', 'semantic', 'class']
        }

        self.input_type = self.get_input_type()
        self._check_params()
        self._call_buffer = {}

    @abstractmethod
    def get_input_type(self):
        pass

    def _check_params(self):
        if self.input_type not in self.possible_inp2out:
            raise UserWarning('{} input type is not supported. See {} for the details'.format(
                self.input_type, self.possible_inp2out))

        for output_type in self.output_type_list:
            if output_type not in self.possible_inp2out[self.input_type]:
                raise UserWarning('{} output type is not supported for {} input type. See {} for the details'.format(
                    output_type, self.input_type, self.possible_inp2out))

        if 'class' in self.output_type_list:
            assert self.num_classes > 0

        if not (0 <= self.overlap_ratio < 1):
            raise UserWarning('Wrong overlap_ratio value')

        if self.persp_trans < 0:
            raise UserWarning('persp_trans must be >= 0')

        if (self.persp_trans > 0.2) and (self.bboxes):
            raise UserWarning('Bounding box is not supported for strong perspective transform yet')

        if self.background not in ['img', 'none']:
            raise UserWarning('background type is not supported')

        if (self.background == 'img') and (self.background_image_list is None):
            raise UserWarning('"background" "img" expects "background_image_list" to be not None')

        if (self.background == 'img') and (len(self.background_image_list) == 0):
            raise UserWarning('background_image_list should be non-empty')

        if (self.background == 'img') and (len(self.background_image_list) > self.max_background_images):
            self.background_image_list = self.background_image_list[:self.max_background_images]
            print('Too many background images ({}). Some will be ignored'.format(len(self.background_image_list)))

        for i, img in enumerate(self.background_image_list):
            try:
                check_is_image(img)
            except:
                try:
                    # print('reading image')
                    self.background_image_list[i] = read(img)
                except:
                    raise UserWarning('Cannot read an image')

        if self.salt < 0:
            raise UserWarning('Wrong value')

        if self.pepper < 0:
            raise UserWarning('Wrong value')

        if self.gauss_var < 0:
            raise UserWarning('Wrong value')

        if (self.smooth_kernel_size < 1) or (self.smooth_kernel_size % 2 == 0):
            raise UserWarning('Wrong value')

    def _check_input(self, img_list, mask_list, class_list=None):
        assert len(img_list) == len(mask_list)

        for i in range(len(img_list)):
            check_is_image(img_list[i])
            assert img_list[i].shape[:2] == mask_list[i].shape[:2]

            if mask_list[i].shape[2] == 1:
                mask_list[i] = single2multi(mask_list[i])

            check_is_image(mask_list[i])

        if 'class' in self.output_type_list:
            assert class_list is not None
            assert len(class_list) == len(img_list)

        return img_list, mask_list, class_list

    def _get_scene_size(self):
        objects_height_ends = [
            self._call_buffer['objects_positions'][i][0] + self._call_buffer['real_img_sizes'][i][0]
            for i in range(len(self._call_buffer['objects_positions']))]

        objects_width_ends = [
            self._call_buffer['objects_positions'][i][1] + self._call_buffer['real_img_sizes'][i][1]
            for i in range(len(self._call_buffer['objects_positions']))]

        max_height = max(objects_height_ends)
        max_width = max(objects_width_ends)

        if self.persp_trans > 0:
            added_width = int(max_width * self.persp_trans)
            added_width = added_width if added_width % 2 == 0 else added_width + 1
        else:
            added_width = 0

        self._call_buffer['added_width'] = added_width

        self._call_buffer['scene_size'] = [max_height, max_width + added_width]

    @staticmethod
    def _transform_background(back):
        return flip(back, 0.5)

    def _get_scene(self):
        if self.background == 'none':
            self._call_buffer['scene'] = np.zeros((*self._call_buffer['scene_size'], 3))
        elif self.background == 'img':
            background_image = random.choice(self.background_image_list).copy()

            background_image = self._transform_background(background_image)

            if ((background_image.shape[0] > self._call_buffer['scene_size'][0])
                    and (background_image.shape[1] > self._call_buffer['scene_size'][1])):
                self._call_buffer['scene'] = random_crop(background_image, self._call_buffer['scene_size'])
            else:
                self._call_buffer['scene'] = resize(background_image, self._call_buffer['scene_size'])

    @abstractmethod
    def _transform_masks(self):
        pass

    def _transform_pairs(self):
        for i in range(len(self._call_buffer['img_list'])):
            self._call_buffer['img_list'][i] = format_image(self._call_buffer['img_list'][i])
            self._call_buffer['mask_list'][i] = format_image(self._call_buffer['mask_list'][i])

            self._call_buffer['img_list'][i], self._call_buffer['mask_list'][i] = rotate_pair(
                self._call_buffer['img_list'][i], self._call_buffer['mask_list'][i], self.max_rotate_degree
            )

            self._call_buffer['img_list'][i], self._call_buffer['mask_list'][i] = flip_pair(
                self._call_buffer['img_list'][i], self._call_buffer['mask_list'][i], self.flip_prob
            )

            if self.adjust_mask:
                [(x_min, y_max), (_, y_min), (x_max, _), (x_max, _), (_, _)] = mask2bbox(self._call_buffer['mask_list'][i])
                self._call_buffer['img_list'][i] = self._call_buffer['img_list'][i][y_min:y_max, x_min:x_max, :]
                self._call_buffer['mask_list'][i] = self._call_buffer['mask_list'][i][y_min:y_max, x_min:x_max, :]

                if self.pad_mask > 0:
                    self._call_buffer['img_list'][i] = pad(self._call_buffer['img_list'][i], self.pad_mask)
                    self._call_buffer['mask_list'][i] = pad(self._call_buffer['mask_list'][i], self.pad_mask)

    def _add_main_masks(self):
        self._call_buffer['main_masks'] = {}

        for mask in self.output_type_list:
            self._call_buffer['main_masks'][mask] = np.zeros_like(self._call_buffer['scene'])

    def _embed_pairs(self):
        for i in range(len(self._call_buffer['img_list'])):
            self._call_buffer['scene'], self._call_buffer['main_masks'] = embed_pair(
                self._call_buffer['img_list'][i],
                self._call_buffer['mask_list'][i],
                self._call_buffer['scene'],
                self._call_buffer['small_masks'][i],
                self._call_buffer['main_masks'],
                self._call_buffer['added_width'],
                self._call_buffer['objects_positions'][i]
            )

    def _embed_bbox(self, bbox, start):
        (x_min, y_max), (_, y_min), (x_max, _), (_, _), (_, _) = bbox
        h = start[0]
        w = start[1]

        x_min += w
        x_max += w
        y_min += h
        y_max += h

        return (x_min, y_max), (x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)

    def _get_bboxes(self):
        if self.bboxes:

            if 'multi-object' in self.output_type_list:
                mo_bboxes = []
                for obj in self._call_buffer['small_masks']:
                    mo_bboxes.append(mask2bbox(obj['multi-object']))

                self._call_buffer['bboxes']['multi-object'] = [self._embed_bbox(
                    mo_bboxes[i], self._call_buffer['objects_positions'][i])
                    for i in range(len(mo_bboxes))]

                if self.persp_trans > 0:
                    self._call_buffer['bboxes']['multi-object'] = [bbox_perspective_transform(
                        np.array([bbox], np.float32),
                        self._call_buffer['added_width'],
                        [self._call_buffer['scene'].shape[0],
                         self._call_buffer['scene'].shape[1] + self._call_buffer['added_width']])
                        for bbox in self._call_buffer['bboxes']['multi-object']]

    def _transform_pipeline(self, i_list, m_list, class_list=None):
        img_list = [i.copy() for i in i_list]
        mask_list = [m.copy() for m in m_list]

        self._call_buffer = {
            'bboxes': {}
        }

        self._call_buffer['img_list'], self._call_buffer['mask_list'], self._call_buffer[
            'class_list'] = self._check_input(img_list, mask_list, class_list)

        self._transform_pairs()
        self._transform_masks()

        self._call_buffer['real_img_sizes'] = [[img.shape[0], img.shape[1]] for img in self._call_buffer['img_list']]
        self._call_buffer['shrink_img_sizes'] = [
            [int(img[0] * (1 - self.overlap_ratio)), int(img[1] * (1 - self.overlap_ratio))]
            for img in self._call_buffer['real_img_sizes']]

        self._call_buffer['objects_positions'] = get_pack_coords(self._call_buffer['shrink_img_sizes'])

        self._get_scene_size()
        self._get_scene()

        self._add_main_masks()

        self._embed_pairs()

        if self.persp_trans > 0:
            self._call_buffer['scene'] = perspective_transform(
                self._call_buffer['scene'], self._call_buffer['added_width'])

            for mask in self._call_buffer['main_masks']:
                self._call_buffer['main_masks'][mask] = perspective_transform(
                    self._call_buffer['main_masks'][mask], self._call_buffer['added_width'])

        self._call_buffer['scene'] = add_salt(self._call_buffer['scene'], self.salt)
        self._call_buffer['scene'] = add_pepper(self._call_buffer['scene'], self.pepper)
        self._call_buffer['scene'] = gauss_noise(self._call_buffer['scene'], self.gauss_var)
        self._call_buffer['scene'] = smooth(self._call_buffer['scene'], self.smooth_kernel_size)

        self._get_bboxes()

        return self._call_buffer

    def transform(self, img_list, mask_list, class_list=None):
        call_buffer = self._transform_pipeline(img_list, mask_list, class_list)

        result = {'scene': call_buffer['scene'], 'masks': {}}

        if self.bboxes:
            result['bboxes'] = self._call_buffer['bboxes']

        for mask in self.output_type_list:
            result['masks'][mask] = call_buffer['main_masks'][mask]

        return result
