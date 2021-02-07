import os
from pathlib import Path
from collections import Counter

from utils import *
from datagen.DataGen import DataGen


class SavingDataGen(DataGen):
    def __init__(self, input_path, input_type, output_path, split_masks=False,
                 img_prefix='rgb', mask_prefix='label',
                 augmentor_params=None, class_mapping=None,
                 balance=True, separable_class=False,
                 img_format='jpg', mask_format='png'):
        super().__init__(input_path, input_type,
                         img_prefix=img_prefix, mask_prefix=mask_prefix,
                         augmentor_params=augmentor_params, class_mapping=class_mapping,
                         balance=balance, separable_class=separable_class)

        self.output_path = output_path
        self.split_masks = split_masks
        self.img_format = img_format
        self.mask_format = mask_format

    def prepare_folders(self):
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        if self.separable_class:
            for c in self.classes:
                class_dir = os.path.join(self.output_path, c)
                Path(class_dir).mkdir(parents=True, exist_ok=True)

                if self.split_masks:
                    for out_type in self.output_type_list:
                        Path(os.path.join(class_dir, out_type)).mkdir(parents=True, exist_ok=True)

                    Path(os.path.join(class_dir, 'images')).mkdir(parents=True, exist_ok=True)

        else:
            if self.split_masks:
                for out_type in self.output_type_list:
                    Path(os.path.join(self.output_path, out_type)).mkdir(parents=True, exist_ok=True)

                Path(os.path.join(self.output_path, 'images')).mkdir(parents=True, exist_ok=True)

    def create_dataset(self, num_samples=1, scene_samples=1):
        self.prepare_folders()

        data_file = os.path.join(self.output_path, 'description.csv')

        try:
            previous_data = read_csv(data_file)
            sample_num = previous_data['sample_num'].max() + 1
        except:
            sample_num = 1

        total_time = 0
        load_time = 0
        transform_time = 0
        streaming_time = 0
        avg_h = []
        avg_w = []

        for sample in range(num_samples):
            transformed_scene, classes = self.get_scene(scene_samples)
            avg_h.append(transformed_scene['scene'].shape[0])
            avg_w.append(transformed_scene['scene'].shape[1])

            samples_per_class_num = dict(Counter(classes))
            for input_class in range(self.num_classes):
                if input_class not in samples_per_class_num.keys():
                    samples_per_class_num[input_class] = 0

            sample_data = {
                'sample_num': sample_num,
                'height': transformed_scene['scene'].shape[0],
                'width': transformed_scene['scene'].shape[1]
            }

            for input_class in range(self.num_classes):
                sample_data[str(input_class) + '_class'] = samples_per_class_num[input_class]

            c = self.num2class[classes[0]] if self.separable_class else ''

            image_path = os.path.join(
                self.output_path, c,
                'images' if self.split_masks else '',
                'generated_{:0>5}_{}.{}'.format(sample_num, self.img_prefix, self.img_format)
            )

            write(image_path, transformed_scene['scene'])
            sample_data['image_path'] = image_path

            for mask_type in self.output_type_list:
                m = mask_type if self.split_masks else ''
                mask_path = os.path.join(
                    self.output_path, c, m,
                    'generated_{:0>5}_{}_{}.{}'.format(sample_num, mask_type, self.mask_prefix, self.mask_format)
                )

                write(mask_path, transformed_scene['masks'][mask_type])
                sample_data[mask_type + '_path'] = mask_path

            if 'bboxes' in transformed_scene:
                subdir = 'bboxes' if self.split_masks else ''
                bbox_dir = os.path.join(self.output_path, subdir)
                Path(bbox_dir).mkdir(parents=True, exist_ok=True)
                bbox_path = os.path.join(bbox_dir, 'generated_{:0>5}_bbox.csv'.format(sample_num))

                x_min_list = []
                x_max_list = []
                y_min_list = []
                y_max_list = []

                for bbox in transformed_scene['bboxes']['multi-object']:
                    [(x_min, y_max), (_, y_min), (x_max, _), (_, _), (_, _)] = bbox
                    x_min_list.append(x_min)
                    x_max_list.append(x_max)
                    y_min_list.append(y_min)
                    y_max_list.append(y_max)

                bboxes_df = pd.DataFrame({
                    'x_min': x_min_list,
                    'x_max': x_max_list,
                    'y_min': y_min_list,
                    'y_max': y_max_list,
                    'class': classes
                })

                write_csv(bboxes_df, bbox_path)
                sample_data['bbox_path'] = bbox_path

            write_csv(pd.DataFrame(sample_data, index=[0]), data_file)
            sample_num += 1

        return (total_time, load_time, transform_time, streaming_time), (sum(avg_h) / len(avg_h), sum(avg_w) / len(avg_w))
