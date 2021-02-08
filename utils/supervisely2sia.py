import os
import sys
import json
from pathlib import Path
import shutil
from utils import *


def supervisely2sia(init_data_folder, sia_data_folder):
    '''
    Prepares data collected in https://app.supervise.ly to work with SIA.
    Returns images and masks lists.
    '''

    #     with open(os.path.join(init_data_folder, 'obj_class_to_machine_color.json'), 'r') as file:
    #         data = file.read()

    #     class_colors_config = json.loads(data)
    #     objects = list(class_colors_config.keys())

    #     instances = [i for i in objects if 'Instance' in i]
    classes = [i for i in os.listdir(init_data_folder) if os.path.isdir(os.path.join(init_data_folder, i))]

    for c in classes:

        Path(os.path.join(sia_data_folder, c)).mkdir(parents=True, exist_ok=True)

        for i in get_images_list(os.path.join(init_data_folder, c, 'img')):
            orig_file = i.split('/')[-1]
            img_num, file_type = orig_file.split('.')
            shutil.copy(i, os.path.join(sia_data_folder, c, img_num + '_rgb.' + file_type))
            shutil.copy(i.replace('img', 'masks_machine').replace('jpg', 'png'),
                        os.path.join(sia_data_folder, c, img_num + '_label.png'))
