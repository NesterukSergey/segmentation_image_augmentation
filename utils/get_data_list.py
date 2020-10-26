import os
from utils.list_files import list_files


def get_data_list(path, img_prefix='rgb', mask_prefix='label'):

    files_path = list_files(path)
    files_path = [os.path.join(path, f) for f in files_path if f.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']]

    images = [f for f in files_path if img_prefix in f]
    masks = [f for f in [f.replace(img_prefix, mask_prefix) for f in images] if f in files_path]

    return images, masks
