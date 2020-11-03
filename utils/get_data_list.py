import os
from utils.list_files import list_files


def get_img_mask_list(img_path, mask_path=None, img_prefix='rgb', mask_prefix='label'):
    if (mask_path is None) or img_path == mask_path:
        files_path = list_files(img_path)
        files_path = [os.path.join(img_path, f) for f in files_path if
                      f.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']]

        images = [f for f in files_path if img_prefix in f]
        masks = [f for f in [f.replace(img_prefix, mask_prefix) for f in images] if f in files_path]

    else:
        files_path = list_files(img_path)
        files_path = [os.path.join(img_path, f) for f in files_path if
                      f.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']]

        mask_files_path = list_files(mask_path)
        mask_files_path = [os.path.join(mask_path, f) for f in mask_files_path if
                           f.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']]

        images = [f for f in files_path if img_prefix in f]
        masks = [f for f in [f.replace(img_prefix, mask_prefix) for f in images] if f in mask_files_path]

    return images, masks


def get_images_list(path):
    files_path = list_files(path)
    files_path = [os.path.join(path, f) for f in files_path if f.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']]
    return files_path
