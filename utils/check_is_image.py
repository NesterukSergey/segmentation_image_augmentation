def check_is_image(img):
    try:
        if not ((img.shape[2] == 3) and (img.shape[0] > 10) and (img.shape[1] > 10)):
            raise UserWarning('Wrong image size')
    except:
        raise UserWarning('Not an numpy image')
