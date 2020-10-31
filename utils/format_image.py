def format_image(img):
    if img.max() <= 1:
        return (img * 255).astype('uint8')
    else:
        return img.astype('uint8')
