from augment.Augmentor import Augmentor


class MultiPartAugmentor(Augmentor):

    def __init__(self, params):
        super().__init__(params)

    def get_input_type(self):
        return 'multi-part'

    def transform(self, img_list, mask_list):
        call_buffer = super().transform(img_list, mask_list)
        scene, mask = call_buffer['scene'], call_buffer['mask']
        return [scene, mask]



