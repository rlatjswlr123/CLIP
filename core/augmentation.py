import albumentations as A
import albumentations.pytorch
import numpy as np
import cv2


class train_aug:
    def __init__(self):
        self.transforms = A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
        A.RandomScale(),
        # A.ColorJitter(),
        A.Resize(224, 224, interpolation=cv2.INTER_CUBIC, always_apply=True),
        A.CenterCrop(224, 224, always_apply=True),
        A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        albumentations.pytorch.transforms.ToTensorV2(),
        ])
    
    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))