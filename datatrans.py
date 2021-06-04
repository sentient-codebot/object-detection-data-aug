#   Required transforms:
#       resize
#       random horizon flip
#       color jitter
#       random affine
#       to tensor

import torch 
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
#from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int
import random

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img, bboxes)

        return img, bboxes

class Resize(transforms.Resize):
    def __init__(self, size):
        super().__init__(self, size)
        self.size = size

    def forward(self, img, bbox):
        return super().forward(self, img), bbox

class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p, B):
        super().__init__()
        self.p = p
        self.B = B
    
    def forward(self, img, bboxes):
        if random.random() <= self.p:
            img = TF.hflip(img)
            rows, cols, labels = bboxes.shape
            B = self.B
            C = labels - B*5
            for row in range(rows):
                for col in range(cols):
                    for b in range(B):
                        bboxes[col,row,C+b*5+1] = 1-bboxes[col,row,C+b*5+1]
            
            return img, bboxes


class ColorJitter(transforms.ColorJitter):
    def __init__(self, brightness, saturation):
        super().__init__(
            self, 
            brightness=brightness, 
            saturation=saturation
        )
    
    def forward(img, bboxes):
        img = super().forward(img)
        return img, bboxes


class RestrictedRandomAffine(torch.nn.Module):
    def __init__(self, translate=None, scale=None):
        super().__init__()
        self.translate = translate
        self.scale = scale

    def forward(self, img, bboxes):
        img_size = TF._get_image_size(img)
        if self.translate is not None:
            max_dx = float(self.translate[0])*img_size[0]
            max_dy = float(self.translate[1])*img_size[1]
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)
        else:
            translations = (0,0)
        if self.scale is not None:
            scale = float(torch.empty(1).uniform_(self.scale[0], self.scale[1]).item())
        else:
            scale = 1.0
        
        return TF.affine(img, translate=translations, scale=scale)