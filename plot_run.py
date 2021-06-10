#%%
import torch 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import VOCDataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import datatrans
import utils

torch.manual_seed(2016)

IMG_DIR = "data/images"
LABEL_DIR = "data/labels"
BATCH_SIZE = 16 # 64 in original paper but I don't have that much vram, grad accum?
NUM_WORKERS = 2
PIN_MEMORY = True

class RandomHueAdjust():
    def __init__(self, max_hue_factor=0.5):
        super().__init__()
        self.max_hue_factor = max_hue_factor

    def __call__(self, img): # max_hue_factor is 0~0.5
        hue_factor = self.max_hue_factor*(torch.rand(1)-0.5)*2
        img = transforms.functional.adjust_hue(img, hue_factor)
        return img

class Compose(object): # have to self-define a compose to deal with both bboxes and img
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img, bboxes)

        return img, bboxes


transform = Compose([
    datatrans.Resize((448,448)),
    datatrans.RandomAffine(
        translate=(0.2,0.2),
        scale=(0.8,1.2)
    ),
    datatrans.RandomHorizontalFlip(p=0.5),
    datatrans.ColorJitter(
        brightness=0.5,
        saturation=0.5
    ),
    datatrans.ToTensor()
])
transform_plain = Compose([
    datatrans.Resize((448,448)),
    datatrans.ToTensor()
])


#%%

def main():
    train_dataset = VOCDataset(
        "data/100examples.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        #num_workers=NUM_WORKERS,
        #pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    (pred_bboxes,true_bboxes,x) = utils.get_batch_bboxes(
        loader=train_loader,
        model=None,
        iou_threshold=0.5,
        threshold=0.4,
        device="cpu",
        pred=False
    )
    #true_bboxes specified as
    #   batch -> idx_box -> 
    #   predicted_class, best_confidence, converted_boxes 
    #   0,              1,              2:

    utils.plot_image(
        x,
        boxes_pred=None,
        boxes_true=true_bboxes,
        nimgs=3
    )
    

if __name__ == "__main__":
    main()

# %%
