#%%
import torch 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import VOCDataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import datatrans

torch.manual_seed(201216)

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

# transform = Compose([
#     transforms.Resize((448, 448)), 
#     #RandomHueAdjust(0.25),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.ColorJitter(
#         brightness=0.5,
#         saturation=0.5),
#     transforms.RandomAffine(
#         degrees=0,
#         #translate=(0.2,0.2),
#         scale=(0.8,1.2)),
#     transforms.ToTensor(),
# ])

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
# transform = Compose([datatrans.ToTensor()])


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

    (x,y) = next(iter(train_loader))
    print(f"x = {x.shape}, y = {y.shape}")

    n = 4  # how many images we will display
    if n > 1:
        plt.figure(figsize=(16, 3))
        for i in range(n):
            target_idx = i+5 # changeable, depending on which images are to be shown

            ax = plt.subplot(1, n, i + 1)
            ax.imshow(x[target_idx].permute(1,2,0))
            for row in range(7):
                for col in range(7):
                    cell_label = y[target_idx, row, col, 20:]
                    if cell_label[0] >= 0.5:
                        center = (
                            x[target_idx].shape[1]/7*(col+cell_label[1]).item(),
                            x[target_idx].shape[2]/7*(row+cell_label[2]).item()
                        )
                        w = x[target_idx].shape[1]/7*cell_label[3].item()
                        h = x[target_idx].shape[2]/7*cell_label[4].item()
                        bottom_left = (
                            round(center[0]-w/2),
                            round(center[1]-h/2)
                        )
                        rect = patches.Rectangle(bottom_left, w, h, linewidth=1, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                    
                    if cell_label[5] >= 0.5:
                        center = (
                            x[target_idx].shape[1]/7*(row+cell_label[6]).item(),
                            x[target_idx].shape[2]/7*(7-col-cell_label[7]).item()
                        )
                        w = x[target_idx].shape[1]/7*cell_label[8].item()
                        h = x[target_idx].shape[2]/7*cell_label[9].item()
                        bottom_left = (
                            round(center[0]-w/2),
                            round(center[1]-h/2)
                        )
                        rect = patches.Rectangle(bottom_left, w, h, linewidth=1, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)

            plt.gray()
            ax.set_xticks([])
            ax.set_yticks([])

        plt.show()
    elif n > 0:
        plt.figure(figsize=(4, 4))
        plt.imshow(x.permute(1,2,0))
        plt.show()


if __name__ == "__main__":
    main()

# %%
