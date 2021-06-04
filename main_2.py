#%%
import torch 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import VOCDataset
import matplotlib.pyplot as plt

import datatrans

torch.manual_seed(47)

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
            img, bboxes = t(img), bboxes

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

transform = datatrans.Resize((448,448))


#%%
def main():
    train_dataset = VOCDataset(
        "data/100examples.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )
    # train_loader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=BATCH_SIZE,
    #     #num_workers=NUM_WORKERS,
    #     #pin_memory=PIN_MEMORY,
    #     shuffle=True,
    #     drop_last=True,
    # )

    (x,y) = next(iter(train_dataset))
    print(f"x = {x.shape}, y = {y.shape}")

    n = 1  # how many images we will display
    plt.figure(figsize=(16, 3))
    for i in range(n):
        # display original
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(x[i].permute(1,2,0))
        plt.gray()
        ax.set_xticks([])
        ax.set_yticks([])

        # display bounding boxes
        #ax = plt.subplot(2, n, i + 1 + n)
        
        #plt.gray()
        #ax.set_xticks([])
        #ax.set_yticks([])
        #ax.set_xlabel(labels_name[i])
    plt.show()


if __name__ == "__main__":
    main()

# %%
