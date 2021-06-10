import math
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
#from model import Yolov1
from backbone import resnet50, googlenet, vgg16
from dataset import VOCDataset
from pathlib import Path
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
import utils
from loss import YoloLoss
import datatrans

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc.

DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 32 


NUM_WORKERS = 2
PIN_MEMORY = True
IN_DRIVE = True
SAVE_PATH = '/content/drive/MyDrive/yolo/ckpts/'
LOAD_MODEL_FILE = "yolov1.pth"
IMG_DIR = "/content/data/images"
LABEL_DIR = "/content/data/labels"

if IN_DRIVE:
    IMG_DIR = "/content/drive/MyDrive/yolo/data/images"
    LABEL_DIR = "/content/drive/MyDrive/yolo/data/labels"

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes # !

        return img, bboxes

transform = datatrans.Compose([
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

transform_test = datatrans.Compose([
    datatrans.Resize((448,448)),
    datatrans.ToTensor()
])




#model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
model = resnet50(split_size=7, num_boxes=2, num_classes=20, pretrained = False).to(DEVICE)

#optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
loss_fn = YoloLoss()


model.load_state_dict(torch.load(LOAD_MODEL_FILE))

train_dataset = VOCDataset(
    "/content/drive/MyDrive/yolo/data/train.csv",
    transform=transform,
    img_dir=IMG_DIR,
    label_dir=LABEL_DIR)

val_dataset = VOCDataset(
    "/content/drive/MyDrive/yolo/data/test.csv", 
    transform=transform_test, 
    img_dir=IMG_DIR, 
    label_dir=LABEL_DIR)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    shuffle=True,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    shuffle=True,
    drop_last=True,
)


import time
start = time.time()
(pred_bboxes,true_bboxes,x) = utils.get_batch_bboxes(
    loader=val_loader,
    model=model,
    iou_threshold=0.5,
    threshold=0.4,
    device="cuda",
    pred=True
)
end = time.time()
print(f"FPS: {BATCH_SIZE/(end-start)}")
#true_bboxes specified as
#   batch -> idx_box -> 
#   predicted_class, best_confidence, converted_boxes 
#   0,              1,              2:

utils.plot_image(
    x,
    boxes_pred=pred_bboxes,
    boxes_true=true_bboxes,
    nimgs=3
) 