import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

class COCO17Pan(Dataset):
    def __init__(self, split, transform=None):
        self.srcFolder="/home/JJ_Group/xutd/git/Conditional_Perceptual_Coding/{}2017".format(split)
        self.annFile="/home/JJ_Group/xutd/git/Conditional_Perceptual_Coding/annotations/panoptic_{}2017.json".format(split)
        self.annFolder="/home/JJ_Group/xutd/git/Conditional_Perceptual_Coding/annotations/panoptic_{}2017".format(split)
        with open(self.annFile) as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data["annotations"])

    def __getitem__(self, idx):
        img_path = self.srcFolder + "/" + self.data["annotations"][idx]["file_name"][:-4] + ".jpg"
        lbl_path = self.annFolder + "/" + self.data["annotations"][idx]["file_name"][:-4] + ".npy"
        img_frame = np.array(Image.open(img_path))
        if(len(img_frame.shape))==2:
            img_frame = np.repeat(img_frame[:,:,None], 3, axis=2)
        lbl_frame = np.load(lbl_path)[:,:,None]
        # print(img_frame.shape, lbl_frame.shape)
        return self.transform(img_frame), self.transform(lbl_frame)

if __name__ == "__main__":
    train_transforms = transforms.Compose(
        [transforms.ToPILImage(),transforms.RandomCrop(256, pad_if_needed=True), transforms.ToTensor()]
    )

    dtset = COCO17Pan("val",train_transforms)
    print(len(dtset))
    x_, xl_ = dtset[0]
    x_ = x_[None]
    xl_ = xl_[None]
    xl_ = xl_ * 255
    xl_ = xl_.to(torch.int64)
    bs, _, h, w = xl_.size()
    ys_ = torch.zeros(bs,28,h,w,dtype=xl_.dtype).scatter_(1,xl_,1)
    ys_ = ys_[:,:,::4,::4]
    print(ys_.size())