import os
import torch
import json
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from torchvision import transforms as TR

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


class CityscapesDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot,mode,flip=True):
        self.dataroot = dataroot
        self.mode = mode
        self.load_size = 512
        self.aspect_ratio = 2.0
        self.flip = flip
        self.images, self.labels, self.paths = self.list_images()

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.paths[0], self.images[idx])).convert('RGB')
        label = Image.open(os.path.join(self.paths[1], self.labels[idx]))
        image, label = self.transforms(image, label)
        label = label * 255
        return image, label, self.images[idx]

    def list_images(self):
        images = []
        path_img = os.path.join(self.dataroot, "leftImg8bit", self.mode)
        for city_folder in sorted(os.listdir(path_img)):
            cur_folder = os.path.join(path_img, city_folder)
            for item in sorted(os.listdir(cur_folder)):
                images.append(os.path.join(city_folder, item))
        labels = []
        path_lab = os.path.join(self.dataroot, "gtFine", self.mode)
        for city_folder in sorted(os.listdir(path_lab)):
            cur_folder = os.path.join(path_lab, city_folder)
            for item in sorted(os.listdir(cur_folder)):
                if item.find("labelIds") != -1:
                    labels.append(os.path.join(city_folder, item))
        assert len(images)  == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
        for i in range(len(images)):
            assert images[i].replace("_leftImg8bit.png", "") == labels[i].replace("_gtFine_labelIds.png", ""),\
                '%s and %s are not matching' % (images[i], labels[i])
        return images, labels, (path_img, path_lab)

    def transforms(self, image, label):
        assert image.size == label.size
        # resize
        new_width, new_height = (int(self.load_size / self.aspect_ratio), self.load_size)
        image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        # flip
        if self.mode == "train" and self.flip:
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)
        # to tensor
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)
        return image, label

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