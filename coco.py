from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import subprocess
from tqdm import tqdm

annFile="/home/JJ_Group/xutd/git/Conditional_Perceptual_Coding/annotations/panoptic_train2017.json"
annFolder="/home/JJ_Group/xutd/git/Conditional_Perceptual_Coding/annotations/panoptic_train2017"
catFile="/home/JJ_Group/xutd/git/panopticapi/panoptic_coco_categories.json"

with open(annFile) as f:
    data = json.load(f)
with open(catFile) as f:
    cdata = json.load(f)

def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


s2i, c2si = {}, {}
sn = 0
for cat in cdata:
    if not cat["supercategory"] in s2i:
        s2i[cat["supercategory"]] = sn
        sn = sn + 1
    if not cat["id"] in c2si:
        c2si[cat["id"]] = s2i[cat["supercategory"]]
print(sn)

for cnt, anno in tqdm(enumerate(data["annotations"])):
    img_path = annFolder + "/" + anno["file_name"]    
    im_frame = Image.open(img_path)
    np_frame = rgb2id(np.array(im_frame))
    h, w = np_frame.shape[0], np_frame.shape[1]
    im_label = np.zeros([h, w], dtype=np.uint8)
    for obj in anno["segments_info"]:
        im_label = im_label + (np_frame == obj["id"]) * c2si[obj["category_id"]]
    im_label = np.array(im_label,dtype=np.uint8)
    label_path = img_path[:-4] + ".npy"
    np.save(label_path, im_label)

