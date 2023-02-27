import argparse
import random
import shutil
import sys
import os

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Cityscapes
from torchvision.transforms.functional import InterpolationMode

import compressai

from compressai.datasets import ImageFolder
from compressai.zoo import image_models
from netext import ScaleHyperpriorExt, MeanScaleHyperpriorWithY, ConMeanScaleHyperpriorWithY
from datasetext import CityscapesDataset

os.environ["CUDA_VISIBLE_DEVICES"]="4"

def test_epoch_con(epoch, loader, model, img_dir):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for xi, xx_ in enumerate(loader):
            x_, xl_, xname = xx_
            x_, xl_ = x_.to(device), xl_.to(device)
            xl_ = xl_.to(torch.int64)
            xl_ = xl_[:,:,::4,::4]
            bs, _ , h, w = xl_.size()
            ys_ = torch.zeros(bs, 34, h, w, dtype=xl_.dtype, device=device).scatter_(1, xl_, 1)
            ys_ = ys_.to(device, dtype=torch.float32)
            out_net = model(x_, ys_)
            xname = xname[0]
            xfold = img_dir + "/" + xname.split('/')[0]
            if not os.path.exists(xfold):
                os.makedirs(xfold)
            with open(img_dir + "/" + xname, 'wb') as f:
                torchvision.utils.save_image(out_net["x_hat"][0], f)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=16,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):

    args = parse_args(argv)
    N, M = 128, 192
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_dataset = CityscapesDataset(args.dataset, mode="train", flip=False)
    test_dataset = CityscapesDataset(args.dataset, mode="val", flip=False)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
        persistent_workers=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
        persistent_workers=True
    )

    G1 = ConMeanScaleHyperpriorWithY(N,M)
    G1 = G1.to(device)

    epoch = 0
    print("Loading", args.checkpoint)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    epoch = checkpoint["epoch"] + 1
    G1.load_state_dict(checkpoint["G1_state_dict"])

    img_dir_train = args.dataset + "/rec_{}_con".format(args.lmbda) + "/train"
    img_dir_test = args.dataset + "/rec_{}_con".format(args.lmbda) + "/val"

    if not os.path.exists(img_dir_train):
        os.makedirs(img_dir_train)
    if not os.path.exists(img_dir_test):
        os.makedirs(img_dir_test)

    test_epoch_con(epoch, train_dataloader, G1, img_dir_train)
    test_epoch_con(epoch, test_dataloader, G1, img_dir_test)

if __name__ == "__main__":
    main(sys.argv[1:])