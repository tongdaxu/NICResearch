# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os

import argparse
import random
import shutil
import sys

import math

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import compressai

from compressai.datasets import ImageFolder
from compressai.zoo import image_models
from netext import ScaleHyperpriorWithY, ScaleHyperpriorYDecoder, Discriminator, MeanScaleHyperpriorWithY

SJOB = os.getenv('SLURM_JOB_ID')

class CropToMod(object):
    def __init__(self, mod):
        self.mod = mod

    def __call__(self, image):
        w, h = image.size
        if w % self.mod != 0: w = self.mod * (w // self.mod)
        if h % self.mod != 0: h = self.mod * (h // self.mod)
        return image.crop((0,0,w,h))

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]

        return out

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def test_epoch(epoch, test_dataloader, G1, G2, criterion, img_dir):
    G1.eval()
    G2.eval()
    device = next(G1.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    mse2_loss = AverageMeter()

    with torch.no_grad():
        for xi, x_ in enumerate(test_dataloader):
            x_ = x_.to(device)
            out_net = G1(x_)
            out_criterion = criterion(out_net, x_)
            G_result = G2(out_net["y_hat_i"])
            g2_mse = torch.mean((G_result.data - x_)**2)

            aux_loss.update(G1.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
            mse2_loss.update(g2_mse)
            if xi<=10 and img_dir != "":
                with open(img_dir + "/{}_o.jpg".format(xi), 'wb') as f:
                    torchvision.utils.save_image(x_[0], f)
                with open(img_dir + "/{}_g1.jpg".format(xi), 'wb') as f:
                    torchvision.utils.save_image(out_net["x_hat"][0], f)
                with open(img_dir + "/{}_g2.jpg".format(xi), 'wb') as f:
                    torchvision.utils.save_image(G_result[0], f)

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSEG1 loss: {mse_loss.avg:.4f} |"
        f"\tMSEG2 loss: {mse2_loss.avg:.4f} |"
        f"\tBpp loss: {bpp_loss.avg:.4f} |"
        f"\tPSNR val: {10.0*torch.log10(1/mse_loss.avg):.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-hyperprior",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
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
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
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
        "--save", action="store_true", default=True, help="Save model to disk"
    )
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
    N, M = 128, 192
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    test_transforms = transforms.Compose(
        [CropToMod(64), transforms.ToTensor()]
    )

    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    G1 = MeanScaleHyperpriorWithY(N,M)
    G2 = ScaleHyperpriorYDecoder(N,M)
    D = Discriminator((3,256,256),(M,16,16),M)
    G1 = G1.to(device)
    G2 = G2.to(device)
    D = D.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        G1 = CustomDataParallel(G1)
        G2 = CustomDataParallel(G2)
        D = CustomDataParallel(D)

    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    print("Loading", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    last_epoch = checkpoint["epoch"] + 1
    G1.load_state_dict(checkpoint["G1_state_dict"])
    G2.load_state_dict(checkpoint["G2_state_dict"])
    D.load_state_dict(checkpoint["D_state_dict"])
    img_dir = args.checkpoint[:-4]+"_imgs"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    test_epoch(last_epoch, test_dataloader, G1, G2, criterion, img_dir)

if __name__ == "__main__":
    main(sys.argv[1:])