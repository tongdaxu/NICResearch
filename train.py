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
from netext import ScaleHyperpriorExt, MeanScaleHyperpriorWithY
from datasetext import CityscapesDataset

SJOB = os.getenv('SLURM_JOB_ID')
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if SJOB is None:
    import time
    SJOB=time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

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


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    G1, criterion, train_dataloader, G1_optimizer, G1_aux_optimizer, epoch, clip_max_norm, patch_size
):
    G1.train()
    device = next(G1.parameters()).device

    for i, dd in enumerate(train_dataloader):
        d, _, _ = dd
        d = d.to(device)
        bs, _, h, w = d.size()
        cx, cy = random.randint(0, h-patch_size[0]), random.randint(0, w-patch_size[1])
        d = d[:,:,cx:cx+patch_size[0],cy:cy+patch_size[1]]

        G1_optimizer.zero_grad()
        G1_aux_optimizer.zero_grad()

        out_net = G1(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(G1.parameters(), clip_max_norm)
        G1_optimizer.step()

        aux_loss = G1.aux_loss()
        aux_loss.backward()
        G1_aux_optimizer.step()

        if i % 500 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.6f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.6f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.6f} |'
                f"\tAux loss: {aux_loss.item():.6f}"
            )


def test_epoch(epoch, test_dataloader, model, criterion, img_dir):
    model.eval()
    device = next(model.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    with torch.no_grad():
        for xi, dd in enumerate(test_dataloader):
            d, _, xname = dd
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)
            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
            if img_dir != "":
                xname = xname[0]
                xname = xname.split('/')[1]
                with open(img_dir + "/{}_ori.jpg".format(xname), 'wb') as f:
                    torchvision.utils.save_image(d[0], f)
                with open(img_dir + "/{}_rec.jpg".format(xname), 'wb') as f:
                    torchvision.utils.save_image(out_net["x_hat"][0], f)

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.6f} |"
        f"\tMSE loss: {mse_loss.avg:.6f} |"
        f"\tBpp loss: {bpp_loss.avg:.6f} |"
        f"\tPSNR val: {10.0*torch.log10(1/mse_loss.avg):.6f} |"
        f"\tAux loss: {aux_loss.avg:.6f}\n"
    )

    return loss.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "checkpoint_best_loss.pth.tar")


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

    if not os.path.exists(SJOB):
        os.makedirs(SJOB)

    args = parse_args(argv)
    N, M = 128, 192
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_dataset = CityscapesDataset(args.dataset, mode="train")
    test_dataset = CityscapesDataset(args.dataset, mode="val")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
        persistent_workers=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
        persistent_workers=True
    )

    G1 = MeanScaleHyperpriorWithY(N,M)
    G1 = G1.to(device)

    G1_optimizer, G1_aux_optimizer = configure_optimizers(G1, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(G1_optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        G1.load_state_dict(checkpoint["G1_state_dict"])
        G1_optimizer.load_state_dict(checkpoint["G1_optimizer"])
        G1_aux_optimizer.load_state_dict(checkpoint["G1_aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        train_one_epoch(
            G1,
            criterion,
            train_dataloader,
            G1_optimizer,
            G1_aux_optimizer,
            epoch,
            args.clip_max_norm,
            args.patch_size
        )
        if epoch%10==0:
            if epoch%100==0:
                CKPT = SJOB + "/ckp_"+str(epoch)+".pth"
                torch.save({
                    'epoch': epoch,
                    'G1_state_dict': G1.state_dict(),
                    'G1_optimizer': G1_optimizer.state_dict(),
                    'lr_scheduler':lr_scheduler.state_dict()
                }, CKPT)
                img_dir = CKPT + "_imgs"
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)
                test_epoch(epoch, test_dataloader, G1, criterion, img_dir)
            else:
                test_epoch(epoch, test_dataloader, G1, criterion, "")
    CKPT = SJOB + "/ckp_"+str(epoch)+".pth"
    torch.save({
        'epoch': epoch,
        'G1_state_dict': G1.state_dict(),
        'G1_optimizer': G1_optimizer.state_dict(),
        'lr_scheduler':lr_scheduler.state_dict()
    }, CKPT)
    img_dir = CKPT + "_imgs"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    test_epoch(epoch, test_dataloader, G1, criterion, img_dir)


if __name__ == "__main__":
    main(sys.argv[1:])