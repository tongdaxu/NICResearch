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
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision import transforms

import compressai

from compressai.datasets import ImageFolder
from compressai.zoo import image_models
from netext import ScaleHyperpriorWithY, ScaleHyperpriorYDecoder, Discriminator, MeanScaleHyperpriorWithY
from srresnet import _NetG, _NetD
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
    G1, G2, D, criterion, train_dataloader, G1_optimizer, G1_aux_optimizer, 
    G2_optimizer, D_optimizer, epoch, clip_max_norm
):
    G1.train()
    G2.train()
    D.train()
    device = next(G1.parameters()).device
    lambda_gp = 10
    lambda_l2 = 0.005
    for i, xx_ in enumerate(train_dataloader):
        x_, xl_ = xx_
        x_ = x_.to(device)
        mini_batch = x_.size()[0]
        # update rate distortion auto encoder
        G1_optimizer.zero_grad()
        G1_aux_optimizer.zero_grad()
        G1_out = G1(x_)
        G1_out_criterion = criterion(G1_out, x_)
        G1_out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(G1.parameters(), clip_max_norm)
        G1_optimizer.step()
        G1_aux_loss = G1.aux_loss()
        G1_aux_loss.backward()
        G1_aux_optimizer.step()

        D.zero_grad()
        # real image
        D_result = D(x_, G1_out["x_hat"].detach()).squeeze()
        D_real_loss = -D_result.mean()

        G_result = G2(G1_out["y_hat_i"])
        D_result = D(G_result.data, G1_out["x_hat"].detach()).squeeze()
        D_fake_loss = D_result.mean()
        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()

        D.zero_grad()
        alpha = torch.rand(x_.size(0), 1, 1, 1)
        alpha1 = alpha.cuda().expand_as(x_)
        interpolated1 = Variable(alpha1 * x_.data + (1 - alpha1) * G_result.data, requires_grad=True)
        interpolated2 = Variable(G1_out["x_hat"].detach(), requires_grad=True)

        out = D(interpolated1, interpolated2).squeeze()

        grad = torch.autograd.grad(outputs=out,
                                    # inputs=[interpolated1, interpolated2],
                                    inputs=interpolated1,
                                    grad_outputs=torch.ones(out.size()).cuda(),
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

        # Backward + Optimize
        gp_loss = lambda_gp * d_loss_gp
        gp_loss.backward()
        D_optimizer.step()

        # train G2
        D.zero_grad()
        G2.zero_grad()

        G_result = G2(G1_out["y_hat_i"])
        D_result = D(G_result, G1_out["x_hat"].detach()).squeeze()

        G2_l2norm = lambda_l2 * F.mse_loss(G1_out["x_hat"].detach(), G_result) 
        G_train_loss = - D_result.mean() + G2_l2norm

        G_train_loss.backward()
        G2_optimizer.step()

        g2_mse = torch.mean((G_result.data - x_)**2)

        if i % 500 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(x_)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tG1 Loss: {G1_out_criterion["loss"].item():.4f} |'
                f'\tG1 MSE loss: {G1_out_criterion["mse_loss"].item():.4f} |'
                f'\tG1 Bpp loss: {G1_out_criterion["bpp_loss"].item():.4f} |'
                f"\tG1 Aux loss: {G1_aux_loss.item():.2f} |"
                f"\tD loss: {D_train_loss.item():.2f} |"
                f"\tG2 MSE loss: {g2_mse.item():.4f} |"
            )

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
        for xi, xx_ in enumerate(test_dataloader):
            x_, xl_ = xx_
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
            if img_dir != "":
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
        default=(128, 128),
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

    train_transforms = transforms.Compose(
        [transforms.Resize(256), transforms.RandomCrop(args.patch_size,pad_if_needed=True), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.Resize(256), CropToMod(64), transforms.ToTensor()]
    )

    train_dataset = Cityscapes(args.dataset, split="train", target_type="semantic", transform=train_transforms, target_transform=train_transforms)
    test_dataset = Cityscapes(args.dataset, split="val", target_type="semantic", transform=test_transforms, target_transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    G1 = MeanScaleHyperpriorWithY(N,M)
    # G2 = ScaleHyperpriorYDecoder(N,M)
    G2 = _NetG() 
    # D = Discriminator((3,256,256),(M,16,16),M)
    D = _NetD()
    G1 = G1.to(device)
    G2 = G2.to(device)
    D = D.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        G1 = CustomDataParallel(G1)
        G2 = CustomDataParallel(G2)
        D = CustomDataParallel(D)

    G1_optimizer, G1_aux_optimizer = configure_optimizers(G1, args)
    G2_optimizer = optim.RMSprop(G2.parameters(), lr=args.learning_rate)
    D_optimizer = optim.RMSprop(D.parameters(), lr=args.learning_rate*2)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(G1_optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        G1.load_state_dict(checkpoint["G1_state_dict"])
        G2.load_state_dict(checkpoint["G2_state_dict"])
        D.load_state_dict(checkpoint["D_state_dict"])
        G1_optimizer.load_state_dict(checkpoint["G1_optimizer"])
        G1_aux_optimizer.load_state_dict(checkpoint["G1_aux_optimizer"])
        G2_optimizer.load_state_dict(checkpoint["G2_optimizer"])
        D_optimizer.load_state_dict(checkpoint["D_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    if not os.path.exists(SJOB):
        os.makedirs(SJOB)

    for epoch in range(last_epoch, args.epochs):
        train_one_epoch(G1, G2, D, criterion, train_dataloader,
            G1_optimizer, G1_aux_optimizer, G2_optimizer, D_optimizer,
            epoch, args.clip_max_norm)
        if epoch%10==0:
            CKPT = SJOB + "/ckp_"+str(epoch)+".pth"
            torch.save({
                'epoch': epoch,
                'G1_state_dict': G1.state_dict(),
                'G2_state_dict': G2.state_dict(),
                'D_state_dict': D.state_dict(),
                'G1_optimizer': G1_optimizer.state_dict(),
                'G1_aux_optimizer': G1_aux_optimizer.state_dict(),
                'G2_optimizer': G2_optimizer.state_dict(),
                'D_optimizer': D_optimizer.state_dict(),
                'lr_scheduler':lr_scheduler.state_dict()
            }, CKPT)
            img_dir = CKPT + "_imgs"
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            test_epoch(epoch, test_dataloader, G1, G2, criterion, img_dir)
        # test for last epoch
        CKPT = SJOB + "/ckp_"+str(epoch)+".pth"
        torch.save({
            'epoch': epoch,
            'G1_state_dict': G1.state_dict(),
            'G2_state_dict': G2.state_dict(),
            'D_state_dict': D.state_dict(),
            'G1_optimizer': G1_optimizer.state_dict(),
            'G1_aux_optimizer': G1_aux_optimizer.state_dict(),
            'G2_optimizer': G2_optimizer.state_dict(),
            'D_optimizer': D_optimizer.state_dict(),
            'lr_scheduler':lr_scheduler.state_dict()
        }, CKPT)
        img_dir = CKPT + "_imgs"
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        test_epoch(epoch, test_dataloader, G1, G2, criterion, img_dir)

if __name__ == "__main__":
    main(sys.argv[1:])