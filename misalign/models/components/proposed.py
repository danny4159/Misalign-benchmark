from misalign.models.components.contextual_loss import Contextual_Loss
from misalign.models.components.autoencoder_kl import Encoder, AutoencoderKL
import monai
from collections import OrderedDict
from torchvision.models import vgg19, vgg16
import torch.nn.functional as F
import torch.nn as nn
import torch
import sys

def image2patch(im, patch_size=5, stride=2):
    if im.dim() == 3:
        N = 1
        C, H, W = im.shape
    elif im.dim() == 4:
        N, C, H, W = im.shape
    else:
        raise ValueError("im must be 3 or 4 dim")

    _patch = F.unfold(im, kernel_size=patch_size, stride=stride, padding=patch_size//2)
    patch = _patch.view(N,C*_patch.shape[1],H//stride, W//stride)
    return patch

class LocalL1Loss(nn.Module):
    def __init__(self, kernel_size=7):
        super(LocalL1Loss, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, inputs, targets):
        assert inputs.size() == targets.size(), "input and target shapes do not match"

        N,C,H,W = inputs.shape
        targets_patch = F.unfold(targets, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        targets_patch = targets_patch.view(N, C, self.kernel_size**2, H, W)
        
        dist = torch.mean(torch.abs(inputs[:,:,None] - targets_patch), dim=1)
        loss = torch.min(dist, dim=1)[0]
        loss = torch.mean(loss)
        return loss

class Distance_Type:
    L2_Distance = 0
    L1_Distance = 1
    Cosine_Distance = 2

class VarAutoencoder_Loss(Contextual_Loss, nn.Module):
    def __init__(
        self,
        layers_weights=(1.0, 1.0),
        cobi=True,
        crop_quarter=False,
        max_1d_size=10000,
        distance_type=Distance_Type.L1_Distance,
        b=1.0,
        h=0.5,
        weight_sp=0.1,
    ):
        nn.Module.__init__(self)

        encoder = Encoder(
            in_channels=1, num_channels=[32, 64], out_channels=64, num_res_blocks=[2, 2]
        )  # You need to define the Encoder class somewhere
        # Load the state dict
        encoder.load_state_dict(
            torch.load(
                "/home/kanghyun/misalign-benchmark/weights/encoder_pretrain_0.0.pth"
            )
        )
        for param in encoder.parameters():
            param.requires_grad = False  # Freeze layers
        self.encoder = encoder

        self.layers_weight = layers_weights
        self.crop_quarter = crop_quarter
        self.distanceType = distance_type
        self.max_1d_size = max_1d_size

        self.cobi = cobi
        self.b = b
        self.h = h
        self.weight_sp = weight_sp

    def forward(self, images, gt, average_over_scales=True, weight=None):
        # First we need to get latent variable as features

        _, images_p = self.encoder(images)  # b, 64, 64, 64
        _, gt_p = self.encoder(gt)  # b, 64, 64, 64

        images_p = images_p[1]
        gt_p = gt_p[1]

        # now these act as features
        N, C, H, W = gt_p.size()
        if self.crop_quarter:
            images_p = self._crop_quarters(images_p)
            gt_p = self._crop_quarters(gt_p)

        if H * W > self.max_1d_size**2:
            images_p = self._random_pooling(images_p, output_1d_size=self.max_1d_size)
            gt_p = self._random_pooling(gt_p, output_1d_size=self.max_1d_size)

        if self.cobi:
            loss = self.calculate_CoBi_Loss(
                images_p, gt_p, average_over_scales=average_over_scales, weight=weight
            )  # need to change this to calculate pixel-wise independent loss
        else:
            loss = self.calculate_CX_Loss(
                images_p, gt_p, average_over_scales=average_over_scales, weight=weight
            )  # need to change this to calculate pixel-wise independent loss
        return loss


class PatchContextualLoss(Contextual_Loss, nn.Module):
    def __init__(
        self,
        patch_size=5,
        cobi=True,
        crop_quarter=True,
        max_1d_size=10000,
        distance_type=Distance_Type.L1_Distance,
        b=1.0,
        h=0.5,
        weight_sp=0.1,
    ):
        nn.Module.__init__(self)
        self.crop_quarter = crop_quarter
        self.distanceType = distance_type
        self.max_1d_size = max_1d_size
        self.patch_size = patch_size

        self.cobi = cobi
        self.b = b
        self.h = h
        self.weight_sp = weight_sp

    def forward(self, images, gt, average_over_scales=True, weight=None):
        # First we need to make patch as features
        images_p = image2patch(images, patch_size=self.patch_size)
        gt_p = image2patch(gt, patch_size=self.patch_size)

        # now these act as features
        N, C, H, W = gt_p.size()
        if self.crop_quarter:
            images_p = self._crop_quarters(self._crop_quarters(images_p))
            gt_p = self._crop_quarters(self._crop_quarters(gt_p))

        if H * W > self.max_1d_size**2:
            images_p = self._random_pooling(images_p, output_1d_size=self.max_1d_size)
            gt_p = self._random_pooling(gt_p, output_1d_size=self.max_1d_size)

        if self.cobi:
            loss = self.calculate_CoBi_Loss(images_p, gt_p, average_over_scales=average_over_scales, weight=weight)
        else:
            loss = self.calculate_CX_Loss(images_p, gt_p, average_over_scales=average_over_scales, weight=weight)
        return loss