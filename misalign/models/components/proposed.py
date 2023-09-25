from misalign.models.components.contextual_loss import Contextual_Loss
from misalign.models.components.autoencoder_kl import Encoder, AutoencoderKL
import monai
from collections import OrderedDict
from torchvision.models import vgg19, vgg16
import torch.nn.functional as F
import torch.nn as nn
import torch
import sys


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
            in_channels=1,
            num_channels=[32, 64],
            out_channels=64,
            num_res_blocks=[2,2]
        )  # You need to define the Encoder class somewhere
        # Load the state dict
        encoder.load_state_dict(torch.load("/home/kanghyun/misalign-benchmark/weights/encoder_pretrain_0.01.pth"))
        for param in encoder.parameters():
            param.requires_grad = False # Freeze layers
        self.encoder = encoder

        self.layers_weight = layers_weights
        self.crop_quarter = crop_quarter
        self.distanceType = distance_type
        self.max_1d_size = max_1d_size

        self.cobi = cobi
        self.b = b
        self.h = h
        self.weight_sp = weight_sp

    def forward(self, images, gt):
        # First we need to get latent variable as features
        
        _, images_p = self.encoder(images) # b, 64, 64, 64
        _, gt_p = self.encoder(gt) # b, 64, 64, 64
        
        images_p = images_p[0]
        gt_p = gt_p[0]
        
        # now these act as features
        N, C, H, W = gt_p.size()
        if self.crop_quarter:
            images_p = self._crop_quarters(images_p)
            gt_p = self._crop_quarters(gt_p)

        if H * W > self.max_1d_size**2:
            images_p = self._random_pooling(images_p, output_1d_size=self.max_1d_size)
            gt_p = self._random_pooling(gt_p, output_1d_size=self.max_1d_size)

        if self.cobi:
            loss = self.calculate_CoBi_Loss(images_p, gt_p) # need to change this to calculate pixel-wise independent loss
        else:
            loss = self.calculate_CX_Loss(images_p, gt_p) # need to change this to calculate pixel-wise independent loss
        return loss
