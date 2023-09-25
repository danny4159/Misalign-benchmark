# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from collections.abc import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from monai.networks.blocks import Convolution
from misalign.models.components.blurpool import BlurPool
from monai.utils import ensure_tuple_rep
import numpy as np
from typing import Union, Optional, Sequence, List, Tuple, Dict


def batch_rotate_p4(batch, k):
    """Rotates by k*90 degrees each sample in a batch.
    Args:
        batch (Tensor): the batch to rotate, format is (N, C, H, W).
        k (list of int): the rotations to perform for each sample k[i]*90 degrees.

    Returns (Tensor):
        The rotated batch.
    """
    batch_size = batch.shape[0]
    assert len(k) == batch_size, "The size of k must be equal to the batch size."

    # Infer the device from the batch tensor
    batch_p4 = []
    for i in range(batch_size):
        batch_p4.append(torch.rot90(batch[i], k=int(k[i]), dims=(1, 2)))
    batch_p4 = torch.stack(batch_p4, dim=0)
    return batch_p4


def get_filter(filt_size=3):
    if filt_size == 1:
        a = np.array(
            [
                1.0,
            ]
        )
    elif filt_size == 2:
        a = np.array([1.0, 1.0])
    elif filt_size == 3:
        a = np.array([1.0, 2.0, 1.0])
    elif filt_size == 4:
        a = np.array([1.0, 3.0, 3.0, 1.0])
    elif filt_size == 5:
        a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
    elif filt_size == 6:
        a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
    elif filt_size == 7:
        a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt


def get_pad_layer(pad_type):
    if pad_type in ["refl", "reflect"]:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ["repl", "replicate"]:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == "zero":
        PadLayer = nn.ZeroPad2d
    else:
        print("Pad type [%s] not recognized" % pad_type)
    return PadLayer


class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.module(x)


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels,
        pad_type="reflect",
        filt_size=3,
        stride=2,
        pad_off=0,
    ):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = in_channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer(
            "filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1))
        )

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            stride=1,
            kernel_size=3,
            padding=1,
            bias=False,
            padding_mode="replicate",
        )

    def forward(self, inp):
        inp = F.relu(inp, inplace=True)
        if self.filt_size == 1:
            if self.pad_off == 0:
                x = inp[:, :, :: self.stride, :: self.stride]
            else:
                x = self.pad(inp)[:, :, :: self.stride, :: self.stride]
        else:
            x = F.conv2d(
                self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1]
            )
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    """
    Convolution-based upsampling layer.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels to the layer.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            stride=1,
            kernel_size=3,
            padding=1,
            bias=False,
            padding_mode="replicate",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.to(torch.float32)

        x = F.interpolate(x, scale_factor=2.0, mode="bilinear")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            x = x.to(dtype)

        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        residual: bool = True,
        channel_attention=True,
        spectral_norm=False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.conv1 = self.spectral_norm_conditional(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                stride=1,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
                bias=False,
            ),
            spectral_norm,
        )

        self.conv2 = self.spectral_norm_conditional(
            nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                stride=1,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
                bias=False,
            ),
            spectral_norm,
        )
        if channel_attention:
            self.channel_attention = ChannelAttention(self.out_channels, reduction=4)
        else:
            self.channel_attention = nn.Identity()

        if self.in_channels != self.out_channels and residual is True:
            self.nin_shortcut = self.spectral_norm_conditional(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    stride=1,
                    kernel_size=1,
                    padding=0,
                ),
                spectral_norm,
            )
        elif residual is True:
            self.nin_shortcut = nn.Identity()
        self.residual = residual

    def spectral_norm_conditional(self, module, apply: bool) -> nn.Module:
        """
        Conditionally apply spectral normalization to the module.

        Args:
            module: PyTorch module to apply spectral normalization to.
            apply: Boolean to control the application of spectral normalization.

        Returns:
            module: PyTorch module with spectral normalization applied conditionally.
        """
        if apply:
            return nn.utils.spectral_norm(module)
        else:
            return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = F.relu(h, inplace=True)
        h = self.conv1(h)

        h = F.relu(h, inplace=True)
        h = self.conv2(h)

        h = self.channel_attention(h)

        if self.residual:
            x = self.nin_shortcut(x)
            h = h + x

        return h


class Encoder(nn.Module):
    """
    Convolutional cascade that downsamples the image into a spatial latent space.

    Args:
        in_channels: number of input channels.
        num_channels: sequence of block output channels.
        out_channels: number of channels in the bottom layer (latent space) of the autoencoder.
        num_res_blocks: number of residual blocks (see ResBlock) per level.
    """

    def __init__(
        self,
        in_channels: int,
        num_channels: Sequence[int],
        out_channels: int,
        num_res_blocks: Sequence[int],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        blocks = []
        # Initial convolution
        blocks.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_channels[0],
                stride=1,
                kernel_size=3,
                padding=1,
                bias=False,
                padding_mode="replicate",
            )
        )

        # Residual and downsampling blocks
        output_channel = num_channels[0]

        for i in range(len(num_channels)):
            input_channel = output_channel
            output_channel = num_channels[i]
            is_final_block = i == len(num_channels) - 1

            for _ in range(self.num_res_blocks[i]):
                blocks.append(
                    ResBlock(
                        in_channels=input_channel,
                        out_channels=output_channel,
                        residual=True,
                    )
                )
                input_channel = output_channel

            blocks.append(Downsample(in_channels=input_channel))

        blocks.append(
            nn.Conv2d(
                in_channels=num_channels[-1],
                out_channels=out_channels,
                stride=1,
                kernel_size=3,
                padding=1,
                bias=False,
                padding_mode="replicate",
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        intermediate_outputs = []
        idx = 1
        for i, block in enumerate(self.blocks):
            x = block(x)
            if isinstance(block, Downsample):
                intermediate_outputs.append(x)
                idx += 1

        return x, intermediate_outputs


class Decoder(nn.Module):
    """
    Convolutional cascade upsampling from a spatial latent space into an image space.

    Args:
        num_channels: sequence of block output channels.
        in_channels: number of channels in the bottom layer (latent space) of the autoencoder.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see ResBlock) per level.
    """

    def __init__(
        self,
        num_channels: Sequence[int],
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int],
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        reversed_block_out_channels = list(reversed(num_channels))

        blocks = []
        # Initial convolution
        blocks.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=reversed_block_out_channels[0],
                stride=1,
                kernel_size=3,
                padding=1,
                bias=False,
                padding_mode="replicate",
            )
        )

        reversed_num_res_blocks = list(reversed(num_res_blocks))
        block_out_ch = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            block_in_ch = block_out_ch
            block_out_ch = reversed_block_out_channels[i]
            is_final_block = i == len(num_channels) - 1

            for _ in range(reversed_num_res_blocks[i]):
                blocks.append(
                    ResBlock(
                        in_channels=block_in_ch,
                        out_channels=block_out_ch,
                        channel_attention=False,
                    )
                )
                block_in_ch = block_out_ch
            blocks.append(Upsample(in_channels=block_in_ch))

        blocks.append(
            nn.Conv2d(
                in_channels=block_in_ch,
                out_channels=out_channels,
                stride=1,
                kernel_size=3,
                padding=1,
                bias=False,
                padding_mode="replicate",
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


# Define the PatchNCE head
class PatchNCEHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=1,
                    kernel_size=1,
                    padding=0,
                    bias=False,
                )
            ),
            nn.ReLU(inplace=False),
            spectral_norm(
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                    kernel_size=1,
                    padding=0,
                    bias=False,
                )
            ),
            Normalize(2),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm + 1e-7)
        return out


class AutoencoderKL(nn.Module):
    """
    Autoencoder model with KL-regularized latent space based on
    Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" https://arxiv.org/abs/2112.10752
    and Pinaya et al. "Brain Imaging Generation with Latent Diffusion Models" https://arxiv.org/abs/2209.07162

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see ResBlock) per level.
        num_channels: sequence of block output channels.
        latent_channels: number of channels in the latent space."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        num_channels: Sequence[int] = (32, 64, 64, 64),
        latent_channels: int = 6,
    ) -> None:
        super().__init__()

        # All number of channels should be multiple of num_groups

        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(num_channels))

        if len(num_res_blocks) != len(num_channels):
            raise ValueError(
                "`num_res_blocks` should be a single integer or a tuple of integers with the same length as "
                "`num_channels`."
            )

        self.encoder = Encoder(
            in_channels=in_channels,
            num_channels=num_channels,
            out_channels=num_channels[-1],
            num_res_blocks=num_res_blocks,
        )

        self.decoder = Decoder(
            num_channels=num_channels,
            in_channels=latent_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
        )

        self.quant_conv_mu = nn.Conv2d(
            in_channels=num_channels[-1],
            out_channels=latent_channels,
            stride=1,
            kernel_size=1,
            padding=0,
            bias=False,
        )

        self.quant_conv_log_sigma = nn.Conv2d(
            in_channels=num_channels[-1],
            out_channels=latent_channels,
            stride=1,
            kernel_size=1,
            padding=0,
            bias=False,
        )

        self.post_quant_conv = nn.Conv2d(
            in_channels=latent_channels,
            out_channels=latent_channels,
            stride=1,
            kernel_size=1,
            padding=0,
            bias=False,
        )

        self.latent_channels = latent_channels

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forwards an image through the spatial encoder, obtaining the latent mean and sigma representations.

        Args:
            x: BxCx[SPATIAL DIMS] tensor

        """
        h, intermediate_outputs = self.encoder(x)

        z_mu = self.quant_conv_mu(h)
        z_log_var = self.quant_conv_log_sigma(h)
        z_log_var = torch.clamp(z_log_var, -30.0, 20.0)
        z_sigma = torch.exp(z_log_var / 2)

        return z_mu, z_sigma, intermediate_outputs

    def NCEHead(self, x: torch.Tensor):
        """
        Forwards an image through the spatial encoder, obtaining the latent mean and sigma representations.

        Args:
            x: BxCx[SPATIAL DIMS] tensor

        """
        _, o = self.encoder(x)

        h1 = o[0].div(o[0].pow(2).sum(1, keepdim=True).pow(1.0 / 2) + 1e-7)
        h2 = o[1].div(o[1].pow(2).sum(1, keepdim=True).pow(1.0 / 2) + 1e-7)

        return h1, h2

    def sampling(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
        """
        From the mean and sigma representations resulting of encoding an image through the latent space,
        obtains a noise sample resulting from sampling gaussian noise, multiplying by the variance (sigma) and
        adding the mean.

        Args:
            z_mu: Bx[Z_CHANNELS]x[LATENT SPACE SIZE] mean vector obtained by the encoder when you encode an image
            z_sigma: Bx[Z_CHANNELS]x[LATENT SPACE SIZE] variance vector obtained by the encoder when you encode an image

        Returns:
            sample of shape Bx[Z_CHANNELS]x[LATENT SPACE SIZE]
        """
        eps = torch.randn_like(z_sigma)
        z_vae = z_mu + eps * z_sigma
        return z_vae

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes and decodes an input image.

        Args:
            x: BxCx[SPATIAL DIMENSIONS] tensor.

        Returns:
            reconstructed image, of the same shape as input
        """
        z_mu, _, _ = self.encode(x)
        reconstruction = self.decode(z_mu)
        return reconstruction

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Based on a latent space sample, forwards it through the Decoder.

        Args:
            z: Bx[Z_CHANNELS]x[LATENT SPACE SHAPE]

        Returns:
            decoded image tensor
        """
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mu, z_sigma, intermediate_output = self.encode(x)
        z = self.sampling(z_mu, z_sigma)
        reconstruction = self.decode(z)
        return reconstruction, z_mu, z_sigma, intermediate_output

    def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        z_mu, z_sigma, _ = self.encode(x)
        z = self.sampling(z_mu, z_sigma)
        return z

    def decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor:
        image = self.decode(z)
        return image


if __name__ == "__main__":
    # 8 time downsampling
    encoder_model = Encoder(
        in_channels=1, num_channels=[16, 32], out_channels=32, num_res_blocks=[2, 2]
    )

    # print(encoder_model)
    x = torch.randn(1, 1, 256, 256)
    y, o = encoder_model(x)  # 1,256,64,64
    print(y.shape)

    for i, o_ in enumerate(o):
        print(f"Downsample layer {i+1} shape: {o_.shape}")

    decoder_model = Decoder(
        num_channels=[16, 32], in_channels=32, out_channels=1, num_res_blocks=[2, 2]
    )
    x_hat = decoder_model(y)

    print(x_hat.shape)

    spatial_dims = 2
    in_channels = 1
    out_channels = 1
    num_res_blocks = (2, 2)
    num_channels = (16, 32)
    latent_channels = 6

    # create an instance of the model
    model = AutoencoderKL(
        in_channels=in_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        num_channels=num_channels,
        latent_channels=latent_channels,
    )

    # create a random tensor with size (1, in_channels, 256, 256)
    input_tensor = torch.randn(1, in_channels, 256, 256)

    # forward pass through the model
    reconstruction, z_mu, z_sigma, o = model(input_tensor)

    # print out the sizes of the outputs
    print("Reconstruction size:", reconstruction.size())
    print("Latent mean size:", z_mu.size())
    print("Latent sigma size:", z_sigma.size())
    for i, h_ in enumerate(o):
        print(f"latent Downsample layer before head {i+1} shape: {h_.shape}")
