# torch
import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn

# system
import os

# torch
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

scale_eval = False

alpha = 0.02
beta = 0.00002

resnet_n_blocks = 1

norm_layer = partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
align_corners = False
up_sample_mode = "bilinear"

# local
sampling_align_corners = False

# The number of filters in each block of the encoding part (down-sampling).
ndf = {
    "A": [32, 64, 64, 64, 64, 64],
}
# The number of filters in each block of the decoding part (up-sampling).
# If len(ndf[cfg]) > len(nuf[cfg]) - then the deformation field is up-sampled to match the input size.
nuf = {
    "A": [64, 64, 64, 64, 64, 32]
}
# Indicate if res-blocks are used in the down-sampling path.
use_down_resblocks = {
    "A": True,
}
# indicate the number of res-blocks applied on the encoded features.
resnet_nblocks = {
    "A": 3,
}
# Indicate if the a final refinement layer is applied on the before deriving the deformation field
refine_output = {
    "A": True,
}
# The activation used in the down-sampling path.
down_activation = {
    "A": "leaky_relu",
}
# The activation used in the up-sampling path.
up_activation = {
    "A": "leaky_relu",
}


def custom_init(m):
    m.data.normal_(0.0, alpha)


def get_init_function(activation, init_function, **kwargs):
    """Get the initialization function from the given name."""
    a = 0.0
    if activation == "leaky_relu":
        a = 0.2 if "negative_slope" not in kwargs else kwargs["negative_slope"]

    gain = 0.02 if "gain" not in kwargs else kwargs["gain"]
    if isinstance(init_function, str):
        if init_function == "kaiming":
            activation = "relu" if activation is None else activation
            return partial(
                torch.nn.init.kaiming_normal_,
                a=a,
                nonlinearity=activation,
                mode="fan_in",
            )
        elif init_function == "dirac":
            return torch.nn.init.dirac_
        elif init_function == "xavier":
            activation = "relu" if activation is None else activation
            gain = torch.nn.init.calculate_gain(nonlinearity=activation, param=a)
            return partial(torch.nn.init.xavier_normal_, gain=gain)
        elif init_function == "normal":
            return partial(torch.nn.init.normal_, mean=0.0, std=gain)
        elif init_function == "orthogonal":
            return partial(torch.nn.init.orthogonal_, gain=gain)
        elif init_function == "zeros":
            return partial(torch.nn.init.normal_, mean=0.0, std=1e-5)
    elif init_function is None:
        if activation in ["relu", "leaky_relu"]:
            return partial(torch.nn.init.kaiming_normal_, a=a, nonlinearity=activation)
        if activation in ["tanh", "sigmoid"]:
            gain = torch.nn.init.calculate_gain(nonlinearity=activation, param=a)
            return partial(torch.nn.init.xavier_normal_, gain=gain)
    else:
        return init_function


def get_activation(activation, **kwargs):
    """Get the appropriate activation from the given name"""
    if activation == "relu":
        return nn.ReLU(inplace=False)
    elif activation == "leaky_relu":
        negative_slope = (
            0.2 if "negative_slope" not in kwargs else kwargs["negative_slope"]
        )
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        return None


class Conv(torch.nn.Module):
    """Defines a basic convolution layer.
    The general structure is as follow:

    Conv -> Norm (optional) -> Activation -----------> + --> Output
                                         |            ^
                                         |__ResBlcok__| (optional)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias=True,
        activation="relu",
        init_func="kaiming",
        use_norm=False,
        use_resnet=False,
        **kwargs
    ):
        super(Conv, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.resnet_block = (
            ResnetTransformer(out_channels, resnet_n_blocks, init_func)
            if use_resnet
            else None
        )
        self.norm = norm_layer(out_channels) if use_norm else None
        self.activation = get_activation(activation, **kwargs)
        # Initialize the weights
        init_ = get_init_function(activation, init_func)
        init_(self.conv2d.weight)
        if self.conv2d.bias is not None:
            self.conv2d.bias.data.zero_()
        if self.norm is not None and isinstance(self.norm, nn.BatchNorm2d):
            nn.init.normal_(self.norm.weight.data, 0.0, 1.0)
            nn.init.constant_(self.norm.bias.data, 0.0)

    def forward(self, x):
        x = self.conv2d(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.resnet_block is not None:
            x = self.resnet_block(x)
        return x


class UpBlock(torch.nn.Module):
    def __init__(
        self,
        nc_down_stream,
        nc_skip_stream,
        nc_out,
        kernel_size,
        stride,
        padding,
        bias=True,
        activation="relu",
        init_func="kaiming",
        use_norm=False,
        refine=False,
        use_resnet=False,
        use_add=False,
        use_attention=False,
        **kwargs
    ):
        super(UpBlock, self).__init__()
        if "nc_inner" in kwargs:
            nc_inner = kwargs["nc_inner"]
        else:
            nc_inner = nc_out
        self.conv_0 = Conv(
            nc_down_stream + nc_skip_stream,
            nc_inner,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            activation=activation,
            init_func=init_func,
            use_norm=use_norm,
            use_resnet=use_resnet,
            **kwargs
        )
        self.conv_1 = None
        if refine:
            self.conv_1 = Conv(
                nc_inner,
                nc_inner,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                activation=activation,
                init_func=init_func,
                use_norm=use_norm,
                use_resnet=use_resnet,
                **kwargs
            )
        self.use_attention = use_attention
        if self.use_attention:
            self.attention_gate = AttentionGate(
                nc_down_stream,
                nc_skip_stream,
                nc_inner,
                use_norm=True,
                init_func=init_func,
            )
        self.up_conv = Conv(
            nc_inner,
            nc_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            activation=activation,
            init_func=init_func,
            use_norm=use_norm,
            use_resnet=False,
            **kwargs
        )
        self.use_add = use_add
        if self.use_add:
            self.output = Conv(
                nc_out,
                2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
                activation=None,
                init_func="zeros",
                use_norm=False,
                use_resnet=False,
            )

    def forward(self, down_stream, skip_stream):
        down_stream_size = down_stream.size()
        skip_stream_size = skip_stream.size()
        if self.use_attention:
            skip_stream = self.attention_gate(down_stream, skip_stream)
        if (
            down_stream_size[2] != skip_stream_size[2]
            or down_stream_size[3] != skip_stream_size[3]
        ):
            down_stream = F.interpolate(
                down_stream,
                (skip_stream_size[2], skip_stream_size[3]),
                mode=up_sample_mode,
                align_corners=align_corners,
            )
        x = torch.cat([down_stream, skip_stream], 1)
        x = self.conv_0(x)
        if self.conv_1 is not None:
            x = self.conv_1(x)
        if self.use_add:
            x = self.output(x) + down_stream
        else:
            x = self.up_conv(x)
        return x


class DownBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias=False,
        activation="relu",
        init_func="kaiming",
        use_norm=False,
        use_resnet=False,
        skip=True,
        refine=False,
        pool=True,
        pool_size=2,
        **kwargs
    ):
        super(DownBlock, self).__init__()
        self.conv_0 = Conv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=bias,
            activation=activation,
            init_func=init_func,
            use_norm=use_norm,
            callback=None,
            use_resnet=use_resnet,
            **kwargs
        )
        self.conv_1 = None
        if refine:
            self.conv_1 = Conv(
                out_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=bias,
                activation=activation,
                init_func=init_func,
                use_norm=use_norm,
                callback=None,
                use_resnet=use_resnet,
                **kwargs
            )
        self.skip = skip
        self.pool = None
        if pool:
            self.pool = nn.MaxPool2d(kernel_size=pool_size)

    def forward(self, x):
        x = skip = self.conv_0(x)
        if self.conv_1 is not None:
            x = skip = self.conv_1(x)
        if self.pool is not None:
            x = self.pool(x)
        if self.skip:
            return x, skip
        else:
            return x


class AttentionGate(torch.nn.Module):
    def __init__(
        self,
        nc_g,
        nc_x,
        nc_inner,
        use_norm=False,
        init_func="kaiming",
        mask_channel_wise=False,
    ):
        super(AttentionGate, self).__init__()
        self.conv_g = Conv(
            nc_g,
            nc_inner,
            1,
            1,
            0,
            bias=True,
            activation=None,
            init_func=init_func,
            use_norm=use_norm,
            use_resnet=False,
        )
        self.conv_x = Conv(
            nc_x,
            nc_inner,
            1,
            1,
            0,
            bias=False,
            activation=None,
            init_func=init_func,
            use_norm=use_norm,
            use_resnet=False,
        )
        self.residual = nn.ReLU(inplace=True)
        self.mask_channel_wise = mask_channel_wise
        self.attention_map = Conv(
            nc_inner,
            nc_x if mask_channel_wise else 1,
            1,
            1,
            0,
            bias=True,
            activation="sigmoid",
            init_function=init_func,
            use_norm=use_norm,
            use_resnet=False,
        )

    def forward(self, g, x):
        x_size = x.size()
        g_size = g.size()
        x_resized = x
        g_c = self.conv_g(g)
        x_c = self.conv_x(x_resized)
        if x_c.size(2) != g_size[2] and x_c.size(3) != g_size[3]:
            x_c = F.interpolate(
                x_c,
                (g_size[2], g_size[3]),
                mode=up_sample_mode,
                align_corners=align_corners,
            )
        combined = self.residual(g_c + x_c)
        alpha = self.attention_map(combined)
        if not self.mask_channel_wise:
            alpha = alpha.repeat(1, x_size[1], 1, 1)
        alpha_size = alpha.size()
        if alpha_size[2] != x_size[2] and alpha_size[3] != x_size[3]:
            alpha = F.interpolate(
                x,
                (x_size[2], x_size[3]),
                mode=up_sample_mode,
                align_corners=align_corners,
            )
        return alpha * x


class ResnetTransformer(torch.nn.Module):
    def __init__(self, dim, n_blocks, init_func):
        super(ResnetTransformer, self).__init__()
        model = []
        for i in range(n_blocks):  # add ResNet blocks
            model += [
                ResnetBlock(
                    dim,
                    padding_type="reflect",
                    norm_layer=norm_layer,
                    use_dropout=False,
                    use_bias=True,
                )
            ]
        self.model = nn.Sequential(*model)

        init_ = get_init_function("relu", init_func)

        def init_weights(m):
            if type(m) == nn.Conv2d:
                init_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            if type(m) == nn.BatchNorm2d:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResUnet(torch.nn.Module):
    def __init__(self, nc_a, nc_b, cfg, init_func, init_to_identity):
        super(ResUnet, self).__init__()
        act = down_activation[cfg]
        # ------------ Down-sampling path
        self.ndown_blocks = len(ndf[cfg])
        self.nup_blocks = len(nuf[cfg])
        assert self.ndown_blocks >= self.nup_blocks
        in_nf = nc_a + nc_b
        conv_num = 1
        skip_nf = {}
        for out_nf in ndf[cfg]:
            setattr(
                self,
                "down_{}".format(conv_num),
                DownBlock(
                    in_nf,
                    out_nf,
                    3,
                    1,
                    1,
                    activation=act,
                    init_func=init_func,
                    bias=True,
                    use_resnet=use_down_resblocks[cfg],
                    use_norm=False,
                ),
            )
            skip_nf["down_{}".format(conv_num)] = out_nf
            in_nf = out_nf
            conv_num += 1
        conv_num -= 1
        if use_down_resblocks[cfg]:
            self.c1 = Conv(
                in_nf,
                2 * in_nf,
                1,
                1,
                0,
                activation=act,
                init_func=init_func,
                bias=True,
                use_resnet=False,
                use_norm=False,
            )
            self.t = (
                (lambda x: x)
                if resnet_nblocks[cfg] == 0
                else ResnetTransformer(2 * in_nf, resnet_nblocks[cfg], init_func)
            )
            self.c2 = Conv(
                2 * in_nf,
                in_nf,
                1,
                1,
                0,
                activation=act,
                init_func=init_func,
                bias=True,
                use_resnet=False,
                use_norm=False,
            )
        # ------------- Up-sampling path
        act = up_activation[cfg]
        for out_nf in nuf[cfg]:
            setattr(
                self,
                "up_{}".format(conv_num),
                Conv(
                    in_nf + skip_nf["down_{}".format(conv_num)],
                    out_nf,
                    3,
                    1,
                    1,
                    bias=True,
                    activation=act,
                    init_fun=init_func,
                    use_norm=False,
                    use_resnet=False,
                ),
            )
            in_nf = out_nf
            conv_num -= 1
        if refine_output[cfg]:
            self.refine = nn.Sequential(
                ResnetTransformer(in_nf, 1, init_func),
                Conv(
                    in_nf,
                    in_nf,
                    1,
                    1,
                    0,
                    use_resnet=False,
                    init_func=init_func,
                    activation=act,
                    use_norm=False,
                ),
            )
        else:
            self.refine = lambda x: x
        self.output = Conv(
            in_nf,
            2,
            3,
            1,
            1,
            use_resnet=False,
            bias=True,
            init_func=("zeros" if init_to_identity else init_func),
            activation=None,
            use_norm=False,
        )

    def forward(self, img_a, img_b):
        x = torch.cat([img_a, img_b], 1)
        skip_vals = {}
        conv_num = 1
        # Down
        while conv_num <= self.ndown_blocks:
            x, skip = getattr(self, "down_{}".format(conv_num))(x)
            skip_vals["down_{}".format(conv_num)] = skip
            conv_num += 1
        if hasattr(self, "t"):
            x = self.c1(x)
            x = self.t(x)
            x = self.c2(x)
        # Up
        conv_num -= 1
        while conv_num > (self.ndown_blocks - self.nup_blocks):
            s = skip_vals["down_{}".format(conv_num)]
            x = F.interpolate(x, (s.size(2), s.size(3)), mode="bilinear")
            x = torch.cat([x, s], 1)
            x = getattr(self, "up_{}".format(conv_num))(x)
            conv_num -= 1
        x = self.refine(x)
        x = self.output(x)
        return x


class Reg(nn.Module):
    def __init__(self, in_channels_a, in_channels_b):
        super(Reg, self).__init__()
        # height,width=256,256
        # in_channels_a,in_channels_b=1,1
        init_func = "kaiming"
        init_to_identity = True

        # paras end------------

        self.in_channels_a = in_channels_a
        self.in_channels_b = in_channels_b
        self.offset_map = ResUnet(
            self.in_channels_a,
            self.in_channels_b,
            cfg="A",
            init_func=init_func,
            init_to_identity=init_to_identity,
        )

    def forward(self, img_a, img_b):
        deformations = self.offset_map(img_a, img_b)

        return deformations


import torch
import torch.nn.functional as F


def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)

        torch.clamp(ix_se, 0, IW - 1, out=ix_se)
        torch.clamp(iy_se, 0, IH - 1, out=iy_se)

    image = image.view(N, C, IH * IW)

    nw_val = torch.gather(
        image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1)
    )
    ne_val = torch.gather(
        image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1)
    )
    sw_val = torch.gather(
        image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1)
    )
    se_val = torch.gather(
        image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1)
    )

    out_val = (
        nw_val.view(N, C, H, W) * nw.view(N, 1, H, W)
        + ne_val.view(N, C, H, W) * ne.view(N, 1, H, W)
        + sw_val.view(N, C, H, W) * sw.view(N, 1, H, W)
        + se_val.view(N, C, H, W) * se.view(N, 1, H, W)
    )

    return out_val


class Transformer_2D(nn.Module):
    def __init__(self):
        super(Transformer_2D, self).__init__()

    # @staticmethod
    def forward(self, src, flow):
        b = flow.shape[0]
        h = flow.shape[2]
        w = flow.shape[3]

        size = (h, w)

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = grid.to(torch.float32)
        grid = grid.repeat(b, 1, 1, 1).to(src.device)
        new_locs = grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]
        warped = grid_sample(src, new_locs)
        # ctx.save_for_backward(src,flow)
        return warped


def smoothing_loss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

    dx = dx * dx
    dy = dy * dy
    d = torch.mean(dx, dim=(1,2,3), keepdim=True) + torch.mean(dy, dim=(1,2,3), keepdim=True)
    return d
