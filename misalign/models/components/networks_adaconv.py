from .networks_v2 import init_net

import warnings

from torch import nn

from math import ceil, floor

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.transforms import transforms

def define_G(
    # input_nc,
    # output_nc,
    netG="adaconv",
    init_type="normal",
    init_gain=0.02,
    gpu_ids=[],
    feat_ch=64,
    demodulate=True,
    requires_grad=True,


):

    net = None
    
    if netG == "adaconv":
        net = AdaConvModel(
            # input_nc,
            # feat_ch,
            # output_nc,
            # demodulate,
            # requires_grad,
        )
    else:
        raise NotImplementedError("Generator model name [%s] is not recognized" % netG)
    return init_net(
        net, init_type, init_gain, gpu_ids, initialize_weights=("stylegan2" not in netG)
    )


class AdaConvModel(nn.Module):
    def __init__(self, style_size=256, style_channels=512, kernel_size=3):
        super().__init__()
        self.encoder = VGGEncoder(normalize=False)

        style_in_shape = (self.encoder.out_channels, style_size // self.encoder.scale_factor, style_size // self.encoder.scale_factor) # self.encoder.out_channels=512, self.encoder.scale_factor=8
        style_out_shape = (style_channels, kernel_size, kernel_size)
        self.style_encoder = GlobalStyleEncoder(in_shape=style_in_shape, out_shape=style_out_shape)
        self.decoder = AdaConvDecoder(style_channels=style_channels, kernel_size=kernel_size)

    def forward(self, content, style, return_embeddings=False): # [1, 1, 592, 384]
        self.encoder.freeze()

        if content.shape[1] == 1:
            content = content.repeat(1, 3, 1, 1)
        if style.shape[1] == 1:
            style = style.repeat(1, 3, 1, 1)

        # Encode -> Decode
        content_embeddings, style_embeddings = self._encode(content, style)
        output = self._decode(content_embeddings[-1], style_embeddings[-1]) # [1, 1, 592, 384]

        # Return embeddings if training
        if return_embeddings:
            output_embeddings = self.encoder(output)
            embeddings = {
                'content': content_embeddings,
                'style': style_embeddings,
                'output': output_embeddings
            }
            return output, embeddings
        else:
            return output

    def _encode(self, content, style):
        content_embeddings = self.encoder(content)
        style_embeddings = self.encoder(style)
        return content_embeddings, style_embeddings

    def _decode(self, content_embedding, style_embedding):
        style_embedding = self.style_encoder(style_embedding)
        output = self.decoder(content_embedding, style_embedding)
        return output


class AdaConvDecoder(nn.Module):
    def __init__(self, style_channels, kernel_size):
        super().__init__()
        self.style_channels = style_channels
        self.kernel_size = kernel_size

        # Inverted VGG with first conv in each scale replaced with AdaConv
        group_div = [1, 2, 4, 8]
        n_convs = [1, 4, 2, 2]
        self.layers = nn.ModuleList([
            *self._make_layers(512, 256, group_div=group_div[0], n_convs=n_convs[0]),
            *self._make_layers(256, 128, group_div=group_div[1], n_convs=n_convs[1]),
            *self._make_layers(128, 64, group_div=group_div[2], n_convs=n_convs[2]),
            *self._make_layers(64, 3, group_div=group_div[3], n_convs=n_convs[3], final_act=False, upsample=False)])

    def forward(self, content, w_style):
        # Checking types is a bit hacky, but it works well.
        for module in self.layers:
            if isinstance(module, KernelPredictor):
                w_spatial, w_pointwise, bias = module(w_style)
            elif isinstance(module, AdaConv2d):
                content = module(content, w_spatial, w_pointwise, bias)
            else:
                content = module(content)
        return torch.tanh(content) # Tanh 내가 추가

    def _make_layers(self, in_channels, out_channels, group_div, n_convs, final_act=True, upsample=True):
        n_groups = in_channels // group_div
        
        if out_channels == 3:  # 이 부분을 변경합니다
            out_channels = 1
        
        layers = []
        for i in range(n_convs):
            last = i == n_convs - 1
            out_channels_ = out_channels if last else in_channels
            if i == 0:
                layers += [
                    KernelPredictor(in_channels, in_channels,
                                    n_groups=n_groups,
                                    style_channels=self.style_channels,
                                    kernel_size=self.kernel_size),
                    AdaConv2d(in_channels, out_channels_, n_groups=n_groups)]
            else:
                layers.append(nn.Conv2d(in_channels, out_channels_, 3,
                                        padding=1, padding_mode='reflect'))

            if not last or final_act:
                layers.append(nn.ReLU())

        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        return layers


class GlobalStyleEncoder(nn.Module):
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.in_shape = in_shape # 512, 75, 75
        self.out_shape = out_shape # 512, 3, 3
        channels = in_shape[0]

        self.downscale = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),
            #
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),
            #
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),
        )

        in_features = self.in_shape[0] * (self.in_shape[1] // 8) * self.in_shape[2] // 8
        out_features = self.out_shape[0] * self.out_shape[1] * self.out_shape[2]
        self.fc = nn.Linear(27648, out_features) # TODO: in_features 하드코딩하자. 어차피 이 코드는 전혀 사이즈 변동에 친화적이지 않다. 아래에 forward에서 ys의 size에 맞춰주면 돼
        # self.fc = nn.Linear(in_features, out_features) # original 코드

    def forward(self, xs):
        ys = self.downscale(xs)
        ys = ys.reshape(len(xs), -1)

        w = self.fc(ys) # [1, 8192]
        w = w.reshape(len(xs), self.out_shape[0], self.out_shape[1], self.out_shape[2])
        return w



class AdaConv2d(nn.Module):
    """
    Implementation of the Adaptive Convolution block. Performs a depthwise seperable adaptive convolution on its input X.
    The weights for the adaptive convolutions are generated by a KernelPredictor module based on the style embedding W.
    The adaptive convolution is followed by a normal convolution.

    References:
        https://openaccess.thecvf.com/content/CVPR2021/papers/Chandran_Adaptive_Convolutions_for_Structure-Aware_Style_Transfer_CVPR_2021_paper.pdf


    Args:
        in_channels: Number of channels in the input image.
        out_channels: Number of channels produced by final convolution.
        kernel_size: The kernel size of the final convolution.
        n_groups: The number of groups for the adaptive convolutions.
            Defaults to 1 group per channel if None.

    Input shape:
        x: Input tensor.
        w_spatial: Weights for the spatial adaptive convolution.
        w_pointwise: Weights for the pointwise adaptive convolution.
        bias: Bias for the pointwise adaptive convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=None):
        super().__init__()
        self.n_groups = in_channels if n_groups is None else n_groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        padding = (kernel_size - 1) / 2
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(kernel_size, kernel_size),
                              padding=(ceil(padding), floor(padding)),
                              padding_mode='reflect')

    def forward(self, x, w_spatial, w_pointwise, bias):
        assert len(x) == len(w_spatial) == len(w_pointwise) == len(bias)
        x = F.instance_norm(x) # [1, 512, 74, 48] -> [1, 512, 74, 48]

        # F.conv2d does not work with batched filters (as far as I can tell)...
        # Hack for inputs with > 1 sample
        ys = []
        for i in range(len(x)):
            y = self._forward_single(x[i:i + 1], w_spatial[i], w_pointwise[i], bias[i])
            ys.append(y)
        ys = torch.cat(ys, dim=0) # [1, 512, 74, 48]

        ys = self.conv(ys)
        return ys

    def _forward_single(self, x, w_spatial, w_pointwise, bias):
        # Only square kernels
        assert w_spatial.size(-1) == w_spatial.size(-2)
        padding = (w_spatial.size(-1) - 1) / 2
        pad = (ceil(padding), floor(padding), ceil(padding), floor(padding))

        # 앞에서 Kernel predictor를 통해 얻은 style정보를 여기서 Depth separable conv 방식에 삽입.  
        x = F.pad(x, pad=pad, mode='reflect')
        x = F.conv2d(x, w_spatial, groups=self.n_groups)
        x = F.conv2d(x, w_pointwise, groups=self.n_groups, bias=bias)
        return x #


class KernelPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups, style_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_channels = style_channels
        self.n_groups = n_groups
        self.kernel_size = kernel_size

        padding = (kernel_size - 1) / 2
        self.spatial = nn.Conv2d(style_channels,
                                 in_channels * out_channels // n_groups,
                                 kernel_size=kernel_size,
                                 padding=(ceil(padding), ceil(padding)),
                                 padding_mode='reflect')
        self.pointwise = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(style_channels,
                      out_channels * out_channels // n_groups,
                      kernel_size=1)
        )
        self.bias = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(style_channels,
                      out_channels,
                      kernel_size=1)
        )

    def forward(self, w):
        w_spatial = self.spatial(w)
        w_spatial = w_spatial.reshape(len(w),
                                      self.out_channels,
                                      self.in_channels // self.n_groups,
                                      self.kernel_size, self.kernel_size)

        w_pointwise = self.pointwise(w)
        w_pointwise = w_pointwise.reshape(len(w),
                                          self.out_channels,
                                          self.out_channels // self.n_groups,
                                          1, 1)

        bias = self.bias(w)
        bias = bias.reshape(len(w),
                            self.out_channels)

        return w_spatial, w_pointwise, bias  # [1, 512, 1, 3, 3], [1, 512, 1, 1, 1], [1, 512]
    

class VGGEncoder(nn.Module):
    def __init__(self, normalize=True, post_activation=True):
        super().__init__()

        if normalize:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            self.normalize = transforms.Normalize(mean=mean, std=std)
        else:
            self.normalize = nn.Identity()

        if post_activation:
            layer_names = {'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'}
        else:
            layer_names = {'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1'}
        blocks, block_names, scale_factor, out_channels = extract_vgg_blocks(models.vgg19(pretrained=True).features,
                                                                             layer_names)

        self.blocks = nn.ModuleList(blocks)
        self.block_names = block_names
        self.scale_factor = scale_factor
        self.out_channels = out_channels

    def forward(self, xs):
        if xs.shape[1] == 1:
            xs = xs.repeat(1, 3, 1, 1)
        
        xs = self.normalize(xs)

        features = []
        for block in self.blocks:
            xs = block(xs)
            features.append(xs)

        return features

    def freeze(self):
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False


# For AdaIn, not used in AdaConv.
class VGGDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        layers = [
            self._conv(512, 256),
            nn.ReLU(),
            self._upsample(),

            self._conv(256, 256),
            nn.ReLU(),
            self._conv(256, 256),
            nn.ReLU(),
            self._conv(256, 256),
            nn.ReLU(),
            self._conv(256, 128),
            nn.ReLU(),
            self._upsample(),

            self._conv(128, 128),
            nn.ReLU(),
            self._conv(128, 64),
            nn.ReLU(),
            self._upsample(),

            self._conv(64, 64),
            nn.ReLU(),
            self._conv(64, 3),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, content):
        ys = self.layers(content)
        return ys

    @staticmethod
    def _conv(in_channels, out_channels, kernel_size=3, padding_mode='reflect'):
        padding = (kernel_size - 1) // 2
        return nn.Conv2d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         padding=padding,
                         padding_mode=padding_mode)

    @staticmethod
    def _upsample(scale_factor=2, mode='nearest'):
        return nn.Upsample(scale_factor=scale_factor, mode=mode)


def extract_vgg_blocks(layers, layer_names):
    blocks, current_block, block_names = [], [], []
    scale_factor, out_channels = -1, -1
    depth_idx, relu_idx, conv_idx = 1, 1, 1
    for layer in layers:
        name = ''
        if isinstance(layer, nn.Conv2d):
            name = f'conv{depth_idx}_{conv_idx}'
            current_out_channels = layer.out_channels
            layer.padding_mode = 'reflect'
            conv_idx += 1
        elif isinstance(layer, nn.ReLU):
            name = f'relu{depth_idx}_{relu_idx}'
            layer = nn.ReLU(inplace=False)
            relu_idx += 1
        elif isinstance(layer, nn.AvgPool2d) or isinstance(layer, nn.MaxPool2d):
            name = f'pool{depth_idx}'
            depth_idx += 1
            conv_idx = 1
            relu_idx = 1
        else:
            warnings.warn(f' Unexpected layer type: {type(layer)}')

        current_block.append(layer)
        if name in layer_names:
            blocks.append(nn.Sequential(*current_block))
            block_names.append(name)
            scale_factor = 1 * 2 ** (depth_idx - 1)
            out_channels = current_out_channels
            current_block = []

    return blocks, block_names, scale_factor, out_channels
