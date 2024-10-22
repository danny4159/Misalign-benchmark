import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops.upfirdn2d import upfirdn2d

# from .networks_v2 import init_net

def define_G(
    input_nc,
    output_nc,
    netG="resnet_4blocks",
    init_type="normal",
    init_gain=0.02,
    gpu_ids=[],
    feat_ch=64,
    demodulate=True,
    requires_grad=True,


):

    net = None
    
    if netG == "dam":
        net = DAModule(
            input_nc,
            feat_ch,
            output_nc,
            demodulate,
            requires_grad,
        )
    else:
        raise NotImplementedError("Generator model name [%s] is not recognized" % netG)
    from .networks_v2 import init_net
    return init_net(
        net, init_type, init_gain, gpu_ids, initialize_weights=("stylegan2" not in netG)
    )


class DAModule(nn.Module):
    def __init__(self, input_nc, feat_ch, output_nc, demodulate=True, load_path=None,
                 requires_grad=True):
        super().__init__()

        self.guide_net = nn.Sequential( # 이 conv들은 image의 style정보를 잘 대표하는 통계값이 나오도록 학습
            nn.Conv2d(input_nc, feat_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1), # style transfer에서 많이 쓰이는 개념.
        )

        self.conv0 = ModulatedStyleConv(input_nc, feat_ch, feat_ch, kernel_size=3,
                                            activate=True, demodulate=demodulate)
        self.conv11 = ModulatedStyleConv(feat_ch, feat_ch, feat_ch, kernel_size=3,
                                downsample=True, activate=True, demodulate=demodulate)
        self.conv12 = ModulatedStyleConv(feat_ch, feat_ch, feat_ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=demodulate)
        self.conv21 = ModulatedStyleConv(feat_ch, feat_ch, feat_ch, kernel_size=3,
                                downsample=True, activate=True, demodulate=demodulate)
        self.conv22 = ModulatedStyleConv(feat_ch, feat_ch, feat_ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=demodulate)
        self.conv31 = ModulatedStyleConv(feat_ch, feat_ch, feat_ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=demodulate)
        self.conv32 = ModulatedStyleConv(feat_ch, feat_ch, feat_ch, kernel_size=3,
                                downsample=False, activate=True, demodulate=demodulate)
        self.conv41 = ModulatedStyleConv(feat_ch, feat_ch, feat_ch, kernel_size=3,
                                upsample=True, activate=True, demodulate=demodulate)
        self.conv42 = ModulatedStyleConv(feat_ch, feat_ch, feat_ch, kernel_size=3,
                                upsample=False, activate=True, demodulate=demodulate)
        self.conv51 = ModulatedStyleConv(feat_ch, feat_ch, feat_ch, kernel_size=3,
                                upsample=True, activate=True, demodulate=demodulate)
        self.conv52 = ModulatedStyleConv(feat_ch, feat_ch, feat_ch, kernel_size=3,
                                upsample=False, activate=True, demodulate=demodulate)
        self.conv6 = ModulatedStyleConv(feat_ch, feat_ch, output_nc, kernel_size=3,
                                activate=False, demodulate=demodulate)

        # if load_path:
        #     checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
        #     if 'state_dict' in checkpoint:
        #         # PyTorch Lightning 형식의 .ckpt 파일
        #         model_state_dict = checkpoint['state_dict']
        #         # 모델의 state_dict에 맞게 키 이름을 조정할 수 있습니다.
        #         adjusted_state_dict = {k.replace('netG_B.', ''): v for k, v in model_state_dict.items()}
        #         self.load_state_dict(adjusted_state_dict, strict=False)
        #     else:
        #         # 일반적인 PyTorch 형식의 .pth 파일
        #         self.load_state_dict(checkpoint)
        
        # if load_path:
        #     self.load_state_dict(torch.load(
        #         load_path, map_location=lambda storage, loc: storage)['params_ema'])

        if not requires_grad:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.train()
            for param in self.parameters():
                param.requires_grad = True

    def forward(self, x, ref):
        if not self.training:
            N, C, H, W = x.shape
            mod_size = 4
            H_pad = mod_size - H % mod_size if not H % mod_size == 0 else 0
            W_pad = mod_size - W % mod_size if not W % mod_size == 0 else 0
            x = F.pad(x, (0, W_pad, 0, H_pad), 'replicate')

        style_guidance = self.guide_net(ref)

        feat0 = self.conv0(x, style_guidance) # [1, 1, 444, 312], [1, 64, 1, 1]
        feat1 = self.conv11(feat0, style_guidance) # [1, 64, 221, 155] 
        feat1 = self.conv12(feat1, style_guidance) # [1, 64, 221, 155]
        feat2 = self.conv21(feat1, style_guidance) # [1, 64, 110, 77]
        feat2 = self.conv22(feat2, style_guidance) # [1, 64, 110, 77]
        feat3 = self.conv31(feat2, style_guidance) # [1, 64, 110, 77]
        feat3 = self.conv32(feat3, style_guidance) # [1, 64, 110, 77]
        feat4 = self.conv41(feat3 + feat2, style_guidance) # [1, 64, 221, 155]
        feat4 = self.conv42(feat4, style_guidance) # [1, 64, 221, 155]
        feat5 = self.conv51(feat4 + feat1, style_guidance) # [1, 64, 443, 311]
        feat5 = self.conv52(feat5, style_guidance) # [1, 64, 443, 311]
        feat6 = self.conv6(feat5 + feat0, style_guidance) # [1, 64, 443, 311], [1, 64, 444, 312]
        out = torch.tanh(feat6) # TODO: tanh 추가함.

        # out = feat6
        if not self.training:
            out = out[:, :, :H, :W]

        return out


class ModulatedStyleConv(nn.Module):
    def __init__(self,
                 input_nc,
                 feat_ch,
                 output_nc,
                 kernel_size,
                 upsample=False,
                 downsample=False,
                 activate=False,
                 blur_kernel=[1, 3, 3, 1],
                 demodulate=True,
                 eps=1e-8,):
        super(ModulatedStyleConv, self).__init__()
        self.eps = eps
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.demodulate = demodulate
        self.upsample = upsample
        self.downsample = downsample
        self.activate = activate
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.style_weight = nn.Sequential(
            nn.Conv2d(feat_ch, input_nc, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),)

        self.style_bias = nn.Sequential(
            nn.Conv2d(feat_ch, output_nc, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),)

        self.weight = nn.Parameter(
            torch.randn(1, output_nc, input_nc, kernel_size, kernel_size))

        # build blurry layer for upsampling
        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, (pad0, pad1), upsample_factor=factor)
        # build blurry layer for downsampling
        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        if activate:
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, style): # style: [1, 64, 1, 1]
        n, c, h, w = x.shape # [1, 1, 444, 312]
        # process style code
        # pdb.set_trace()

        style_w = self.style_weight(style).view(n, 1, c, 1, 1) # [1, 1, 1, 1, 1]
        style_b = self.style_bias(style).view(n, self.output_nc, 1, 1) # [1, 64, 1, 1]

        # combine weight and style
        weight = self.weight * style_w # [1, 64, 1, 3, 3] = [1, 64, 1, 3, 3] * [1, 1, 1, 1, 1] # 컨볼루션 가중치에 sytle 가중치를 곱해줘.
        if self.demodulate: # demodulation: 컨볼루션 안정성 위해
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps) # [1, 64]
            weight = weight * demod.view(n, self.output_nc, 1, 1, 1) #[1, 64, 1, 3, 3]

        weight = weight.view(n * self.output_nc, c, self.kernel_size, # [64, 1, 3, 3]
                             self.kernel_size)

        if self.upsample:
            x = x.view(1, n * c, h, w)
            weight = weight.view(n, self.output_nc, c, self.kernel_size,
                                 self.kernel_size)
            weight = weight.transpose(1, 2).reshape(n * c, self.output_nc,
                                                    self.kernel_size,
                                                    self.kernel_size)
            x = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=n)
            x = x.reshape(n, self.output_nc, *x.shape[-2:])
            x = self.blur(x) # aliasing 방지. 자연스럽게
        elif self.downsample: # 밑에 차원은 하나에 대한 예시.
            x = self.blur(x) # [4, 64, 257, 257]
            x = x.view(1, n * self.input_nc, *x.shape[-2:]) # [1, 256, 257, 257]
            x = F.conv2d(x, weight, stride=2, padding=0, groups=n) # [1, 256, 128, 128] / weight [256, 64, 3, 3]
            x = x.view(n, self.output_nc, *x.shape[-2:]) # [4, 64, 128, 128]
        else:
            x = x.view(1, n * c, h, w) # [1, 1, 444, 312]
            x = F.conv2d(x, weight, stride=1, padding=self.padding, groups=n) # [1, 64, 444, 312]
            x = x.view(n, self.output_nc, *x.shape[-2:]) # [1, 64, 444, 312]

        out = x + style_b # [1, 64, 444, 312] = [1, 64, 444, 312] + [1, 64, 1, 1])

        if self.activate:
            out = self.act(out)

        return out


class Blur(nn.Module):

    def __init__(self, kernel, pad, upsample_factor=1):
        super(Blur, self).__init__()
        kernel = _make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, x):
        return upfirdn2d(x, self.kernel, pad=self.pad)
        # return x

def _make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k