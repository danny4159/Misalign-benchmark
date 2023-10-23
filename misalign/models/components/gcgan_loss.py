# Implementation of gc loss for GC-GAN from https://github.com/yuta-hi/pytorch_similarity
# Cross-modality image synthesis from unpaired data using CycleGAN, Hiasa Yuta International Workshop on Simulation and Synthesis in Medical Imaging

import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.functional as F

_func_conv_nd_table = {
    1: F.conv1d,
    2: F.conv2d,
    3: F.conv3d
}

def spatial_filter_nd(x, kernel, mode='replicate'):
    """ N-dimensional spatial filter with padding.

    Args:
        x (~torch.Tensor): Input tensor.
        kernel (~torch.Tensor): Weight tensor (e.g., Gaussain kernel).
        mode (str, optional): Padding mode. Defaults to 'replicate'.

    Returns:
        ~torch.Tensor: Output tensor
    """

    n_dim = x.dim() - 2
    conv = _func_conv_nd_table[n_dim]

    pad = [None,None]*n_dim
    pad[0::2] = kernel.shape[2:]
    pad[1::2] = kernel.shape[2:]
    pad = [k//2 for k in pad]

    return conv(F.pad(x, pad=pad, mode=mode), kernel)

# NOTE: Gaussian kernel
def _gauss_1d(x, mu, sigma):
    return 1./(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) )

def gauss_kernel_1d(sigma, truncate=4.0):
    sd = float(sigma)
    lw = int(truncate * sd + 0.5)
    x = np.arange(-lw, lw+1)
    kernel_1d = _gauss_1d(x, 0., sigma)
    return kernel_1d / kernel_1d.sum()

def gauss_kernel_2d(sigma, truncate=4.0):
    sd = float(sigma)
    lw = int(truncate * sd + 0.5)
    x = y = np.arange(-lw, lw+1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    kernel_2d = _gauss_1d(X, 0., sigma) \
              * _gauss_1d(Y, 0., sigma)
    return kernel_2d / kernel_2d.sum()

def gauss_kernel_3d(sigma, truncate=4.0):
    sd = float(sigma)
    lw = int(truncate * sd + 0.5)
    x = y = z = np.arange(-lw, lw+1)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    kernel_3d = _gauss_1d(X, 0., sigma) \
              * _gauss_1d(Y, 0., sigma) \
              * _gauss_1d(Z, 0., sigma)
    return kernel_3d / kernel_3d.sum()

def gradient_kernel_1d(method='default'):

    if method == 'default':
        kernel_1d = np.array([-1,0,+1])
    else:
        raise ValueError('unsupported method (got {})'.format(method))

    return kernel_1d

def gradient_kernel_2d(method='default', axis=0):

    if method == 'default':
        kernel_2d = np.array([[0,-1,0],
                              [0,0,0],
                              [0,+1,0]])
    elif method == 'sobel':
        kernel_2d = np.array([[-1,-2,-1],
                              [0,0,0],
                              [+1,+2,+1]])
    elif method == 'prewitt':
        kernel_2d = np.array([[-1,-1,-1],
                              [0,0,0],
                              [+1,+1,+1]])
    elif method == 'isotropic':
        kernel_2d = np.array([[-1,-np.sqrt(2),-1],
                              [0,0,0],
                              [+1,+np.sqrt(2),+1]])
    else:
        raise ValueError('unsupported method (got {})'.format(method))

    return np.moveaxis(kernel_2d, 0, axis)

def gradient_kernel_3d(method='default', axis=0):

    if method == 'default':
        kernel_3d = np.array([[[0, 0, 0],
                               [0, -1, 0],
                               [0, 0, 0]],
                              [[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]],
                              [[0, 0, 0],
                               [0, +1, 0],
                               [0, 0, 0]]])
    elif method == 'sobel':
        kernel_3d = np.array([[[-1, -3, -1],
                               [-3, -6, -3],
                               [-1, -3, -1]],
                              [[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]],
                              [[+1, +3, +1],
                               [+3, +6, +3],
                               [+1, +3, +1]]])
    elif method == 'prewitt':
        kernel_3d = np.array([[[-1, -1, -1],
                               [-1, -1, -1],
                               [-1, -1, -1]],
                              [[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]],
                              [[+1, +1, +1],
                               [+1, +1, +1],
                               [+1, +1, +1]]])
    elif method == 'isotropic':
        kernel_3d = np.array([[[-1, -1, -1],
                               [-1, -np.sqrt(2), -1],
                               [-1, -1, -1]],
                              [[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]],
                              [[+1, +1, +1],
                               [+1, +np.sqrt(2), +1],
                               [+1, +1, +1]]])
    else:
        raise ValueError('unsupported method (got {})'.format(method))

    return np.moveaxis(kernel_3d, 0, axis)

def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return [x, x]


def normalized_cross_correlation(x, y, return_map, reduction='mean', eps=1e-8):
    """ N-dimensional normalized cross correlation (NCC)

    Args:
        x (~torch.Tensor): Input tensor.
        y (~torch.Tensor): Input tensor.
        return_map (bool): If True, also return the correlation map.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'sum'``.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.

    Returns:
        ~torch.Tensor: Output scalar
        ~torch.Tensor: Output tensor
    """

    shape = x.shape
    b = shape[0]

    # reshape
    x = x.view(b, -1)
    y = y.view(b, -1)

    # mean
    x_mean = torch.mean(x, dim=1, keepdim=True)
    y_mean = torch.mean(y, dim=1, keepdim=True)

    # deviation
    x = x - x_mean
    y = y - y_mean

    dev_xy = torch.mul(x,y)
    dev_xx = torch.mul(x,x)
    dev_yy = torch.mul(y,y)

    dev_xx_sum = torch.sum(dev_xx, dim=1, keepdim=True)
    dev_yy_sum = torch.sum(dev_yy, dim=1, keepdim=True)

    ncc = torch.div(dev_xy + eps / dev_xy.shape[1],
                    torch.sqrt( torch.mul(dev_xx_sum, dev_yy_sum)) + eps)
    ncc_map = ncc.view(b, *shape[1:])

    # reduce
    if reduction == 'mean':
        ncc = torch.mean(torch.sum(ncc, dim=1))
    elif reduction == 'sum':
        ncc = torch.sum(ncc)
    else:
        raise KeyError('unsupported reduction type: %s' % reduction)

    if not return_map:
        return ncc

    return ncc, ncc_map

def _grad_param(ndim, method, axis):

    if ndim == 1:
        kernel = gradient_kernel_1d(method)
    elif ndim == 2:
        kernel = gradient_kernel_2d(method, axis)
    elif ndim == 3:
        kernel = gradient_kernel_3d(method, axis)
    else:
        raise NotImplementedError

    kernel = kernel.reshape(1, 1, *kernel.shape)
    return Parameter(torch.Tensor(kernel).float())

def _gauss_param(ndim, sigma, truncate):

    if ndim == 1:
        kernel = gauss_kernel_1d(sigma, truncate)
    elif ndim == 2:
        kernel = gauss_kernel_2d(sigma, truncate)
    elif ndim == 3:
        kernel = gauss_kernel_3d(sigma, truncate)
    else:
        raise NotImplementedError

    kernel = kernel.reshape(1, 1, *kernel.shape)
    return Parameter(torch.Tensor(kernel).float())

class GradientDifference2d(nn.Module):
    """ Two-dimensional gradient difference

    Args:
        grad_method (str, optional): Type of the gradient kernel. Defaults to 'default'.
        gauss_sigma (float, optional): Standard deviation for Gaussian kernel. Defaults to None.
        gauss_truncate (float, optional): Truncate the Gaussian kernel at this value. Defaults to 4.0.
        return_map (bool, optional): If True, also return the correlation map. Defaults to False.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'mean'``.
    """
    def __init__(self,
                 grad_method='default',
                 gauss_sigma=None,
                 gauss_truncate=4.0,
                 return_map=False,
                 reduction='mean'):

        super(GradientDifference2d, self).__init__()

        self.grad_method = grad_method
        self.gauss_sigma = _pair(gauss_sigma)
        self.gauss_truncate = gauss_truncate

        self.grad_u_kernel = None
        self.grad_v_kernel = None

        self.gauss_kernel_x = None
        self.gauss_kernel_y = None

        self.return_map = return_map
        self.reduction = reduction

        self._initialize_params()
        self._freeze_params()

    def _initialize_params(self):
        self._initialize_grad_kernel()
        self._initialize_gauss_kernel()

    def _initialize_grad_kernel(self):
        self.grad_u_kernel = _grad_param(2, self.grad_method, axis=0)
        self.grad_v_kernel = _grad_param(2, self.grad_method, axis=1)

    def _initialize_gauss_kernel(self):
        if self.gauss_sigma[0] is not None:
            self.gauss_kernel_x = _gauss_param(2, self.gauss_sigma[0], self.gauss_truncate)
        if self.gauss_sigma[1] is not None:
            self.gauss_kernel_y = _gauss_param(2, self.gauss_sigma[1], self.gauss_truncate)

    def _check_type_forward(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(x.dim()))

    def _freeze_params(self):
        self.grad_u_kernel.requires_grad = False
        self.grad_v_kernel.requires_grad = False
        if self.gauss_kernel_x is not None:
            self.gauss_kernel_x.requires_grad = False
        if self.gauss_kernel_y is not None:
            self.gauss_kernel_y.requires_grad = False

    def forward(self, x, y):

        self._check_type_forward(x)
        self._check_type_forward(y)
        self._freeze_params()

        if x.shape[1] != y.shape[1]:
            x = torch.mean(x, dim=1, keepdim=True)
            y = torch.mean(y, dim=1, keepdim=True)

        # reshape
        b, c = x.shape[:2]
        spatial_shape = x.shape[2:]

        x = x.view(b*c, 1, *spatial_shape)
        y = y.view(b*c, 1, *spatial_shape)

        # smoothing
        if self.gauss_kernel_x is not None:
            x = spatial_filter_nd(x, self.gauss_kernel_x)
        if self.gauss_kernel_y is not None:
            y = spatial_filter_nd(y, self.gauss_kernel_y)

        # gradient magnitude
        x_grad_u = torch.abs(spatial_filter_nd(x, self.grad_u_kernel))
        x_grad_v = torch.abs(spatial_filter_nd(x, self.grad_v_kernel))

        y_grad_u = torch.abs(spatial_filter_nd(y, self.grad_u_kernel))
        y_grad_v = torch.abs(spatial_filter_nd(y, self.grad_v_kernel))

        # absolute difference
        diff_u = torch.abs(x_grad_u - y_grad_u)
        diff_v = torch.abs(x_grad_v - y_grad_v)

        # reshape back
        diff_u = diff_u.view(b, c, *spatial_shape)
        diff_v = diff_v.view(b, c, *spatial_shape)

        diff_map = 0.5 * (diff_u + diff_v)

        if self.reduction == 'mean':
            diff = torch.mean(diff_map)
        elif self.reduction == 'sum':
            diff = torch.sum(diff_map)
        else:
            raise KeyError('unsupported reduction type: %s' % self.reduction)

        if self.return_map:
            return diff, diff_map

        return diff


class GradientCorrelation2d(GradientDifference2d):
    """ Two-dimensional gradient correlation (GC)

    Args:
        grad_method (str, optional): Type of the gradient kernel. Defaults to 'default'.
        gauss_sigma (float, optional): Standard deviation for Gaussian kernel. Defaults to None.
        gauss_truncate (float, optional): Truncate the Gaussian kernel at this value. Defaults to 4.0.
        return_map (bool, optional): If True, also return the correlation map. Defaults to False.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'mean'``.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
    """
    def __init__(self,
                 grad_method='default',
                 gauss_sigma=None,
                 gauss_truncate=4.0,
                 return_map=False,
                 reduction='mean',
                 crop_quarters=False,
                 eps=1e-8):

        super().__init__(grad_method,
                        gauss_sigma,
                        gauss_truncate,
                        return_map,
                        reduction)

        self.eps = eps
        self.crop_quarters = crop_quarters

    @staticmethod
    def _crop_quarters(feature):
        N, fC, fH, fW = feature.size()
        quarters_list = []
        quarters_list.append(feature[..., 0 : round(fH / 2), 0 : round(fW / 2)])
        quarters_list.append(feature[..., 0 : round(fH / 2), round(fW / 2) :])
        quarters_list.append(feature[..., round(fH / 2) :, 0 : round(fW / 2)])
        quarters_list.append(feature[..., round(fH / 2) :, round(fW / 2) :])

        feature_tensor = torch.cat(quarters_list, dim=0)
        return feature_tensor

    def forward(self, x, y):
        if self.crop_quarters:
            x = self._crop_quarters(x)
            y = self._crop_quarters(y)

        self._check_type_forward(x)
        self._check_type_forward(y)
        self._freeze_params()

        if x.shape[1] != y.shape[1]:
            x = torch.mean(x, dim=1, keepdim=True)
            y = torch.mean(y, dim=1, keepdim=True)

        # reshape
        b, c = x.shape[:2]
        spatial_shape = x.shape[2:]

        x = x.view(b*c, 1, *spatial_shape)
        y = y.view(b*c, 1, *spatial_shape)

        # smoothing
        if self.gauss_kernel_x is not None:
            x = spatial_filter_nd(x, self.gauss_kernel_x)
        if self.gauss_kernel_y is not None:
            y = spatial_filter_nd(y, self.gauss_kernel_y)

        # gradient magnitude
        x_grad_u = torch.abs(spatial_filter_nd(x, self.grad_u_kernel))
        x_grad_v = torch.abs(spatial_filter_nd(x, self.grad_v_kernel))

        y_grad_u = torch.abs(spatial_filter_nd(y, self.grad_u_kernel))
        y_grad_v = torch.abs(spatial_filter_nd(y, self.grad_v_kernel))

        # gradient correlation
        gc_u, gc_map_u = normalized_cross_correlation(x_grad_u, y_grad_u, True, self.reduction, self.eps)
        gc_v, gc_map_v = normalized_cross_correlation(x_grad_v, y_grad_v, True, self.reduction, self.eps)

        gc = 0.5 * (gc_u + gc_v)

        return 1-gc

class GradientCorrelationLoss2d(GradientCorrelation2d):
    """ Two-dimensional gradient correlation loss (GC-loss)

    Args:
        grad_method (str, optional): Type of the gradient kernel. Defaults to 'default'.
        gauss_sigma (float, optional): Standard deviation for Gaussian kernel. Defaults to None.
        gauss_truncate (float, optional): Truncate the Gaussian kernel at this value. Defaults to 4.0.
        return_map (bool, optional): If True, also return the correlation map. Defaults to False.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'mean'``.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
    """
    def forward(self, x, y):
        gc = super().forward(x, y)

        if not self.return_map:
            return 1.0 - gc

        return 1.0 - gc[0], gc[1]