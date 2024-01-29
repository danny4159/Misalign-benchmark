# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from scipy import ndimage
# from . import transformer_configs as configs

import ml_collections

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size##paraphrase

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3,input_dim=3,old = 1, train=False):
        super(Embeddings, self).__init__()
        self.config = config
        img_size = _pair(img_size) # 256 -> 256,256
        grid_size = config.patches["grid"] # 16, 16
        #TODO:
        #1. patch size만 1 되게 train일땐 14 * 14 test일땐 n*14 되도록
        #2. patch_size_real 지우고, positional_encoding만 순간 데이터 크기에 맞게 되도록 바꿔주면

        # patch_size: 이미지를 얼마 단위로 나눌것인지
        patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1]) # 1, 1  //: 몫 
        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16) # 16, 16
        # n_patches: 이미지가 몇 개의 패치로 나뉘는지
        n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
        in_channels = 1024
        #Learnable patch embeddings
        self.patch_embeddings = Conv2d(in_channels=in_channels, # 1024
                                       out_channels=config.hidden_size, # 768
                                       kernel_size=patch_size, # 1, 1
                                       stride=patch_size) # 1, 1
        #learnable positional encodings

        # Case1. Case2할때는 주석처리
        # self.positional_encoding = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size)) # 1, 256, 768
        
        self.dropout = Dropout(config.transformer["dropout_rate"])
    
    # 내가 추가한 코드
    def create_sinusoidal_embeddings(self, n_patches, hidden_size):
        """Create sinusoidal positional embeddings."""
        position = torch.arange(n_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        sinusoidal_emb = torch.zeros(n_patches, hidden_size)
        sinusoidal_emb[:, 0::2] = torch.sin(position * div_term)
        sinusoidal_emb[:, 1::2] = torch.cos(position * div_term)
        return sinusoidal_emb.unsqueeze(0)

    def forward(self, x):
        
        x = self.patch_embeddings(x) # [1, 1024, 30, 20] -> [1, 768, 30, 20] // 256x256: [2, 1024, 16, 16] -> [2, 768, 16, 16] // 196x196: [1, 1024, 28, 20]->[1, 768, 28, 20]
        x = x.flatten(2) # [1, 768, 600] // [1, 768, 256]
        x = x.transpose(-1, -2) # [1, 600, 768] // [1, 256, 768]

        # Case1: Original encoding
        # positional_encoding = self.positional_encoding #.to(x.device)

        # Case2: sinusoidal encoding
        _,i,j = x.shape
        positional_encoding = self.create_sinusoidal_embeddings(n_patches=i,hidden_size=j).to(x.device)

        embeddings = x + positional_encoding # [1, 560, 768] -> [1, 196, 768]
        embeddings = self.dropout(embeddings)
        return embeddings # // [1, 256, 768]

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states): # [1, 256, 768]
        attn_weights = [] 
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self,config, img_size, vis,in_channels=3,old = 1):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config,img_size=img_size,input_dim=in_channels,old = old)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, features

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias,dim2=None):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        #use_dropout= use_dropo
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ART_block(nn.Module):
    def __init__(self,config, input_dim, img_size=224, transformer = None):
        super(ART_block, self).__init__()
        self.transformer = transformer
        self.config = config
        ngf = 64
        mult = 4
        use_bias = False
        norm_layer = nn.BatchNorm2d
        padding_type = 'reflect'
        if self.transformer:
            # Downsample
            model = [nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3,
                               stride=2, padding=1, bias=use_bias),
                     norm_layer(ngf * 8),
                     nn.ReLU(True)]
            model += [nn.Conv2d(ngf * 8, 1024, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(1024),
                      nn.ReLU(True)]
            setattr(self, 'downsample', nn.Sequential(*model))
            #Patch embedings
            self.embeddings = Embeddings(config, img_size=img_size, input_dim=input_dim)
            # Upsampling block
            model = [nn.ConvTranspose2d(self.config.hidden_size, ngf * 8,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                     norm_layer(ngf * 8),
                     nn.ReLU(True)]
            model += [nn.ConvTranspose2d(ngf * 8, ngf * 4,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(ngf * 4),
                      nn.ReLU(True)]
            setattr(self, 'upsample', nn.Sequential(*model))
            #Channel compression
            self.cc = channel_compression(ngf * 8, ngf * 4)
        # Residual CNN
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False,
                             use_bias=use_bias)]
        setattr(self, 'residual_cnn', nn.Sequential(*model))

    def forward(self, x):
        if self.transformer:
            # downsample
            down_sampled = self.downsample(x) # [1, 256, 64, 64] -> [1, 1024, 16, 16]
            # embed
            embedding_output = self.embeddings(down_sampled) # [1, 256, 768]
            # feed to transformer
            transformer_out, attn_weights = self.transformer(embedding_output) # [1, 256, 768]
            B, n_patch, hidden = transformer_out.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
            _, _, h, w = down_sampled.shape
            # h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch)) # 16, 16 #TODO: 이거 그냥 위에 down_sampled에서 크기 받아와서 나누도록
            transformer_out = transformer_out.permute(0, 2, 1) #  [1, 256, 768] -> [1, 768, 256]
            transformer_out = transformer_out.contiguous().view(B, hidden, h, w) # [1, 768, 16, 16]
            # upsample transformer output
            transformer_out = self.upsample(transformer_out) # [1, 256, 64, 64]
            # concat transformer output and resnet output
            x = torch.cat([transformer_out, x], dim=1) # [1, 512, 64, 64]
            # channel compression
            x = self.cc(x) # [1, 256, 64, 64]
        # residual CNN
        x = self.residual_cnn(x) # [1, 256, 64, 64]
        return x

########Generator############
class ResViT(nn.Module):
    def __init__(self,config, input_dim, output_dim=3, ngf=64, img_size=224, vis=False):
        super(ResViT, self).__init__()
        self.transformer_encoder = Encoder(config, vis)
        self.config = config
        use_bias = False
        norm_layer = nn.BatchNorm2d
        padding_type = 'reflect'
        mult = 4

        ############################################################################################
        # Layer1-Encoder1
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_dim, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        setattr(self, 'encoder_1', nn.Sequential(*model))
        ############################################################################################
        # Layer2-Encoder2
        n_downsampling = 2
        model = []
        i = 0
        mult = 2 ** i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_2', nn.Sequential(*model))
        ############################################################################################
        # Layer3-Encoder3
        model = []
        i = 1
        mult = 2 ** i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_3', nn.Sequential(*model))
        ####################################ART Blocks##############################################
        mult = 4
        self.art_1 = ART_block(self.config, input_dim, img_size,transformer = self.transformer_encoder)
        self.art_2 = ART_block(self.config, input_dim, img_size,transformer = None)
        self.art_3 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_4 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_5 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_6 = ART_block(self.config, input_dim, img_size,transformer = self.transformer_encoder)
        self.art_7 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_8 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_9 = ART_block(self.config, input_dim, img_size, transformer=None)
        ############################################################################################
        # Layer13-Decoder1
        n_downsampling = 2
        i = 0
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(int(ngf * mult / 2)),
                 nn.ReLU(True)]
        setattr(self, 'decoder_1', nn.Sequential(*model))
        ############################################################################################
        # Layer14-Decoder2
        i = 1
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(int(ngf * mult / 2)),
                 nn.ReLU(True)]
        setattr(self, 'decoder_2', nn.Sequential(*model))
        ############################################################################################
        # Layer15-Decoder3
        model = []
        model = [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_dim, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        setattr(self, 'decoder_3', nn.Sequential(*model))

    ############################################################################################
        
    def forward(self, x):
        # Pass input through cnn encoder of ResViT
        x = self.encoder_1(x) # [1, 1, 434, 308] ->
        x = self.encoder_2(x)
        x = self.encoder_3(x)

        #Information Bottleneck
        x = self.art_1(x) # [1, 256, 109, 77] -> 
        x = self.art_2(x)
        x = self.art_3(x)
        x = self.art_4(x)
        x = self.art_5(x)
        x = self.art_6(x)
        x = self.art_7(x)
        x = self.art_8(x)
        x = self.art_9(x)

        #decoder
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        return x


    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            if self.config.name == 'b16':
                self.art_1.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
                self.art_1.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

                self.art_6.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
                self.art_6.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer_encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer_encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.art_1.embeddings.positional_encoding
            if posemb.size() == posemb_new.size():
                self.art_1.embeddings.positional_encoding.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.art_1.embeddings.positional_encoding1.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.art_1.embeddings.positional_encoding.copy_(np2th(posemb))

            #############
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.art_6.embeddings.positional_encoding
            if posemb.size() == posemb_new.size():
                self.art_6.embeddings.positional_encoding.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.art_6.embeddings.positional_encoding.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.art_6.embeddings.positional_encoding.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer_encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)


class Res_CNN(nn.Module):
    def __init__(self, config, input_dim, img_size=224, output_dim=3, vis=False):
        super(Res_CNN, self).__init__()
        self.config = config
        output_nc = output_dim
        ngf = 64
        use_bias = False
        norm_layer = nn.BatchNorm2d
        padding_type = 'reflect'
        mult = 4

        ############################################################################################
        # Layer1-Encoder1
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_dim, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        setattr(self, 'encoder_1', nn.Sequential(*model))
        ############################################################################################
        # Layer2-Encoder2
        n_downsampling = 2
        model = []
        i = 0
        mult = 2 ** i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_2', nn.Sequential(*model))
        ############################################################################################
        # Layer3-Encoder3
        model = []
        i = 1
        mult = 2 ** i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_3', nn.Sequential(*model))
        ####################################ART Blocks##############################################
        mult = 4
        self.art_1 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_2 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_3 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_4 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_5 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_6 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_7 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_8 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_9 = ART_block(self.config, input_dim, img_size, transformer=None)
        ############################################################################################
        # Layer13-Decoder1
        n_downsampling = 2
        i = 0
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(int(ngf * mult / 2)),
                 nn.ReLU(True)]
        setattr(self, 'decoder_1', nn.Sequential(*model))
        ############################################################################################
        # Layer14-Decoder2
        i = 1
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(int(ngf * mult / 2)),
                 nn.ReLU(True)]
        setattr(self, 'decoder_2', nn.Sequential(*model))
        ############################################################################################
        # Layer15-Decoder3
        model = []
        model = [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_dim, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        setattr(self, 'decoder_3', nn.Sequential(*model))

    ############################################################################################

    def forward(self, x):
        # Encoder
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        x = self.encoder_3(x)
        # Information bottleneck
        x = self.art_1(x)
        x = self.art_2(x)
        x = self.art_3(x)
        x = self.art_4(x)
        x = self.art_5(x)
        x = self.art_6(x)
        x = self.art_7(x)
        x = self.art_8(x)
        x = self.art_9(x)
        # Decoder
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        return x




class channel_compression(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(channel_compression, self).__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
          self.skip = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))
        else:
          self.skip = None

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.block(x)
        out += (x if self.skip is None else self.skip(x))
        out = F.relu(out)
        return out
    
####################################################################################

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1


    config.pretrained_path = './model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    config.activation = 'softmax'
    return config


def get_resvit_b16_config():
    """Returns the residual ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.name = 'b16'

    config.pretrained_path = './model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'

    return config

def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1


    config.pretrained_path = './model/vit_checkpoint/imagenet21k/ViT-L_16.npz'
    return config


def get_resvit_l16_config():
    """Returns the residual ViT-L/16 configuration. customized """
    config = get_l16_config()
    config.patches.grid = (16, 16)
    
    config.name = 'l16'
    config.pretrained_path = './model/vit_checkpoint/imagenet21k/ViT-L_16.npz'
    return config



def get_b14_config():
    """Returns the ViT-B/14 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1


    config.pretrained_path = './model/vit_checkpoint/imagenet21k/ViT-B_14.npz'
    config.patch_size = 14

    config.activation = 'softmax'
    return config

def get_resvit_b14_config():
    """Returns the residual ViT-B/14 configuration."""
    config = get_b14_config()
    config.patches.grid = (14, 14)
    config.name = 'b14'

    config.pretrained_path = './model/vit_checkpoint/imagenet21k/R50+ViT-B_14.npz'

    return config


def get_b8_config():
    """Returns the ViT-B/8 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8, 8)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1


    config.pretrained_path = './model/vit_checkpoint/imagenet21k/ViT-B_8.npz'
    config.patch_size = 8

    config.activation = 'softmax'
    return config

def get_resvit_b8_config():
    """Returns the residual ViT-B/8 configuration."""
    config = get_b8_config()
    config.patches.grid = (8, 8)
    config.name = 'b8'

    config.pretrained_path = './model/vit_checkpoint/imagenet21k/R50+ViT-B_8.npz'

    return config

CONFIGS = {
    'ViT-B_16': get_b16_config(),
    'ViT-L_16': get_l16_config(),
    'ViT-B-14': get_b14_config(),
    'ViT-B-8': get_b8_config(),
    'Res-ViT-B_16': get_resvit_b16_config(),
    'Res-ViT-L_16': get_resvit_l16_config(),
    'Res-ViT-B_14': get_resvit_b14_config(),
    'Res-ViT-B_8': get_resvit_b8_config(),
}
