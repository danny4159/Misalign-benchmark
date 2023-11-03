from typing import Any
import itertools

import torch
from misalign.models.components.networks import ImagePool
from misalign.models.components.networks_v2 import GANLoss
from misalign.models.base_module_A_to_B import BaseModule_A_to_B
from misalign import utils
import numpy as np

from misalign.models.components.transformer import Reg, Transformer_2D, smoothing_loss
from misalign.models.components.contextual_loss import (
    Contextual_Loss,
)  # this is the CX loss

from misalign.data.components.transforms_fly import dataset_IXI_FLY, dataset_synthRAD_FLY_RAM, dataset_synthRAD_FLY
from torch.utils.data import DataLoader

log = utils.get_pylogger(__name__)

gray2rgb = lambda x: torch.cat((x, x, x), dim=1)

import higher


class ProposedModule_A_to_B(BaseModule_A_to_B):
    # 1. Regist: for real_B or fake_A
    # 2. meta-learning: batch or spatial
    # 3. Feature-descriptor: VGG or ResNet50(RadNet) or self-supervised

    def __init__(
        self,
        netG_A: torch.nn.Module,
        netD_A: torch.nn.Module,
        optimizer,
        params,
        **kwargs: Any
    ):
        super().__init__()
        self.netG_A = netG_A
        self.netD_A = netD_A
        self.save_hyperparameters(logger=False, ignore=["netG_A", "netD_A"])
        self.automatic_optimization = False  # perform manual
        self.params = params

        if self.params.flag_register:
            self.netR = Reg(1, 1)  # use registration inside the network (RegGAN)
            self.spatial_transform = Transformer_2D()

        self.params = params
        self.optimizer = optimizer

        if self.params.flag_feature_descriptor == "VGG":
            log.info("Using VGG as feature descriptor")
            style_feat_layers = {"conv_2_2": 1.0, "conv_3_2": 1.0, "conv_4_2": 1.0}
            if self.params.flag_ctx:
                log.info("Using Contextual Loss")
                self.style_loss = Contextual_Loss(style_feat_layers, cobi=True)
            else:
                # regular Perceptual loss
                log.info("Using Perceptual Loss")
                self.style_loss = Contextual_Loss(style_feat_layers, l1=True)

        else:
            log.info("Using Resnet50 as feature descriptor (from RadNet)")
            style_feat_layers = {"maxpool": 1.0, "layer1": 1.0, "layer2": 1.0}
            if self.params.flag_ctx:
                log.info("Using Contextual Loss")
                self.style_loss = Contextual_Loss(
                    style_feat_layers, cobi=False, vgg=False
                )
            else:
                log.info("Using Perceptual Loss")
                self.style_loss = Contextual_Loss(style_feat_layers, l1=True, vgg=False)

        if self.params.flag_meta_learning:
            log.info("Using meta-learning")
            # self.data_meta = dataset_IXI_FLY( #TODO: 수정. 이거 if문으로 처리될 수 있도록
            self.data_meta = dataset_synthRAD_FLY_RAM(
                rand_crop=True,
                data_dir=self.params.dir_meta_learning,
                crop_size=80,
                reverse=self.params.reverse,
                aug=True,
                return_msk=True,
            )
            # item = self.data_meta[0]
            self.clean_dloader = DataLoader(self.data_meta, batch_size=32, shuffle=True)

        else:
            log.info("Not using meta-learning")

        # loss function
        self.criterionGAN = GANLoss(gan_mode="lsgan", reduce=False)

        self.criterionL1 = torch.nn.L1Loss(reduction="none")
        style_feat_layers = {"conv_2_2": 1.0, "conv_3_2": 1.0, "conv_4_2": 1.0}

        # Image Pool
        self.fake_B_pool = ImagePool(params.pool_size)

    def backward_G_train(
        self, real_a, real_b, fake_b, weight=None, reduce=False, gan=True
    ):
        loss_G = torch.zeros(1, device=self.device, requires_grad=True)

        if weight is not None:
            weight_spatial = weight # batch, 1 , width, heith
            weight_batch = torch.mean(weight_spatial, dim=(1, 2, 3),keepdim=True) # batch, 1 , 1, 1

        if self.params.flag_register:
            Trans_A = self.netR(fake_b, real_b)

            if weight is not None:
                loss_smooth = (
                    (smoothing_loss(Trans_A)) * weight_batch * self.params.lambda_smooth
                )
                loss_smooth = torch.mean(loss_smooth)
            else:
                loss_smooth = (
                    torch.mean(smoothing_loss(Trans_A)) * self.params.lambda_smooth
                )

            loss_G = loss_G + loss_smooth

            reg_fake_b = self.spatial_transform(fake_b, Trans_A)
            real_a_trans = self.spatial_transform(real_a, Trans_A)
            fake_b_trans = self.netG_A(real_a_trans)

            loss_reg_consistency = (
                torch.nn.functional.l1_loss(reg_fake_b, fake_b_trans)
                * self.params.lambda_reg_consistency
            )
            loss_G = loss_G + loss_reg_consistency
        else:
            reg_fake_b = fake_b

        loss_L1_B = self.criterionL1(reg_fake_b, real_b)
        if weight is not None:
            loss_L1_B = torch.mean(loss_L1_B * weight_spatial)
        else:
            loss_L1_B = torch.mean(loss_L1_B)

        loss_G = loss_G + loss_L1_B * self.params.lambda_l1

        if self.params.flag_GAN and gan == True:
            pred_fake = self.netD_A(fake_b)
            loss_G_A = self.criterionGAN(pred_fake, True)
            if weight is not None:
                loss_GAN = torch.mean(loss_G_A * weight_batch)
            else:
                loss_GAN = torch.mean(loss_G_A)
            loss_G = loss_G + loss_GAN
        if self.params.flag_ctx:
            loss_style_B = self.style_loss(fake_b, real_b)
        else:
            loss_style_B = self.style_loss(reg_fake_b, real_b)

        if weight is not None:
            loss_style_B = torch.mean(loss_style_B * weight_batch)
        else:
            loss_style_B = torch.mean(loss_style_B)

        loss_G = loss_G + loss_style_B * self.params.lambda_style

        return loss_G

    def determine_weight_LRE(self, real_a, real_b, meta_real_a, meta_real_b, mask=None):
        """
        weight: output과 label간에 loss를 통해 어느 pixel에 weight를 더 줄지 학습해
        """
        if mask is not None:
            mask = mask.float() # Meta mask 그냥 Mask있으면 Mask 씌우는거 학습은 없어
        else:
            mask = 1.0

        with higher.innerloop_ctx(self.netG_A, self.optimizer_G) as (
            meta_model,
            meta_opt,
        ):
            fake_b = meta_model(real_a)
            # Register the fake_b to real_b
            if self.params.flag_register:
                Trans_A = self.netR(fake_b, real_b)
                fake_b = self.spatial_transform(fake_b, Trans_A)
            if self.params.flag_meta_use_spatial:
                weight = torch.zeros(
                    real_a.size(), device=self.device, requires_grad=True # 0으로 초기화는 하지만 학습 가능한 매개변수
                )
            else:
                weight = torch.zeros(
                    real_a.shape[0], 1, 1, 1, device=self.device, requires_grad=True
                )

            _loss = self.criterionL1(real_b, fake_b)
            loss = torch.mean(_loss * weight)
            meta_opt.step(loss) #TODO: 이게 weight 학습에 어떤 영향을 주는지 사실 잘 모르겠어.

            meta_val_loss = self.criterionL1(
                (meta_real_b + 1) * mask, (meta_model(meta_real_a) + 1) * mask
            )
            meta_val_loss = torch.mean(meta_val_loss)

            eps_grad = torch.autograd.grad(meta_val_loss, weight)[ #아 이런씩으로 학습시킬수도 있구나?
                0
            ].detach()  # Gradient

        w_tilde = torch.clamp(-eps_grad, min=0)
        l1_norm = torch.sum(w_tilde)
        if l1_norm != 0:
            w_b = w_tilde / l1_norm
        else:
            w_b = w_tilde + 0.1 / l1_norm

        # scaling:
        if self.params.flag_meta_use_spatial:
            w_b = w_b * real_a.size(0) * real_a.size(2) * real_a.size(3)
        else:
            w_b = w_b * real_a.size(0)

        w_b = w_b.detach()  # w_a is the weight for a
        return w_b

    def training_step(self, batch: Any, batch_idx: int):
        optimizer_G, optimizer_D = self.optimizers()
        if len(batch) == 3:
            real_a, real_b, slice_idx = batch
        else:
            real_a, real_b = batch

        if self.current_epoch > -1 and self.params.flag_meta_learning:
            ################## Getting meta data ##################

            meta_real_a, meta_real_b, meta_msk = next(iter(self.clean_dloader))

            meta_real_a, meta_real_b, meta_msk = (
                meta_real_a.to(device=self.device),
                meta_real_b.to(device=self.device),
                meta_msk.to(device=self.device),
            )
            ########################################################
            # Meta-learning type: LRE
            if self.params.flag_use_mask:
                mask = meta_msk
            else:
                mask = None

            if self.params.meta_type == "LRE":
                w = self.determine_weight_LRE( 
                    real_a, real_b, meta_real_a, meta_real_b, mask=mask
                )
                optimizer_G.zero_grad()
            else:
                raise NotImplementedError
            del meta_real_a, meta_real_b, meta_msk  # remove this

        else:
            w = None

        with optimizer_G.toggle_model():
            fake_b = self.netG_A(real_a)
            loss_G = self.backward_G_train(
                real_a, real_b, fake_b, weight=w, reduce=True
            )
            self.manual_backward(loss_G)
            optimizer_G.step()
            optimizer_G.zero_grad()

        if self.params.flag_GAN:
            with optimizer_D.toggle_model():
                loss_D_A = self.backward_D_A(real_b, fake_b)
                if w is not None:
                    weight_batch = torch.mean(w, dim=(1, 2, 3),keepdim=True)
                    loss_D_A = torch.mean(loss_D_A * weight_batch)
                else:
                    loss_D_A = torch.mean(loss_D_A)
                self.manual_backward(loss_D_A)
                optimizer_D.step()
                optimizer_D.zero_grad()

        if w is None:
            self.loss_G = loss_G.detach() * 0.1 + self.loss_G * 0.9
            self.log("G_loss", self.loss_G, prog_bar=True)
        else:
            if torch.sum(w) > 1:
                self.loss_G = loss_G.detach() * 0.1 + self.loss_G * 0.9
                self.log("G_loss", self.loss_G, prog_bar=True)
        
        if self.params.flag_weight_saving:
            return {'w': w} # weight 저장을 위해. weight saving callback함수에서 사용

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """

        self.optimizer_G = self.hparams.optimizer(params=self.netG_A.parameters())

        if self.params.flag_register:
            optimizer_G = self.hparams.optimizer(
                params=itertools.chain(
                    self.netG_A.parameters(),
                    self.netR.parameters(),
                )
            )
        else:
            optimizer_G = self.hparams.optimizer(params=self.netG_A.parameters())
        optimizer_D = self.hparams.optimizer(params=self.netD_A.parameters())

        return optimizer_G, optimizer_D
