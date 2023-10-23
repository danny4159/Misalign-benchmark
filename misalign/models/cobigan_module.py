from typing import Any
import itertools

import torch
from misalign.models.components.networks import GANLoss, ImagePool
from misalign.models.base_module import BaseModule
from misalign import utils
from torchvision import models
from misalign.models.components.contextual_loss import Contextual_Loss, PatchContextualLoss  # this is the CX loss

log = utils.get_pylogger(__name__)

gray2rgb = lambda x : torch.cat((x, x, x), dim=1)
class COBIGANModule(BaseModule):
    """from: Zoom to Learn, Learn to Zoom
    """
    def __init__(
        self,
        netG_A: torch.nn.Module,
        netG_B: torch.nn.Module,
        netD_A: torch.nn.Module,
        netD_B: torch.nn.Module,
        optimizer,
        params,
        **kwargs: Any
    ):
        super().__init__()
        # assign generator
        self.netG_A = netG_A
        self.netG_B = netG_B
        # assign discriminator
        self.netD_A = netD_A
        self.netD_B = netD_B

        self.automatic_optimization = False # perform manual
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.params = params
        self.optimizer = optimizer

        # assign contextual loss

        self.patch_loss = PatchContextualLoss()

        style_feat_layers = {
            "conv_2_2": 1.0,
            "conv_3_2": 1.0,
            "conv_4_2": 1.0
        }

        self.style_loss = Contextual_Loss(style_feat_layers, cobi=True)
        # Image Pool
        self.fake_AB_pool = ImagePool(params.pool_size)

        # loss function
        self.no_lsgan = False
        self.criterionGAN = GANLoss(use_lsgan=not self.no_lsgan)
        self.criterionL1 = torch.nn.L1Loss()

        # Image Pool
        self.fake_A_pool = ImagePool(params.pool_size)
        self.fake_B_pool = ImagePool(params.pool_size)

    def backward_G(self, real_a, real_b, fake_a, fake_b, lambda_patch, lambda_style):
        # GAN loss D_A(G_A(A))
        pred_fake = self.netD_A(fake_b)
        loss_G_A = self.criterionGAN(pred_fake, True)
        # GAN loss D_B(G_B(B))
        pred_fake = self.netD_B(fake_a)
        loss_G_B = self.criterionGAN(pred_fake, True)

        loss_GAN = (loss_G_A + loss_G_B)

        #Content loss
        loss_patch_A = self.patch_loss(real_a, fake_a)
        loss_patch_B = self.patch_loss(real_b, fake_b)
        loss_patch = (loss_patch_A + loss_patch_B) * lambda_patch

        loss_style_A = self.style_loss(real_a, fake_a)
        loss_style_B = self.style_loss(real_b, fake_b)
        loss_style = (loss_style_A + loss_style_B) * lambda_style
        
        loss_G = loss_GAN + loss_patch + loss_style

        return loss_G

    def training_step(self, batch: Any, batch_idx: int):

        optimizer_G, optimizer_D_A, optimizer_D_B = self.optimizers()
        real_a, real_b, fake_a, fake_b = self.model_step(batch)

        with optimizer_G.toggle_model():
            loss_G = self.backward_G(real_a, real_b, fake_a, fake_b, self.params.lambda_patch, self.params.lambda_style)
            self.manual_backward(loss_G)
            optimizer_G.step()
            optimizer_G.zero_grad()

        with optimizer_D_A.toggle_model():
            loss_D_A = self.backward_D_A(real_b, fake_b)
            self.manual_backward(loss_D_A)
            optimizer_D_A.step()
            optimizer_D_A.zero_grad()

        with optimizer_D_B.toggle_model():
            loss_D_B = self.backward_D_B(real_a, fake_a)
            self.manual_backward(loss_D_B)
            optimizer_D_B.step()
            optimizer_D_B.zero_grad()
            
        self.loss_G = loss_G.detach() * 0.1 + self.loss_G * 0.9
        self.log("G_loss", self.loss_G, prog_bar=True)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer_G = self.hparams.optimizer(params=itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()))
        optimizer_D_A = self.hparams.optimizer(params=self.netD_A.parameters())
        optimizer_D_B = self.hparams.optimizer(params=self.netD_B.parameters())

        return optimizer_G, optimizer_D_A, optimizer_D_B
