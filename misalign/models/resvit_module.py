from typing import Any
import itertools

import torch
from misalign.models.components.networks import GANLoss, ImagePool
from misalign.models.base_module import BaseModule
from misalign import utils
from torchvision import models

log = utils.get_pylogger(__name__)

gray2rgb = lambda x : torch.cat((x, x, x), dim=1)
class ResViTModule(BaseModule):
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

        # Image Pool
        self.fake_AB_pool = ImagePool(params.pool_size)
        self.fake_BA_pool = ImagePool(params.pool_size)

        # loss function
        self.no_lsgan = False
        self.criterionGAN = GANLoss(use_lsgan=not self.no_lsgan)
        self.criterionL1 = torch.nn.L1Loss()

    def backward_G(self, real_a, real_b, fake_a, fake_b, lambda_l1):

        fake_ab = torch.cat((real_a, fake_b), 1) # [12, 1, 256, 256] + [12, 1, 256, 256]
        pred_fake_ab = self.netD_B(fake_ab)
        loss_GAN_AB = self.criterionGAN(pred_fake_ab, True)

        fake_ba = torch.cat((real_b, fake_a), 1)
        pred_fake_ba = self.netD_A(fake_ba)
        loss_GAN_BA = self.criterionGAN(pred_fake_ba, True)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) * 0.5

        loss_L1 = (self.criterionL1(fake_a, real_a) + self.criterionL1(fake_b, real_b) * lambda_l1) * 0.5
       
        loss_G = loss_GAN + loss_L1

        return loss_G
    
    def backward_D_A(self, real_a, real_b, fake_a, fake_b):
        fake_ba = self.fake_BA_pool.query(torch.cat((real_b, fake_a), 1).data)
        pred_fake_ba = self.netD_A(fake_ba.detach())
        loss_D_A_fake = self.criterionGAN(pred_fake_ba, False)

        real_ba = torch.cat((real_b, real_a), 1)
        pred_real_ba = self.netD_A(real_ba)
        loss_D_A_real = self.criterionGAN(pred_real_ba, True)

        loss_D_A = (loss_D_A_fake + loss_D_A_real) * 0.5
        
        return loss_D_A
    
    def backward_D_B(self, real_a, real_b, fake_a, fake_b):
        fake_ab = self.fake_AB_pool.query(torch.cat((real_a,fake_b), 1).data)
        pred_fake_ab = self.netD_B(fake_ab.detach())
        loss_D_B_fake = self.criterionGAN(pred_fake_ab, False)

        real_ab = torch.cat((real_a, real_b), 1)
        pred_real_ab = self.netD_B(real_ab)
        loss_D_B_real = self.criterionGAN(pred_real_ab, True)

        loss_D_B = (loss_D_B_fake + loss_D_B_real) * 0.5
    
        return loss_D_B



    def training_step(self, batch: Any, batch_idx: int):

        optimizer_G, optimizer_D_A, optimizer_D_B = self.optimizers()
        real_a, real_b, fake_a, fake_b = self.model_step(batch)

        with optimizer_G.toggle_model():
            loss_G = self.backward_G(real_a, real_b, fake_a, fake_b, self.params.lambda_l1)
            self.manual_backward(loss_G)
            self.clip_gradients(optimizer_G, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            optimizer_G.step()
            optimizer_G.zero_grad()

        with optimizer_D_A.toggle_model():
            loss_D_A = self.backward_D_A(real_a, real_b, fake_a, fake_b)
            self.manual_backward(loss_D_A)
            self.clip_gradients(optimizer_D_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            optimizer_D_A.step()
            optimizer_D_A.zero_grad()

        with optimizer_D_B.toggle_model():
            loss_D_B = self.backward_D_B(real_a, real_b, fake_a, fake_b)
            self.manual_backward(loss_D_B)
            self.clip_gradients(optimizer_D_B, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            optimizer_D_B.step()
            optimizer_D_B.zero_grad()
        self.log("G_loss", loss_G.detach(), prog_bar=True)

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


if __name__ == "__main__":
    _ = ResViTModule(None, None, None)
