from typing import Any, Optional
import itertools

import torch
from misalign.models.components.perceptual import PerceptualLoss
from misalign.models.components.networks import GANLoss, ImagePool
from misalign.models.base_module import BaseModule
from misalign import utils
from misalign.models.components.contextual_loss import Contextual_Loss

log = utils.get_pylogger(__name__)

gray2rgb = lambda x : torch.cat((x, x, x), dim=1)
class UnitModule(BaseModule):

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

        # loss function
        self.no_lsgan = False
        self.criterionGAN = GANLoss(use_lsgan=not self.no_lsgan)
        self.criterionRecon = torch.nn.L1Loss()
        self.criterionPerceptual = PerceptualLoss(spatial_dims=2, network_type="radimagenet_resnet50")
        style_feat_layers = {
            "conv_2_2": 1.0,
            "conv_3_2": 1.0,
            "conv_4_2": 1.0
        }
        self.contextual_loss = Contextual_Loss(style_feat_layers)

        # Image Pool
        self.fake_A_pool = ImagePool(params.pool_size)
        self.fake_B_pool = ImagePool(params.pool_size)

    def __compute_kl(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def backward_G(self, real_a, real_b, hidden_a, hidden_b, recon_a, recon_b, recon_b_a, recon_a_b, hidden_b_a, hidden_a_b, recon_a_b_a, recon_b_a_b, lambda_content, lambda_kl, lambda_cycle, lambda_kl_cross, lambda_perceptual, lambda_contextual):

        # loss GAN
        pred_fake = self.netD_A(recon_b_a)
        loss_G_adv_a = self.criterionGAN(pred_fake, True)
        pred_fake = self.netD_B(recon_a_b)
        loss_G_adv_b = self.criterionGAN(pred_fake, True)
        loss_GAN = (loss_G_adv_a + loss_G_adv_b)
        # loss content (recon)
        loss_G_recon_a = self.criterionRecon(recon_a, real_a)
        loss_G_recon_b = self.criterionRecon(recon_b, real_b)
        loss_content = (loss_G_recon_a + loss_G_recon_b) * lambda_content
        # loss kl
        loss_G_recon_kl_a = self.__compute_kl(hidden_a)
        loss_G_recon_kl_b = self.__compute_kl(hidden_b)
        loss_kl = (loss_G_recon_kl_a + loss_G_recon_kl_b) * lambda_kl
        # loss cycle
        loss_G_cyc_recon_a = self.criterionRecon(recon_a_b_a, real_a)
        loss_G_cyc_recon_b = self.criterionRecon(recon_b_a_b, real_b)
        loss_cycle = (loss_G_cyc_recon_a + loss_G_cyc_recon_b) * lambda_cycle
        # loss kl cross
        loss_G_recon_kl_ab = self.__compute_kl(hidden_a_b)
        loss_G_recon_kl_ba = self.__compute_kl(hidden_b_a)
        loss_kl_cross = (loss_G_recon_kl_ab + loss_G_recon_kl_ba) * lambda_kl_cross
        # perceptual loss (extra)
        loss_G_vgg_a = self.criterionPerceptual(recon_b_a, real_b)
        loss_G_vgg_b = self.criterionPerceptual(recon_a_b, real_a)
        loss_perceptual = (loss_G_vgg_a + loss_G_vgg_b) * lambda_perceptual
        # contextual loss (extra)
        loss_contextual_a = torch.mean(self.contextual_loss(real_a, recon_b_a))
        loss_contextual_b = torch.mean(self.contextual_loss(real_b, recon_a_b))
        loss_contextual = (loss_contextual_a + loss_contextual_b) * lambda_contextual

        # total loss
        loss_G = loss_GAN + loss_content + loss_kl + loss_cycle + loss_kl_cross + loss_perceptual + loss_contextual
        return loss_G

    def model_step_unit(self, batch: Any):
        real_a, real_b = batch

        hidden_a, noise_a = self.netG_A.encode(real_a)
        hidden_b, noise_b = self.netG_B.encode(real_b)
        # decode (within domain)
        recon_a = self.netG_A.decode(hidden_a + noise_a)
        recon_b = self.netG_B.decode(hidden_b + noise_b)
        # decode (cross domain)
        recon_b_a = self.netG_A.decode(hidden_b + noise_b)
        recon_a_b = self.netG_B.decode(hidden_a + noise_a)
        # encode again
        hidden_b_a, noise_b_a = self.netG_A.encode(recon_b_a)
        hidden_a_b, noise_a_b = self.netG_B.encode(recon_a_b)
        # decode again (if needed)
        recon_a_b_a = self.netG_A.decode(hidden_a_b + noise_a_b)
        recon_b_a_b = self.netG_B.decode(hidden_b_a + noise_b_a)

        return real_a, real_b, hidden_a, hidden_b, recon_a, recon_b, recon_b_a, recon_a_b, hidden_b_a, hidden_a_b, recon_a_b_a, recon_b_a_b

    def training_step(self, batch: Any, batch_idx: int):

        optimizer_G, optimizer_D_A, optimizer_D_B = self.optimizers()
        real_a, real_b, hidden_a, hidden_b, recon_a, recon_b, recon_b_a, recon_a_b, hidden_b_a, hidden_a_b, recon_a_b_a, recon_b_a_b = self.model_step_unit(batch)

        with optimizer_G.toggle_model():
            loss_G = self.backward_G(real_a, real_b, hidden_a, hidden_b, recon_a, recon_b, recon_b_a, recon_a_b, hidden_b_a, hidden_a_b, recon_a_b_a, recon_b_a_b, 
                                     self.params.lambda_content, self.params.lambda_kl, self.params.lambda_cycle, self.params.lambda_kl_cross, self.params.lambda_perceptual, self.params.lambda_contextual)
            self.manual_backward(loss_G)
            optimizer_G.step()
            optimizer_G.zero_grad()

        with optimizer_D_A.toggle_model():        
            loss_D_A = self.backward_D_A(real_a, recon_b_a)
            self.manual_backward(loss_D_A)
            optimizer_D_A.step()
            optimizer_D_A.zero_grad()

        with optimizer_D_B.toggle_model():
            loss_D_B = self.backward_D_B(real_b, recon_a_b)
            self.manual_backward(loss_D_B)
            optimizer_D_B.step()
            optimizer_D_B.zero_grad()

        # update and log metrics
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
    _ = UnitModule(None, None, None)