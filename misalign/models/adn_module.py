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
class AdnModule(BaseModule):
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

    def backward_G(self, real_a, real_b, pred_ll, pred_lh, pred_hl, pred_hh, pred_hlh, pred_lhl, lambda_recon, lambda_self_reduction, lambda_art_consistency, lambda_gen_high, lambda_art_syn):

        # loss GAN
        pred_fake = self.netD_A(pred_lh)
        loss_G_adv_b = self.criterionGAN(pred_fake, True)
        pred_fake = self.netD_B(pred_hl)
        loss_G_adv_a = self.criterionGAN(pred_fake, True)
        loss_GAN = (loss_G_adv_a +loss_G_adv_b)

        # recon loss
        loss_gen_recon_a = self.criterionRecon(pred_ll, real_a)
        loss_gen_recon_b = self.criterionRecon(pred_hh, real_b)
        loss_recon = (loss_gen_recon_a + loss_gen_recon_b) * lambda_recon

        # self reduction loss
        loss_self_reduction = self.criterionRecon(pred_hlh, real_b) * lambda_self_reduction

        # artifact consistency loss
        loss_art_consistency = self.criterionRecon(real_a - pred_lh, pred_hl - real_b) * lambda_art_consistency

        # for paired dataset (default=0)
        loss_gen_high = self.criterionRecon(pred_lh, real_b) * lambda_gen_high
        loss_art_syn_loss = self.criterionRecon(pred_lhl, real_a) * lambda_art_syn

        loss_G = loss_GAN + loss_recon + loss_self_reduction + loss_art_consistency + loss_gen_high + loss_art_syn_loss

        return loss_G

    def model_step_adn(self, batch: Any):
        real_a, real_b = batch # real_a: low image,  real_b: high image
        pred_ll, pred_lh = self.netG_A.forward1(real_a)
        pred_hl, pred_hh = self.netG_A.forward2(real_a, real_b)
        pred_hlh = self.netG_A.forward_lh(pred_hl)
        pred_lhl = self.netG_A.forward_hl(pred_hl, pred_lh)
        
        return real_a, real_b, pred_ll, pred_lh, pred_hl, pred_hh, pred_hlh, pred_lhl

    def training_step(self, batch: Any, batch_idx: int):
        optimizer_G, optimizer_D_A, optimizer_D_B = self.optimizers()
        real_a, real_b, pred_ll, pred_lh, pred_hl, pred_hh, pred_hlh, pred_lhl = self.model_step_adn(batch)
        with optimizer_G.toggle_model():
            loss_G = self.backward_G(real_a, real_b, pred_ll, pred_lh, pred_hl, pred_hh, pred_hlh, pred_lhl, 
                                     self.params.lambda_recon, self.params.lambda_self_reduction, self.params.lambda_art_consistency, self.params.lambda_gen_high, self.params.lambda_art_syn)
            self.manual_backward(loss_G)
            optimizer_G.step()
            optimizer_G.zero_grad()

        with optimizer_D_A.toggle_model():        
            loss_D_A = self.backward_D_A(real_a, pred_hl) # 이것도 고려
            self.manual_backward(loss_D_A)
            optimizer_D_A.step()
            optimizer_D_A.zero_grad()

        with optimizer_D_B.toggle_model():
            loss_D_B = self.backward_D_B(real_b, pred_lh) # 이것도 고려
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
    _ = AdnModule(None, None, None)