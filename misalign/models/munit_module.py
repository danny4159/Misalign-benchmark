from typing import Any, Optional
import itertools

import torch
from misalign.models.components.perceptual import PerceptualLoss
from misalign.models.components.networks import GANLoss, ImagePool
from misalign.models.base_module_for_fid import BaseModule
# from misalign.models.base_module import BaseModule # TODO: FID 평가위해 잠깐 바꾼거
from misalign import utils
from misalign.models.components.contextual_loss import Contextual_Loss

log = utils.get_pylogger(__name__)

gray2rgb = lambda x : torch.cat((x, x, x), dim=1)
class MunitModule(BaseModule): # FID 평가위해 잠깐 바꾼거
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

        # fix the noise used in sampling
        valid_size = 1 # TODO: hyperparameters['valid_size']
        self.style_dim = 8 # TODO:  hyperparameters['gen']['style_dim']
        self.s_a = torch.randn(valid_size, self.style_dim, 1, 1).cuda(params['devices'][0])
        self.s_b = torch.randn(valid_size, self.style_dim, 1, 1).cuda(params['devices'][0])

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

    def backward_G(self, real_a, real_b, s_a, s_b, c_a, s_a_prime, c_b, s_b_prime, x_a_recon, x_b_recon, x_ba, x_ab, c_b_recon, s_a_recon, c_a_recon, s_b_recon, x_aba, x_bab, lambda_image, lambda_style, lambda_content, lambda_cycle, lambda_perceptual, lambda_contextual):

        # loss GAN
        pred_fake = self.netD_A(x_ba)
        loss_G_adv_a = self.criterionGAN(pred_fake, True)
        pred_fake = self.netD_B(x_ab)
        loss_G_adv_b = self.criterionGAN(pred_fake, True)
        loss_GAN = (loss_G_adv_a + loss_G_adv_b)
        # loss recon
        loss_gen_recon_x_a = self.criterionRecon(x_a_recon, real_a)
        loss_gen_recon_x_b = self.criterionRecon(x_b_recon, real_b)
        loss_image = (loss_gen_recon_x_a + loss_gen_recon_x_b) * lambda_image

        loss_gen_recon_s_a = self.criterionRecon(s_a_recon, s_a)
        loss_gen_recon_s_b = self.criterionRecon(s_b_recon, s_b)
        loss_style = (loss_gen_recon_s_a + loss_gen_recon_s_b) * lambda_style

        loss_gen_recon_c_a = self.criterionRecon(c_a_recon, c_a)
        loss_gen_recon_c_b = self.criterionRecon(c_b_recon, c_b)
        loss_content = (loss_gen_recon_c_a+ loss_gen_recon_c_b) * lambda_content

        loss_gen_cycrecon_x_a = self.criterionRecon(x_aba, real_a)
        loss_gen_cycrecon_x_b = self.criterionRecon(x_bab, real_b)
        loss_cycle = (loss_gen_cycrecon_x_a + loss_gen_cycrecon_x_b) * lambda_cycle

        # domain-invariant perceptual loss
        loss_G_vgg_b = self.criterionPerceptual(x_ba, real_b)  
        loss_G_vgg_a = self.criterionPerceptual(x_ab, real_a)  
        loss_perceptual = (loss_G_vgg_a + loss_G_vgg_b) * lambda_perceptual
        
        # contextual loss (extra)
        loss_contextual_a = torch.mean(self.contextual_loss(x_ba, real_a))
        loss_contextual_b = torch.mean(self.contextual_loss(x_ab, real_b))
        loss_contextual = (loss_contextual_a + loss_contextual_b) * lambda_contextual

        # total loss 
        loss_G = loss_GAN + loss_image + loss_style + loss_content + loss_cycle + loss_perceptual + loss_contextual
        return loss_G

    def model_step_munit(self, batch: Any):
        real_a, real_b = batch
        s_a = torch.randn(real_a.size(0), self.style_dim, 1, 1).cuda()
        s_b = torch.randn(real_b.size(0), self.style_dim, 1, 1).cuda()
        # encode
        c_a, s_a_prime = self.netG_A.encode(real_a)
        c_b, s_b_prime = self.netG_B.encode(real_b)
        # decode (within domain)
        x_a_recon = self.netG_A.decode(c_a, s_a_prime)
        x_b_recon = self.netG_B.decode(c_b, s_b_prime)
        # decode (cross domain)
        x_ba = self.netG_A.decode(c_b, s_a)
        x_ab = self.netG_B.decode(c_a, s_b)
        # encode again
        c_b_recon, s_a_recon = self.netG_A.encode(x_ba)
        c_a_recon, s_b_recon = self.netG_B.encode(x_ab)
        # decode again (if needed)
        x_aba = self.netG_A.decode(c_a_recon, s_a_prime) #if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.netG_B.decode(c_b_recon, s_b_prime) #if hyperparameters['recon_x_cyc_w'] > 0 else None

        return real_a, real_b, s_a, s_b, c_a, s_a_prime, c_b, s_b_prime, x_a_recon, x_b_recon, x_ba, x_ab, c_b_recon, s_a_recon, c_a_recon, s_b_recon, x_aba, x_bab

    def training_step(self, batch: Any, batch_idx: int):
        optimizer_G, optimizer_D_A, optimizer_D_B = self.optimizers()
        real_a, real_b, s_a, s_b, c_a, s_a_prime, c_b, s_b_prime, x_a_recon, x_b_recon, x_ba, x_ab, c_b_recon, s_a_recon, c_a_recon, s_b_recon, x_aba, x_bab = self.model_step_munit(batch)
        with optimizer_G.toggle_model():
            loss_G = self.backward_G(real_a, real_b, s_a, s_b, c_a, s_a_prime, c_b, s_b_prime, x_a_recon, x_b_recon, x_ba, x_ab, c_b_recon, s_a_recon, c_a_recon, s_b_recon, x_aba, x_bab, 
                                     self.params.lambda_image, self.params.lambda_style, self.params.lambda_content, self.params.lambda_cycle, self.params.lambda_perceptual, self.params.lambda_contextual)
            self.manual_backward(loss_G)
            optimizer_G.step()
            optimizer_G.zero_grad()

        with optimizer_D_A.toggle_model():        
            loss_D_A = self.backward_D_A(real_a, x_ba) # 이것도 고려
            self.manual_backward(loss_D_A)
            optimizer_D_A.step()
            optimizer_D_A.zero_grad()

        with optimizer_D_B.toggle_model():
            loss_D_B = self.backward_D_B(real_b, x_ab) # 이것도 고려
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
    _ = MunitModule(None, None, None)