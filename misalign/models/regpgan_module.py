from typing import Any
import itertools

import torch
from misalign.models.components.networks import GANLoss, ImagePool
from misalign.models.base_module import BaseModule
from misalign import utils
from torchvision import models
from misalign.models.components.transformer import *

log = utils.get_pylogger(__name__)

gray2rgb = lambda x: torch.cat((x, x, x), dim=1)


class RegGANModule(BaseModule):
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

        self.automatic_optimization = False  # perform manual
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.params = params
        self.optimizer = optimizer

        # assign perceptual network
        self.vgg = VGG16()
        self.netR_A_B = Reg(1, 1)  # Register fake B to real B
        self.netR_B_A = Reg(1, 1)  # Register fake A to real A
        self.spatial_transform = Transformer_2D()

        # Image Pool
        self.fake_AB_pool = ImagePool(params.pool_size)

        # loss function
        self.no_lsgan = False
        self.criterionGAN = GANLoss(use_lsgan=not self.no_lsgan)
        self.criterionL1 = torch.nn.L1Loss()

        # Image Pool
        self.fake_A_pool = ImagePool(params.pool_size)
        self.fake_B_pool = ImagePool(params.pool_size)

    def backward_G(self, real_a, real_b, fake_a, fake_b, lambda_l1, lambda_vgg, lambda_smooth):
        # GAN loss D_A(G_A(A))
        pred_fake = self.netD_A(fake_b)
        loss_G_A = self.criterionGAN(pred_fake, True)
        # GAN loss D_B(G_B(B))
        pred_fake = self.netD_B(fake_a)
        loss_G_B = self.criterionGAN(pred_fake, True)

        loss_GAN = (loss_G_A + loss_G_B) * 0.5
        # G(A) = B

        Trans_B = self.netR_B_A(fake_a, real_a)
        reg_fake_a = self.spatial_transform(fake_a, Trans_B)

        Trans_A = self.netR_A_B(fake_b, real_b)
        reg_fake_b = self.spatial_transform(fake_b, Trans_A)

        loss_L1 = (
            self.criterionL1(reg_fake_a, real_a) * lambda_l1
            + self.criterionL1(reg_fake_b, real_b) * lambda_l1
        ) * 0.5

        # Perceptual Loss
        VGG_real_A = self.vgg(
            real_a.expand(
                [int(real_a.size()[0]), 3, int(real_a.size()[2]), int(real_a.size()[3])]
            )
        )[0]
        VGG_fake_A = self.vgg(
            reg_fake_a.expand(
                [int(real_a.size()[0]), 3, int(real_a.size()[2]), int(real_a.size()[3])]
            )
        )[0]

        VGG_real_B = self.vgg(
            real_b.expand(
                [int(real_b.size()[0]), 3, int(real_b.size()[2]), int(real_b.size()[3])]
            )
        )[0]
        VGG_fake_B = self.vgg(
            reg_fake_b.expand(
                [int(real_b.size()[0]), 3, int(real_b.size()[2]), int(real_b.size()[3])]
            )
        )[0]

        VGG_loss = (
            self.criterionL1(VGG_fake_A, VGG_real_A) * lambda_vgg
            + self.criterionL1(VGG_fake_B, VGG_real_B) * lambda_vgg
        ) * 0.5

        # combined loss
        loss_G = (
            loss_GAN
            + loss_L1
            + VGG_loss
            + (smoothing_loss(Trans_A) + smoothing_loss(Trans_B)) * lambda_smooth
        )

        return loss_G

    def training_step(self, batch: Any, batch_idx: int):
        optimizer_G, optimizer_D_A, optimizer_D_B = self.optimizers()
        real_a, real_b, fake_a, fake_b = self.model_step(batch)

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

        with optimizer_G.toggle_model():
            loss_G = self.backward_G(
                real_a,
                real_b,
                fake_a,
                fake_b,
                self.params.lambda_l1,
                self.params.lambda_vgg,
                self.params.lambda_smooth
            )

            self.manual_backward(loss_G)
            self.clip_gradients(
                optimizer_G, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
            )

            optimizer_G.step()
            optimizer_G.zero_grad()
        self.loss_G = loss_G.detach() * 0.1 + self.loss_G * 0.9
        self.log("G_loss", self.loss_G, prog_bar=True)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """

        optimizer_G = self.hparams.optimizer(
            params=itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(), self.netR_A_B.parameters(), self.netR_B_A.parameters())
        )
        optimizer_D_A = self.hparams.optimizer(params=self.netD_A.parameters())
        optimizer_D_B = self.hparams.optimizer(params=self.netD_B.parameters())

        return optimizer_G, optimizer_D_A, optimizer_D_B


# Extracting VGG feature maps before the 2nd maxpooling layer
class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        for x in range(4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.stage1(X)
        h_relu2 = self.stage2(h_relu1)
        return h_relu2


if __name__ == "__main__":
    _ = RegGANModule(None, None, None)
