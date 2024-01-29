from typing import Any
import itertools

import torch
import torch.nn.functional as F
from misalign.models.components.networks import GANLoss, ImagePool
from misalign.models.base_module import BaseModule
from misalign import utils
from torchvision import models
from misalign.models.components.contextual_loss import Contextual_Loss # this is the CX loss
from misalign.models.components.scgan_loss import MINDLoss
from misalign.models.components.adaconv_loss import MomentMatchingStyleLoss

log = utils.get_pylogger(__name__)

gray2rgb = lambda x : torch.cat((x, x, x), dim=1)
class AdaConvModule(BaseModule):

    def __init__(
        self,
        netG_A: torch.nn.Module,
        netG_B: torch.nn.Module,
        optimizer,
        params,
        **kwargs: Any
    ):
        super().__init__()
        # assign generator
        self.netG_A = netG_A
        self.netG_B = netG_B

        self.automatic_optimization = False # perform manual
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.params = params
        self.optimizer = optimizer

        # assign contextual loss
        style_feat_layers = {
            # "conv_2_2": 1.0,
            # "conv_3_2": 1.0,
            # "conv_4_2": 1.0,
            "conv_4_4": 1.0
        }

        # loss function
        self.criterionContextual = Contextual_Loss(style_feat_layers)
        self.criterionMSE = torch.nn.MSELoss()
        self.criterionStyle = MomentMatchingStyleLoss()       
        self.criterionL1 = torch.nn.L1Loss()

    def backward_G(self, real_a, real_b, fake_a, fake_b, embed_net_a, embed_net_b):
        
        loss_content_A = self.criterionMSE(embed_net_a['content'][-1], embed_net_a['output'][-1])
        
        loss_contextual_A = self.criterionContextual(real_a, fake_a)

        loss_style_A = []
        for (style_features, output_features) in zip(embed_net_a['style'], embed_net_a['output']):
            loss_style_A.append(self.criterionStyle(style_features, output_features))
        loss_style_A = sum(loss_style_A)

        fake_b_a = self.netG_B(fake_b, real_a, return_embeddings=False)
        loss_cycle_A = self.criterionL1(real_a, fake_b_a)




        loss_content_B = self.criterionMSE(embed_net_b['content'][-1], embed_net_b['output'][-1])

        loss_contextual_B = self.criterionContextual(real_b, fake_b)

        loss_style_B = []
        for (style_features, output_features) in zip(embed_net_b['style'], embed_net_b['output']):
            loss_style_B.append(self.criterionStyle(style_features, output_features))
        loss_style_B = sum(loss_style_B)

        fake_a_b = self.netG_A(fake_a, real_b, return_embeddings=False)
        loss_cycle_B = self.criterionL1(real_b, fake_a_b)

        loss_G = (loss_content_A + loss_content_B) * 1 + (loss_contextual_A + loss_contextual_B) * 5 + (loss_style_A + loss_style_B) * 1 + (loss_cycle_A + loss_cycle_B) * 0

        return loss_G
    
    def model_step(self, batch: Any):
        real_a, real_b = batch
        fake_b, embed_net_a = self.netG_A(real_a, real_b, return_embeddings=True)
        fake_a, embed_net_b = self.netG_B(real_b, real_a, return_embeddings=True)
        return real_a, real_b, fake_a, fake_b, embed_net_a, embed_net_b
    
    def training_step(self, batch: Any, batch_idx: int):

        optimizer_G = self.optimizers()
        real_a, real_b, fake_a, fake_b, embed_net_a, embed_net_b = self.model_step(batch)

        with optimizer_G.toggle_model():
            loss_G = self.backward_G(real_a, real_b, fake_a, fake_b, embed_net_a, embed_net_b)
            self.manual_backward(loss_G)
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
        optimizer_G = self.hparams.optimizer(params=itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()))

        return optimizer_G

    def validation_step(self, batch: Any, batch_idx: int):
        real_A, real_B, fake_A, fake_B, _, _ = self.model_step(batch)
        loss = (F.l1_loss(real_A, fake_A) + F.l1_loss(real_B, fake_B)) / 2
        # Perform metric
        self.val_psnr_A(real_A, fake_A)
        self.val_ssim_A(real_A, fake_A)
        # self.val_lpips_A(gray2rgb(real_A), gray2rgb(fake_A)) #TODO: 잠깐 수정

        self.val_psnr_B(real_B, fake_B)
        self.val_ssim_B(real_B, fake_B)
        # self.val_lpips_B(gray2rgb(real_B), gray2rgb(fake_B))

        self.log("val/loss", loss.detach(), prog_bar=True)
        
    def test_step(self, batch: Any, batch_idx: int):
        real_A, real_B, fake_A, fake_B, _, _ = self.model_step(batch)

        loss = (F.l1_loss(real_A, fake_A) + F.l1_loss(real_B, fake_B)) / 2
        # Perform metric
        _psnr_A = self.test_psnr_A(real_A, fake_A)
        _ssim_A = self.test_ssim_A(real_A, fake_A)
        # _lpips_A = self.test_lpips_A(gray2rgb(real_A), gray2rgb(fake_A))

        _psnr_B = self.test_psnr_B(real_B, fake_B)
        _ssim_B = self.test_ssim_B(real_B, fake_B)
        # _lpips_B = self.test_lpips_B(gray2rgb(real_B), gray2rgb(fake_B))

        self.stats_psnr_A.update(_psnr_A)
        self.stats_psnr_B.update(_psnr_B)
        
        self.stats_ssim_A.update(_ssim_A)
        self.stats_ssim_B.update(_ssim_B)
        
        # self.stats_lpips_A.update(_lpips_A)
        # self.stats_lpips_B.update(_lpips_B) 

        self.log("test/loss", loss.detach(), prog_bar=True)