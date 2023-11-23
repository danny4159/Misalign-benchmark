from typing import Any
import itertools

import torch
from misalign.models.components.networks import GANLoss, ImagePool
from misalign.models.base_module import BaseModule
from misalign import utils
from torchvision import models
from misalign.models.components.contextual_loss import Contextual_Loss # this is the CX loss

log = utils.get_pylogger(__name__)

gray2rgb = lambda x : torch.cat((x, x, x), dim=1)
class DAMModule(BaseModule):

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

        self.style_loss = Contextual_Loss(style_feat_layers)

        # loss function
        self.criterionL1 = torch.nn.L1Loss()


    def backward_G(self, real_a, real_b, fake_a, fake_b, lambda_style, lambda_cycle_a, lambda_cycle_b, lambda_cycle_style):
        
        ## Cycle loss
        # MR > CT > MR
        # rec_a = self.netG_A(fake_b, fake_a)
        rec_a = self.netG_A(fake_b, real_a)
        loss_cycle_A = self.criterionL1(real_a, rec_a) * lambda_cycle_a
        # CT > MR > CT
        # rec_b = self.netG_B(fake_a, fake_b) 
        rec_b = self.netG_B(fake_a, real_b) 
        loss_cycle_B = self.criterionL1(real_b, rec_b) * lambda_cycle_b
        
        ## Contextual loss
        loss_style_A = self.style_loss(real_a, fake_a)
        loss_style_B = self.style_loss(real_b, fake_b)
        loss_style = (loss_style_A + loss_style_B) * lambda_style

        ## Contextual loss
        loss_cycle_style_A = self.style_loss(rec_b, real_b)
        loss_cycle_style_B = self.style_loss(rec_a, real_a)
        loss_cycle_style = (loss_cycle_style_A + loss_cycle_style_B) * lambda_cycle_style

        loss_G = loss_style + loss_cycle_A + loss_cycle_B + loss_cycle_style

        return loss_G

    def training_step(self, batch: Any, batch_idx: int):

        optimizer_G = self.optimizers()
        real_a, real_b, fake_a, fake_b = self.model_step(batch)

        with optimizer_G.toggle_model():
            loss_G = self.backward_G(real_a, real_b, fake_a, fake_b, self.params.lambda_style, self.params.lambda_cycle_a, self.params.lambda_cycle_b, self.params.lambda_cycle_style)
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
