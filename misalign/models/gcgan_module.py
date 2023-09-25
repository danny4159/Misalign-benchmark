from typing import Any, Optional
import itertools

import torch
from misalign.models.components.networks import GANLoss, ImagePool
from misalign.models.base_module import BaseModule
from misalign import utils
from misalign.models.components.gcgan_loss import GradientCorrelation2d

log = utils.get_pylogger(__name__)

gray2rgb = lambda x : torch.cat((x, x, x), dim=1)
class GCCycleGANModule(BaseModule):
    """ Lightning module for Y Hiasa "Cross-modality image synthesis from unpaired data using cyclegan: Effects of gradient consistency loss and training data size
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
        self.automatic_optimization = False # perform manual
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.params = params
        self.optimizer = optimizer
        
        # assign generator
        self.netG_A = netG_A
        self.netG_B = netG_B
        
        self.netD_A = netD_A
        self.netD_B = netD_B
        
        # loss function
        
        self.no_lsgan = False
        self.criterionGAN = GANLoss(use_lsgan=not self.no_lsgan)
        
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdentity = torch.nn.L1Loss()
        self.criterionGradientCorrelation = GradientCorrelation2d() # Added here
        
        # Image Pool
        self.fake_A_pool = ImagePool(params.pool_size)
        self.fake_B_pool = ImagePool(params.pool_size)

    def backward_G(self, real_a, real_b, fake_a, fake_b, lambda_cycle=100, lambda_identity=1, lambda_gc=1):
        # GAN loss D_A(G_A(A))
        pred_fake = self.netD_A(fake_b)
        loss_G_A = self.criterionGAN(pred_fake, True)

        # GAN loss D_B(G_B(B))
        pred_fake = self.netD_B(fake_a)
        loss_G_B = self.criterionGAN(pred_fake, True)

        loss_G = (loss_G_A + loss_G_B)
        
        # Forward cycle loss
        rec_A = self.netG_B(fake_b)
        loss_cycle_A = self.criterionCycle(rec_A, real_a) * lambda_cycle

        # Backward cycle loss
        rec_B = self.netG_A(fake_a)
        loss_cycle_B = self.criterionCycle(rec_B, real_b) * lambda_cycle
        loss_cycle = (loss_cycle_A + loss_cycle_B)
        
        # Identity loss
        rec_A = self.netG_B(fake_b)
        loss_identity_A = self.criterionCycle(rec_A, real_a) * lambda_identity

        # Identity loss
        rec_B = self.netG_A(fake_a)
        loss_identity_B = self.criterionCycle(rec_B, real_b) * lambda_identity
        
        loss_identity = (loss_identity_A + loss_identity_B)
        
        # GC loss
        loss_gc_A = self.criterionGradientCorrelation(real_a, fake_b) * lambda_gc
        loss_gc_B = self.criterionGradientCorrelation(real_b, fake_a) * lambda_gc
        loss_gc = (loss_gc_A + loss_gc_B)
        
        # combined loss
        loss_G = loss_G + loss_cycle + loss_identity + loss_gc
        return loss_G
    
    def training_step(self, batch: Any, batch_idx: int):
        
        optimizer_G, optimizer_D_A, optimizer_D_B = self.optimizers()
        real_a, real_b, fake_a, fake_b = self.model_step(batch)

        with optimizer_G.toggle_model():
            loss_G = self.backward_G(real_a, real_b, fake_a, fake_b, self.params.lambda_cycle, self.params.lambda_identity, self.params.lambda_gc)
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
    _ = GCCycleGANModule(None, None, None)