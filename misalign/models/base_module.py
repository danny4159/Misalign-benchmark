from typing import Any, Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.aggregation import CatMetric

gray2rgb = lambda x : torch.cat((x, x, x), dim=1)

class BaseModule(LightningModule):

    def __init__(
        self,
        *args: Any,
        **kwargs: Any
    ):
        super().__init__()
        # define metrics (for image A)
        self.val_psnr_A, self.val_ssim_A, self.val_lpips_A = self.define_metrics()
        self.test_psnr_A, self.test_ssim_A, self.test_lpips_A = self.define_metrics()
        self.stats_psnr_A, self.stats_ssim_A, self.stats_lpips_A = self.define_cat_metrics()

        # define metrics (for image B)
        self.val_psnr_B, self.val_ssim_B, self.val_lpips_B = self.define_metrics()
        self.test_psnr_B, self.test_ssim_B, self.test_lpips_B = self.define_metrics()
        self.stats_psnr_B, self.stats_ssim_B, self.stats_lpips_B = self.define_cat_metrics()

        # create list (for reset)
        self.val_metrics = [self.val_psnr_A, self.val_ssim_A, self.val_lpips_A, self.val_psnr_B, self.val_ssim_B, self.val_lpips_B]
        self.test_metrics = [self.test_psnr_A, self.test_ssim_A, self.test_lpips_A, self.test_psnr_B, self.test_ssim_B, self.test_lpips_B]

    @staticmethod
    def define_metrics():
        psnr = PeakSignalNoiseRatio()
        ssim = StructuralSimilarityIndexMeasure()
        lpips = LearnedPerceptualImagePatchSimilarity()
        return psnr, ssim, lpips
    
    @staticmethod
    def define_cat_metrics():
        psnr = CatMetric()
        ssim = CatMetric()
        lpips = CatMetric()
        return psnr, ssim, lpips

    def forward(self, a: torch.Tensor, b: Optional[torch.Tensor]=None):        
        if b is None:
            return self.netG_A(a)
        else:
            return self.netG_A(a), self.netG_B(b)

    def model_step(self, batch: Any):
        real_a, real_b = batch
        if self.netG_A._get_name() == 'G_Resnet': # For UNIT
            hidden_a, _ = self.netG_A.encode(real_a)
            fake_b = self.netG_B.decode(hidden_a)
            hidden_b, _ = self.netG_B.encode(real_b)
            fake_a = self.netG_A.decode(hidden_b)

        elif self.netG_A._get_name() == 'AdaINGen': # For MUNIT
            c_a, s_a_fake = self.netG_A.encode(real_a)
            c_b, s_b_fake = self.netG_B.encode(real_b)
            fake_a = self.netG_A.decode(c_b, self.s_a)
            fake_b = self.netG_B.decode(c_a, self.s_b)
        
        elif self.netG_A._get_name() == 'ADN': # For ADN
            pred_ll, pred_lh = self.netG_A.forward1(real_a)
            pred_hl, pred_hh = self.netG_A.forward2(real_a, real_b)
            fake_a = pred_hl
            fake_b = pred_lh
        
        elif self.netG_A._get_name() == 'DAModule': # For DAM
            fake_b = self.netG_B.forward(real_a, real_b)
            fake_a = self.netG_A.forward(real_b, real_a)
        
        else:
            fake_b, fake_a = self.forward(real_a, real_b)
        return real_a, real_b, fake_a, fake_b

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        for metrics in self.val_metrics:
            metrics.reset()
        for metrics in self.test_metrics:
            metrics.reset()
        return super().on_train_start()

    def backward_G(self, real_a, real_b, fake_a, fake_b, *args, **kwargs):
        pass


    def backward_D_A(self, real_b, fake_b):
        fake_b = self.fake_B_pool.query(fake_b)
        loss_D_A = self.backward_D_basic(self.netD_A, real_b, fake_b)
        return loss_D_A

    def backward_D_B(self, real_a, fake_a):
        fake_a = self.fake_A_pool.query(fake_a)
        loss_D_B = self.backward_D_basic(self.netD_B, real_a, fake_a)
        return loss_D_B
    
    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def training_step(self, batch: Any, batch_idx: int):
        return super().training_step(batch, batch_idx)

    def on_train_epoch_start(self) -> None:
        self.loss_G = 0
        return super().on_train_epoch_start()

    def on_train_epoch_end(self):
        return super().on_train_epoch_end()

    def validation_step(self, batch: Any, batch_idx: int):
        real_A, real_B, fake_A, fake_B = self.model_step(batch)
        loss = (F.l1_loss(real_A, fake_A) + F.l1_loss(real_B, fake_B)) / 2
        # Perform metric
        self.val_psnr_A(real_A, fake_A)
        self.val_ssim_A(real_A, fake_A)
        self.val_lpips_A(gray2rgb(real_A), gray2rgb(fake_A))

        self.val_psnr_B(real_B, fake_B)
        self.val_ssim_B(real_B, fake_B)
        self.val_lpips_B(gray2rgb(real_B), gray2rgb(fake_B))

        self.log("val/loss", loss.detach(), prog_bar=True)

    def on_validation_epoch_end(self):

        psnr = self.val_psnr_A.compute()
        ssim = self.val_ssim_A.compute()
        lpips = self.val_lpips_A.compute()

        self.log("val/psnr_A", psnr.detach())
        self.log("val/ssim_A", ssim.detach())
        self.log("val/lpips_A", lpips.detach())

        psnr = self.val_psnr_B.compute()
        ssim = self.val_ssim_B.compute()
        lpips = self.val_lpips_B.compute()

        self.log("val/psnr_B", psnr.detach())
        self.log("val/ssim_B", ssim.detach())
        self.log("val/lpips_B", lpips.detach())

        for metrics in self.val_metrics:
            metrics.reset()

    def test_step(self, batch: Any, batch_idx: int):
        real_A, real_B, fake_A, fake_B = self.model_step(batch)

        loss = (F.l1_loss(real_A, fake_A) + F.l1_loss(real_B, fake_B)) / 2
        # Perform metric
        _psnr_A = self.test_psnr_A(real_A, fake_A)
        _ssim_A = self.test_ssim_A(real_A, fake_A)
        _lpips_A = self.test_lpips_A(gray2rgb(real_A), gray2rgb(fake_A))

        _psnr_B = self.test_psnr_B(real_B, fake_B)
        _ssim_B = self.test_ssim_B(real_B, fake_B)
        _lpips_B = self.test_lpips_B(gray2rgb(real_B), gray2rgb(fake_B))

        self.stats_psnr_A.update(_psnr_A)
        self.stats_psnr_B.update(_psnr_B)
        
        self.stats_ssim_A.update(_ssim_A)
        self.stats_ssim_B.update(_ssim_B)
        
        self.stats_lpips_A.update(_lpips_A)
        self.stats_lpips_B.update(_lpips_B) 

        self.log("test/loss", loss.detach(), prog_bar=True)

    def on_test_epoch_end(self):
        psnr = self.stats_psnr_A.compute()
        ssim = self.stats_ssim_A.compute()
        lpips = self.stats_lpips_A.compute()

        self.log("test/psnr_A_mean", psnr.mean())
        self.log("test/ssim_A_mean", ssim.mean())
        self.log("test/lpips_A_mean", lpips.mean())

        self.log("test/psnr_A_std", psnr.std())
        self.log("test/ssim_A_std", ssim.std())
        self.log("test/lpips_A_std", lpips.std())

        psnr = self.stats_psnr_B.compute()
        ssim = self.stats_ssim_B.compute()
        lpips = self.stats_lpips_B.compute()

        self.log("test/psnr_B_mean", psnr.mean()) # option : rounding
        self.log("test/ssim_B_mean", ssim.mean())
        self.log("test/lpips_B_mean", lpips.mean())

        self.log("test/psnr_B_std", psnr.std())
        self.log("test/ssim_B_std", ssim.std())
        self.log("test/lpips_B_std", lpips.std())

        for metrics in self.val_metrics:
            metrics.reset()

    def configure_optimizers(self):
        pass