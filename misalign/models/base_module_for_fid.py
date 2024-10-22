from typing import Any, Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics.clustering import NormalizedMutualInfoScore
from torchmetrics.image.fid import FrechetInceptionDistance

from misalign.metrics.gradient_correlation import GradientCorrelationMetric
from misalign.metrics.sharpness import SharpnessMetric

from torchmetrics.aggregation import CatMetric

gray2rgb = lambda x : torch.cat((x, x, x), dim=1)
# norm_0_to_1 = lambda x: (x + 1) / 2
flatten_to_1d = lambda x: x.view(-1)
norm_to_uint8 = lambda x: ((x + 1) / 2 * 255).to(torch.uint8)

class BaseModule(LightningModule):

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

        self.val_gc_A, self.val_nmi_A, self.val_fid_A, self.val_sharpness_A = self.define_metrics()
        self.test_gc_A, self.test_nmi_A, self.test_fid_A, self.test_sharpness_A = self.define_metrics()

        self.val_gc_B, self.val_nmi_B, self.val_fid_B, self.val_sharpness_B = self.define_metrics()
        self.test_gc_B, self.test_nmi_B, self.test_fid_B, self.test_sharpness_B = self.define_metrics()

        self.val_metrics = [self.val_gc_A, self.val_nmi_A, self.val_fid_A, self.val_sharpness_A, self.val_gc_B, self.val_nmi_B, self.val_fid_B, self.val_sharpness_B]
        self.test_metrics = [self.test_gc_A, self.test_nmi_A, self.test_fid_A, self.test_sharpness_A, self.test_gc_B, self.test_nmi_B, self.test_fid_B, self.test_sharpness_B]

        self.nmi_scores_A = []
        self.nmi_scores_B = []


    @staticmethod
    def define_metrics():
        gc = GradientCorrelationMetric()
        nmi = NormalizedMutualInfoScore()
        fid = FrechetInceptionDistance()
        sharpness = SharpnessMetric()

        return gc, nmi, fid, sharpness
    
    
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

        self.val_gc_A.update(norm_to_uint8(real_B), norm_to_uint8(fake_A))
        nmi_score_A = self.val_nmi_A(flatten_to_1d(norm_to_uint8(real_B)), flatten_to_1d(norm_to_uint8(fake_A)))
        self.nmi_scores_A.append(nmi_score_A)
        self.val_fid_A.update(gray2rgb(norm_to_uint8(real_A)), real=True)
        self.val_fid_A.update(gray2rgb(norm_to_uint8(fake_A)), real=False)
        self.val_sharpness_A.update(norm_to_uint8(fake_A))

        self.val_gc_B.update(norm_to_uint8(real_A), norm_to_uint8(fake_B))
        nmi_score_B = self.val_nmi_B(flatten_to_1d(norm_to_uint8(real_A)), flatten_to_1d(norm_to_uint8(fake_B)))
        self.nmi_scores_B.append(nmi_score_B)
        self.val_fid_B.update(gray2rgb(norm_to_uint8(real_B)), real=True)
        self.val_fid_B.update(gray2rgb(norm_to_uint8(fake_B)), real=False)
        self.val_sharpness_B.update(norm_to_uint8(fake_B))

    def on_validation_epoch_end(self):

        gc_A = self.val_gc_A.compute()
        nmi_A = torch.mean(torch.stack(self.nmi_scores_A))
        fid_A = self.val_fid_A.compute()
        sharpness_A = self.val_sharpness_A.compute()

        self.log("val/gc_A", gc_A.detach(), sync_dist=True)
        self.log("val/nmi_A", nmi_A.detach(), sync_dist=True)
        self.log("val/fid_A", fid_A.detach(), sync_dist=True)
        self.log("val/sharpness_A", sharpness_A.detach(), sync_dist=True)

        gc_B = self.val_gc_B.compute()
        nmi_B = torch.mean(torch.stack(self.nmi_scores_B))
        fid_B = self.val_fid_B.compute()
        sharpness_B = self.val_sharpness_B.compute()

        self.log("val/gc_B", gc_B.detach(), sync_dist=True)
        self.log("val/nmi_B", nmi_B.detach(), sync_dist=True)
        self.log("val/fid_B", fid_B.detach(), sync_dist=True)
        self.log("val/sharpness_B", sharpness_B.detach(), sync_dist=True)

        for metrics in self.val_metrics:
            metrics.reset()
        self.nmi_scores_A = []
        self.nmi_scores_B = []

    def test_step(self, batch: Any, batch_idx: int):

        real_A, real_B, fake_A, fake_B = self.model_step(batch)

        self.test_gc_A.update(norm_to_uint8(real_B), norm_to_uint8(fake_A))
        nmi_score_A = self.test_nmi_A(flatten_to_1d(norm_to_uint8(real_B)), flatten_to_1d(norm_to_uint8(fake_A)))
        self.nmi_scores_A.append(nmi_score_A)
        self.test_fid_A.update(gray2rgb(norm_to_uint8(real_A)), real=True)
        self.test_fid_A.update(gray2rgb(norm_to_uint8(fake_A)), real=False)
        self.test_sharpness_A.update(norm_to_uint8(fake_A))
       
        self.test_gc_B.update(norm_to_uint8(real_A), norm_to_uint8(fake_B))
        nmi_score_B = self.test_nmi_B(flatten_to_1d(norm_to_uint8(real_A)), flatten_to_1d(norm_to_uint8(fake_B)))
        self.nmi_scores_B.append(nmi_score_B)
        self.test_fid_B.update(gray2rgb(norm_to_uint8(real_B)), real=True)
        self.test_fid_B.update(gray2rgb(norm_to_uint8(fake_B)), real=False)
        self.test_sharpness_B.update(norm_to_uint8(fake_B))


    def on_test_epoch_end(self):
        gc_A = self.test_gc_A.compute()
        nmi_A = torch.mean(torch.stack(self.nmi_scores_A))
        fid_A = self.test_fid_A.compute()
        sharpness_A = self.test_sharpness_A.compute()
        
        self.log("test/gc_A_mean", gc_A.detach(), sync_dist=True)
        self.log("test/nmi_A_mean", nmi_A.detach(), sync_dist=True)
        self.log("test/fid_A_mean", fid_A.detach(), sync_dist=True)
        self.log("test/sharpness_A_mean", sharpness_A.detach(), sync_dist=True)

        gc_B = self.test_gc_B.compute()
        nmi_B = torch.mean(torch.stack(self.nmi_scores_B))
        fid_B = self.test_fid_B.compute()
        sharpness_B = self.test_sharpness_B.compute()
        
        self.log("test/gc_B_mean", gc_B.detach(), sync_dist=True)
        self.log("test/nmi_B_mean", nmi_B.detach(), sync_dist=True)
        self.log("test/fid_B_mean", fid_B.detach(), sync_dist=True)
        self.log("test/sharpness_B_mean", sharpness_B.detach(), sync_dist=True)

        for metrics in self.test_metrics:
            metrics.reset()
        self.nmi_scores_A = []
        self.nmi_scores_B = []

    def configure_optimizers(self):
        pass