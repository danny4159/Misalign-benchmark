from typing import Any
import itertools

import torch
from misalign.models.components.networks import GANLoss, ImagePool
from misalign.models.base_module import BaseModule
from misalign import utils
from torchvision import models

from misalign.data.components.transforms_fly import dataset_IXI_FLY, dataset_synthRAD_FLY_RAM, dataset_synthRAD_FLY
from torch.utils.data import DataLoader
from misalign.models.components.contextual_loss import (
    Contextual_Loss,
)  # this is the CX loss
from pytorch_msssim import ssim


import higher


log = utils.get_pylogger(__name__)

gray2rgb = lambda x : torch.cat((x, x, x), dim=1)
class PixelGANModule(BaseModule):
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

        # assign perceptual network
        self.vgg = VGG16()

        # Image Pool
        self.fake_AB_pool = ImagePool(params.pool_size)

        # loss function
        self.no_lsgan = False
        self.criterionGAN = GANLoss(use_lsgan=not self.no_lsgan)
        self.criterionL1 = torch.nn.L1Loss(reduction="none")
        self.criterionL2 = torch.nn.MSELoss(reduction="none")  # L2 Loss 정의
        self.criterionSSIM = ssim  # SSIM Loss 정의

        style_feat_layers = {"conv_2_2": 1.0, "conv_3_2": 1.0, "conv_4_2": 1.0}
        if self.params.lambda_style != 0:
            self.criterionStyle = Contextual_Loss(style_feat_layers, cobi=True)

        # Image Pool
        self.fake_A_pool = ImagePool(params.pool_size)
        self.fake_B_pool = ImagePool(params.pool_size)

        if self.params.flag_meta_learning:
            log.info("Using meta-learning")
            self.data_meta = dataset_IXI_FLY( #TODO: 수정. 이거 if문으로 처리될 수 있도록
            # self.data_meta = dataset_synthRAD_FLY_RAM(
                rand_crop=True,
                data_dir=self.params.dir_meta_learning,
                crop_size=80,
                reverse=self.params.reverse,
                aug=True,
                return_msk=True,
            )
            # item = self.data_meta[0]
            self.clean_dloader = DataLoader(self.data_meta, batch_size=32, shuffle=True)

        else:
            log.info("Not using meta-learning")

    def check_nan(self, loss, loss_name):
        if torch.isnan(loss).any():
            raise RuntimeError(f"NaN detected in {loss_name}!")
    
    def print_meta_data_ranges(self, meta_real_a, meta_real_b, meta_msk):
        print(f"meta_real_a: Min = {meta_real_a.min().item()}, Max = {meta_real_a.max().item()}, "
          f"meta_real_b: Min = {meta_real_b.min().item()}, Max = {meta_real_b.max().item()}, "
          f"meta_msk: Min = {meta_msk.min().item()}, Max = {meta_msk.max().item()}")
    
    def backward_G(self, real_a, real_b, fake_a, fake_b, lambda_l1, lambda_vgg, lambda_style, lambda_l2, lambda_ssim, weight_a=None, weight_b=None):
        loss_G = torch.zeros(1, device=self.device, requires_grad=True)

        if weight_a is not None:
            weight_spatial_a = weight_a # batch, 1 , width, heith
            weight_batch_a = torch.mean(weight_spatial_a, dim=(1, 2, 3),keepdim=True)

        if weight_b is not None:
            weight_spatial_b = weight_b # batch, 1 , width, heith
            weight_batch_b = torch.mean(weight_spatial_b, dim=(1, 2, 3),keepdim=True)

        # if self.params.flag_register:
        #     Trans_A = self.netR(fake_b, real_b)

        # GAN loss D_A(G_A(A))
        pred_fake_b = self.netD_A(fake_b)
        loss_G_A = self.criterionGAN(pred_fake_b, True)
        self.check_nan(loss_G_A, "GAN Loss A")


        pred_fake_a = self.netD_B(fake_a)
        loss_G_B = self.criterionGAN(pred_fake_a, True)
        self.check_nan(loss_G_B, "GAN Loss B")


        if weight_a is not None:
            loss_G_A = torch.mean(loss_G_A * weight_batch_a)
        else:
            loss_G_A = torch.mean(loss_G_A)

        # GAN loss D_B(G_B(B))
        pred_fake_a = self.netD_B(fake_a)
        loss_G_B = self.criterionGAN(pred_fake_a, True)

        if weight_b is not None:
            loss_G_B = torch.mean(loss_G_B * weight_batch_b)
        else:
            loss_G_B = torch.mean(loss_G_B)

        loss_GAN = (loss_G_A + loss_G_B) * 0.5
        self.log("loss_GAN", loss_GAN.detach(), prog_bar=True)  # GAN loss 로그 추가
        loss_G = loss_G + loss_GAN

        # G(A) = B
        loss_L1_A = self.criterionL1(fake_a, real_a) * lambda_l1
        if weight_b is not None:
            loss_L1_A = torch.mean(loss_L1_A * weight_spatial_b)
        else:
            loss_L1_A = torch.mean(loss_L1_A)
        self.check_nan(loss_L1_A, "L1 Loss A")

        loss_L1_B = self.criterionL1(fake_b, real_b) * lambda_l1
        if weight_a is not None:
            loss_L1_B = torch.mean(loss_L1_B * weight_spatial_a)
        else:
            loss_L1_B = torch.mean(loss_L1_B)
        self.check_nan(loss_L1_B, "L1 Loss B") 

        loss_L1 = (loss_L1_A + loss_L1_B) * 0.5
        self.log("loss_L1", loss_L1.detach(), prog_bar=True)  # L1 loss 로그 추가
        loss_G = loss_G + loss_L1

        # Perceptual Loss
        # [128,64,64]로 batch차원이 융합돼있음. meta weight batch를 그냥 multiple. 그럼 브로드캐스트돼서 해당 배치가 얼마나 유의미한지에 따라 반영됨
        VGG_real_A = self.vgg(real_a.expand([int(real_a.size()[0]),3,int(real_a.size()[2]),int(real_a.size()[3])]))[0] 
        VGG_fake_A = self.vgg(fake_a.expand([int(real_a.size()[0]),3,int(real_a.size()[2]),int(real_a.size()[3])]))[0]
        VGG_loss_A = self.criterionL1(VGG_fake_A,VGG_real_A) * lambda_vgg
        if weight_b is not None:
            VGG_loss_A = torch.mean(VGG_loss_A * weight_batch_b)
        else:
            VGG_loss_A = torch.mean(VGG_loss_A)
        self.check_nan(VGG_loss_A, "VGG Loss A")

        VGG_real_B = self.vgg(real_b.expand([int(real_b.size()[0]),3,int(real_b.size()[2]),int(real_b.size()[3])]))[0]
        VGG_fake_B = self.vgg(fake_b.expand([int(real_b.size()[0]),3,int(real_b.size()[2]),int(real_b.size()[3])]))[0]
        VGG_loss_B = self.criterionL1(VGG_fake_B,VGG_real_B) * lambda_vgg
        if weight_a is not None:
            VGG_loss_B = torch.mean(VGG_loss_B * weight_batch_a)
        else:
            VGG_loss_B = torch.mean(VGG_loss_B)
        self.check_nan(VGG_loss_B, "VGG Loss B") 

        VGG_loss = (VGG_loss_A + VGG_loss_B) * 0.5
        self.log("loss_VGG", VGG_loss.detach(), prog_bar=True)  # VGG loss 로그 추가
        loss_G = loss_G + VGG_loss

        if self.params.lambda_style != 0:
            Style_loss_A = self.criterionStyle(real_a, fake_a) * lambda_style
            if weight_b is not None:
                Style_loss_A = torch.mean(Style_loss_A * weight_batch_b)
            else:
                Style_loss_A = torch.mean(Style_loss_A)
            self.check_nan(Style_loss_A, "Style_loss_A")

            Style_loss_B = self.criterionStyle(real_b, fake_b) * lambda_style
            if weight_a is not None:
                Style_loss_B = torch.mean(Style_loss_B * weight_batch_a)
            else:
                Style_loss_B = torch.mean(Style_loss_B)
            self.check_nan(Style_loss_B, "Style_loss_B")
            
            Style_loss = (Style_loss_A + Style_loss_B) * 0.5
            self.log("loss_Style", Style_loss.detach(), prog_bar=True)
            loss_G = loss_G + Style_loss
            
        
        if self.params.lambda_l2 != 0:
            loss_L2_A = self.criterionL2(fake_a, real_a) * lambda_l2
            loss_L2_A = torch.mean(loss_L2_A * weight_spatial_b) if weight_b is not None else torch.mean(loss_L2_A)
            self.check_nan(loss_L2_A, "L2 Loss A")
            
            loss_L2_B = self.criterionL2(fake_b, real_b) * lambda_l2
            loss_L2_B = torch.mean(loss_L2_B * weight_spatial_a) if weight_a is not None else torch.mean(loss_L2_B)
            self.check_nan(loss_L2_B, "L2 Loss B")
            
            loss_L2 = (loss_L2_A + loss_L2_B) * 0.5
            self.log("loss_L2", loss_L2.detach(), prog_bar=True)
            loss_G = loss_G + loss_L2

        # SSIM Loss
        if self.params.lambda_ssim != 0:
            loss_SSIM_A = (1 - self.criterionSSIM(fake_a, real_a)) * lambda_ssim  # SSIM의 값이 높을수록 유사하므로 1에서 뺌
            loss_SSIM_A = torch.mean(loss_SSIM_A * weight_spatial_b) if weight_b is not None else torch.mean(loss_SSIM_A)
            self.check_nan(loss_SSIM_A, "SSIM Loss A")

            loss_SSIM_B = (1 - self.criterionSSIM(fake_b, real_b)) * lambda_ssim
            loss_SSIM_B = torch.mean(loss_SSIM_B * weight_spatial_a) if weight_a is not None else torch.mean(loss_SSIM_B)
            self.check_nan(loss_SSIM_B, "SSIM Loss B")

            loss_SSIM = (loss_SSIM_A + loss_SSIM_B) * 0.5
            self.log("loss_SSIM", loss_SSIM.detach(), prog_bar=True)
            loss_G = loss_G + loss_SSIM

        return loss_G
    
    def check_model_weights(self, model, model_name="Model"):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if torch.isnan(param).any():
                    raise RuntimeError(f"NaN detected in {model_name} parameter: {name}")
                if torch.isinf(param).any():
                    raise RuntimeError(f"Inf detected in {model_name} parameter: {name}")
                # print(f"{model_name} parameter: {name} - Min: {param.min().item()}, Max: {param.max().item()}")

    def determine_weight_LRE_For_A(self, real_a, real_b, meta_real_a, meta_real_b, mask=None):
        """
        weight: output과 label간에 loss를 통해 어느 pixel에 weight를 더 줄지 학습해
        """
        if mask is not None:
            mask = mask.float() # Meta mask 그냥 Mask있으면 Mask 씌우는거 학습은 없어
        else:
            mask = 1.0

        # Check weights of the original model
        self.check_model_weights(self.netG_A, "netG_A")

        with higher.innerloop_ctx(self.netG_A, self.optimizer_G_A) as (
            meta_model,
            meta_opt,
        ):
        
            self.check_model_weights(meta_model, "meta_model_A")

            fake_b = meta_model(real_a)
            # for name, param in meta_model.named_parameters():
            #     print(f"{name}: min={param.min().item()}, max={param.max().item()}")
            # print("fake min: ", fake_b.min(), " fake max: ",fake_b.max(), " fake mean: ", fake_b.mean())
            self.check_nan(fake_b, "fake_b after meta_model(real_a)")  # NaN 체크

            # Register the fake_b to real_b
            if self.params.flag_register:
                Trans_A = self.netR(fake_b, real_b)
                fake_b = self.spatial_transform(fake_b, Trans_A)
                self.check_nan(Trans_A, "Trans_A")  # NaN 체크
                self.check_nan(fake_b, "fake_b after spatial_transform")  # NaN 체크

            if self.params.flag_meta_use_spatial:
                weight = torch.zeros(
                    real_a.size(), device=self.device, requires_grad=True # 0으로 초기화는 하지만 학습 가능한 매개변수
                )
            else:
                weight = torch.zeros(
                    real_a.shape[0], 1, 1, 1, device=self.device, requires_grad=True
                )

            _loss = self.criterionL1(real_b, fake_b)
            
            self.check_nan(_loss, "L1 Loss between real_b and fake_b")  # NaN 체크
            loss = torch.mean(_loss * weight) # weight가 어떻게 변해야 meta_model의 loss가 더 줄어드는지 meta_model의 파라미터를 학습하는것 ?
            self.check_nan(loss, "Weighted loss")  # NaN 체크
            meta_opt.step(loss)

            meta_val_loss = self.criterionL1(
                (meta_real_b + 1) * mask, (meta_model(meta_real_a) + 1) * mask
            )
            
            self.check_nan(meta_val_loss, "meta_val_loss")  # NaN 체크
            meta_val_loss = torch.mean(meta_val_loss)
            self.check_nan(meta_val_loss, "Mean meta_val_loss")  # NaN 체크
            # print("meta_val_loss:", meta_val_loss)
            # print("weight:", weight,"weight shape:", weight.size())
            # print("weight min:", weight.min(), " weight max:", weight.max())
            eps_grad = torch.autograd.grad(meta_val_loss, weight)[ # weight의 gradient를 구하는것. 
                0                                                  # 양의 gradient는 해당 픽셀의 weight를 증가시켜야 손실이 감소됨.
            ].detach()  # Gradient                                 # 음의 gradient는 해당 픽셀의 weight를 감소시켜야 손실이 감소됨.
            self.check_nan(eps_grad, "eps_grad")  # NaN 체크
            # print("eps_grad:", eps_grad)
        # print(f"eps_grad min: {eps_grad.min().item()}, max: {eps_grad.max().item()}")
        # print(f"Number of negative values in eps_grad: {(eps_grad < 0).sum().item()}")

        w_tilde = torch.clamp(-eps_grad, min=0) # 음수시켜서 0이하값은 0이 되도록

        # print(f"w_tilde min: {w_tilde.min().item()}, max: {w_tilde.max().item()}")
        # print(f"Number of zero values in w_tilde: {(w_tilde == 0).sum().item()}")

        self.check_nan(w_tilde, "w_tilde")  # NaN 체크
        l1_norm = torch.sum(w_tilde)
        self.check_nan(l1_norm, "l1_norm")  # NaN 체크
        if l1_norm != 0:
            w_b = w_tilde / l1_norm
        else:
            w_b = w_tilde + 0.1 / l1_norm

        # scaling:
        if self.params.flag_meta_use_spatial:
            w_b = w_b * real_a.size(0) * real_a.size(2) * real_a.size(3)
        else:
            w_b = w_b * real_a.size(0)
        self.check_nan(w_b, "Final w_b")  # NaN 체크

        w_b = w_b.detach()  # w_a is the weight for a
        return w_b
    
    def determine_weight_LRE_For_B(self, real_a, real_b, meta_real_a, meta_real_b, mask=None):
        """
        weight: output과 label간에 loss를 통해 어느 pixel에 weight를 더 줄지 학습해
        """
        if mask is not None:
            mask = mask.float()  # Meta mask가 있으면 mask 적용
        else:
            mask = 1.0

        # Check weights of the meta model
        self.check_model_weights(self.netG_B, "netG_B")

        with higher.innerloop_ctx(self.netG_B, self.optimizer_G_B) as (
            meta_model,
            meta_opt,
        ):
            self.check_model_weights(meta_model, "meta_model_B")
            fake_a = meta_model(real_b)
            # for name, param in meta_model.named_parameters():
            #     print(f"{name}: min={param.min().item()}, max={param.max().item()}")
            self.check_nan(fake_a, "fake_a after meta_model(real_b)")  # NaN 체크

            # Register the fake_a to real_a
            if self.params.flag_register:
                Trans_B = self.netR(fake_a, real_a)
                self.check_nan(Trans_B, "Trans_B")  # NaN 체크
                fake_a = self.spatial_transform(fake_a, Trans_B)
                self.check_nan(fake_a, "fake_a after spatial_transform")  # NaN 체크

            if self.params.flag_meta_use_spatial:
                weight = torch.zeros(
                    real_b.size(), device=self.device, requires_grad=True  # 0으로 초기화된 학습 가능한 매개변수
                )
            else:
                weight = torch.zeros(
                    real_b.shape[0], 1, 1, 1, device=self.device, requires_grad=True
                )

            _loss = self.criterionL1(real_a, fake_a)
            self.check_nan(_loss, "L1 Loss between real_a and fake_a")  # NaN 체크

            loss = torch.mean(_loss * weight)  # weight가 어떻게 변해야 meta_model의 loss가 더 줄어드는지 meta_model의 파라미터를 학습하는 것
            self.check_nan(loss, "Weighted loss")  # NaN 체크

            meta_opt.step(loss)

            meta_val_loss = self.criterionL1(
                (meta_real_a + 1) * mask, (meta_model(meta_real_b) + 1) * mask
            )
            # print(f"meta_val_loss min: {meta_val_loss.min().item()}, max: {meta_val_loss.max().item()}")
            self.check_nan(meta_val_loss, "meta_val_loss")  # NaN 체크

            meta_val_loss = torch.mean(meta_val_loss)
            self.check_nan(meta_val_loss, "Mean meta_val_loss")  # NaN 체크

            eps_grad = torch.autograd.grad(meta_val_loss, weight)[0].detach()
            self.check_nan(eps_grad, "eps_grad")  # NaN 체크

        # print(f"eps_grad min: {eps_grad.min().item()}, max: {eps_grad.max().item()}")
        # print(f"Number of negative values in eps_grad: {(eps_grad < 0).sum().item()}")

        w_tilde = torch.clamp(-eps_grad, min=0) # 음수시켜서 0이하값은 0이 되도록

        # print(f"w_tilde min: {w_tilde.min().item()}, max: {w_tilde.max().item()}")
        # print(f"Number of zero values in w_tilde: {(w_tilde == 0).sum().item()}")

        # w_tilde = torch.clamp(-eps_grad, min=0)  # 음수시켜서 0이하 값은 0이 되도록
        self.check_nan(w_tilde, "w_tilde")  # NaN 체크

        l1_norm = torch.sum(w_tilde)
        self.check_nan(l1_norm, "l1_norm")  # NaN 체크

        if l1_norm != 0:
            w_b = w_tilde / l1_norm
        else:
            w_b = w_tilde + 0.1 / l1_norm
        
        # scaling:
        if self.params.flag_meta_use_spatial:
            w_b = w_b * real_b.size(0) * real_b.size(2) * real_b.size(3)
        else:
            w_b = w_b * real_b.size(0)

        self.check_nan(w_b, "Final w_b")  # NaN 체크

        w_b = w_b.detach()  # w_a is the weight for a
        return w_b
    
    def training_step(self, batch: Any, batch_idx: int):

        optimizer_G, optimizer_D_A, optimizer_D_B = self.optimizers()
        
        if len(batch) == 3:
            real_a, real_b, slice_idx = batch
        else:
            real_a, real_b = batch

        if self.current_epoch > -1 and self.params.flag_meta_learning:
            ################## Getting meta data ##################

            meta_real_a, meta_real_b, meta_msk = next(iter(self.clean_dloader))

            meta_real_a, meta_real_b, meta_msk = (
                meta_real_a.to(device=self.device),
                meta_real_b.to(device=self.device),
                meta_msk.to(device=self.device),
            )

            # self.print_meta_data_ranges(meta_real_a, meta_real_b, meta_msk)
            ########################################################
            # Meta-learning type: LRE
            if self.params.flag_use_mask:
                mask = meta_msk
            else:
                mask = None

            if self.params.meta_type == "LRE":
                w_a = self.determine_weight_LRE_For_A( 
                    real_a, real_b, meta_real_a, meta_real_b, mask=mask
                )

                w_b = self.determine_weight_LRE_For_B( 
                    real_a, real_b, meta_real_a, meta_real_b, mask=mask
                )
                optimizer_G.zero_grad()
            else:
                raise NotImplementedError
            del meta_real_a, meta_real_b, meta_msk  # remove this

        else:
            w_a = None
            w_b = None

        with optimizer_G.toggle_model():
            real_a, real_b, fake_a, fake_b = self.model_step(batch)
            loss_G = self.backward_G(real_a, real_b, fake_a, fake_b, self.params.lambda_l1, self.params.lambda_vgg, self.params.lambda_style, self.params.lambda_l2, self.params.lambda_ssim, weight_a=w_a, weight_b=w_b)
            self.manual_backward(loss_G)
            self.clip_gradients(optimizer_G, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            optimizer_G.step()
            optimizer_G.zero_grad()

        with optimizer_D_A.toggle_model():
            loss_D_A = self.backward_D_A(real_b, fake_b)
            if w_a is not None:
                weight_batch_a = torch.mean(w_a, dim=(1, 2, 3),keepdim=True)
                loss_D_A = torch.mean(loss_D_A * weight_batch_a)
            else:
                loss_D_A = torch.mean(loss_D_A)
            self.manual_backward(loss_D_A)
            self.clip_gradients(optimizer_D_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            optimizer_D_A.step()
            optimizer_D_A.zero_grad()

        with optimizer_D_B.toggle_model():
            loss_D_B = self.backward_D_B(real_a, fake_a)
            if w_b is not None:
                weight_batch_b = torch.mean(w_b, dim=(1, 2, 3),keepdim=True)
                loss_D_B = torch.mean(loss_D_B * weight_batch_b)
            else:
                loss_D_B = torch.mean(loss_D_B)
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
        self.optimizer_G_A = self.hparams.optimizer(params=itertools.chain(self.netG_A.parameters()))
        self.optimizer_G_B = self.hparams.optimizer(params=itertools.chain(self.netG_B.parameters()))
        
        optimizer_G = self.hparams.optimizer(params=itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()))
        #TODO: netR추가되면 ??
        optimizer_D_A = self.hparams.optimizer(params=self.netD_A.parameters())
        optimizer_D_B = self.hparams.optimizer(params=self.netD_B.parameters())

        return optimizer_G, optimizer_D_A, optimizer_D_B

#Extracting VGG feature maps before the 2nd maxpooling layer
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
    _ = PixelGANModule(None, None, None)
