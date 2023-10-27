from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.utils import make_grid
from torchvision.transforms import CenterCrop
import torch
from typing import Any, List, Optional
from lightning.pytorch import Callback
from misalign import utils
import numpy as np
import nibabel as nib
import h5py


import os

log = utils.get_pylogger(__name__)

class ImageLoggingCallback(Callback):
    def __init__(
        self,
        val_batch_idx: List[int] = [10, 20, 30, 40, 50],
        tst_batch_idx: List[int] = [7, 8, 9, 10, 11],
        center_crop: int = 256,
        every_epoch=5,
        log_test: bool = False,
    ):
        """_summary_

        Args:
            batch_idx (List[int], optional): _description_. Defaults to [10,20,30,40,50].
            log_test (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.val_batch_idx = val_batch_idx  # log images on the validation stage
        self.tst_batch_idx = tst_batch_idx  # log images on the testing stage

        self.every_epoch = every_epoch
        self.log_test = log_test  # log images on the testing stage as well
        self.center_crop = center_crop  # center crop the images to this size

    def on_validation_start(self, trainer, pl_module):
        self.img_grid = []
        self.err_grid = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (
            batch_idx in self.val_batch_idx
            and trainer.current_epoch % self.every_epoch == 0
        ):
            res = pl_module.model_step(batch)

            if len(res) == 4:
                a, b, preds_a, preds_b = res
                self.ngrid = 4
                # if size of a is bigger than centercrop, then center crop
                # if a.shape[-1] > self.center_crop:
                #     a = CenterCrop(self.center_crop)(a)
                #     b = CenterCrop(self.center_crop)(b)
                #     preds_a = CenterCrop(self.center_crop)(preds_a)
                #     preds_b = CenterCrop(self.center_crop)(preds_b)

                err_a = torch.abs(a - preds_a)
                err_b = torch.abs(b - preds_b)

                # log.info(f'a shape: <{a.shape}>, b shape: <{b.shape}>, preds_a shape: <{preds_a.shape}>, preds_b shape: <{preds_b.shape}>')

                # log outputs to the tensorboard

                self.img_grid = self.img_grid + [
                    (a[0] + 1) / 2,
                    (preds_a[0] + 1) / 2,
                    (b[0] + 1) / 2,
                    (preds_b[0] + 1) / 2,
                ]
                self.err_grid = self.err_grid + [err_a[0], err_b[0]]

            elif len(res) == 3:
                a, b, preds_b = res
                self.ngrid = 3
                # a, b, preds_a, preds_b = pl_module.model_step(batch)

                # if a.shape[-1] > self.center_crop:
                #     a = CenterCrop(self.center_crop)(a)
                #     b = CenterCrop(self.center_crop)(b)
                #     preds_b = CenterCrop(self.center_crop)(preds_b)

                err_b = torch.abs(b - preds_b)

                # log.info(f'a shape: <{a.shape}>, b shape: <{b.shape}>, preds_a shape: <{preds_a.shape}>, preds_b shape: <{preds_b.shape}>')

                # log outputs to the tensorboard

                self.img_grid = self.img_grid + [
                    (a[0] + 1) / 2,
                    (preds_b[0] + 1) / 2,
                    (b[0] + 1) / 2,
                ]
                self.err_grid = self.err_grid + [err_b[0]]

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if len(self.img_grid) > 0 and trainer.current_epoch % self.every_epoch == 0:
            log.info(f"Saving validation img_grid shape: <{len(self.img_grid)}>")

            img_grid = make_grid(self.img_grid, nrow=self.ngrid)
            err_grid = make_grid(
                self.err_grid, nrow=self.ngrid // 2, value_range=(0, 1)
            )

            # Log to TensorBoard
            trainer.logger.experiment.add_image(
                f"val/images", img_grid, trainer.current_epoch
            )
            trainer.logger.experiment.add_image(
                f"val/error", err_grid, trainer.current_epoch
            )
            self.img_grid = []
            self.err_grid = []
        else:
            log.debug(f"No images to log for validation")

    def on_test_start(self, trainer, pl_module):
        self.img_grid = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (
            self.log_test and batch_idx in self.tst_batch_idx
        ):  # log every indexes for slice number in test set
            res = pl_module.model_step(batch)
            if len(res) == 4:
                self.ngrid = 4
                a, b, preds_a, preds_b = res

                # if size of a is bigger than centercrop, then center crop
                # if a.shape[-1] > self.center_crop:
                #     a = CenterCrop(self.center_crop)(a)
                #     b = CenterCrop(self.center_crop)(b)
                #     preds_a = CenterCrop(self.center_crop)(preds_a)
                #     preds_b = CenterCrop(self.center_crop)(preds_b)

                # log.info(f'a shape: <{a.shape}>, b shape: <{b.shape}>, preds_a shape: <{preds_a.shape}>, preds_b shape: <{preds_b.shape}>')

                # log outputs to the tensorboard
                self.img_grid = self.img_grid + [
                    (a[0] + 1) / 2,
                    (preds_a[0] + 1) / 2,
                    (b[0] + 1) / 2,
                    (preds_b[0] + 1) / 2,
                ]
            elif len(res) == 3:
                self.ngrid = 3
                a, b, preds_b = res

                # if a.shape[-1] > self.center_crop:
                #     a = CenterCrop(self.center_crop)(a)
                #     b = CenterCrop(self.center_crop)(b)
                #     preds_b = CenterCrop(self.center_crop)(preds_b)

                self.img_grid = self.img_grid + [
                    (a[0] + 1) / 2,
                    (preds_b[0] + 1) / 2,
                    (b[0] + 1) / 2,
                ]

    def on_test_end(self, trainer, pl_module):
        log.info(f"Saving test img_grid shape: <{len(self.img_grid)}>")

        # Create a grid of images
        if len(self.img_grid) > 0:
            img_grid = make_grid(self.img_grid, nrow=self.ngrid)
            # Log to TensorBoard
            trainer.logger.experiment.add_image(f"test/final_image", img_grid)
        else:
            log.warning(f"No images to log for testing")


class ImageSavingCallback(Callback):
    def __init__(self, center_crop: int = 256, subject_number_length: int = 3):
        """_summary_
        Image saving callback : Save images in nii format for each subject

        """
        super().__init__()
        self.center_crop = center_crop  # center crop the images to this size
        self.subject_number_length = subject_number_length

    @staticmethod
    def change_torch_numpy(a, b, c, d):
        assert (
            a.ndim == b.ndim == c.ndim == d.ndim
        ), "All input arrays must have the same number of dimensions"
        if a.ndim == 4:
            a_np = a.cpu().detach().numpy()[0, 0]
            b_np = b.cpu().detach().numpy()[0, 0]
            c_np = c.cpu().detach().numpy()[0, 0]
            d_np = d.cpu().detach().numpy()[0, 0]  # d_np : (256,256)
        else:
            raise NotImplementedError("This function has not been implemented yet.")
        return a_np, b_np, c_np, d_np

    @staticmethod
    def change_numpy_nii(a, b, c, d):
        assert (
            a.ndim == b.ndim == c.ndim == d.ndim == 3
        ), "All input arrays must have the same number of dimensions (3)"

        # scale to [0, 1] and [0, 255]
        a, b, c, d = (
            ((a + 1) / 2) * 255,
            ((b + 1) / 2) * 255,
            ((c + 1) / 2) * 255,
            ((d + 1) / 2) * 255,
        )

        # type to np.int16
        a, b, c, d = (
            a.astype(np.int16),
            b.astype(np.int16),
            c.astype(np.int16),
            d.astype(np.int16),
        )

        # transpose 1, 2 dim (for viewing on ITK-SNAP)
        a, b, c, d = (
            np.transpose(a, axes=(1, 0, 2))[:,::-1],
            np.transpose(b, axes=(1, 0, 2))[:,::-1],
            np.transpose(c, axes=(1, 0, 2))[:,::-1],
            np.transpose(d, axes=(1, 0, 2))[:,::-1],
        )

        # Create Nifti1Image for each
        a_nii, b_nii, c_nii, d_nii = (
            nib.Nifti1Image(a, np.eye(4)),
            nib.Nifti1Image(b, np.eye(4)),
            nib.Nifti1Image(c, np.eye(4)),
            nib.Nifti1Image(d, np.eye(4)),
        )

        return a_nii, b_nii, c_nii, d_nii

    @staticmethod
    def save_nii(a_nii, b_nii, c_nii, d_nii, subject_number, folder_path):
        nib.save(a_nii, os.path.join(folder_path, f"a_{subject_number}.nii.gz"))
        nib.save(b_nii, os.path.join(folder_path, f"b_{subject_number}.nii.gz"))
        nib.save(c_nii, os.path.join(folder_path, f"preds_a_{subject_number}.nii.gz"))
        if d_nii is not None:
            nib.save(d_nii, os.path.join(folder_path, f"preds_b_{subject_number}.nii.gz"))
        return

    def on_test_start(self, trainer, pl_module):
        # make save folder
        folder_name = os.path.join(trainer.default_root_dir, "results")
        log.info(f"Saving test images to nifti files to {folder_name}")

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        self.save_folder_name = folder_name

        self.img_a = []
        self.img_b = []
        self.img_preds_a = []
        self.img_preds_b = []
        self.i = 0
        self.subject_slice_num = []
        self.subject_number = 1

        ################################################################
        # synthRAD위해 추가한 코드!
        head, _ = os.path.split(trainer.default_root_dir)
        while head:
            # 분할된 경로의 마지막 부분을 확인합니다.
            tail = os.path.basename(head)
            if tail == "logs":
                # "logs"를 찾았을 경우 그 전까지의 경로를 반환합니다.
                code_root_dir = os.path.dirname(head)
                break
            head, _= os.path.split(head)
        data_path = os.path.join(code_root_dir, 'data', 'SynthRAD_MR_CT_Pelvis', 'test', 'prepared_data_0_0_0_0_0_MASK_Norm.h5') # TODO: 수정. 데이터셋 이름 바뀌면.
        # h5 파일에서 MR 그룹의 모든 데이터셋을 리스트로 불러오기
        with h5py.File(data_path, 'r') as file:
            mr_group = file['MR']
            dataset_list = [mr_group[key][()] for key in mr_group.keys()]
        # dataset_list의 각 데이터셋에서 3번째 channel을 self.img_a에 추가
        for dataset in dataset_list:
            self.subject_slice_num.append(dataset.shape[2])  # 3번째 channel


    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        res = pl_module.model_step(batch)
        if len(res) == 4:
            a, b, preds_a, preds_b = res

            # # if size of a is bigger than centercrop, then center crop
            # if a.shape[-1] > self.center_crop:
            #     a = CenterCrop(self.center_crop)(a)
            #     b = CenterCrop(self.center_crop)(b)
            #     preds_a = CenterCrop(self.center_crop)(preds_a)
            #     preds_b = CenterCrop(self.center_crop)(preds_b)

            # Change a,b,preds_a,preds_b to numpy array
            a, b, preds_a, preds_b = self.change_torch_numpy(a, b, preds_a, preds_b)

            self.img_a.append(a)
            self.img_b.append(b)
            self.img_preds_a.append(preds_a)
            self.img_preds_b.append(preds_b)

            # if len(img_a) == 91, stack file and save to nii
            if len(self.img_a) == self.subject_slice_num[0]:
            # if len(self.img_a) == 91:
                a_nii = np.stack(self.img_a, -1)
                b_nii = np.stack(self.img_b, -1)
                preds_a_nii = np.stack(self.img_preds_a, -1)
                preds_b_nii = np.stack(self.img_preds_b, -1)
                # convert numpy to nii
                a_nii, b_nii, preds_a_nii, preds_b_nii = self.change_numpy_nii(
                    a_nii, b_nii, preds_a_nii, preds_b_nii
                )
                # save nii image to (.nii) file
                self.save_nii(
                    a_nii,
                    b_nii,
                    preds_a_nii,
                    preds_b_nii,
                    subject_number=self.subject_number,
                    folder_path=self.save_folder_name,
                )

                # empty list
                self.img_a = []
                self.img_b = []
                self.img_preds_a = []
                self.img_preds_b = []
                self.subject_number += 1
                self.subject_slice_num.pop(0)
                               

            if self.subject_number > self.subject_number_length:
                log.info(f"Saving test images up to {self.subject_number_length}")
                return

        elif len(res) == 3:
            a, b, preds_a = res

            # # if size of a is bigger than centercrop, then center crop
            # if a.shape[-1] > self.center_crop:
            #     a = CenterCrop(self.center_crop)(a)
            #     b = CenterCrop(self.center_crop)(b)
            #     preds_a = CenterCrop(self.center_crop)(preds_a)

            # Change a,b,preds_a to numpy array
            a, b, preds_a, _ = self.change_torch_numpy(a, b, preds_a, a*0)

            self.img_a.append(a)
            self.img_b.append(b)
            self.img_preds_a.append(preds_a)

            if len(self.img_a) == self.subject_slice_num[0]:
            # if len(self.img_a) == 91:
                a_nii = np.stack(self.img_a, -1)
                b_nii = np.stack(self.img_b, -1)
                preds_a_nii = np.stack(self.img_preds_a, -1)

                # convert numpy to nii
                a_nii, b_nii, preds_a_nii, _ = self.change_numpy_nii(
                    a_nii, b_nii, preds_a_nii, a_nii*0
                )
                # save nii image to (.nii) file
                self.save_nii(
                    a_nii,
                    b_nii,
                    preds_a_nii,
                    None,
                    subject_number=self.subject_number,
                    folder_path=self.save_folder_name,
                )

                # empty list
                self.img_a = []
                self.img_b = []
                self.img_preds_a = []
                self.subject_number += 1
                self.subject_slice_num.pop(0)
                
            if self.subject_number > self.subject_number_length:
                log.info(f"Saving test images up to {self.subject_number_length}")
                return
        else:
            raise NotImplementedError("This function has not been implemented yet.")
        return