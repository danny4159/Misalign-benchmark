from typing import Any, Dict, Optional
from monai.transforms import Affine
import numpy as np
import torch
import h5py
from misalign import utils
import os
from torch.utils.data import Dataset
from monai.transforms import RandFlipd, RandRotate90d, Compose, RandCropd
import monai
from monai.utils.type_conversion import convert_to_tensor
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import CenterCrop
from torchvision.transforms.functional import center_crop

import torchio as tio
import math
import random
from scipy.fft import fftn, ifftn, fftshift, ifftshift
from scipy.ndimage import binary_dilation, generate_binary_structure

log = utils.get_pylogger(__name__)

def random_crop(tensorA, tensorB, output_size=(128, 128)):
    """
    Crop randomly the image in a sample.

    Args:
        tensor (Tensor): Image to be cropped.
        output_size (tuple or int): Desired output size. If int, square crop
            is made.

    Returns:
        Tensor: Cropped image.
    """
    # Handle the case where the output size is an integer
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    # Ensure the tensor has the correct dimensions
    assert len(tensorA.shape) == 3, "Input tensor A must have 3 dimensions (C, H, W)"
    assert len(tensorB.shape) == 3, "Input tensor B must have 3 dimensions (C, H, W)"

    _, h, w = tensorA.shape

    # Calculate the top left corner of the random crop
    top = torch.randint(0, h - output_size[0] + 1, size=(1,)).item()
    left = torch.randint(0, w - output_size[1] + 1, size=(1,)).item()

    # Perform the crop
    tensorA = F.crop(tensorA, top, left, output_size[0], output_size[1])
    tensorB = F.crop(tensorB, top, left, output_size[0], output_size[1])

    return tensorA, tensorB

def random_crop2(tensorA, tensorB, tensorC, output_size=(128, 128)):
    """
    Crop randomly the image in a sample.

    Args:
        tensor (Tensor): Image to be cropped.
        output_size (tuple or int): Desired output size. If int, square crop
            is made.

    Returns:
        Tensor: Cropped image.
    """
    # Handle the case where the output size is an integer
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    # Ensure the tensor has the correct dimensions
    assert len(tensorA.shape) == 3, "Input tensor A must have 3 dimensions (C, H, W)"
    assert len(tensorB.shape) == 3, "Input tensor B must have 3 dimensions (C, H, W)"
    assert len(tensorC.shape) == 3, "Input tensor C must have 3 dimensions (C, H, W)"

    _, h, w = tensorA.shape

    # Calculate the top left corner of the random crop
    top = torch.randint(0, h - output_size[0] + 1, size=(1,)).item()
    left = torch.randint(0, w - output_size[1] + 1, size=(1,)).item()

    # Perform the crop
    tensorA = F.crop(tensorA, top, left, output_size[0], output_size[1])
    tensorB = F.crop(tensorB, top, left, output_size[0], output_size[1])
    tensorC = F.crop(tensorC, top, left, output_size[0], output_size[1])

    return tensorA, tensorB, tensorC


def motion_region(raw_img, motion_img, region=5):
    # raw_img : numpy
    # motion_img : numpy
    raw_k = fftshift(fftn(raw_img, axes=(-1, -2)), axes=(-1, -2))
    motion_k = fftshift(fftn(motion_img, axes=(-1, -2)), axes=(-1, -2))
    width = raw_k.shape[-1] // 2  # center
    diff = math.ceil(2.56 * region)

    raw_k[..., width - diff : width - 5] = motion_k[..., width - diff : width - 5]
    raw_k[..., width + 5 : width + diff] = motion_k[..., width + 5 : width + diff]

    res = np.real(ifftn(ifftshift(raw_k, axes=(-1, -2)), axes=(-1, -2)))
    return res


def motion_artifact(img_A, region_range=None):
    random_motion = tio.RandomMotion(
        degrees=(5, 5),  # Maximum rotation angle in degrees
        translation=(10, 10),  # Maximum translation in mm
        num_transforms=10,  # Number of motion transformations to apply
    )
    im_a = img_A.numpy()
    motion_A = random_motion(im_a[None])
    if region_range is None:
        region = np.random.randint(8, 10)
    else:
        region = region_range
    motion_A = motion_region(im_a, motion_A[0], region)

    return torch.from_numpy(motion_A)


def translate_images(
    A: torch.Tensor,
    B: torch.Tensor,
    misalign_x: float,
    misalign_y: float,
    degree: float,
):
    """Translates two given images (A and B) by random misalignments along x and y dimensions.

    Args:
        A (torch.Tensor): The first input image tensor.
        B (torch.Tensor): The second input image tensor.
        misalign_x (float): Maximum allowable misalignment along the x dimension.
        misalign_y (float): Maximum allowable misalignment along the y dimension.

    Returns:
        Tuple[torch.Tensor]: A pair of tensors representing the translated images.
    """

    # translation (changed : misalign is the smae but magnitude is different)
    _misalign_x = np.random.uniform(-1, 1, size=2)
    _misalign_y = np.random.uniform(-1, 1, size=2)
    _degree = np.random.uniform(-1, 1, size=2)

    misalign_x = misalign_x * _misalign_x
    misalign_y = misalign_y * _misalign_y
    rot_degree = _degree * degree * 1 / 180 * math.pi

    # misalign_x = np.random.uniform(-misalign_x, misalign_x, size=2) # This is the previous version
    # misalign_y = np.random.uniform(-misalign_y, misalign_y, size=2)

    translate_params_A = (misalign_y[0], misalign_x[0])  # (y, x) for image A
    translate_params_B = (misalign_y[1], misalign_x[1])  # (y, x) for image B

    # create affine transform
    affine_A = Affine(
        translate_params=translate_params_A,
        rotate_params=rot_degree[0],
        padding_mode="reflection",
        mode="nearest",
    )
    affine_B = Affine(
        translate_params=translate_params_B,
        rotate_params=rot_degree[1],
        padding_mode="reflection",
        mode="nearest",
    )

    moved_A = affine_A(A)
    moved_B = affine_B(B)

    return moved_A[0], moved_B[0]


class dataset_IXI_FLY(Dataset):
    def __init__(
        self,
        data_dir: str,
        rand_crop: bool = False,
        misalign_x: float = 0.0,
        misalign_y: float = 0.0,
        degree: float = 0.0,
        motion_prob: float = 0.0,
        deform_prob: float = 0.0,
        aug: bool = False,
        reverse: bool = False,
        return_msk: bool = False,
        crop_size=256,
    ):
        super().__init__()
        self.rand_crop = rand_crop
        self.data_dir = data_dir
        if return_msk:
            self.aug_func = Compose(
                [
                    RandFlipd(keys=["A", "B", "M"], prob=0.5, spatial_axis=[0, 1]),
                    RandRotate90d(keys=["A", "B", "M"], prob=0.5, spatial_axes=[0, 1]),
                ]
            )

        else:
            self.aug_func = Compose(
                [
                    RandFlipd(keys=["A", "B"], prob=0.5, spatial_axis=[0, 1]),
                    RandRotate90d(keys=["A", "B"], prob=0.5, spatial_axes=[0, 1]),
                ]
            )


        self.misalign_x = misalign_x
        self.misalign_y = misalign_y
        self.degree = degree
        self.motion_prob = motion_prob
        self.deform_prob = deform_prob
        self.aug = aug
        self.reverse = reverse
        self.return_msk = return_msk
        self.crop_size = crop_size

    def __len__(self):
        """Returns the number of samples in the dataset."""
        os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
        with h5py.File(self.data_dir, "r") as f:
            siz = len(f["data_A"])
        return siz
    
    def __getitem__(self, idx):
        """Fetches a sample from the dataset given an index.

        Args:
            idx (int): The index for the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of tensors representing the samples for A and B.
        """
        os.environ["HDF5_USE_FILE_LOCKING"] = "True"
        with h5py.File(self.data_dir, "r") as hr:
            A = hr["data_A"][idx]
            if (
                self.deform_prob > 0
                and idx > 2
                and idx < self.__len__() - 2
                and np.random.rand() < self.deform_prob
            ):
                idx_new = np.random.randint(idx + 1, idx + 2)
                A = hr["data_A"][idx]
                B = hr["data_B"][idx_new]
            else:
                B = hr["data_B"][idx]

            if self.return_msk:
                M = hr["data_M"][idx]
                M = torch.from_numpy(M[None])

            A = A.astype(np.float32)
            B = B.astype(np.float32)

            A = torch.from_numpy(A[None])
            B = torch.from_numpy(B[None])

        # Create a dictionary for the data
        if self.return_msk:
            data_dict = {"A": A, "B": B, "M": M}
        else:
            data_dict = {"A": A, "B": B}

        # Apply the random flipping
        if self.aug:
            data_dict = self.aug_func(data_dict)

        A = data_dict["A"]
        A = convert_to_tensor(A)
        B = data_dict["B"]
        B = convert_to_tensor(B)

        if self.return_msk:
            M = data_dict["M"]
            M = convert_to_tensor(M)

        # Perform misalignment (Rigid)
        # First: translation
        if self.misalign_x == 0 and self.misalign_y == 0 and self.degree == 0:
            pass
        else:
            A, B = translate_images(A, B, self.misalign_x, self.misalign_y, self.degree)

        if np.random.rand() < self.motion_prob:
            if self.reverse:
                A = motion_artifact(A)  # A is the label
            else:
                B = motion_artifact(B)  # B is the label

        A, B = torch.clamp(A, min=-1, max=1), torch.clamp(
            B, min=-1, max=1
        )  # make sure -1, 1
        
        if self.rand_crop:
            if self.return_msk:
                A, B, M = random_crop2(A, B, M, (self.crop_size, self.crop_size))

            else:
                A, B = random_crop(A, B, (self.crop_size, self.crop_size))

        if self.reverse:
            if self.return_msk:
                return B, A, M
            else:
                return B, A
        else:
            if self.return_msk:
                return A, B, M
            else:
                return A, B

###################################################
class dataset_synthRAD_FLY_RAM(Dataset):
    def __init__(
        self,
        data_dir: str,
        rand_crop: bool = False,
        misalign_x: float = 0.0,
        misalign_y: float = 0.0,
        degree: float = 0.0,
        motion_prob: float = 0.0,
        deform_prob: float = 0.0,
        aug: bool = False,
        reverse: bool = False,
        return_msk: bool = False,
        crop_size=256,
    ):
        super().__init__()
        self.rand_crop = rand_crop
        self.data_dir = data_dir
        
        if return_msk:
            self.aug_func = Compose(
                [
                    RandFlipd(keys=["A", "B", "M"], prob=0.5, spatial_axis=[0, 1]),
                    RandRotate90d(keys=["A", "B", "M"], prob=0.5, spatial_axes=[0, 1]),
                ]
            )

        else:
            self.aug_func = Compose(
                [
                    RandFlipd(keys=["A", "B"], prob=0.5, spatial_axis=[0, 1]),
                    RandRotate90d(keys=["A", "B"], prob=0.5, spatial_axes=[0, 1]),
                ]
            )

        self.misalign_x = misalign_x
        self.misalign_y = misalign_y
        self.degree = degree
        self.motion_prob = motion_prob
        self.deform_prob = deform_prob
        self.aug = aug
        self.reverse = reverse
        self.return_msk = return_msk
        self.crop_size = crop_size

        #RAM으로 하는 코드
        with h5py.File(self.data_dir, 'r') as file:
            self.patient_keys = list(file['MR'].keys())
            self.MR_data = [file['MR'][key][:] for key in self.patient_keys]
            self.CT_data = [file['CT'][key][:] for key in self.patient_keys]
            if self.return_msk:
                self.MASK_data = [file['MASK'][key][:] for key in self.patient_keys]
            else:
                self.MASK_data = [None] * len(self.patient_keys)
            self.slice_counts = [data.shape[-1] for data in self.MR_data]
            self.cumulative_slice_counts = np.cumsum([0] + self.slice_counts)

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return self.cumulative_slice_counts[-1]

    def __getitem__(self, idx):
        """Fetches a sample from the dataset given an index.

        Args:
            idx (int): The index for the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of tensors representing the samples for A and B.
        """

        # RAM 코드
        patient_idx = np.searchsorted(self.cumulative_slice_counts, idx+1) - 1
        slice_idx = idx - self.cumulative_slice_counts[patient_idx]
        
        # with h5py.File(self.data_dir, "r") as hr:
        # A = self.h5file["MR"][patient_key][..., slice_idx]
        # if (
        #     self.deform_prob > 0
        #     and idx > 2
        #     and idx < self.__len__() - 2
        #     and np.random.rand() < self.deform_prob
        # ):
        #     slice_idx_new = np.random.randint(slice_idx + 1, slice_idx + 2)
        #     A = self.h5file["MR"][patient_key][..., slice_idx]
        #     B = self.h5file["CT"][patient_key][..., slice_idx_new]
        # else:
        # B = self.h5file["CT"][patient_key][..., slice_idx]

        #RAM 코드
        A = self.MR_data[patient_idx][..., slice_idx]
        B = self.CT_data[patient_idx][..., slice_idx]
        if self.return_msk:
            M = self.MASK_data[patient_idx][..., slice_idx]
            M = torch.from_numpy(M[None])

        # if self.return_msk:
        #     M = self.h5file["MASK"][patient_key][..., slice_idx]
        #     M = torch.from_numpy(M[None])

        A = A.astype(np.float32)
        B = B.astype(np.float32)

        A = torch.from_numpy(A[None])
        B = torch.from_numpy(B[None])

        # Create a dictionary for the data
        if self.return_msk:
            data_dict = {"A": A, "B": B, "M": M}
        else:
            data_dict = {"A": A, "B": B}

        # Apply the random flipping
        if self.aug:
            data_dict = self.aug_func(data_dict)

        A = data_dict["A"]
        A = convert_to_tensor(A)
        B = data_dict["B"]
        B = convert_to_tensor(B)

        if self.return_msk:
            M = data_dict["M"]
            M = convert_to_tensor(M)

        # Perform misalignment (Rigid)
        # First: translation
        # if self.misalign_x == 0 and self.misalign_y == 0 and self.degree == 0:
        #     pass
        # else:
        #     A, B = translate_images(A, B, self.misalign_x, self.misalign_y, self.degree)

        # if np.random.rand() < self.motion_prob:
        #     if self.reverse:
        #         A = motion_artifact(A)  # A is the label
        #     else:
        #         B = motion_artifact(B)  # B is the label

        A, B = torch.clamp(A, min=-1, max=1), torch.clamp(
            B, min=-1, max=1
        )  # make sure -1, 1
        
        if self.rand_crop:
            if self.return_msk:
                A, B, M = random_crop2(A, B, M, (self.crop_size, self.crop_size))

            else:
                A, B = random_crop(A, B, (self.crop_size, self.crop_size))

        if self.reverse:
            if self.return_msk:
                return B, A, M
            else:
                return B, A
        else:
            if self.return_msk:
                return A, B, M
            else:
                return A, B
            
# 원래코드
class dataset_synthRAD_FLY(Dataset):
    def __init__(
        self,
        data_dir: str,
        rand_crop: bool = False,
        misalign_x: float = 0.0,
        misalign_y: float = 0.0,
        degree: float = 0.0,
        motion_prob: float = 0.0,
        deform_prob: float = 0.0,
        aug: bool = False,
        reverse: bool = False,
        return_msk: bool = False,
        crop_size=256,
    ):
        super().__init__()
        self.rand_crop = rand_crop
        self.data_dir = data_dir
        
        os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"

        # Each patient has a different number of slices        
        self.patient_keys = []
        with h5py.File(self.data_dir, 'r') as file:
            self.patient_keys = list(file['MR'].keys())
            self.slice_counts = [file['MR'][key].shape[-1] for key in self.patient_keys]
            self.cumulative_slice_counts = np.cumsum([0] + self.slice_counts)
        
                
        if return_msk:
            self.aug_func = Compose(
                [
                    RandFlipd(keys=["A", "B", "M"], prob=0.5, spatial_axis=[0, 1]),
                    RandRotate90d(keys=["A", "B", "M"], prob=0.5, spatial_axes=[0, 1]),
                ]
            )

        else:
            self.aug_func = Compose(
                [
                    RandFlipd(keys=["A", "B"], prob=0.5, spatial_axis=[0, 1]),
                    RandRotate90d(keys=["A", "B"], prob=0.5, spatial_axes=[0, 1]),
                ]
            )

        self.misalign_x = misalign_x
        self.misalign_y = misalign_y
        self.degree = degree
        self.motion_prob = motion_prob
        self.deform_prob = deform_prob
        self.aug = aug
        self.reverse = reverse
        self.return_msk = return_msk
        self.crop_size = crop_size

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return self.cumulative_slice_counts[-1]

    def __getitem__(self, idx):
        """Fetches a sample from the dataset given an index.

        Args:
            idx (int): The index for the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of tensors representing the samples for A and B.
        """
        patient_idx = np.searchsorted(self.cumulative_slice_counts, idx+1) - 1
        slice_idx = idx - self.cumulative_slice_counts[patient_idx]
        patient_key = self.patient_keys[patient_idx]

        with h5py.File(self.data_dir, 'r') as file:
            A = file["MR"][patient_key][..., slice_idx]
        # if (
        #     self.deform_prob > 0
        #     and idx > 2
        #     and idx < self.__len__() - 2
        #     and np.random.rand() < self.deform_prob
        # ):
        #     slice_idx_new = np.random.randint(slice_idx + 1, slice_idx + 2)
        #     A = self.h5file["MR"][patient_key][..., slice_idx]
        #     B = self.h5file["CT"][patient_key][..., slice_idx_new]
        # else:
            B = file["CT"][patient_key][..., slice_idx]

            if self.return_msk:
                M = file["MASK"][patient_key][..., slice_idx]
                M = torch.from_numpy(M[None])

        A = A.astype(np.float32)
        B = B.astype(np.float32)

        A = torch.from_numpy(A[None])
        B = torch.from_numpy(B[None])

        # Create a dictionary for the data
        if self.return_msk:
            data_dict = {"A": A, "B": B, "M": M}
        else:
            data_dict = {"A": A, "B": B}

        # Apply the random flipping
        if self.aug:
            data_dict = self.aug_func(data_dict)

        A = data_dict["A"]
        A = convert_to_tensor(A)
        B = data_dict["B"]
        B = convert_to_tensor(B)

        if self.return_msk:
            M = data_dict["M"]
            M = convert_to_tensor(M)

        # Perform misalignment (Rigid)
        # First: translation
        # if self.misalign_x == 0 and self.misalign_y == 0 and self.degree == 0:
        #     pass
        # else:
        #     A, B = translate_images(A, B, self.misalign_x, self.misalign_y, self.degree)

        # if np.random.rand() < self.motion_prob:
        #     if self.reverse:
        #         A = motion_artifact(A)  # A is the label
        #     else:
        #         B = motion_artifact(B)  # B is the label

        A, B = torch.clamp(A, min=-1, max=1), torch.clamp(
            B, min=-1, max=1
        )  # make sure -1, 1
        
        if self.rand_crop:
            if self.return_msk:
                A, B, M = random_crop2(A, B, M, (self.crop_size, self.crop_size))

            else:
                A, B = random_crop(A, B, (self.crop_size, self.crop_size))

        if self.reverse:
            if self.return_msk:
                return B, A, M
            else:
                return B, A
        else:
            if self.return_msk:
                return A, B, M
            else:
                return A, B            