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
import torch.nn.functional as nnF

import torchio as tio
import math
import random
from scipy.fft import fftn, ifftn, fftshift, ifftshift

def padding_target_size(tensorA, tensorB, min_size=(256,256), pad_value=-1): # TODO: Adaconv 쓸 시 수정 adaconv -> min_size=(592,384) (16배수), 원래 256,256 (나머지 다)
    """
    Pad and crop the image tensors to the minimum size with padding value.
    If the image size is less than min_size, pad it to min_size.
    If padding needs to be odd, add the extra padding to the top and left.

    Args:
        tensorA (Tensor): Image to be processed.
        tensorB (Tensor): Second Image to be processed.
        min_size (int): Minimum size to pad and crop the image.
        pad_value (float): Value to use for padding.

    Returns:
        Tensor: Processed images.
    """
    if isinstance(min_size, int):
        min_size = (min_size, min_size)

    _, h, w = tensorA.shape

    # Calculate padding
    pad_top = (min_size[0] - h + 1) // 2 if h < min_size[0] else 0
    pad_bottom = (min_size[0] - h) // 2 if h < min_size[0] else 0
    pad_left = (min_size[1] - w + 1) // 2 if w < min_size[1] else 0
    pad_right = (min_size[1] - w) // 2 if w < min_size[1] else 0

    # Apply padding
    if pad_top != 0 or pad_left != 0:
        tensorA = nnF.pad(tensorA, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)
        tensorB = nnF.pad(tensorB, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)

    return tensorA, tensorB

def random_crop(tensorA, tensorB, output_size=(128,128), adjacent_slices=False):
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
    assert len(tensorA.shape) == 3, 'Input tensor A must have 3 dimensions (C, H, W)'
    assert len(tensorB.shape) == 3, 'Input tensor B must have 3 dimensions (C, H, W)'

    slice, h, w = tensorA.shape
    
    # Calculate the top left corner of the random crop

    top = torch.randint(0, h - output_size[0] + 1, size=(1,)).item() if h > output_size[0] else 0
    left = torch.randint(0, w - output_size[1] + 1, size=(1,)).item() if w > output_size[1] else 0
    # top = torch.randint(0, h - output_size[0] + 1, size=(1,)).item()
    # left = torch.randint(0, w - output_size[1] + 1, size=(1,)).item()
    
    if adjacent_slices:  # Apply the same crop to each slice
        cropped_slices_A = [F.crop(tensorA[i], top, left, output_size[0], output_size[1]) for i in range(slice)]
        cropped_slices_B = [F.crop(tensorB[i], top, left, output_size[0], output_size[1]) for i in range(slice)]
        tensorA = torch.stack(cropped_slices_A)  # Stack back to shape (slice_count, output_size[0], output_size[1])
        tensorB = torch.stack(cropped_slices_B)
    else:
        # Perform the crop
        tensorA = F.crop(tensorA, top, left, output_size[0], output_size[1])
        tensorB = F.crop(tensorB, top, left, output_size[0], output_size[1])
        
    return tensorA, tensorB

def even_crop(tensorA, tensorB, target_size):
    """
    Crop the image to the target size evenly from all sides.

    Args:
        tensorA (Tensor): Image to be cropped.
        tensorB (Tensor): Second Image to be cropped.
        target_size (tuple): Desired output size (height, width).

    Returns:
        Tensor: Cropped images.
    """
    _, h, w = tensorA.shape
    new_h, new_w = target_size

    # Calculate cropping dimensions
    top = (h - new_h) // 2
    bottom = h - new_h - top
    left = (w - new_w) // 2
    right = w - new_w - left

    # Crop images
    tensorA = F.crop(tensorA, top, left, new_h, new_w)
    tensorB = F.crop(tensorB, top, left, new_h, new_w)

    return tensorA, tensorB

log = utils.get_pylogger(__name__)

def JHL_deformation(A:np.ndarray,
                    B:np.ndarray,
                    deform_prob:float):
    """Apply dense random elastic deformation.

    Args:
        A (np.ndarray: An input image ndarray.
        deform_prob (float): The probability of performing deformation.

    Returns:
        Tuple[np.ndarray]: A deformed image.
    """

    elastic_transform = tio.transforms.RandomElasticDeformation(
        num_control_points=9,  # Number of control points along each dimension.
        max_displacement=5,    # Maximum displacement along each dimension at each control point.
    )

    elastic_A = elastic_transform(A)
    elastic_B = elastic_transform(B)
    num_samples = A.shape[1]
    num_transform_samples = int(num_samples * deform_prob)
    indices_A = random.sample(range(num_samples), num_transform_samples)
    indices_B = random.sample(range(num_samples), num_transform_samples)

    for index in indices_A: # apply random slice not each patient
        A[:,index,:,:] = elastic_A[:,index,:,:]

    for index in indices_B:
        B[:,index,:,:] = elastic_B[:,index,:,:]

    return A, B

def JHL_motion_region(raw_img, motion_img, prob):
    raw_k = fftshift(fftn(raw_img))
    motion_k = fftshift(fftn(motion_img))
    diff = math.ceil(2.56*prob)

    raw_k[:,128-diff : 128,:] = motion_k[:,128-diff : 128,:]

    res = abs(ifftn(ifftshift(raw_k)))
    return res

def JHL_motion_artifacts(A:np.ndarray,
                         B:np.ndarray,
                         motion_prob:float):
    """Simulates motion artifacts.

    Args:
        A (np.ndarray): An input image ndarray.
        motion_prob (float): The probability of occurrence of motion.

    Returns:
        Tuple[np.ndarray]: A motion artifacts-injected tensor image.
    """

    # Define the 3D-RandomMotion transform
    random_motion = tio.RandomMotion(
        degrees=(5,5),              # Maximum rotation angle in degrees
        translation=(10,10),         # Maximum translation in mm
        num_transforms=10         # Number of motion transformations to apply
    )

    A = np.transpose(A, [0,2,3,1])
    B = np.transpose(B, [0,2,3,1])
    motion_A = random_motion(A)
    motion_A_prob = JHL_motion_region(A, motion_A, prob=6)
    motion_B = random_motion(B)
    motion_B_prob = JHL_motion_region(B, motion_B, prob=6)

    num_samples = A.shape[3]
    num_transform_samples = int(num_samples * motion_prob)
    indices_A = random.sample(range(num_samples), num_transform_samples)
    indices_B = random.sample(range(num_samples), num_transform_samples)

    for index in indices_A:
        A[:,:,:,index] = motion_A_prob[:,:,:,index]

    for index in indices_B:
        B[:,:,:,index] = motion_B_prob[:,:,:,index]

    A = np.transpose(A, [0,3,1,2])
    B = np.transpose(B, [0,3,1,2])

    return A, B

def JHL_rotate_images(A:np.ndarray,
                      B:np.ndarray,
                      degree:float):
    """Rotates a given image (A) by random degree along z dimensions.

    Args:
        A (np.ndarray): An input image ndarray.
        degree (float): Maximum allowable degree along the z dimension.

    Returns:
        Tuple[np.ndarray]: A rotated tensor image.
    """

    # rotation
    transform = tio.RandomAffine(
        scales=0,
        degrees=(degree,0,0),        # z-axis 3D-Rotation range in degrees.
        translation=(0,0,0),
        default_pad_value='otsu',  # edge control, fill value is the mean of the values at the border that lie under an Otsu threshold.
    )

    rotated_A = transform(A)
    rotated_B = transform(B)
    
    return rotated_A, rotated_B

def translate_images(A: torch.Tensor,
              B: torch.Tensor,
              misalign_x: float,
              misalign_y: float):
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

    misalign_x = misalign_x * _misalign_x
    misalign_y = misalign_y * _misalign_y

    # misalign_x = np.random.uniform(-misalign_x, misalign_x, size=2) # This is the previous version
    # misalign_y = np.random.uniform(-misalign_y, misalign_y, size=2)

    translate_params_A = (misalign_y[0], misalign_x[0])  # (y, x) for image A
    translate_params_B = (misalign_y[1], misalign_x[1])  # (y, x) for image B

    # create affine transform
    affine_A = Affine(translate_params=translate_params_A)
    affine_B = Affine(translate_params=translate_params_B)

    moved_A = affine_A(A)
    moved_B = affine_B(B)

    return moved_A[0], moved_B[0]
    
def download_process_IXI(data_dir: str,
                         write_dir: str,
                         misalign_x: float = 0.0,  # maximum misalignment in x direction (float)
                         misalign_y: float = 0.0,  # maximum misalignment in y direction (float)
                         degree: float = 0.0,      # maximum rotation in z direction (float)
                         motion_prob: float = 0.0, # the probability of occurrence of motion.
                         deform_prob: float = 0.0, # the probability of performing deformation.
                         ret: bool = False
                        ):
    """Downloads, preprocesses and saves the IXI dataset. The images are randomly translated.

    Args:
        data_dir (str): The directory where the input dataset is located.
        write_dir (str): The directory where the processed dataset will be saved.
        misalign_x (float): Maximum allowable misalignment along the x dimension. Defaults to 0.0.
        misalign_y (float): Maximum allowable misalignment along the y dimension. Defaults to 0.0.
        degree (float): Maximum rotation in z direction. Defaults to 0.0.
        motion_prob (float): The probability of occurrence of motion. Defaults to 0.0.
        deform_prob (float): The probability of performing deformation. Defaults to 0.0.
        ret (bool): If True, the function also returns the processed dataset. Defaults to False.

    Returns:
        Optional[Tuple[torch.Tensor, torch.Tensor]]: A pair of tensors representing the processed datasets
        for A and B. This is returned only when 'ret' is set to True.
    """

    with h5py.File(data_dir, "r") as f:
        data_A = np.array(f["data_x"][:, :, :, 1:2])
        data_B = np.array(f["data_y"][:, :, :, 1:2])

    # create torch tensors
    data_A = np.transpose(data_A, (3, 2, 0, 1))
    data_B = np.transpose(data_B, (3, 2, 0, 1))

    # ensure that data is in range [-1,1]
    data_A[data_A < 0] = 0
    data_B[data_B < 0] = 0

    data_A, data_B = JHL_deformation(data_A, data_B, deform_prob)
    data_A, data_B = JHL_motion_artifacts(data_A, data_B, motion_prob)

    data_A = (data_A - 0.5) / 0.5
    data_B = (data_B - 0.5) / 0.5

    log.info(f"Preparing the misalignment from <{data_dir}>")

    data_A, data_B = JHL_rotate_images(data_A, data_B, degree)

    for sl in range(data_A.shape[1]):
        A = torch.from_numpy(data_A[:, sl, :, :])
        B = torch.from_numpy(data_B[:, sl, :, :])

        A, B = translate_images(A, B, misalign_x, misalign_y)

        if sl == 0:
            final_data_A = A
            final_data_B = B
        else:
            final_data_A = torch.cat((final_data_A, A), dim=0)
            final_data_B = torch.cat((final_data_B, B), dim=0)

    log.info(f"Saving the prepared dataset to <{write_dir}>")

    with h5py.File(write_dir, "w") as hw:
        hw.create_dataset("data_A", data=final_data_A.numpy())
        hw.create_dataset("data_B", data=final_data_B.numpy())

    if ret:
        return final_data_A, final_data_B
    else:
        return


class dataset_IXI(Dataset):
    def __init__(self, data_dir: str, flip_prob: float = 0.5, rot_prob: float = 0.5, rand_crop: bool = False, reverse=False, adjacent_slices: bool = False, *args, **kwargs):
        """Initializes the IXI dataset for loading during model training/testing.

        Args:
            data_dir (str): The directory where the processed dataset is located.
            flip_prob (float): The probability of flipping the data. Defaults to 0.5.
        """
        super().__init__()
        self.rand_crop = rand_crop
        self.data_dir = data_dir
        self.aug_func = Compose(
            [
                RandFlipd(keys=["A", "B"], prob=flip_prob, spatial_axis=[0, 1]),
                RandRotate90d(keys=["A", "B"], prob=rot_prob, spatial_axes=[0, 1]),
            ]
        )
        self.reverse = reverse
        self.adjacent_slices = adjacent_slices
        self.slices_per_patient = 91  # 각 환자당 91개의 슬라이스가 있다고 가정

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
        os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
        with h5py.File(self.data_dir, "r") as hr:

            A = np.array(hr["data_A"][idx]) # (256,256)
            B = np.array(hr["data_B"][idx])

            if self.adjacent_slices:
                patient_idx = idx // self.slices_per_patient
                slice_idx_within_patient = idx % self.slices_per_patient
                patient_start = patient_idx * self.slices_per_patient
                patient_end = patient_start + self.slices_per_patient - 1
                idx_prev = max(patient_start, idx - 1)  # 이전 슬라이스 (경계 조건 처리)
                idx_next = min(patient_end, idx + 1)    # 다음 슬라이스 (경계 조건 처리)

                A_prev = np.array(hr["data_A"][idx_prev])
                A_next = np.array(hr["data_A"][idx_next])
                B_prev = np.array(hr["data_B"][idx_prev])
                B_next = np.array(hr["data_B"][idx_next])
                
                # Convert to torch and add channel dimension
                A = torch.from_numpy(np.stack([A_prev, A, A_next])) # (3,256,256)
                B = torch.from_numpy(np.stack([B_prev, B, B_next]))           
            else:
                A = torch.from_numpy(A[None]).clone() # (1,256,256)
                B = torch.from_numpy(B[None]).clone()

        A = np.float32(A)
        B = np.float32(B)
        # Create a dictionary for the data
        data_dict = {"A": A, "B": B}

        # Apply the random flipping
        if self.adjacent_slices is False:
            data_dict = self.aug_func(data_dict)

        A = data_dict["A"]
        A = convert_to_tensor(A)
        B = data_dict["B"]
        B = convert_to_tensor(B)

        if self.rand_crop:
            A, B = random_crop(A, B, (128,128), adjacent_slices=self.adjacent_slices) # (1,128,128) or (3,128,128) # only trainset
        if self.reverse:
            return B, A
        else:
            return A, B



def download_process_Kaggle(data_dir: str,
                         write_dir: str,
                         misalign_x: float = 0.0,  # maximum misalignment in x direction (float)
                         misalign_y: float = 0.0,  # maximum misalignment in y direction (float)
                         degree: float = 0.0,      # maximum rotation in z direction (float)
                         motion_prob: float = 0.0, # the probability of occurrence of motion.
                         deform_prob: float = 0.0, # the probability of performing deformation.
                         ret: bool = False
                        ):
    """Downloads, preprocesses and saves the IXI dataset. The images are randomly translated.

    Args:
        data_dir (str): The directory where the input dataset is located.
        write_dir (str): The directory where the processed dataset will be saved.
        misalign_x (float): Maximum allowable misalignment along the x dimension. Defaults to 0.0.
        misalign_y (float): Maximum allowable misalignment along the y dimension. Defaults to 0.0.
        degree (float): Maximum rotation in z direction. Defaults to 0.0.
        motion_prob (float): The probability of occurrence of motion. Defaults to 0.0.
        deform_prob (float): The probability of performing deformation. Defaults to 0.0.
        ret (bool): If True, the function also returns the processed dataset. Defaults to False.

    Returns:
        Optional[Tuple[torch.Tensor, torch.Tensor]]: A pair of tensors representing the processed datasets
        for A and B. This is returned only when 'ret' is set to True.
    """

    with h5py.File(data_dir, "r") as f:
        data_A = np.array(f["data_x"])
        data_B = np.array(f["data_y"])

    # create torch tensors
    # data_A = np.transpose(data_A, (2, 1, 0))
    # data_A = np.flipud(data_A)
    # data_A = np.transpose(data_A, (2, 0, 1))
    data_A = np.expand_dims(data_A, axis=0)

    # data_B = np.transpose(data_B, (2, 1, 0))
    # data_B = np.flipud(data_B)
    # data_B = np.transpose(data_B, [2,0,1])
    data_B = np.expand_dims(data_B, axis=0)

    # ensure that data is in range [-1,1]
    data_A[data_A < 0] = 0
    data_B[data_B < 0] = 0

    data_A, data_B = JHL_deformation(data_A.copy(), data_B.copy(), deform_prob)
    data_A, data_B = JHL_motion_artifacts(data_A, data_B, motion_prob)

    data_A = (data_A - 0.5) / 0.5
    data_B = (data_B - 0.5) / 0.5

    log.info(f"Preparing the misalignment from <{data_dir}>")

    data_A, data_B = JHL_rotate_images(data_A, data_B, degree)

    for sl in range(data_A.shape[1]):
        A = torch.from_numpy(data_A[:, sl, :, :])
        B = torch.from_numpy(data_B[:, sl, :, :])

        A, B = translate_images(A, B, misalign_x, misalign_y)

        if sl == 0:
            final_data_A = A
            final_data_B = B
        else:
            final_data_A = torch.cat((final_data_A, A), dim=0)
            final_data_B = torch.cat((final_data_B, B), dim=0)

    log.info(f"Saving the prepared dataset to <{write_dir}>")

    with h5py.File(write_dir, "w") as hw:
        hw.create_dataset("data_A", data=final_data_A.numpy())
        hw.create_dataset("data_B", data=final_data_B.numpy())

    if ret:
        return final_data_A, final_data_B
    else:
        return
    


class dataset_Kaggle(Dataset):
    def __init__(self, data_dir: str, flip_prob: float = 0.5, rot_prob: float = 0.5, rand_crop: bool = False, *args, **kwargs):
        """Initializes the IXI dataset for loading during model training/testing.

        Args:
            data_dir (str): The directory where the processed dataset is located.
            flip_prob (float): The probability of flipping the data. Defaults to 0.5.
        """
        super().__init__()
        self.rand_crop = rand_crop
        self.data_dir = data_dir
        self.aug_func = Compose(
            [
                RandFlipd(keys=["A", "B"], prob=flip_prob, spatial_axis=[0, 1]),
                RandRotate90d(keys=["A", "B"], prob=rot_prob, spatial_axes=[0, 1]),
            ]
        )

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
        os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
        with h5py.File(self.data_dir, "r") as hr:
            A = np.array(hr["data_A"][idx])
            B = np.array(hr["data_B"][idx])
        A = torch.from_numpy(A[None]).clone()
        B = torch.from_numpy(B[None]).clone()
        # Create a dictionary for the data
        data_dict = {"A": A, "B": B}

        # Apply the random flipping
        data_dict = self.aug_func(data_dict)

        A = data_dict["A"]
        A = convert_to_tensor(A)
        B = data_dict["B"]
        B = convert_to_tensor(B)

        if self.rand_crop:
            A, B = random_crop(A, B, (192,192))

        return A, B
    

def download_process_SynthRAD_MR_CT_Pelvis(data_dir: str, # h5를 만들기위한 함수. nifti -> h5로 해야해.
                         write_dir: str,
                         misalign_x: float = 0.0,  # maximum misalignment in x direction (float)
                         misalign_y: float = 0.0,  # maximum misalignment in y direction (float)
                         degree: float = 0.0,      # maximum rotation in z direction (float)
                         motion_prob: float = 0.0, # the probability of occurrence of motion.
                         deform_prob: float = 0.0, # the probability of performing deformation.
                         ret: bool = False
                        ):
    #TODO:
    #1. nifti를 가져올거야 (data_dir + 1PC + mr.nii.gz등)
    #2. dataset으로 근데 random crop 적용
    #3. 그다음 h5로 저장
    
    
    with h5py.File(data_dir, "r") as f: #TODO nifti로 가져오도록 코드 바꿔주기.
        data_A = np.array(f["data_x"])
        data_B = np.array(f["data_y"])

    # create torch tensors
    data_A = np.transpose(data_A, (2, 0, 1))
    data_A = np.expand_dims(data_A, axis=0)
    print('data_A shape: ', data_A.shape)

    data_B = np.transpose(data_B, (2, 0, 1))
    data_B = np.expand_dims(data_B, axis=0)
    print('data_B shape: ', data_B.shape)

    # ensure that data is in range [-1,1]
    data_A, data_B = JHL_deformation(data_A.copy(), data_B.copy(), deform_prob)
    data_A, data_B = JHL_motion_artifacts(data_A, data_B, motion_prob)

    data_A = (data_A - 0.5) / 0.5
    data_B = (data_B - 0.5) / 0.5

    log.info(f"Preparing the misalignment from <{data_dir}>")

    data_A, data_B = JHL_rotate_images(data_A, data_B, degree)

    for sl in range(data_A.shape[1]):
        A = torch.from_numpy(data_A[:, sl, :, :])
        B = torch.from_numpy(data_B[:, sl, :, :])

        A, B = translate_images(A, B, misalign_x, misalign_y)

        if sl == 0:
            final_data_A = A
            final_data_B = B
        else:
            final_data_A = torch.cat((final_data_A, A), dim=0)
            final_data_B = torch.cat((final_data_B, B), dim=0)

    log.info(f"Saving the prepared dataset to <{write_dir}>")

    with h5py.File(write_dir, "w") as hw:
        hw.create_dataset("data_A", data=final_data_A.numpy())
        hw.create_dataset("data_B", data=final_data_B.numpy())

    if ret:
        return final_data_A, final_data_B
    else:
        return

# TODO: Proposed 빨리 돌리기위해서 ram에서 돌아가도록 바꾼 코드
class dataset_SynthRAD_MR_CT_Pelvis_RAM(Dataset):
    def __init__(self, data_dir: str, flip_prob: float = 0.5, rot_prob: float = 0.5, rand_crop: bool = False, reverse=False, *args, **kwargs):
        super().__init__()
        self.rand_crop = rand_crop
        self.data_dir = data_dir
        self.reverse = reverse
        
        # Each patient has a different number of slices        
        with h5py.File(data_dir, 'r') as file:
            self.patient_keys = list(file['MR'].keys())
            
            # Loading all MR and CT data into memory
            self.MR_data = [file['MR'][key][:] for key in self.patient_keys]
            self.CT_data = [file['CT'][key][:] for key in self.patient_keys]
            
            self.slice_counts = [data.shape[-1] for data in self.MR_data]
            self.cumulative_slice_counts = np.cumsum([0] + self.slice_counts)
        
        self.aug_func = Compose(
            [
                RandFlipd(keys=["A", "B"], prob=flip_prob, spatial_axis=[0, 1]),
                RandRotate90d(keys=["A", "B"], prob=rot_prob, spatial_axes=[0, 1]),
            ]
        )

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

        # Retrieve the data from memory instead of reading from the h5py file
        A = self.MR_data[patient_idx][..., slice_idx]
        B = self.CT_data[patient_idx][..., slice_idx]

        A = torch.from_numpy(A).unsqueeze(0).float()
        B = torch.from_numpy(B).unsqueeze(0).float()

        # Create a dictionary for the data
        data_dict = {"A": A, "B": B}

        # Apply the random flipping
        data_dict = self.aug_func(data_dict)

        A = data_dict["A"]
        A = convert_to_tensor(A)
        B = data_dict["B"]
        B = convert_to_tensor(B)

        if self.rand_crop:
            A, B = random_crop(A, B, (320,192))
        else:
            _, h, w = A.shape
            A, B = random_crop(A, B, (h//4*4,w//4*4)) # under nearest multiple of four
        
        if self.reverse:
            return B, A
        else:
            return A, B
    
#원래코드
class dataset_SynthRAD_MR_CT_Pelvis(Dataset):
    def __init__(self, data_dir: str, flip_prob: float = 0.5, rot_prob: float = 0.5, rand_crop: bool = False, reverse=False, padding: bool = False, *args, **kwargs):
        super().__init__()
        self.rand_crop = rand_crop
        self.data_dir = data_dir
        self.reverse = reverse
        self.padding = padding

        os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
        
        # Each patient has a different number of slices        
        self.patient_keys = []
        with h5py.File(self.data_dir, 'r') as file:
            self.patient_keys = list(file['MR'].keys())
            self.slice_counts = [file['MR'][key].shape[-1] for key in self.patient_keys]
            self.cumulative_slice_counts = np.cumsum([0] + self.slice_counts)
        
        self.aug_func = Compose(
            [
                RandFlipd(keys=["A", "B"], prob=flip_prob, spatial_axis=[0, 1]),
                RandRotate90d(keys=["A", "B"], prob=rot_prob, spatial_axes=[0, 1]),
            ]
        )

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
            A = file['MR'][patient_key][..., slice_idx]
            B = file['CT'][patient_key][..., slice_idx]

        A = torch.from_numpy(A).unsqueeze(0).float()
        B = torch.from_numpy(B).unsqueeze(0).float()

        # Create a dictionary for the data
        data_dict = {"A": A, "B": B}

        # TODO: auf func 원래 위치

        A = data_dict["A"]
        A = convert_to_tensor(A)
        B = data_dict["B"]
        B = convert_to_tensor(B)

        if self.padding:
            A, B = padding_target_size(A, B)

        # Apply the random flipping
        data_dict = self.aug_func(data_dict)

        if self.rand_crop:
            # A, B = random_crop(A, B, (320,192)) # 이게 지금까지 계속 써왔던 것
            A, B = random_crop(A, B, (256,256)) # resvit 용
            # A, B = random_crop(A, B, (416,256)) # 이건 weight 맞출때
        else:
            _, h, w = A.shape
            # A, B = even_crop(A, B, (256,256)) # under nearest multiple of 14 # TODO: resvit돌리기위한 코드
            A, B = even_crop(A, B, (h//16*16,w//16*16)) # under nearest multiple of 16 # adaconv랑 다른것들도 다 이거
            # A, B = even_crop(A, B, (h//4*4,w//4*4)) # under nearest multiple of 16 
            # A, B = random_crop(A, B, (h//4*4,w//4*4)) # under nearest multiple of four

        if self.reverse:
            return B, A
        else:
            return A, B
            
    def get_patient_slice_idx(self, idx):
        """ 주어진 샘플 인덱스에 대한 환자 인덱스와 슬라이스 인덱스를 반환합니다. """
        patient_idx = np.searchsorted(self.cumulative_slice_counts, idx+1) - 1
        slice_idx = idx - self.cumulative_slice_counts[patient_idx]
        return patient_idx, slice_idx
    


#원래코드
class dataset_SynthRAD_EPI_MPRAGE_Pelvis(Dataset):
    def __init__(self, data_dir: str, flip_prob: float = 0.5, rot_prob: float = 0.5, rand_crop: bool = False, reverse=False, padding: bool = False, *args, **kwargs):
        super().__init__()
        self.rand_crop = rand_crop
        self.data_dir = data_dir
        self.reverse = reverse
        self.padding = padding

        os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
        
        # Each patient has a different number of slices        
        self.patient_keys = []
        with h5py.File(self.data_dir, 'r') as file:
            self.patient_keys = list(file['epi'].keys())
            self.slice_counts = [file['epi'][key].shape[-1] for key in self.patient_keys]
            self.cumulative_slice_counts = np.cumsum([0] + self.slice_counts)
        
        self.aug_func = Compose(
            [
                RandFlipd(keys=["A", "B"], prob=flip_prob, spatial_axis=[0, 1]),
                RandRotate90d(keys=["A", "B"], prob=rot_prob, spatial_axes=[0, 1]),
            ]
        )

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
            A = file['epi'][patient_key][..., slice_idx]
            B = file['mprage'][patient_key][..., slice_idx]

        A = torch.from_numpy(A).unsqueeze(0).float()
        B = torch.from_numpy(B).unsqueeze(0).float()

        # Create a dictionary for the data
        data_dict = {"A": A, "B": B}

        # TODO: auf func 원래 위치

        A = data_dict["A"]
        A = convert_to_tensor(A)
        B = data_dict["B"]
        B = convert_to_tensor(B)

        if self.padding:
            A, B = padding_target_size(A, B)

        # Apply the random flipping
        data_dict = self.aug_func(data_dict)

        if self.rand_crop:
            # A, B = random_crop(A, B, (320,192)) # 이게 지금까지 계속 써왔던 것
            A, B = random_crop(A, B, (256,256)) # resvit 용
            # A, B = random_crop(A, B, (416,256)) # 이건 weight 맞출때
        else:
            _, h, w = A.shape
            # A, B = even_crop(A, B, (256,256)) # under nearest multiple of 14 # TODO: resvit돌리기위한 코드
            A, B = even_crop(A, B, (h//16*16,w//16*16)) # under nearest multiple of 16 # adaconv랑 다른것들도 다 이거
            # A, B = even_crop(A, B, (h//4*4,w//4*4)) # under nearest multiple of 16 
            # A, B = random_crop(A, B, (h//4*4,w//4*4)) # under nearest multiple of four

        if self.reverse:
            return B, A
        else:
            return A, B
            
    def get_patient_slice_idx(self, idx):
        """ 주어진 샘플 인덱스에 대한 환자 인덱스와 슬라이스 인덱스를 반환합니다. """
        patient_idx = np.searchsorted(self.cumulative_slice_counts, idx+1) - 1
        slice_idx = idx - self.cumulative_slice_counts[patient_idx]
        return patient_idx, slice_idx