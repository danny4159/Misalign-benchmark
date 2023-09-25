from typing import Any, Dict, Optional

import os
import torch
import h5py
import numpy as np
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from misalign.data.components.transforms import download_process_Kaggle, dataset_Kaggle

class KaggleDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        misalign_x: float = 0.0, # maximum misalignment in x direction (float)
        misalign_y: float = 0.0, # maximum misalignment in y direction (float)
        degree: float = 0.0, # The rotation range in z axis (float)
        motion_prob: float = 0.0, # The probability of occurrence of motion (float)
        deform_prob: float = 0.0, # Deformation probability (float)
        flip_prob: float = 0.0, # augmentation for training (flip)
        rot_prob: float = 0.0, # augmentation for training (rot90)
        rand_crop: bool = False, # augmentation for training (random crop)
        batch_size: int = 64,
        num_workers: int = 5,
        pin_memory: bool = False,
        **kwargs: Any
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.misalign_x = misalign_x
        self.misalign_y = misalign_y
        self.degree = degree
        self.motion_prob = motion_prob
        self.deform_prob = deform_prob
        
        self.data_dir = data_dir
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def misalign(self):
        return 'Misalignment x:{}, y:{}, R:{}, M:{}, D:{}'.format(self.misalign_x, self.misalign_y, self.degree, self.motion_prob, self.deform_prob)

    def prepare_data(self):
        """Prepares the data for usage.

        This function is responsible for the misalignment of training data and saving the data to hdf5 format.
        It doesn't assign any state variables.
        """
        
        for phase in ['train','val','test']:
            target_file = os.path.join(self.data_dir, phase, 'masked_data.mat')
            
            if phase == 'train': # misalign only for training data
                mis_x, mis_y, Rot_z, M_prob, D_prob = self.misalign_x, self.misalign_y, self.degree, self.motion_prob, self.deform_prob
                write_dir = os.path.join(self.data_dir, phase, 'prepared_data_{}_{}_{}_{}_{}.h5'.format(mis_x,mis_y,Rot_z,M_prob,D_prob)) # save to hdf5
                self.train_dir = write_dir

            
            elif phase == 'val' or 'test': # no misalignment for validation and test data
                mis_x, mis_y, Rot_z, M_prob, D_prob = 0.0, 0.0, 0.0, 0.0, 0.0
                write_dir = os.path.join(self.data_dir, phase, 'prepared_data_{}_{}_{}_{}_{}.h5'.format(mis_x,mis_y,Rot_z,M_prob,D_prob)) # save to hdf5  
                if phase == 'val':
                    self.val_dir = write_dir
                elif phase == 'test':
                    self.test_dir = write_dir
                
            if os.path.exists(write_dir):
                print('path exists for {}'.format(write_dir))
            else:
                download_process_Kaggle(target_file, write_dir, mis_x, mis_y, Rot_z, M_prob, D_prob) # call function
        
        
    def setup(self, stage: Optional[str] = None):
        """Sets up the datasets.

        This function is responsible for loading the data and assigning the datasets.

        Args:
            stage (str, optional): The stage for which to setup the data. Can be None, 'fit' or 'test'. Defaults to None.
        """
        # load and split datasets only if not loaded already
        self.data_train = dataset_Kaggle(self.train_dir, flip_prob=self.hparams.flip_prob, rot_prob=self.hparams.rot_prob, rand_crop=self.hparams.rand_crop) # Use flip and crop augmentation for training data
        self.data_val = dataset_Kaggle(self.val_dir, flip_prob=0.0, rot_prob=0.0)
        self.data_test = dataset_Kaggle(self.test_dir, flip_prob=0.0, rot_prob=0.0)
     

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = KaggleDataModule('C:/Users/NUGURI/Workspace/misalign-benchmark/data/Kaggle')
    _.prepare_data()
    _.setup()
