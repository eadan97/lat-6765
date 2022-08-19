import os
from typing import Optional, Tuple

import numpy as np
from pl_bolts.datamodules import AsynchronousLoader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from src.datamodules.datasets.class_restricted_image_dataset import ClassRestrictedImageDataset
from src.datamodules.datasets.image_dataset import ImageDataset


class CycleGanDatamodule(LightningDataModule):
    """
    LightningDataModule for CycleGan datasets.
    Datasets consists of images from two domains.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
            self,
            image_size: int = 256,
            image_margin: int = 30,
            data_dir: str = "data",
            dataset_name: str = "",
            train_val_test_split: Tuple = (0.8, 0.1, 0.1),
            dataset_percentage: float = 1.0,
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            use_gpu: bool = False,
            serial_batches: bool = False,
            restricted: bool = False,
            **kwargs,
    ):
        super().__init__()
        assert len(train_val_test_split) == 2 or len(
            train_val_test_split) == 3, "there should be only two or three splits"
        self.share_val_test = len(train_val_test_split) == 2
        assert np.abs(train_val_test_split[0] + train_val_test_split[1] - 1) <= 1e-8 if self.share_val_test else np.abs(
            train_val_test_split[0] + train_val_test_split[1] + train_val_test_split[
                2] - 1) <= 1e-8, "the splits should add up to 1"

        self.data_dir = os.path.join(data_dir, dataset_name)
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_gpu = use_gpu
        self.restricted = restricted
        self.serial_batches = serial_batches
        self.percentage = dataset_percentage

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size + image_margin),
            transforms.RandomCrop(image_size),
            # transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # self.dims is returned when you call datamodule.size()
        self.dims = (3, image_size, image_size)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        if self.restricted:
            dataset_train = ClassRestrictedImageDataset(self.data_dir, transform=self.transforms,
                                                        percentage=self.percentage,
                                                        use_gpu=self.use_gpu)
        else:
            dataset_train = ImageDataset(self.data_dir, transform=self.transforms, percentage=self.percentage,
                                         use_gpu=self.use_gpu,
                                         serial_batches=self.serial_batches)

        # dataset_test = ImageDataset(self.data_dir, transform=self.transforms, use_gpu=self.use_gpu,
        #                             serial_batches=True)

        train_len = int(self.train_val_test_split[0] * len(dataset_train))
        if self.share_val_test:
            train_set, test_set = random_split(dataset_train, [train_len, len(dataset_train) - train_len])
            val_set = test_set
        else:
            val_len = int(self.train_val_test_split[1] * len(dataset_train))
            train_set, val_set, test_set = random_split(dataset_train,
                                                        [train_len, val_len, len(dataset_train) - train_len - val_len])

        self.data_train = train_set
        self.data_test = test_set
        self.data_val = val_set

    def train_dataloader(self):
        return AsynchronousLoader(DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        ))

    def val_dataloader(self):
        return AsynchronousLoader(DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        ))

    def test_dataloader(self):
        return AsynchronousLoader(DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        ))
