import os
from typing import Optional, Tuple

import numpy as np
from pl_bolts.datamodules import AsynchronousLoader
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


class ClassifierDataModule(LightningDataModule):
    
    def __init__(
            self,
            data_dir: str = "data/",
            dataset_name: str = "",
            train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
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

        self.train_transforms = transforms.Compose(
            [transforms.RandomRotation(30), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])]
        )
        self.val_test_transforms = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224),
             transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])]
        )

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, 224, 224)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.num_classes = 0

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        train_dataset = ImageFolder(self.data_dir, transform=self.train_transforms)
        test_dataset = ImageFolder(self.data_dir, transform=self.val_test_transforms)

        targets = train_dataset.targets
        self.classes = train_dataset.classes
        all_idx = np.arange(len(targets))

        train_idx, test_idx = train_test_split(
            all_idx,
            test_size=self.train_val_test_split[1] + self.train_val_test_split[2],
            shuffle=True,
            stratify=targets)
        if not self.share_val_test:
            targets_test = [targets[idx] for idx in test_idx]
            test_idx, val_idx = train_test_split(
                test_idx,
                test_size=self.train_val_test_split[2] / (self.train_val_test_split[1] + self.train_val_test_split[2]),
                shuffle=True,
                stratify=targets_test)
        else:
            val_idx = test_idx

        # train_len = int(self.train_val_test_split[0] * len(dataset))
        # if self.share_val_test:
        #     train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])
        #     val_set = test_set
        # else:
        #     val_len = int(self.train_val_test_split[1] * len(dataset))
        #     train_set, val_set, test_set = random_split(dataset,
        #                                                 [train_len, val_len, len(dataset) - train_len - val_len])
        self.data_train = Subset(train_dataset, train_idx)
        self.data_test = Subset(test_dataset, test_idx)
        self.data_val = Subset(test_dataset, val_idx)

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
