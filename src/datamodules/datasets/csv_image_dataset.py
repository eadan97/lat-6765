import os
import random
from typing import Optional, Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.io import read_file, decode_jpeg

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def is_image_file(filename: str):
    return filename.lower().endswith(IMG_EXTENSIONS)


def make_dataset(root):
    images = []
    targets = []
    assert os.path.isdir(root), '%s is not a valid directory' % root

    for root, _, fnames in sorted(os.walk(root)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                targets.append(os.path.dirname(path))

    return images, targets


class CsvImageDataset(Dataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    Similarly, you need to prepare two directories:
    """

    def __init__(self,
                 csv_path: str,
                 data_path: str,
                 transform: Optional[Callable],
                 serial_batches: bool = False,
                 use_gpu: bool = True,
                 ):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(CsvImageDataset, self).__init__()
        self.data_path = data_path
        self.transforms = transform
        self.device = 'cuda' if use_gpu else 'cpu'
        self.serial_batches = serial_batches
        images_frame = pd.read_csv(csv_path)
        classes = images_frame.Type.unique()

        self.paths_A = images_frame[images_frame['Type'] == classes[0]].Filename.tolist()
        self.paths_B = images_frame[images_frame['Type'] == classes[1]].Filename.tolist()

        self.A_size = len(self.paths_A)  # get the size of dataset A
        self.B_size = len(self.paths_B)  # get the size of dataset B

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        index_A = index % self.A_size
        if self.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        data_A = read_file(os.path.join(self.data_path, self.paths_A[index_A]))  # raw data is on CPU
        data_B = read_file(os.path.join(self.data_path, self.paths_B[index_B]))  # raw data is on CPU
        img_A = decode_jpeg(data_A, device=self.device)  # decoded image in on GPU
        img_B = decode_jpeg(data_B, device=self.device)  # decoded image in on GPU
        img_A = self.transforms(img_A)
        img_B = self.transforms(img_B)
        return img_A, img_B

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
