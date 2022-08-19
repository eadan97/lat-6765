import os
import random
from typing import Optional, Callable

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder, DatasetFolder
from torchvision.io import read_file, decode_jpeg

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def is_image_file(filename: str):
    return filename.lower().endswith(IMG_EXTENSIONS)


def make_dataset(root):
    assert os.path.isdir(root), '%s is not a valid directory' % root
    all_classes = [cls.lower() for cls in next(os.walk(root))[1]]

    # images = {cl: [] for cl in all_classes}
    images = {}
    for root, _, fnames in sorted(os.walk(root)):
        class_name = os.path.basename(root).lower()
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                if class_name not in images:
                    images[class_name] = [path]
                else:
                    images[class_name].append(path)
    return images


class ClassRestrictedImageDataset(Dataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    Similarly, you need to prepare two directories:
    """

    def __init__(self,
                 data_path: str,
                 transform: Optional[Callable] = None,
                 percentage: float = 1.0,
                 use_gpu: bool = True,
                 ):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(ClassRestrictedImageDataset, self).__init__()
        self.transforms = transform
        self.device = 'cuda' if use_gpu else 'cpu'
        dir_A = os.path.join(data_path, 'A')
        dir_B = os.path.join(data_path, 'B')

        self.paths_A = make_dataset(dir_A)
        self.paths_B = make_dataset(dir_B)

        set_A = set(self.paths_A.keys())
        set_B = set(self.paths_B.keys())
        for key in sorted(set_A - set_B):
            del self.paths_A[key]
        for key in sorted(set_B - set_A):
            del self.paths_B[key]
        self.lens = {}
        self.idx_to_key = {}
        self.total_len = 0
        for key in sorted(self.paths_A.keys()) if percentage == 1 else random.sample(sorted(self.paths_A.keys()),
                                                                                     int(len(sorted(
                                                                                             self.paths_A.keys())) * percentage)):
            len_A = len(self.paths_A[key])
            len_B = len(self.paths_B[key])
            new_len = max(len_A, len_B)
            self.lens[key] = self.total_len
            self.idx_to_key.update({self.total_len + i: (key, i % len_A, i % len_B) for i in range(new_len)})
            self.total_len += new_len
        # self.dataset_A = ImageFolder(dir_A, transform=transform)
        # self.dataset_B = ImageFolder(dir_B, transform=transform)
        # self.A_size = len(self.paths_A)  # get the size of dataset A
        # self.B_size = len(self.paths_B)  # get the size of dataset B

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
        key, index_A, index_B = self.idx_to_key[index]
        # data_A = read_file(self.paths_A[index_A])  # raw data is on CPU
        # data_B = read_file(self.paths_B[index_B])  # raw data is on CPU
        # img_A = decode_jpeg(data_A, device=self.device).float()  # decoded image in on GPU
        # img_B = decode_jpeg(data_B, device=self.device).float()  # decoded image in on GPU
        img_A = Image.open(self.paths_A[key][index_A]).convert('RGB')  # decoded image in on GPU
        img_B = Image.open(self.paths_B[key][index_B]).convert('RGB')  # decoded image in on GPU
        if self.transforms is not None:
            img_A = self.transforms(img_A)
            img_B = self.transforms(img_B)
        return img_A, img_B

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.total_len
