import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler, SequentialSampler

from data.patch_sampling import PatchSampler
from copy import deepcopy

from collections import namedtuple


patch_dataset_split = namedtuple("patch_dataset_split", ["name", "indices"])


class PatchDataset(Dataset):
    # custom dataset must overwrite this
    @property
    def num_ref_images(self):
        raise NotImplementedError()

    # custom dataset must overwrite this
    @property
    def num_dist_images(self):
        raise NotImplementedError()

    DEFAULT_SPLIT_NAME = "DEFAULT"

    def __init__(self,
                 name,
                 path,
                 patch_dim=16,
                 patch_count=256,
                 patch_num_scales=1,
                 patch_sampler_config=None,
                 normalize=True,  # allow normalization
                 normalize_imagenet=True,  # force normalization with Imagenet mean and std dev
                 debug=False,
                 **kwargs  # NOTE: this avoids crash when unused parameters are passed
                 ):
        super(PatchDataset, self).__init__()

        if len(kwargs) > 0:
            print("WARNING: PatchDataset has unused kwargs={}".format(kwargs))

        self.debug = debug

        self.name = name
        self.path = path

        print("Setting up {} Dataset.".format(self.name))

        if isinstance(patch_dim, int):
            self.patch_dim = (patch_dim, patch_dim)  # h x w
        elif isinstance(patch_dim, tuple):
            self.patch_dim = patch_dim  # tuple
        else:
            raise ValueError("PatchDataset: unsupported patch_dim [{}]".format(patch_dim))

        self.patch_count = patch_count
        self.patch_num_scales = max(1, patch_num_scales)

        if patch_sampler_config is None:
            patch_sampler_config = {}
        self.patch_sampler = PatchSampler(**patch_sampler_config)

        self.normalize = normalize

        # force using ImageNet mean and std deviation for normalizing
        self.normalize_imagenet = normalize_imagenet
        self.imagenet_norm_mean = [0.485, 0.456, 0.406]
        self.imagenet_norm_std = [0.229, 0.224, 0.225]

        # default normalizing parameters: ImageNet mean and std deviation values
        # note: each dataset can override this to something else, which would be used if self.normalize_imagenet is False
        self.norm_mean = deepcopy(self.imagenet_norm_mean)
        self.norm_std = deepcopy(self.imagenet_norm_std)

        # read dataset files and collect paths
        self.data = self.read_dataset()

        self.splits_dict_ref = {}
        self.splits_dict = {}
        self.split_crt = None

    def comparisons_before_image(self, i):
        return self.num_dist_images * i

    def comparisons_per_image(self, i):
        return self.num_dist_images

    def add_split(self, split: patch_dataset_split = None):
        """
        computes dataset entry indices for each split
        :param split: named tuple containing name and indices for new split
                    indices are given along the reference image dimension
        :return:
        """

        if split is None:
            # default to simply all images in the dataset
            split = patch_dataset_split(
                name=self.DEFAULT_SPLIT_NAME,
                indices=[i for i in range(self.num_ref_images)],
            )

        self.splits_dict_ref[split.name] = split

        # expand from image indices to judgement (each ref. image has K distorted images) indices
        indices = [
            [
                (self.comparisons_before_image(i) + j)
                for j in range(self.comparisons_per_image(i))
            ] for i in split.indices
        ]

        indices = [item for sublist in indices for item in sublist]  # flatten

        if split.name in self.splits_dict:
            print("Warning: {} dataset overwriting an existing split with name '{}'.".format(
                self.name, split.name))

        split = patch_dataset_split(split.name, indices)

        self.splits_dict[split.name] = split

    def set_split(self, split_name):
        if split_name not in self.splits_dict:
            raise ValueError("PatchDataset: split_name [{}] not in splits.".format(split_name))
        self.split_crt = split_name

    def has_split(self, split_name):
        return split_name in self.splits_dict

    def get_norm_mean_std(self):
        if self.normalize:
            if self.normalize_imagenet:
                # force imagenet normalization parameters
                return self.imagenet_norm_mean, self.imagenet_norm_std
            else:
                return self.norm_mean, self.norm_std
        return [0., 0., 0.], [1., 1., 1.]  # parameters for no normalization

    @staticmethod
    def crop_params_pos(i, j, hw):
        xy = np.array([i, j], np.float32) / hw  # rescale to [0, 1]
        xy = xy * 2 - 1  # rescale to [-1, 1]
        return torch.as_tensor(xy, dtype=torch.float32)

    @staticmethod
    def get_height_width_factor(img, patch_dim):
        # constant for normalizing crop parameters (for uv)
        return np.array([img.height - patch_dim[0], img.width - patch_dim[1]])

    @staticmethod
    def get_height_width_factor_hw(height, width, patch_dim):
        # constant for normalizing crop parameters (for uv)
        return np.array([height - patch_dim[0], width - patch_dim[1]])

    def read_dataset(self):
        raise NotImplementedError("CustomDataset {} read_dataset() not implemented.".format(self.name))

    def __len__(self):
        return len(self.splits_dict[self.split_crt].indices)

    def __getitem__(self, item):
        raise NotImplementedError("CustomDataset {} __getitem__() not implemented.".format(self.name))


class PatchDatasetSampler(Sampler):
    def __init__(self, data_source: PatchDataset, split_name, patch_count, allow_img_flip, shuffle):
        self.split_name = split_name
        self.patch_count = patch_count
        self.allow_img_flip = allow_img_flip

        self.patch_dataset = data_source
        self.patch_dataset.set_split(split_name)

        if shuffle:
            self.sampler = RandomSampler(data_source)
        else:
            self.sampler = SequentialSampler(data_source)

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        # this runs when dataset loader __iter__ is called thereby modifying patch_dataset as needed by current split
        self.patch_dataset.set_split(self.split_name)  # set the correct split
        self.patch_dataset.patch_count = self.patch_count
        self.patch_dataset.allow_img_flip = self.allow_img_flip
        return len(self.patch_dataset)


class PatchDatasetLoader(DataLoader):
    def __init__(self, dataset, split_name, shuffle, patch_count, allow_img_flip, **kwargs):
        assert isinstance(dataset, PatchDataset), "PatchDatasetLoader must be paired with a PatchDataset."
        assert dataset.has_split(split_name), "PatchDatasetLoader must be paired with a PatchDataset with a split"
        sampler = PatchDatasetSampler(dataset, split_name, patch_count, allow_img_flip, shuffle)
        super(PatchDatasetLoader, self).__init__(dataset, sampler=sampler, **kwargs)

    def __iter__(self):
        return super().__iter__()
