import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler, SequentialSampler

from data.patch_sampling import PatchSampler, get_iqa_patches
from data.utils import transform_img, normalize_values, reverse_values, imread, get_imagenet_transfor_params
from utils.misc.correlations import FitFunction
from utils.misc.miscelaneous import check_unused_kwargs
from utils.logging import log_warn

from collections import namedtuple

dataset_split = namedtuple("dataset_split", ["name", "indices"])

DATASETS_DEFAULT_PATH = "G:/Datasets"


class ImageDataset(Dataset):
    @property
    def img_dim(self):
        # resolution of images in the dataset
        raise NotImplementedError()

    def __init__(self,
                 name,
                 path,
                 is_hdr=False,
                 normalize=True,  # allow normalization
                 normalize_imagenet=False,  # force normalization with Imagenet mean and std dev
                 **kwargs  # NOTE: this avoids crash when unused parameters are passed
                 ):
        super(ImageDataset, self).__init__()
        check_unused_kwargs("DatasetBase", **kwargs)

        self.name = name
        self.path = f"{DATASETS_DEFAULT_PATH}/{path}"

        print(f"Setting up {self.name} dataset.")

        self.is_hdr = is_hdr
        if is_hdr:
            print(f"Dataset {self.name} is HDR.")

        # normalization parameters
        self.normalize = normalize
        if normalize and is_hdr:
            self.normalize = False
            log_warn("Image data normalization is disabled for HDR dataset.")

        # default normalization parameters
        self.norm_mean = [0.5, 0.5, 0.5]
        self.norm_std = [0.5, 0.5, 0.5]

        # normalization parameters when using ImageNet mean and std deviation
        self.normalize_imagenet = normalize_imagenet  # if True, use Imagenet params instead of default
        self.imagenet_norm_mean, self.imagenet_norm_std = get_imagenet_transfor_params()

    def get_norm_mean_std(self):
        if self.normalize:
            if self.normalize_imagenet:
                # force imagenet normalization parameters
                return self.imagenet_norm_mean, self.imagenet_norm_std
            else:
                return self.norm_mean, self.norm_std
        return None, None  # parameters for no normalization

    @staticmethod
    def crop_params_tensor(i, j, max_h, max_w):
        return torch.as_tensor([i / max_h, j / max_w], dtype=torch.float32) * 2. - 1.  # rescale to [-1, 1]

    @staticmethod
    def get_max_height_width(height, width, patch_height, patch_width):
        # constant for normalizing crop parameters (for uv)
        return height - patch_height, width - patch_width

    @staticmethod
    def get_height_width_factor_hw(height, width, patch_height, patch_width):
        # constant for normalizing crop parameters (for uv)
        return np.array([height - patch_height, width - patch_width])

    def __len__(self):
        raise NotImplementedError(f"DatasetBase {self.name} __len__() not implemented.")

    def __getitem__(self, item):
        raise NotImplementedError(f"DatasetBase {self.name} __getitem__() not implemented.")


class IQADataset(ImageDataset):
    __SPLIT_NAME_FULL = "FullDataset"

    @property
    def num_ref_images(self):
        # total number of reference images across which the dataset can be split
        raise NotImplementedError()

    @property
    def num_dist_images(self):
        # total number of distorted images for each reference image
        raise NotImplementedError()

    @property
    def num_distortions(self):
        # total number of distorted images for each reference image
        raise NotImplementedError()

    def __init__(
            self,
            name,
            path,

            # processing params for quality values
            qs_normalize=True,
            qs_reverse=True,
            qs_normalize_mean_std=False,
            qs_linearize=True,
            qs_plot=False,

            **kwargs
    ):
        super().__init__(name, path, **kwargs)

        self.qs_normalize = qs_normalize
        self.qs_reverse = qs_reverse
        self.qs_normalize_mean_std = qs_normalize_mean_std
        self.qs_linearize = qs_linearize
        self.qs_plot = qs_plot

        if qs_linearize and qs_normalize_mean_std:
            raise ValueError("{}: qs_normalize_mean_std is mutually exclusive with qs_linearize.".format(self.name))

        self.splits_dict_ref = {}  # records ref image indices
        self.splits_dict = {}  # also record dist image indices (e.g., there are K dist images per ref image)
        self.split_name_crt = None  # current split index

        self.qs = None  # list of quality values

        self.dist_images_per_image = None
        self.dist_images_before_image = None

        # initialize dataset fields
        self.read_dataset()

        # apply data processing
        self.process_qs()

    def read_dataset(self):
        # must return qs and dist_images_per_image
        raise NotImplementedError(f"IQADataset {self.name} read_dataset() not implemented.")

    def get_current_index(self, relative_index):
        # convert from split-relative index to a global index in the dataset's data arrays
        # ex: current split holds data items at indices 25-50 (qs[25:50]),
        # the first item (at relative_index=0) is at global index 25
        # TODO: fix self.num_repeats_data unresolved attribute reference (move from subclass to this class)
        index_mapped = relative_index % (len(self) // self.num_repeats_data)  # wrap around repeated indices
        index_mapped = self.splits_dict[self.split_name_crt].indices[index_mapped]
        return index_mapped

    def add_split(self, split: dataset_split = None):
        """
        initializes and stores dataset image indices for a new split
        :param split: named tuple containing name and indices for new split
                    indices are given along the reference image dimension
        :return:
        """
        if split is None or split.indices is None:
            # default to simply all images in the dataset
            split = dataset_split(
                name=self.__SPLIT_NAME_FULL if split is None else split.name,
                indices=[i for i in range(self.num_ref_images)],
            )

        if len(split.indices) < 5000:
            print(f"Dataset {self.name} adding new split [{split.name}] with ref image indices {split.indices}")
        else:
            print(
                f"Dataset {self.name} adding new split [{split.name}] with a total of {len(split.indices)} ref images.")

        # if a split with this name already exists
        if split.name in self.splits_dict:
            log_warn(f"Dataset {self.name} overwriting an existing split with name {split.name}.", self.name)

        # build new split
        # expand from ref image indices to ref/dist pair indices (i.e., each ref. image has K distorted images)
        indices = [
            [
                (self.dist_images_before_image[i] + j) for j in range(self.dist_images_per_image[i])
            ] for i in split.indices
        ]
        indices = np.concatenate(indices).flatten()  # 1d array

        # record new split
        self.splits_dict_ref[split.name] = split  # only reference image index per split
        self.splits_dict[split.name] = dataset_split(split.name, indices)

    def set_split_crt(self, split_name):
        if split_name not in self.splits_dict:
            raise KeyError(f"Dataset {self.name} does not contain split with name [{split_name}].")
        self.split_name_crt = split_name

    def has_split(self, split_name):
        return split_name in self.splits_dict

    def __len__(self):
        return len(self.splits_dict[self.split_name_crt].indices) * self.num_repeats_data

    def process_qs(self):
        # convert all arrays to np.array and preprocess Q array
        qs_raw = np.array(self.qs, float)
        qs = qs_raw.copy()

        print("Before processing Qs (min/mean/max):", qs.min(), qs.mean(), qs.max())

        qs = normalize_values(qs, self.qs_normalize, self.qs_normalize_mean_std)

        if self.qs_linearize:
            print("Linearizing dataset (histogram equalization)...")

            sorted_indices = np.argsort(qs)
            qs_counts = np.arange(len(qs))
            qs_lin = qs_counts.flatten() / len(qs) * qs.max() - qs.min()
            qs_sort = qs[sorted_indices]

            try:
                self.fit_function = FitFunction(qs_sort, qs_lin, residuals_func="L2")
                qs = self.fit_function(qs)  # apply fit

            except OverflowError:
                print(self.name, ": Overflow during Q array linearization. Using raw quality values instead.")

            # normalize post linearization
            qs = normalize_values(qs, self.qs_normalize, self.qs_normalize_mean_std)

        # relative flip of values (e.g., from 0-1 to 1-0)
        qs = reverse_values(qs, self.qs_reverse)

        # overwrite with processed q values
        self.qs = qs

        print("After processing Qs (min/mean/max):", qs.min(), qs.mean(), qs.max())
        self.plot_process_qs(qs_raw, qs)

    def plot_process_qs(self, qs_raw, qs, xrange=None, yrange=None):
        if self.qs_plot:
            from matplotlib import pyplot as plt

            plt.figure(dpi=300)
            plt.title(f"{self.name}: Qs after processing")
            plt.plot(qs_raw, qs, 'b,', label="Fit function", alpha=1.0)  # PIXELS
            plt.plot(qs_raw, qs, 'bo', markersize=0.1, label="Fit function", alpha=0.1)  # CIRCLES

            qmin = qs.min()
            qmax = qs.max()
            qrawmin = qs_raw.min()
            qrawmax = qs_raw.max()

            # plt.figure()
            hist, bins = np.histogram(qs_raw, bins=100)
            plt.fill_between([(bins[i] + bins[i + 1]) / 2 for i in range(len(hist))],
                             y1=qmin, y2=qmin + hist / hist.max() * (qmax - qmin),
                             color="r", label="Raw Q", alpha=0.5)
            # qs_raw_max = qs_raw.max() if (abs(qs_raw.min()) < abs(qs_raw.max())) else qs_raw.min()
            hist, bins = np.histogram(qs, bins=100)
            plt.fill_betweenx([(bins[i] + bins[i + 1]) / 2 for i in range(len(hist))],
                                x1=qrawmin, x2=qrawmin + hist / hist.max() * (qrawmax - qrawmin),
                                color="b", label="Processed Q", alpha=0.25)
            # plt.plot([i for i in range(len(qs_raw))], qs_raw[sorted_indices])
            if xrange is not None:
                plt.xlim(xrange)
            if yrange is not None:
                plt.ylim(yrange)
            plt.xlabel("Dataset Q")
            plt.ylabel("Processed Q")
            plt.legend()
            plt.show()


class PatchFRIQADataset(IQADataset):
    def __init__(
            self,
            name,
            path,

            # patch parameters
            patch_dim=16,
            patch_count=256,
            patch_num_scales=1,
            patch_sampler_config=None,

            # data augmentation params
            allow_img_flip=False,  # allow horizontal flip
            img_zero_error_q_prob=0.,  #
            patch_sampling_num_scales_ratio=2.,
            use_aligned_patches=1,
            use_ref_img_cache=False,
            use_dist_img_cache=False,
            num_repeats_data=1,

            return_paths=False,  # for debugging
            return_full_imgs=False,  # for debugging

            **kwargs
    ):
        # read_dataset() called in super().__init__() will initialize paths
        self.paths_ref = None
        self.paths_dist = None

        super(PatchFRIQADataset, self).__init__(name, path, **kwargs)

        if patch_dim == -1:
            raise ValueError("Unsupported patch dimensions.")

        if isinstance(patch_dim, int):
            self.patch_dim = patch_dim
        else:
            raise ValueError(f"PatchDataset: unsupported patch_dim format [{patch_dim}]")

        self.patch_count = patch_count
        self.patch_num_scales = max(1, patch_num_scales)

        print(f"Dataset sampled at {patch_num_scales} image scales with base patch size {patch_dim}.")

        if patch_sampler_config is None:
            patch_sampler_config = {}

        self.patch_sampler = PatchSampler(**patch_sampler_config)

        self.allow_img_flip = allow_img_flip  # allow random horizontal/vertical flipping
        self.img_zero_error_q_prob = img_zero_error_q_prob
        self.patch_sampling_num_scales_ratio = patch_sampling_num_scales_ratio
        self.use_aligned_patches = use_aligned_patches
        self.num_repeats_data = num_repeats_data

        self.zero_error_q = torch.tensor(np.min(self.qs))  # assume minimum value corresponds to zero error

        self.use_ref_img_cache = use_ref_img_cache
        self.use_dist_img_cache = use_dist_img_cache
        self.img_cache = dict() if (use_ref_img_cache or use_dist_img_cache) else None
        if self.img_cache is not None:
            log_warn(f"use_ref_img_cache={use_ref_img_cache} and use_dist_img_cache={use_dist_img_cache}", self.name)

        self.return_paths = return_paths
        self.return_full_imgs = return_full_imgs

    def process_dataset_data(self, qs, paths_ref, paths_dist, dist_images_per_image):
        self.qs = np.array(qs, float)
        self.paths_ref = paths_ref
        self.paths_dist = paths_dist
        self.dist_images_per_image = np.array(dist_images_per_image, int)
        self.dist_images_before_image = self.compute_dist_images_before_image(dist_images_per_image)

    @staticmethod
    def compute_dist_images_before_image(dist_images_per_image):
        # NOTE: dist images before current ref image equals cumulative sum minus dist count for current ref image
        return np.cumsum(dist_images_per_image) - dist_images_per_image

    def img_pretransform(self, img):
        # nothing to do in the base case, simply return the img
        return img

    def get_img(self, path, is_ref_img=False):
        # get from cache or read from disk
        use_cache = (is_ref_img and self.use_ref_img_cache) or (not is_ref_img and self.use_dist_img_cache)
        if use_cache and path in self.img_cache:
            img = self.img_cache[path]
        else:
            img = imread(path, self.is_hdr)
            img = self.img_pretransform(img)
            if use_cache:
                self.img_cache[path] = img
        return img

    def get_img_random_flip(self):
        r_samples = torch.rand(2)
        h_flip = self.allow_img_flip and (r_samples[0] < 0.5)
        v_flip = self.allow_img_flip and (r_samples[1] < 0.5)
        return h_flip, v_flip

    def is_hdr_image(self, path):
        return False

    def __getitem__(self, index):
        index = self.get_current_index(index)

        path_ref = self.paths_ref[index]
        path_dist = self.paths_dist[index]
        q = torch.as_tensor(self.qs[index])

        img_ref = self.get_img(path_ref, is_ref_img=True)
        img_dist = self.get_img(path_dist, is_ref_img=False)
        # TODO: handle images that get corrupted: exception on imread()

        if self.return_full_imgs:
            log_warn("DEBUG return_full_imgs instead of patches...")
            data_out = (q, np.array(img_ref), np.array(img_dist))
            return data_out

        norm_mean, norm_std = self.get_norm_mean_std()

        # transform full images to tensors without cropping
        h_flip, v_flip = self.get_img_random_flip()
        tensor_ref = transform_img(img_ref, None, h_flip, v_flip, norm_mean, norm_std)
        tensor_dist = transform_img(img_dist, None, h_flip, v_flip, norm_mean, norm_std)

        patches, pos, scales = get_iqa_patches(
            (img_ref, img_dist),
            (tensor_ref, tensor_dist),
            self.patch_count, self.patch_dim, self.patch_sampler, self.patch_num_scales,
            scale_num_samples_ratio=self.patch_sampling_num_scales_ratio,
            use_aligned_patches=self.use_aligned_patches,
        )

        if scales is None:
            scales = -1  # NOTE: this fixes "TypeError: default_collate: batch must not contain None"

        data_out = (q, patches, pos, scales)

        # NOTE December 2023: path_ref, path_dist added for visualization only
        if self.return_paths:
            data_out += (path_ref, path_dist)

        if self.is_hdr:
            data_out += (self.is_hdr_image(path_ref),)

        return data_out


class PairwiseFRIQAPatchDataset(PatchFRIQADataset):
    def __init__(self,
                 name,
                 **kwargs,
                 ):
        self.paths_ref = None
        self.paths_dist1 = None
        self.paths_dist2 = None

        super(PairwiseFRIQAPatchDataset, self).__init__(
            name=name,
            **kwargs
        )

        if self.paths_ref is None or self.paths_dist1 is None or self.paths_dist2 is None:
            raise AttributeError("PairwiseFRIqaPatchDataset: required data fields are not initialized.")

    def __getitem__(self, index):
        index = self.get_current_index(index)

        path_ref = self.paths_ref[index]
        path_dist1 = self.paths_dist1[index]
        path_dist2 = self.paths_dist2[index]
        q = torch.as_tensor(self.qs[index])

        img_ref = self.get_img(path_ref, True)
        img_dist1 = self.get_img(path_dist1)
        img_dist2 = self.get_img(path_dist2)

        norm_mean, norm_std = self.get_norm_mean_std()

        # transform full images to tensors without cropping and with flipping disabled
        h_flip, v_flip = self.get_img_random_flip()
        tensor_ref = transform_img(img_ref, None, h_flip, v_flip, norm_mean, norm_std)
        tensor_dist1 = transform_img(img_dist1, None, h_flip, v_flip, norm_mean, norm_std)
        tensor_dist2 = transform_img(img_dist2, None, h_flip, v_flip, norm_mean, norm_std)

        patches, pos, scales = get_iqa_patches(
            (img_ref, img_dist1, img_dist2),
            (tensor_ref, tensor_dist1, tensor_dist2),
            self.patch_count, self.patch_dim, self.patch_sampler, self.patch_num_scales,
            scale_num_samples_ratio=self.patch_sampling_num_scales_ratio,
            use_aligned_patches=self.use_aligned_patches,
        )

        data_out = (q, patches, pos, scales)

        # NOTE December 2023: path_ref, path_dist added for visualization only
        if self.return_paths:
            data_out += (path_ref, path_dist1, path_dist2)

        if self.is_hdr:
            data_out += (self.is_hdr_image(path_ref),)

        return data_out

    def process_qs(self):
        # do nothing, as we are dealing with preference data
        log_warn("process_qs() is disabled for pairwise preference dataset.", self.name)


# TODO: fix NR dataset; this needs a custom __getitem__ (?)
class NRIqaPatchDataset(PatchFRIQADataset):
    def __init__(self, **kwargs):
        if "full_reference" in kwargs:
            log_warn("full_reference arg can't be modified for NR-IQA dataset. Setting full_reference=False.",
                     self.name)

        # force full_reference off
        kwargs["full_reference"] = False

        super(NRIqaPatchDataset, self).__init__(
            **kwargs
        )


class PatchDatasetSampler(Sampler):
    def __init__(self, data_source: PatchFRIQADataset, shuffle, split_name, patch_count, num_repeats_data,
                 allow_img_flip, img_zero_error_q_prob, use_aligned_patches):
        # NOTE: calling super().__init__(data_source) is redundant (not needed) as it does nothing
        self.split_name = split_name
        self.patch_count = patch_count
        self.allow_img_flip = allow_img_flip
        self.img_zero_error_q_prob = img_zero_error_q_prob
        self.use_aligned_patches = use_aligned_patches
        self.patch_dataset = data_source
        self.num_repeats_data = max(1, int(num_repeats_data))

        self.notify_patch_dataset()

        if shuffle:
            self.sampler = RandomSampler(data_source)
        else:
            self.sampler = SequentialSampler(data_source)

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        # this runs when dataset loader __iter__ is called and modifies patch_dataset as needed by current split
        self.notify_patch_dataset()
        return len(self.patch_dataset)

    def notify_patch_dataset(self):
        self.patch_dataset.set_split_crt(self.split_name)  # set the correct split
        self.patch_dataset.patch_count = self.patch_count
        self.patch_dataset.allow_img_flip = self.allow_img_flip
        self.patch_dataset.img_zero_error_q_prob = self.img_zero_error_q_prob
        self.patch_dataset.use_aligned_patches = self.use_aligned_patches
        self.patch_dataset.num_repeats_data = self.num_repeats_data


class PatchDatasetLoader(DataLoader):
    @property
    def split_name(self):
        return self.patch_sampler.split_name

    def __init__(self, dataset, split_name, shuffle, patch_count, num_repeats_data=1,
                 allow_img_flip=False, img_zero_error_q_prob=-1, use_aligned_patches=True, **kwargs
                 ):
        if not isinstance(dataset, PatchFRIQADataset):
            raise TypeError("PatchDatasetLoader must be paired with a PatchDataset.")
        if not dataset.has_split(split_name):
            raise ValueError("Attempting to pair PatchDatasetLoader with PatchDataset that does not have the required "
                             f"split (split_name={split_name})")
        self.patch_sampler = PatchDatasetSampler(dataset, shuffle, split_name, patch_count, num_repeats_data,
                                                 allow_img_flip, img_zero_error_q_prob, use_aligned_patches)
        super(PatchDatasetLoader, self).__init__(dataset, sampler=self.patch_sampler, **kwargs)

    def __iter__(self):
        self.patch_sampler.notify_patch_dataset()  # update dataset variables
        return super().__iter__()