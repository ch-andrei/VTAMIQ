from PIL import Image
import numpy as np
import torch

from data.utils import transform as transform_pil, get_iqa_patches
from data.patch_datasets import PatchDataset


class FRIqaPatchDataset(PatchDataset):
    def __init__(self,
                 qs_normalize=True,
                 qs_reverse=True,
                 qs_linearize=True,
                 qs_linearize_plot=False,
                 allow_img_flip=False,
                 **kwargs):
        super(FRIqaPatchDataset, self).__init__(**kwargs)

        self.qs_normalize = qs_normalize
        self.qs_reverse = qs_reverse
        self.qs_linearize = qs_linearize
        self.qs_linearize_plot = qs_linearize_plot

        self.allow_img_flip = allow_img_flip  # allow random horizontal flipping

        full_reference = kwargs.pop("full_reference", True)
        if not full_reference:
            raise DeprecationWarning("The use of NR-IQA was disabled on 9/30/2021.")

        self.data = self.process_data(self.data)

    def read_dataset(self):
        raise NotImplementedError()

    def process_data(self, data):
        # convert all arrays to np.array and preprocess Q array
        qs = np.array(data[-1])

        if self.qs_normalize:
            qs -= qs.min()
            qs /= qs.max()

        if self.qs_reverse:
            qs = qs.max() - qs

        print("After normalize/reverse qs", qs.min(), qs.mean(), qs.max())

        if self.qs_linearize:
            print("Linearizing dataset (histogram equalization)...")

            from utils.correlations import FitFunction

            qs_counts = np.arange(len(qs))
            qs_lin = qs_counts.flatten() / len(qs)
            qs_sort = np.sort(qs)
            try:
                fit_function = FitFunction(qs_sort, qs_lin)
                qs = fit_function(qs)

                if self.qs_linearize_plot:
                    from matplotlib import pyplot as plt
                    plt.plot(qs_sort, qs_counts, 'ro', markersize=0.5, label='before fit')
                    plt.plot(fit_function(qs_sort), qs_counts, 'bo', markersize=0.5, label='after fit')
                    plt.legend()
                    plt.show()
            except OverflowError:
                print(self.name, ": Overflow during Q array linearization. Using raw quality values instead.")

            # normalize again after fitting to avoid going outside [0,1]
            if self.qs_normalize:
                qs -= qs.min()
                qs /= qs.max()

        str_data = tuple()
        for array in data[:-1]:  # except qs
            # NOTE: not sure if this is still relevant in 2021
            # use np.string to avoid pytorch memory leak (high usage) for non-string arrays when num_workers > 1
            # https://github.com/pytorch/pytorch/issues/13246
            array = np.array(array, dtype=np.string_)

            str_data += (array,)

        return str_data + (qs,)

    def __getitem__(self, index):
        index = self.splits_dict[self.split_crt].indices[index]

        paths_ref, paths_dist, qs = self.data
        path_ref = paths_ref[index]

        path_dist = paths_dist[index]
        q = torch.tensor(qs[index])

        r_samples = torch.rand(1) <= 0.5  # random horizontal flip, random flip between ref/dist patch return order

        img_ref = Image.open(path_ref).convert('RGB')
        img_dist = Image.open(path_dist).convert('RGB')

        norm_mean, norm_std = self.get_norm_mean_std()

        # transform full images to tensors without cropping
        r_flip = self.allow_img_flip and r_samples[0]

        tensor_ref = transform_pil(img_ref, None, r_flip, (norm_mean, norm_std))
        tensor_dist = transform_pil(img_dist, None, r_flip, (norm_mean, norm_std))

        p1, p2, pos, scales = get_iqa_patches(
            (img_ref, img_dist),
            (tensor_ref, tensor_dist),
            self.patch_count, self.patch_dim, self.patch_sampler, self.patch_num_scales
        )

        return q, p1, p2, pos, scales


class PairwiseFRIqaPatchDataset(FRIqaPatchDataset):
    def __init__(self,
                 **kwargs,
                 ):

        # remove some config values
        kwargs.pop("full_reference", None)  # can only be full_reference

        super(PairwiseFRIqaPatchDataset, self).__init__(
            full_reference=True,
            **kwargs
        )

        assert 4 == len(self.data), "Pairwise dataset needs two distorted image path lists."

    def __getitem__(self, index):
        index = self.splits_dict[self.split_crt].indices[index]

        paths_ref, paths_dist1, paths_dist2, qs = self.data
        path_ref = paths_ref[index]

        path_dist1 = paths_dist1[index]
        path_dist2 = paths_dist2[index]
        q = torch.tensor(qs[index])

        r_samples = torch.rand(1) <= 0.5  # random horizontal flip, random flip between ref/dist patch return order

        img_ref = Image.open(path_ref).convert('RGB')
        img_dist1 = Image.open(path_dist1).convert('RGB')
        img_dist2 = Image.open(path_dist2).convert('RGB')

        norm_mean, norm_std = self.get_norm_mean_std()

        # transform full images to tensors without cropping and with flipping disabled
        r_flip = self.allow_img_flip and r_samples[0]

        tensor_ref = transform_pil(img_ref, None, r_flip, (norm_mean, norm_std))
        tensor_dist1 = transform_pil(img_dist1, None, r_flip, (norm_mean, norm_std))
        tensor_dist2 = transform_pil(img_dist2, None, r_flip, (norm_mean, norm_std))

        p1, p2, p3, pos, scales = get_iqa_patches(
            (img_ref, img_dist1, img_dist2),
            (tensor_ref, tensor_dist1, tensor_dist2),
            self.patch_count, self.patch_dim, self.patch_sampler, self.patch_num_scales
        )

        return q, p1, p2, p3, pos, scales

    def process_data(self, data):
        # do nothing, since we are dealing with preferences, there is no need to preprocess Qs
        return data


class NRIqaPatchDataset(FRIqaPatchDataset):
    def __init__(self, **kwargs):
        if "full_reference" in kwargs:
            print("Warning: full_reference arg can't be modified for NR-IQA dataset. Setting full_reference=False")

        # force full_reference off
        kwargs["full_reference"] = False

        super(NRIqaPatchDataset, self).__init__(
            **kwargs
        )
