from PIL import Image
import numpy as np
import torch

from data.utils import transform as transform_pil
from data.sampling.patch_sampling import get_iqa_patches
from data.patch_datasets import PatchDataset
from utils.timer import Timer

from utils.correlations import FitFunction


class FRIqaPatchDataset(PatchDataset):
    def __init__(self,
                 qs_normalize=True,
                 qs_reverse=True,
                 qs_normalize_mean_std=False,
                 qs_linearize=True,
                 qs_plot=True,
                 allow_img_flip=False,  # allow horizontal flip
                 img_zero_error_q_prob=0.,  #
                 **kwargs):
        super(FRIqaPatchDataset, self).__init__(**kwargs)

        self.qs_normalize = qs_normalize
        self.qs_reverse = qs_reverse
        self.qs_normalize_mean_std = qs_normalize_mean_std
        self.qs_linearize = qs_linearize
        self.qs_plot = qs_plot

        self.allow_img_flip = allow_img_flip  # allow random horizontal flipping
        self.img_zero_error_q_prob = img_zero_error_q_prob

        assert not (qs_normalize_mean_std or qs_linearize) or (qs_normalize_mean_std ^ qs_linearize), \
            "{}: qs_normalize_mean_std is mutually exclusive with qs_linearize. select only one.".format(self.name)

        full_reference = kwargs.pop("full_reference", True)
        if not full_reference:
            raise DeprecationWarning("The use of NR-IQA was disabled on 9/30/2021.")

        self.data = self.process_data(self.data)

        self.timer_transform = Timer()
        self.timer_patches = Timer()

        self.zero_error_q = torch.tensor(np.min(self.data[-1]))  # assume minimum value corresponds to zero error

    def read_dataset(self):
        raise NotImplementedError()

    def normalize_qs(self, qs):
        if self.qs_reverse:
            qs = qs.min() + qs.max() - qs

        if self.qs_normalize:
            qs -= qs.min()
            qs /= qs.max()

        if self.qs_normalize_mean_std:
            qs -= qs.mean()
            qs /= qs.std()

        return qs

    def process_data(self, data):
        # convert all arrays to np.array and preprocess Q array
        qs_raw = np.array(data[-1])
        qs = qs_raw.copy()

        print("Processing Qs raw:", qs.min(), qs.mean(), qs.max())

        qs = self.normalize_qs(qs)

        sorted_indices = np.argsort(qs_raw)

        print("After normalize/reverse qs", qs.min(), qs.mean(), qs.max())

        if self.qs_linearize:
            print("Linearizing dataset (histogram equalization)...")

            qs_counts = np.arange(len(qs))
            qs_lin = qs_counts.flatten() / len(qs)
            qs_sort = qs[sorted_indices]
            try:
                self.fit_function = FitFunction(qs_sort, qs_lin)
                qs = self.fit_function(qs)  # apply fit
                qs = self.normalize_qs(qs)  # reapply normalizations

            except OverflowError:
                print(self.name, ": Overflow during Q array linearization. Using raw quality values instead.")

        if self.qs_plot:
            from matplotlib import pyplot as plt

            plt.figure()
            plt.title("Q mapping and histogram")
            plt.plot(qs_raw, qs, 'ro', markersize=0.5, label="Processed Q vs raw Q")
            # plt.show()

            # plt.figure()
            hist, bins = np.histogram(qs_raw, bins=100)
            plt.fill_between([(bins[i]+bins[i+1])/2 for i in range(len(hist))], hist/hist.max()*qs.max(), y2=0)
            # plt.plot([i for i in range(len(qs_raw))], qs_raw[sorted_indices])
            plt.xlabel("Dataset Q")
            plt.ylabel("Processed Q")
            plt.legend()
            plt.show()

        str_data = tuple()
        for array in data[:-1]:  # except qs
            # NOTE: fix from 2019
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

        r_samples = torch.rand(4)  # random horizontal flip, random flip between ref/dist patch return order

        img_ref = Image.open(path_ref).convert('RGB')
        img_dist = Image.open(path_dist).convert('RGB')

        norm_mean, norm_std = self.get_norm_mean_std()

        # self.timer_transform.start()
        # transform full images to tensors without cropping
        h_flip = self.allow_img_flip and (r_samples[0] < 0.5)
        v_flip = self.allow_img_flip and (r_samples[1] < 0.5)
        tensor_ref = transform_pil(img_ref, None, h_flip, v_flip, (norm_mean, norm_std))
        tensor_dist = transform_pil(img_dist, None, h_flip, v_flip, (norm_mean, norm_std))
        # print('transform: ', self.timer_transform.stop(), "avg:", self.timer_transform.delta_avg)

        # self.timer_patches.start()
        p1, p2, pos, scales = get_iqa_patches(
            (img_ref, img_dist),
            (tensor_ref, tensor_dist),
            self.patch_count, self.patch_dim, self.patch_sampler, self.patch_num_scales
        )
        # print('get_iqa_patches: ', self.timer_patches.stop(), "avg:", self.timer_patches.delta_avg)

        if r_samples[2] < self.img_zero_error_q_prob:
            data = (self.zero_error_q, p1, p1, pos, scales) if r_samples[3] < 0.5 else \
                (self.zero_error_q, p2, p2, pos, scales)
        else:
            data = q, p1, p2, pos, scales

        return data


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

        r_samples = torch.rand(2) <= 0.5  # random horizontal flip, random flip between ref/dist patch return order

        img_ref = Image.open(path_ref).convert('RGB')
        img_dist1 = Image.open(path_dist1).convert('RGB')
        img_dist2 = Image.open(path_dist2).convert('RGB')

        norm_mean, norm_std = self.get_norm_mean_std()

        # transform full images to tensors without cropping and with flipping disabled
        h_flip = self.allow_img_flip and (r_samples[0] < 0.5)
        v_flip = self.allow_img_flip and (r_samples[1] < 0.5)
        tensor_ref = transform_pil(img_ref, None, h_flip, v_flip, (norm_mean, norm_std))
        tensor_dist1 = transform_pil(img_dist1, None, h_flip, v_flip, (norm_mean, norm_std))
        tensor_dist2 = transform_pil(img_dist2, None, h_flip, v_flip, (norm_mean, norm_std))

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
