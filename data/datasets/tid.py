import numpy as np
from data.datasets.iqa_datasets import FRIqaPatchDataset


class TID2013Dataset(FRIqaPatchDataset):
    num_ref_images = 25
    num_dist_images = 120

    def __init__(self,
                 name="TID2013",
                 path="I:/Datasets/tid2013",
                 **kwargs
                 ):
        """
        The original TID2013 dataset provides 3000 quality comparisons for 25 references images (512x384 resolution).
        :param path: path to TID2013 directory
        :param patch_dim: int or tuple (h x w), will sample a random square patch of this size
        :param patch_count: number of samples to return for each image

        """
        super(TID2013Dataset, self).__init__(
            name=name,
            path=path,
            **kwargs
        )

        # # TID2013 statistics computed using 3000 test images + 25 reference images
        # self.norm_mean = [0.4735, 0.4304, 0.3593]
        # self.norm_std = [0.2008, 0.2029, 0.1905]

        self.img_dim = (384, 512)

    def read_dataset(self,
                     # to support reading datasets similar to TID, the following vars were added
                     reference_images_path="/reference_images",
                     distorted_images_path="/distorted_images",
                     q_file_name="mos_with_names.txt",
                     split_char=" ",
                     q_ind=0,
                     filename_ind=1,
                     filename_ext="bmp",
                     has_header=False
                     ):
        """
        returns a list of tuples (reference_image_path, distorted_image_path, quality)
        where q is MOS for original TID2013 or JOD for TID2013+.
        :return:
        """
        reference_images_path = self.path + reference_images_path
        distorted_images_path = self.path + distorted_images_path

        paths_ref, paths_dist, qs = [], [], []
        q_file_path = self.path + "/" + q_file_name
        with open(q_file_path, 'r') as q_file:
            if has_header:
                q_file.__next__()  # skip header line

            for line in q_file:
                line = line.strip().split(split_char)  # split by comma or space

                # the first 3 letters are the reference file name
                path_reference = reference_images_path + '/' + line[filename_ind][0:3] + "." + filename_ext
                path_distorted = distorted_images_path + '/' + line[filename_ind]
                q = float(line[q_ind])

                paths_ref.append(path_reference)
                paths_dist.append(path_distorted)
                qs.append(q)

        return paths_ref, paths_dist, qs


class TID2008Dataset(TID2013Dataset):
    # num_ref_images inherited from TID2013Dataset
    num_dist_images = 68

    def __init__(self,
                 path='I:/Datasets/tid2008',
                 **kwargs):
        super(TID2008Dataset, self).__init__(path=path, name="TID2008", **kwargs)


class TID2013PlusDataset(TID2013Dataset):
    def __init__(self,
                 path='I:/Datasets/tid2013+',  # THIS WONT WORK SINCE WE REUSE TID2013, TODO: fix path issues
                 **kwargs):
        super(TID2013PlusDataset, self).__init__(path=path, name="TID2013+", **kwargs)

    def read_dataset(self):
        return super().read_dataset(q_file_name="JOD.csv",
                                    split_char=",",
                                    q_ind=1,
                                    filename_ind=0,
                                    has_header=True
                                    )




# class TID2013PairwiseDataset(Dataset):
#     def __init__(self,
#                  num_patches=1,
#                  num_patches_per_pair=1,
#                  output_single_q=False,
#                  **tid_kwargs,
#                  ):
#         self.ppi = num_patches
#         self.ppp = num_patches_per_pair
#         self.output_single_q = output_single_q
#         self.tid2013 = TID2013Dataset(**tid_kwargs)
#         self.out_dim = self.tid2013.out_dim
#         self.index_pairs = {}
#
#     def train_mode(self):
#         self.tid2013.train_mode()
#
#     def validation_mode(self):
#         self.tid2013.validation_mode()
#
#     def get_pair_index(self, index):
#         # get 2 random indices from tid2013 (for the same image)
#         index = int(index / self.ppi)
#
#         c1 = c2 = 0
#         while c1 == c2:
#             c1, c2 = np.random.randint(120, size=(2,))
#         index1 = 120 * index + c1
#         index2 = 120 * index + c2
#
#         return index1, index2
#
#     def reset_iter(self):
#         # reset index pairs
#         self.index_pairs = {}
#
#     def __getitem__(self, index):
#         index1, index2 = self.get_pair_index(index)
#
#         return [self.__get_patch(index1, index2) for _ in range(self.ppp)]
#
#     def __get_patch(self, index1, index2):
#         r_crop_params = get_random_crop_params(self.tid2013.img_dim[0], self.tid2013.img_dim[1],
#                                                self.out_dim[0], self.out_dim[1])
#         r_flip = np.random.rand(1) <= 0.5
#         img_ref, img_dist1, q1 = self.tid2013.__getitem__(index1, r_crop_params, r_flip)
#         img_ref, img_dist2, q2 = self.tid2013.__getitem__(index2, r_crop_params, r_flip)
#
#         return img_ref, img_dist1, img_dist2, (q1 < q2) if self.output_single_q else (q1, q2)
#
#     def __len__(self):
#         return int(self.tid2013.__len__() * self.ppi / 120)


# class TID2013PairwiseDataLoader(DataLoader):
#     def __init__(self, dataset, **kwargs):
#         super().__init__(dataset, **kwargs)
#         self.tid_dataset = self.dataset
#
#     def __iter__(self):
#         if isinstance(self.tid_dataset, TID2013PairwiseDataset):
#             self.tid_dataset.reset_iter()  # this is needed to remove cached indices
#         return super().__iter__()


# class TID2013PairwiseSampler(Sampler):
#     def __init__(self, data_source, shuffle=False):
#         super().__init__(data_source)  # doesn't do anything, but to respect inheritance syntax
#         self.tid_dataset = data_source
#         self.spi = self.tid_dataset.samples_per_image
#         self.shuffle = shuffle
#
#     def __iter__(self):
#         n = int(len(self.tid_dataset) / self.spi)
#         if self.shuffle:
#             indices = torch.randperm(n)
#         else:
#             indices = torch.arange(n, dtype=torch.int)
#         indices = torch.repeat_interleave(indices, repeats=self.spi, dim=0)  # stack
#         return iter(indices.tolist())
#
#     def __len__(self):
#         return len(self.tid_dataset)

#
# class TID2013Sampler(Sampler):
#     def __init__(self,
#                  data_source,
#                  shuffle=False,
#                  shuffle_type="image-comparison"):
#         """
#         :param data_source:
#         :param shuffle:
#         :param shuffle_type:
#             controls shuffling order when {shuffle} is True (choose from ["image", "image-comparison", "comparison", ])
#
#             Expected outputs:
#                 Let B = batch size of dataloader.
#
#                 shuffle_type == "image":
#                     shuffling is done across the image dimension such that each 120 x B
#                     subsequent indices correspond to a particular random image.
#                     Original comparisons ordering is maintained for each image (1, 2, 3, ..., 120).
#
#                 shuffle_type == "comparison":
#                     shuffling is done across the comparison dimension such that each B
#                     subsequent indices correspond to a particular random comparison for some random image.
#
#                 shuffle_type == "image-comparison":
#                     most thorough shuffling, where each sample has no ordering.
#
#         """
#         super().__init__(data_source)  # doesn't do anything, but to respect inheritance syntax
#         self.tid_dataset = data_source
#
#         self.spi = self.tid_dataset.num_patches
#         self.shuffle = shuffle
#
#         if shuffle_type is None:
#             self.shuffle = False
#
#         elif shuffle_type == "image":
#             # shuffling based on length of dataset in range 0-25 (counting by images)
#             self.shuffle_factor = self.spi * 120
#             self.range_factor = lambda: range(self.shuffle_factor)
#
#         elif shuffle_type == "comparison":
#             # shuffling based on length of dataset in range 0-3000 (counting by comparisons)
#             self.shuffle_factor = self.spi
#             self.range_factor = lambda: range(self.shuffle_factor)
#
#         elif shuffle_type == "image-comparison":
#             self.shuffle_factor = self.spi * 120
#             self.range_factor = lambda: torch.randperm(self.shuffle_factor)
#
#         else:
#             raise TypeError("Unsupported shuffle type: [{}].".format(shuffle_type))
#
#     def __iter__(self):
#         if self.shuffle:
#             n = int(len(self.tid_dataset) / self.shuffle_factor)
#             indices = torch.randperm(n)
#             # print('indices', indices)
#             indices = [(self.shuffle_factor * i + j) for i in indices for j in self.range_factor()]
#             # print('indices factored', indices)
#         else:
#             indices = torch.arange(len(self), dtype=torch.int).tolist()
#         return iter(indices)
#
#     def __len__(self):
#         return len(self.tid_dataset)




