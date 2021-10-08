import numpy as np
import pandas as pd
from data.datasets.iqa_datasets import FRIqaPatchDataset


class CSIQDataset(FRIqaPatchDataset):
    num_ref_images = 30
    num_dist_images = -1  # special case, can be 28 or 29

    def __init__(self,
                 name="CSIQ",
                 path="I:/Datasets/CSIQ",
                 **kwargs
                 ):
        self.distortions = {
            1: "awgn",
            2: "jpeg",
            3: "jpeg2000",
            4: "fnoise",
            5: "blur",
            6: "contrast",
        }

        self.img_dim = (512, 512)

        super(CSIQDataset, self).__init__(
            name=name,
            path=path,
            qs_reverse=False,
            **kwargs
        )

    def read_dataset(self):
        """
        returns a list of tuples (reference_image_path, distorted_image_path, quality)
        where q is DMOS from CSIQ.
        :return:
        """
        ref_imgs_path = self.path + "/src_imgs"
        dis_imgs_path = self.path + "/dst_imgs"
        q_file_path = self.path + "/DMOS.csv"

        q_ind = 5
        filename_ind = 0
        dst_type_ind = 1
        dst_lev_ind = 3

        filename_ext = "png"

        comparisons_per_image = {}

        with open(q_file_path, 'r') as q_file:
            q_file.__next__()  # skip header line

            for line in q_file:
                line = line.strip().split(',')  # split by comma or space

                img_name = line[filename_ind]
                dst_type = self.distortions[int(line[dst_type_ind])]  # remove spaces
                dst_lev = line[dst_lev_ind]

                # the first 3 letters are the reference file name
                path_ref = ref_imgs_path + '/' + img_name + "." + filename_ext
                path_dist = "{}/{}/{}.{}.{}.{}".format(dis_imgs_path, dst_type, img_name, dst_type, dst_lev, filename_ext)
                q = float(line[q_ind])

                if img_name not in comparisons_per_image:
                    comparisons_per_image[img_name] = []
                comparisons_per_image[img_name].append((path_ref, path_dist, q))

        paths_ref, paths_dist, qs = [], [], []

        images = sorted(list(comparisons_per_image.keys()))

        # add comparison counts for each image to the images list
        # simulatenously, add corresponding paths
        for i in range(len(images)):
            img_name_ref = images[i]
            comparisons = comparisons_per_image[img_name_ref]

            for comparison in comparisons:
                path_ref, path_dist, q = comparison

                paths_ref.append(path_ref)
                paths_dist.append(path_dist)
                qs.append(q)

            images[i] = (img_name_ref, len(comparisons))

        self.images = images

        return paths_ref, paths_dist, qs

    # override
    def comparisons_before_image(self, i):
        return 0 if i < 1 else sum([item[1] for item in self.images[:i]])

    # override
    def comparisons_per_image(self, i):
        return self.images[i][1]