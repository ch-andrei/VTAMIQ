import numpy as np
import pandas as pd
from data.patch_datasets import PatchFRIQADataset


class CSIQDataset(PatchFRIQADataset):
    num_ref_images = 30
    num_dist_images = -1  # special case, can be 28 or 29
    img_dim = (512, 512)

    def __init__(self,
                 name="CSIQ",
                 path="CSIQ",
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

        super(CSIQDataset, self).__init__(
            name=name,
            path=path,
            # Dataset reports DMOS values [0-1]; larger values correspond to larger distortions; no need to reverse
            qs_reverse=False,
            **kwargs
        )

    def read_dataset(self):
        """
        returns a list of tuples (reference_image_path, distorted_image_path, quality)
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

        # count number of distorted images per reference image;
        # setup ref/dist paths
        paths_ref, paths_dist, qs = [], [], []
        image_names = sorted(list(comparisons_per_image.keys()))
        dist_images_per_image = np.zeros(len(image_names), int)
        for i, img_name_ref in enumerate(image_names):
            comparisons = comparisons_per_image[img_name_ref]

            for comparison in comparisons:
                path_ref, path_dist, q = comparison

                paths_ref.append(path_ref)
                paths_dist.append(path_dist)
                qs.append(q)

            dist_images_per_image[i] = len(comparisons)

        self.process_dataset_data(qs, paths_ref, paths_dist, dist_images_per_image)
