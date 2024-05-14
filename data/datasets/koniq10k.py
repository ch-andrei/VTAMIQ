import numpy as np
from data.patch_datasets import NRIqaPatchDataset


# TODO: fix this
class KONIQ10k(NRIqaPatchDataset):
    num_ref_images = 10073
    num_dist_images = 1

    def __init__(self,
                 path='koniq10k',
                 resolution="half",
                 use_mos_zscore=False,
                 **kwargs
                 ):
        if resolution == "full":
            print("KONIQ10k using full resolution images")
            self.img_dim = (768, 1024)
            self.images_path = path + '/1024x768'
        elif resolution == "half":
            print("KONIQ10k using half resolution images")
            self.img_dim = (384, 512)
            self.images_path = path + '/512x384'
        else:
            raise ValueError("KONIQ10k: Resolution must be 'full' or 'half'")

        self.q_file_path = path + '/koniq10k_scores_and_distributions.csv'
        self.use_mos_zscore = use_mos_zscore

        super(KONIQ10k, self).__init__(
            name='KONIQ10k',
            path=path,
            **kwargs
        )

        # self.norm_mean = [0.4618, 0.4178, 0.3700]
        # self.norm_std = [0.2298, 0.2159, 0.2132]

    def read_dataset(self):
        """
        returns a list of tuples (reference_image_path, distorted_image_path, quality)
        :return:
        """

        q_index = 9 if self.use_mos_zscore else 7

        paths, qs = [], []
        with open(self.q_file_path, 'r') as q_file:
            q_file.__next__()  # skip header line

            # the file is formatted as "mos_value distorted_image_filename"
            for line in q_file:
                line = line.strip().split(",")  # split by comma or space

                path = self.images_path + '/' + line[0].replace("\"", "")
                q = float(line[q_index])

                paths.append(path)
                qs.append(q)

        self.qs = qs
        self.paths = paths
