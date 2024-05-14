import numpy as np

from data.patch_datasets import PatchFRIQADataset
from utils.logging import log_warn


class KADIS700kDataset(PatchFRIQADataset):
    # the full dataset should have ~140000 images with 5 distortions;
    # distortion 15 was too slow to compute using the original generation script so it was skipped.
    # total number of distorted images in dataset is 700000; without distortion 15 which has 28700 denoise images,
    # there are now (700000 - 28700) = 671300 -> 671300 / 5 = 134260 ref images
    num_ref_images = 134260  # 134260 * 5 = 671300
    num_dist_images = 5

    # Note: both "colorsaturate" and "saturate" are tagged with 7,8 in "kadis700k_ref_imgs.csv" so we map them together
    distortion_types = {
        "gblur": 1,
        "lblur": 2,
        "mblur": 3,
        "colordiffuse": 4,
        "colorshift": 5,
        "colorquantize": 6,
        "colorsaturate": 7,
        "saturate": 7,
        "jp2k": 9,
        "jpeg": 10,
        "noisegauss": 11,
        "noisecolorcomp": 12,
        "noiseimpulse": 13,
        "noisemultiplicative": 14,
        "denoise": 15,
        "brighten": 16,
        "darken": 17,
        "meanshift": 18,
        "jitter": 19,
        "noneccentricity": 20,
        "pixelate": 21,
        "noisequantize": 22,
        "colorblock": 23,
        "sharpenHi": 24,
        "contrastchange": 25,
    }

    # Our dataset contains the following categories and image counts (without distortion 15)
    # {'brighten': 27590,
    #  'colorblock': 26860,
    #  'colordiffuse': 27995,
    #  'colorquantize': 28995,
    #  'colorsaturate': 27870,
    #  'colorshift': 28045,
    #  'contrastchange': 28335,
    #  'darken': 28350,
    #  'denoise': 0,
    #  'gblur': 27940,
    #  'jitter': 28290,
    #  'jp2k': 28355,
    #  'jpeg': 27645,
    #  'lblur': 28175,
    #  'mblur': 27815,
    #  'meanshift': 28255,
    #  'noisecolorcomp': 28090,
    #  'noisegauss': 28035,
    #  'noiseimpulse': 27595,
    #  'noisemultiplicative': 27255,
    #  'noisequantize': 28055,
    #  'noneccentricity': 27770,
    #  'pixelate': 28250,
    #  'saturate': 27645,
    #  'sharpenHi': 28090}

    def __init__(self,
                 preprocess=False,  # can disable Q preprocessing
                 version=1,  # 0 for original, 1 and 2 for modified
                 **kwargs
                 ):

        self.preprocess = preprocess
        self.version = version

        if version == 0:
            self.scores_file = "kadis700k_friqa_no15.csv"  # original
        elif version == 1:
            self.scores_file = "kadis700k_vtamiq.csv"  # vtamiq trained on kadid
        elif version == 2:
            self.scores_file = "kadis700k_v2.csv"  # vtamiq trained on display model and PU-encoded pieapp
        else:
            raise ValueError("Incorrect dataset version selected.")

        super(KADIS700kDataset, self).__init__(
            path="kadis700k",
            name="KADIS700k",
            qs_reverse=False,
            qs_linearize=False,
            use_ref_img_cache=False,  # the dataset is too large for caching the reference images
            **kwargs
        )

    def read_dataset(self):
        """
        :return:
        """
        reference_images_path = self.path + "/kadis700k/ref_imgs"
        distorted_images_path = self.path + "/kadis700k/dist_imgs"

        q_index = 6 if self.version == 0 else -1  # VSI for original version of KADIS or VTAMIQ for customized
        # q_vsi = float(line[6])
        # q_mdsi = float(line[5])
        # q_fsim = float(line[7])
        # q_sff = float(line[9])
        # q_vtamiq = float(line[-1])

        paths_ref, paths_dist, qs = [], [], []
        q_file_path = self.path + "/" + self.scores_file
        with open(q_file_path, 'r') as q_file:
            q_file.__next__()  # skip header line

            for i, line in enumerate(q_file):

                line = line.strip().split(',')  # split by comma or space

                path_distorted = line[0][:-4]
                path_reference = line[1]

                distorted_split = path_distorted.split('_')

                distortion_type_digit = self.distortion_types[distorted_split[-2]]
                distortion_level = int(distorted_split[-1])

                if distortion_type_digit == 15:
                    continue  # skip distortion type 15

                path_distorted = "{}_{:02d}_{:02d}.bmp".format(path_reference[:-4], distortion_type_digit, distortion_level)

                q = float(line[q_index])

                paths_ref.append(reference_images_path + '/' + path_reference)
                paths_dist.append(distorted_images_path + '/' + path_distorted)
                qs.append(q)

        dist_images_per_image = [self.num_dist_images for _ in range(self.num_ref_images)]
        self.process_dataset_data(qs, paths_ref, paths_dist, dist_images_per_image)

    def process_qs(self):
        if self.preprocess:
            log_warn(f"Dataset {self.name} running self.process_qs()...")
            super().process_qs()
            log_warn(f"Dataset {self.name} self.process_qs() completed.")
