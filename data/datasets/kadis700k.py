import numpy as np

from data.datasets.iqa_datasets import FRIqaPatchDataset
from utils.image_tools import normalize_array


class KADIS700kDataset(FRIqaPatchDataset):
    # the full dataset should have ~140000 images with 5 distortions;
    # distortion 15 was too slow to generate using the originally provided MATLAB script so it was skipped.
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
                 path='I:/Datasets/kadis700k',
                 preprocess=True,
                 kadis_version=2,  # 0 for original, 1 and 2 for modified
                 **kwargs
                 ):

        self.preprocess = preprocess

        if kadis_version == 0:
            self.scores_file = "kadis700k_friqa_no15.csv"
        elif kadis_version == 1:
            self.scores_file = "kadis700k_vtamiq.csv"
        elif kadis_version == 2:
            self.scores_file = "kadis700k_v2.csv"
        else:
            raise ValueError("Incorrect dataset version selected.")

        super(KADIS700kDataset, self).__init__(
            path=path,
            name="KADIS700k",
            qs_reverse=False,
            **kwargs
        )

    def read_dataset(self):
        """
        :return:
        """
        reference_images_path = self.path + "/kadis700k/ref_imgs"
        distorted_images_path = self.path + "/kadis700k/dist_imgs"

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

                q_v = float(line[-1])
                # q_mdsi = float(line[5])
                # q_vsi = float(line[6])
                # q_fsim = float(line[7])
                # q_sff = float(line[9])

                q = q_v

                paths_ref.append(reference_images_path + '/' + path_reference)
                paths_dist.append(distorted_images_path + '/' + path_distorted)
                qs.append(q)

        return paths_ref, paths_dist, qs

    def process_data(self, data):
        print(self.name, "processing Q array. This will take about 20s...")

        if not self.preprocess:
            return data

        data = super().process_data(data)

        # on top of the regular data processing, KADIS also needs to apply a normalzing fit for the Q values

        qs = data[-1]

        from utils.correlations import FitFunction

        qs_counts = np.arange(len(qs))
        qs_lin = qs_counts.flatten() / len(qs)
        qs_sort = np.sort(qs)
        try:
            fit_function = FitFunction(qs_sort, qs_lin)
            qs_lin = fit_function(qs)
            qs_lin = normalize_array(qs_lin)

            from matplotlib import pyplot as plt
            plt.plot(qs_sort, qs_counts, 'ro', markersize=0.5, label='before hist eq.')
            plt.plot(normalize_array(fit_function(qs_sort)), qs_counts, 'bo', markersize=0.5, label='after hist eq.')
            plt.legend()
            plt.show()
        except OverflowError:
            print(self.name, ": Overflow during Q array linear fit. Using raw quality values instead.")
            qs_lin = qs

        return data[:-1] + (qs_lin, )
