import numpy as np
import os
from data.datasets.iqa_datasets import FRIqaPatchDataset


class PIPAL(FRIqaPatchDataset):
    num_ref_images = 200
    num_dist_images = 116

    def __init__(self,
                 name="PIPAL",
                 path="I:/Datasets/PIPAL",
                 **kwargs
                 ):
        super(PIPAL, self).__init__(
            name=name,
            path=path,

            # qs_linearize=False,

            **kwargs
        )

        self.img_dim = (288, 288)

    def read_dataset(self):
        """
        returns a list of tuples (reference_image_path, distorted_image_path, quality)
        where q is DMOS
        :return:
        """

        reference_images_path = self.path + "/Train_Ref"
        distorted_images_path = self.path + "/Train_Dist"
        labels_path = self.path + "/Train_Label"

        paths_ref, paths_dist, qs = [], [], []

        for filename in os.listdir(labels_path):
            with open(labels_path + "/" + filename, 'r') as input_file:
                for line in input_file:
                    line = line.strip()
                    line = line.split(",")

                    dist_name = line[0]
                    ref_name = dist_name[:5] + ".bmp"  # first 5 characters
                    q = float(line[1])

                    path_reference = reference_images_path + '/' + ref_name
                    path_distorted = distorted_images_path + '/' + dist_name

                    paths_ref.append(path_reference)
                    paths_dist.append(path_distorted)
                    qs.append(q)

        return paths_ref, paths_dist, qs


class PIPALTest(PIPAL):
    num_ref_images = 25
    num_dist_images = 66

    def __init__(self,
                 name="PIPALTest",
                 suffix="Test",
                 **kwargs
                 ):
        self.suffix = suffix

        super(PIPALTest, self).__init__(
            name=name,
            qs_plot=False,
            qs_normalize=False,
            qs_reverse=False,
            qs_normalize_mean_std=False,
            **kwargs
        )

    def read_dataset(self):
        """
        returns a list of tuples (reference_image_path, distorted_image_path, quality)
        where q is DMOS
        :return:
        """

        reference_images_path = "{}/{}_Ref".format(self.path, self.suffix)
        distorted_images_path = "{}/{}_Dist".format(self.path, self.suffix)

        paths_ref, paths_dist, qs = [], [], []

        for dist_name in os.listdir(distorted_images_path):
            ref_name = dist_name[:5] + ".bmp"  # first 5 characters

            path_reference = reference_images_path + '/' + ref_name
            path_distorted = distorted_images_path + '/' + dist_name

            paths_ref.append(path_reference)
            paths_dist.append(path_distorted)
            qs.append(-1)

        return paths_ref, paths_dist, qs


class PIPALVal(PIPALTest):
    num_ref_images = 25
    num_dist_images = 40

    def __init__(self,
                 **kwargs
                 ):
        super(PIPALVal, self).__init__(
            name="PIPALVal",
            suffix="Val",
            **kwargs
        )


class PIPALVal22(PIPALTest):
    num_ref_images = 25
    num_dist_images = 66

    def __init__(self,
                 **kwargs
                 ):
        super(PIPALVal22, self).__init__(
            name="PIPALVal22",
            suffix="NTIRE2022_FR_Valid",
            **kwargs
        )


class PIPALTest22(PIPALTest):
    num_ref_images = 25
    num_dist_images = 66

    def __init__(self,
                 **kwargs
                 ):
        super(PIPALTest22, self).__init__(
            name="PIPALTest22",
            suffix="NTIRE2022_FR_Testing",
            **kwargs
        )
