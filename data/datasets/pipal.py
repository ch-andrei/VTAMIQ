import numpy as np
import os
from data.patch_datasets import PatchFRIQADataset


class PIPAL(PatchFRIQADataset):
    num_ref_images = 200
    num_dist_images = 116
    num_distortions = 75
    img_dim = (288, 288)

    def __init__(self,
                 name="PIPAL",
                 path="PIPAL",
                 **kwargs
                 ):
        super(PIPAL, self).__init__(
            name=name,
            path=path,

            # PIPAL provides ELO scores for images; lower ELO scores correspond to lower quality; must reverse.
            # disabling reverse before linearize yields a less aggressive fit; with reverse, better normalization, but
            # more extreme bad quality samples which seems inadequate.
            qs_reverse=True,
            qs_linearize=True,

            **kwargs
        )

    def read_dataset(self):
        """
        returns a list of tuples (reference_image_path, distorted_image_path, quality)
        :return:
        """

        reference_images_path = self.path + "/Train_Ref"
        distorted_images_path = self.path + "/Train_Dist"
        labels_path = self.path + "/Train_Label"

        paths_ref, paths_dist, qs = [], [], []

        for filename in sorted(os.listdir(labels_path)):
            with open(labels_path + "/" + filename, 'r') as input_file:
                for line in input_file:
                    line = line.strip()
                    line = line.split(",")

                    dist_name = line[0]
                    ref_name = dist_name[:5] + ".bmp"  # first 5 characters
                    q = float(line[1])

                    path_reference = reference_images_path + '/' + ref_name
                    path_distorted = distorted_images_path + '/' + dist_name

                    qs.append(q)
                    paths_ref.append(path_reference)
                    paths_dist.append(path_distorted)

        dist_images_per_image = [self.num_dist_images for _ in range(self.num_ref_images)]
        self.process_dataset_data(qs, paths_ref, paths_dist, dist_images_per_image)


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
            **kwargs
        )

    def read_dataset(self):
        """
        returns a list of tuples (reference_image_path, distorted_image_path, quality)
        :return:
        """

        reference_images_path = "{}/{}_Ref".format(self.path, self.suffix)
        distorted_images_path = "{}/{}_Dist".format(self.path, self.suffix)

        paths_ref, paths_dist, qs = [], [], []

        for dist_name in sorted(os.listdir(distorted_images_path)):
            ref_name = dist_name[:5] + ".bmp"  # first 5 characters

            path_reference = reference_images_path + '/' + ref_name
            path_distorted = distorted_images_path + '/' + dist_name

            qs.append(-1)
            paths_ref.append(path_reference)
            paths_dist.append(path_distorted)

        dist_images_per_image = [self.num_dist_images for _ in range(self.num_ref_images)]
        self.process_dataset_data(qs, paths_ref, paths_dist, dist_images_per_image)


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
