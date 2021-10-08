import numpy as np
import os
from data.datasets.iqa_datasets import FRIqaPatchDataset, PairwiseFRIqaPatchDataset


class PieAPPTrainPairwise(PairwiseFRIqaPatchDataset):
    num_ref_images = 140
    num_dist_images = 483

    def __init__(self,
                 name="PieAPPTrainPairwise",
                 path="I:/Datasets/PieAPP_dataset",
                 **kwargs
                 ):
        super(PieAPPTrainPairwise, self).__init__(
            name=name,
            path=path,
            **kwargs
        )

        self.img_dim = (256, 256)

    def read_dataset(self):
        """
        returns a list of tuples (reference_image_path, distorted_image_path, quality)
        where q is DMOS
        :return:
        """

        reference_images_path = self.path + "/reference_images/train"
        distorted_images_path = self.path + "/distorted_images/train"
        labels_path = self.path + "/labels/train"

        paths_ref, paths_dist1, paths_dist2, qs = [], [], [], []
        label_files = os.listdir(labels_path)
        for label_filename in label_files:
            with open("{}/{}".format(labels_path, label_filename), 'r') as label_file:
                label_file.__next__()  # skip header

                for line in label_file:
                    line = line.strip()
                    line = line.split(",")

                    ref_name = line[0]
                    ref_name_no_ext = ref_name[:-4]

                    path_reference = reference_images_path + '/' + line[0]
                    path_distorted1 = distorted_images_path + '/' + ref_name_no_ext + "/" + line[1]
                    path_distorted2 = distorted_images_path + '/' + ref_name_no_ext + "/" + line[2]
                    q = float(line[4])  # column 5, processed probability of preference for image A

                    paths_ref.append(path_reference)
                    paths_dist1.append(path_distorted1)
                    paths_dist2.append(path_distorted2)
                    qs.append(q)

        return paths_ref, paths_dist1, paths_dist2, qs


class PieAPPTestset(FRIqaPatchDataset):
    num_ref_images = 40
    num_dist_images = 15

    def __init__(self,
                 name="PieAPPTestset",
                 path="I:/Datasets/PieAPP_dataset",
                 **kwargs
                 ):
        super(PieAPPTestset, self).__init__(
            name=name,
            path=path,
            qs_reverse=False,  # no need to compute "q = 1.0 - q"
            **kwargs
        )

        self.img_dim = (256, 256)

    def read_dataset(self):
        """
        returns a list of tuples (reference_image_path, distorted_image_path, quality)
        where q is DMOS
        :return:
        """

        reference_images_path = self.path + "/reference_images/test"
        distorted_images_path = self.path + "/distorted_images/test"
        ref_names_filename = self.path + "/test_reference_list.txt"

        paths_ref, paths_dist, qs = [], [], []
        with open(ref_names_filename, 'r') as ref_names_file:

            for line in ref_names_file:
                ref_name = line.strip()
                ref_name_no_ext = ref_name[:-4]  # remove extension

                # get labels
                labels_path = self.path + "/labels/test/{}_per_image_score.csv".format(ref_name_no_ext)
                with open(labels_path, 'r') as labels_file:
                    labels_file.__next__()  # skip header line

                    for line in labels_file:
                        line = line.strip().split(',')

                        dist_img_name = line[1]

                        # the first 3 letters are the reference file name
                        path_reference = reference_images_path + '/' + ref_name
                        path_distorted = distorted_images_path + '/' + ref_name_no_ext + '/' + dist_img_name
                        q = float(line[2])

                        paths_ref.append(path_reference)
                        paths_dist.append(path_distorted)
                        qs.append(q)

        return paths_ref, paths_dist, qs
