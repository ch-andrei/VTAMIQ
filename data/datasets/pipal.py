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
            # disable all q preprocessing
            qs_normalize=False,
            qs_reverse=False,
            qs_linearize=False,
            qs_linearize_plot=False,
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


if __name__ == "__main__":
    # preprocess for NTIRE2021 competition submission format

    pipal = PIPALTest()
    path = "./output/1623772844-PIPALVal-p16-VTAMIQ-B16-6L-4R-5e-18b-256p-TESTSET"
    vtamiq_file_name = "test.txt"
    output_file_name = "output.txt"
    pipal_test_distored_path = pipal.path + "/Val_Dist"
    dist_files = os.listdir(pipal_test_distored_path)
    output_path = path + "/submission"

    # read predictions
    qs = []
    with open("{}/{}".format(path, vtamiq_file_name), 'r') as input_file:
        for line in input_file:
            line = line.strip()
            line = line.split(" ")
            line = line[2].split(",")
            for q in line:
                qs.append(float(q))

    # normalize
    qs = np.array(qs)
    qs -= qs.min()
    qs /= qs.max()

    # sort
    scores = []
    for dist_file, q in zip(dist_files, qs):
        scores.append((dist_file, q))
    scores = sorted(scores, key=lambda x: x[0])

    # build output folders
    os.makedirs(output_path, exist_ok=True)

    # write predictions file
    # assume predictions have the same order as file names (should be the case when shuffle=False in dataset)
    with open("{}/{}".format(output_path, output_file_name), 'w') as output_file:
        for name, q in scores:
            output_file.write("{},{}\n".format(name, q))

    # write readme file
    with open("{}/{}".format(output_path, "readme.txt"), 'w') as output_file:
        output_file.write("runtime per image [s] : 0.01\n"
                          "CPU[1] / GPU[0] : 1\n"
                          "Extra Data [1] / No Extra Data [0] : 1\n"
                          "Other description : RTC\n"
                          "Full-Reference [1] / Non-Reference [0] : 1\n"
                          )

    # zip everything
    import zipfile

    output_files = os.listdir(output_path)
    zipf = zipfile.ZipFile(output_path + "/submission.zip", 'w', zipfile.ZIP_DEFLATED)
    for file in output_files:
        zipf.write(os.path.join(output_path, file), file)

    zipf.close()
