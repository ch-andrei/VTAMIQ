import numpy as np
import scipy.io

from data.patch_datasets import PatchFRIQADataset


class LIVEDataset(PatchFRIQADataset):
    num_ref_images = 29
    num_dist_images = -1  # special case, LIVE can have different number of dist images
    img_dim = None  # special case, can vary
    num_distortions = 5

    # distortion types and number of comparisons for each distortion
    distortions = [
        ('jp2k', 227),
        ('jpeg', 233),
        ('wn', 174),
        ('gblur', 174),
        ('fastfading', 174)
    ]  # do not modify this list; keeping the ordering is important as it is used to count dataset file scores

    def __init__(self, **kwargs):

        self.ref_path = 'refimgs'

        # will be set by read_dataset()
        # format: similar to self.distortions, but with image name instead of distortion type
        self.ref_images = []  # [(img_name1, comparisons_count1), ...]

        super(LIVEDataset, self).__init__(
            name='LIVE',
            path='LIVE',
            # Raw scores are “Bad”, “Poor”, “Fair”, “Good”, and “Excellent” rescaled to 1-100;
            # Scores are then converted to DMOS values in range 0-100, with 0 being perfect quality.
            # Hence, no need to reverse the scores.
            qs_reverse=False,
            qs_linearize=True,
            **kwargs
        )

        # self.norm_mean = [0.4803, 0.4598, 0.4018]
        # self.norm_std = [0.2196, 0.2202, 0.2221]

    def read_dataset(self):
        """
        returns a list of tuples (reference_image_path, distorted_image_path, quality)
        :return:
        """
        dmos_mat = scipy.io.loadmat(self.path + '/dmos_realigned.mat')
        refnames_mat = scipy.io.loadmat(self.path + '/refnames_all.mat')

        refnames = [item[0] for item in refnames_mat['refnames_all'].flatten()]
        dmos = dmos_mat['dmos_new'].flatten()
        orgs = dmos_mat['orgs'].flatten()

        def get_distortion_info(i):
            i = max(0, i)
            k = 0
            while k + 1 < len(self.distortions) and i - self.distortions[k][1] >= 0:
                i = i - self.distortions[k][1]
                k += 1
            type, count = self.distortions[k]
            return type, 1 + min(i, count)

        comparisons_per_image = {}

        for i, q in enumerate(dmos):
            q = dmos[i]
            org = orgs[i]
            img_name_ref = refnames[i]
            distortion_type, i_dist_img = get_distortion_info(i)

            if org == 1:
                continue

            path_ref = "{}/{}/{}".format(self.path, self.ref_path, img_name_ref)
            path_dist = "{}/{}/{}".format(self.path, distortion_type, "img{}.bmp".format(i_dist_img))

            if img_name_ref not in comparisons_per_image:
                comparisons_per_image[img_name_ref] = []
            comparisons_per_image[img_name_ref].append((path_ref, path_dist, q))

        paths_ref, paths_dist, qs = [], [], []
        ref_image_names = sorted(list(comparisons_per_image.keys()))

        # add comparison counts for each image to the ref_image_names list
        # simultaneously, add corresponding paths
        for i in range(len(ref_image_names)):
            img_name_ref = ref_image_names[i]
            comparisons = comparisons_per_image[img_name_ref]

            for comparison in comparisons:
                path_ref, path_dist, q = comparison

                paths_ref.append(path_ref)
                paths_dist.append(path_dist)
                qs.append(q)

            ref_image_names[i] = (img_name_ref, len(comparisons))

        dist_images_per_image = [num_dist_images for _, num_dist_images in ref_image_names]
        self.process_dataset_data(qs, paths_ref, paths_dist, dist_images_per_image)
