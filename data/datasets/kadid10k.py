from data.datasets.tid import TID2013Dataset


class KADID10kDataset(TID2013Dataset):
    num_ref_images = 81
    num_dist_images = 125

    def __init__(self,
                 path='I:/Datasets/kadid10k',
                 **kwargs
                 ):
        super(KADID10kDataset, self).__init__(
            path=path,
            name="KADID10k",
            **kwargs
        )

        # TODO: recompute mean/std for KADID10k
        # currently using TID2013 mean/std values
        self.norm_mean = [0.4372, 0.4634, 0.4204]
        self.norm_std = [0.2421, 0.2150, 0.2291]

    def read_dataset(self):
        return super().read_dataset(
            reference_images_path='/images',
            distorted_images_path='/images',
            q_file_name="dmos.csv",
            split_char=",",
            q_ind=2,
            filename_ind=0,
            has_header=True,
            filename_ext='png',
            )