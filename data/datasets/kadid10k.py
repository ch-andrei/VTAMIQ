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