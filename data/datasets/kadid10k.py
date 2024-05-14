from data.datasets.tid import TID2013Dataset


class KADID10kDataset(TID2013Dataset):
    num_ref_images = 81
    num_dist_images = 125

    def __init__(self,
                 path='kadid10k',
                 **kwargs
                 ):
        super(KADID10kDataset, self).__init__(
            path=path,
            name="KADID10k",
            # From KADID paper:
            # quality values over a 5-point scale, i.e.,
            # imperceptible (5), perceptible but not annoying, slightly annoying, annoying, and very annoying (1).
            # Highest score corresponds to perfect quality, hence need to reverse.
            # qs_reverse=True,  # already set to True by parent TID2013
            # qs_linearize=True,  # already set to True by parent TID2013
            **kwargs
        )

    def read_dataset(self):
        super().read_dataset(
            reference_images_path='/images',
            distorted_images_path='/images',
            q_file_name="dmos.csv",
            split_char=",",
            q_ind=2,
            filename_ind=0,
            has_header=True,
            filename_ext='png',
            )