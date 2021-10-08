import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp
# from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import hashlib
import math


class PatchSampler(object):
    def __init__(self,
                 centerbias_weight=0.,
                 diffbased_weight=0.,
                 uniform_weight=1.,
                 diffbased_pow=2,
                 cache_use=False,
                 cache_size=1024,  # TODO: implement this
                 ):
        """
        class to generate i,j coordinates for sampling patches from a 2D image
        :param centerbias_weight:
        :param diffbased_weight:
        :param uniform_weight:
        """

        self.uniform_weight = max(0., uniform_weight)
        self.centerbias_weight = max(0., centerbias_weight)
        self.diffbased_weight = max(0., diffbased_weight)
        self.diffbased_pow = diffbased_pow

        total_weight = self.uniform_weight + self.diffbased_weight + self.centerbias_weight
        assert total_weight != 0, "Must specify non-zero weights."

        self.cache_use = cache_use
        self.cache_size = cache_size
        self.prob_cache = {}  # will store precomputed probability maps for speedup

        if self.centerbias_weight > 0:
            centerbias_image_path = "modules/Attention/deepgaze2/centerbias.npy"
            self.centerbias_template = np.load(centerbias_image_path)

    def __call__(self, h, w, ho, wo, imgs=None, num_samples=1):
        return self.get_sample_params(h, w, ho, wo, imgs, num_samples)

    def get_sample_params(self, h, w, ho, wo, imgs=None, num_samples=1):
        if self.diffbased_weight == 0 and self.centerbias_weight == 0:
            return [get_random_crop_params(h, w, ho, wo) for _ in range(num_samples)]
        else:
            # # remove out of bounds area
            hi = h - ho
            wi = w - wo

            if self.diffbased_weight > 0:
                assert imgs is not None, "PatchSampler: 'imgs' input must be specified for difference-based sampling."
                ref_img = np.array(imgs[0])
                diff = np.zeros_like(ref_img)
                for dist_img in imgs[1:]:
                    diff += np.abs(ref_img - np.array(dist_img))
                diff = diff / (len(imgs) - 1)  # average the difference
                cached_name = compute_hashname(diff, hi, wi)
            else:
                cached_name = (hi, wi)

            # check cached probability maps
            if self.cache_use and cached_name in self.prob_cache:
                sample_prob = self.prob_cache[cached_name]
            else:
                # if not in cache, recompute
                # print('computing for', cached_name)

                centerbias = 0
                # compute centerbiased component
                if self.centerbias_weight > 0:
                    centerbias = centerbias_prob(self.centerbias_template, h, w)
                    centerbias = centerbias / np.max(centerbias)
                    centerbias = centerbias[ho//2: hi+ho//2, wo//2: wi+wo//2]

                diffbased = 0
                # compute diffbased component
                if self.diffbased_weight > 0:
                    diffbased = np.abs(diff)

                    pow = self.diffbased_pow
                    if len(diffbased.shape) > 2:
                        diffbased = np.sum(diffbased * diffbased, axis=2)  # L2 distance over color channels
                        pow = pow / 2

                    diffbased = np.power(diffbased, pow)

                    if np.sum(diffbased) < 1e-6:  # failure case when there is no difference
                        diffbased += 1  # handles zero error cases by adding uniform error

                    dmax = np.max(diffbased)
                    diffbased = diffbased / (1. if dmax == 0 else dmax)  # normalize to [0-1]

                    # apply gaussian kernel to smoothen the difference map
                    # image resolution influences its apparent blurriness when using same sigma for all resolutions;
                    # want to approximately blur all resolutions similarly -> select sigma based on resolution
                    gaussian_sigma = np.power(ho*ho+wo*wo, 1/4)  # empirically, decent heuristic for size vs blurriness
                    diffbased = gaussian_filter(diffbased, sigma=gaussian_sigma, mode="constant", cval=0)

                    # crop to probability space
                    diffbased = diffbased[ho//2: hi+ho//2, wo//2: wi+wo//2]
                    dmax = np.max(diffbased)
                    diffbased = diffbased / (1. if dmax == 0 else dmax)

                # add uniform weight
                sample_prob = self.centerbias_weight * centerbias + \
                              self.diffbased_weight * diffbased + \
                              self.uniform_weight

                # normalize probability; ensure that the total sum is equal 1
                sample_prob = sample_prob / np.sum(sample_prob)

                # # debug plotting
                # plt_show("sample_prob", sample_prob)

                # compute cumulative sum over all probs
                # note: the last value in this list equals ~1.0 -> sample_prob[-1] = ~1.0
                sample_prob = np.cumsum(sample_prob.flatten())

                if self.cache_use:
                    self.prob_cache[cached_name] = sample_prob

            return get_n_random_crop_params_bias(hi, wi, ho, wo, sample_prob, num_samples)


def compute_hashname(img, a, b):
    """
    encodes the image using its str representation (similar to chechsum).
    Considers a small part of the array, its numpy ToString() representation and its average value.
    The above should be unique enough to represent most plausible images without cache repetition.
    :param img:
    :return:
    """
    to_str = lambda v: str(abs(v))[:6]
    as_str = "{}-{}-{}-{}-{}".format(a, b, str(img), to_str(img.min()), to_str(img.mean()), to_str(img.max()))
    return str(hashlib.sha256(as_str.encode()).hexdigest())


def get_random_crop_params(h, w, ho, wo):
    """Get parameters for ``crop`` for a random crop.
    Args:
        :param w: input width
        :param h: input height
        :param wo: output width
        :param ho: output height

    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    """
    if w <= wo and h <= ho:
        return 0, 0, h, w

    i = np.random.randint(0, h - ho)
    j = np.random.randint(0, w - wo)
    return i, j, ho, wo


def centerbias_prob(bias, h, w):
    # rescale to match image size
    centerbias = zoom(bias, (h / 1024, w / 1024), order=0, mode='nearest')
    # renormalize log density after performing zoom
    centerbias -= logsumexp(centerbias)
    # softmax for probabilities
    centerbias_p = np.exp(centerbias) / np.sum(np.exp(centerbias))
    return centerbias_p


def binary_search(arr, x, low, high):
    if low < high:
        mid = (high + low) // 2
        if x < arr[mid]:
            return binary_search(arr, x, low, mid - 1)
        else:
            return binary_search(arr, x, mid + 1, high)
    else:
        return low - 1


def get_n_random_crop_params_bias(h, w, ho, wo, sample_prob, num_samples=1):
    rsamples = np.random.rand(num_samples).astype(np.float32)
    sample_prob = sample_prob.reshape(-1, 1).astype(np.float32)

    samples = []

    for i in range(num_samples):
        sample = rsamples[i]

        ind = binary_search(sample_prob, sample, 0, len(sample_prob))

        # convert to i, j indices
        pos_i = int(ind / w)
        pos_j = int(ind % w)

        samples.append((pos_i, pos_j, ho, wo))

    return samples

def get_random_crop_params_bias(h, w, ho, wo, sample_prob):
    """Get parameters for ``crop`` for a random crop given a probability bias.
    Args:
        :param w: input width
        :param h: input height
        :param wo: output width
        :param ho: output height
        :param sample_prob: bias map for sampling input image. Used as a probability map for sampling.

    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    """
    # get random sample over [0, 1], and find its location in probability list
    sample = np.random.rand(1)
    sample_index = binary_search(sample_prob, sample, 0, len(sample_prob) - 1)

    # convert to i, j indices
    i = int(sample_index / w)
    j = sample_index % w

    return i, j, ho, wo


def plt_show(title, img):
    plt.title(title)
    plt.imshow(img)
    plt.colorbar()
    plt.show()


if __name__=='__main__':
    from matplotlib import pyplot as plt
    from PIL import Image

    def imread(path):
        im = Image.open(path).convert("RGB")
        return np.array(im, np.float) / 255

    path = "I:/Datasets/tid2013"
    im1 = imread(path + "/reference_images/I01.BMP")
    # im2 = imread(path + "/distorted_images/i01_01_4.BMP")  # random noise
    # im2 = imread(path + "/distorted_images/i01_18_5.BMP")  # color diff
    im2 = imread(path + "/distorted_images/i01_15_5.BMP")
    # im2 = imread(path + "/distorted_images/i01_07_5.BMP")
    # im2 = imread(path + "/distorted_images/i01_08_5.BMP")
    diff = im2 - im1
    ps = PatchSampler(
        centerbias_weight=2,
        diffbased_weight=10,
        uniform_weight=0.5,
    )
    h, w = im1.shape[0], im1.shape[1]
    out_dim = (16, 16)
    hits = np.zeros((h, w))
    num_samples = 15000
    samples = ps.get_sample_params(h, w, out_dim[0], out_dim[1], imgs=(im1, im2), num_samples=num_samples)
    for sample in samples:
        x, y = sample[:2]
        hits[x: x+out_dim[0], y: y+out_dim[1]] += 1
        hit = hits[x: x+out_dim[0], y: y+out_dim[1]]
        shape = hit.shape
        if shape[0] < out_dim[0] or shape[1] < out_dim[1]:
            print("NOT A GOOD SQUARE {}".format(shape))

    # plt_show("im", im1)
    plt_show("MSE", np.sqrt(np.sum(diff * diff, axis=2)))
    plt_show("Patches", hits)
    plt_show("Image 1", im1)
    plt_show("Image 2", im2)

    centerbias_image_path = "modules/Attention/deepgaze2/centerbias.npy"
    centerbias_template = np.load(centerbias_image_path)
    centerbias = centerbias_prob(centerbias_template, h, w)
    centerbias = centerbias / np.max(centerbias)
    plt_show("Centerbias", centerbias)
