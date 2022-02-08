import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp
from skimage.util.shape import view_as_windows


class PatchSampler(object):
    __centerbias_image_path__ = "./modules/Attention/deepgaze2/centerbias.npy"

    def __init__(self,
                 centerbias_weight=0.,
                 diffbased_weight=0.,
                 uniform_weight=1.,
                 diffbased_pow=2,
                 prenormalize=True
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
        self.prenormalize = prenormalize

        total_weight = self.uniform_weight + self.diffbased_weight + self.centerbias_weight
        assert total_weight != 0, "Must specify non-zero weights."

        if self.centerbias_weight > 0:
            self.centerbias_template = np.load(self.__centerbias_image_path__)

    def __call__(self, h, w, ho, wo, imgs=None, num_samples=1):
        # return self.get_sample_params(h, w, ho, wo, imgs, num_samples)
        return self.get_sample_params(h, w, ho, wo, imgs, num_samples)

    def get_sample_params(self, h, w, ho, wo, imgs=None, num_samples=1):
        if self.diffbased_weight == 0 and self.centerbias_weight == 0:
            # stratified grid with equal probability for each cell
            return stratified_grid_sampling(h, w, ho, wo, sample_prob=np.ones((h, w)), num_samples=num_samples)
        else:
            # compute centerbiased component
            centerbias = 0
            if self.centerbias_weight > 0:
                centerbias = centerbias_prob(self.centerbias_template, h, w)
                centerbias = centerbias / np.max(centerbias)

            diffbased = 0
            # compute diffbased component
            if self.diffbased_weight > 0:
                assert imgs is not None, "PatchSampler: 'imgs' input must be specified for difference based sampling."

                def pil2np(img):
                    im = np.array(img).astype(float)
                    if self.prenormalize:
                        im -= im.min()
                        im /= im.max()
                    return im

                ref_img = pil2np(imgs[0])
                diff = np.zeros_like(ref_img)
                for dist_img in imgs[1:]:
                    dist_img = pil2np(dist_img)
                    diff += np.abs(ref_img - dist_img)
                diff = diff / (len(imgs) - 1)  # average the difference

                diffbased = np.abs(diff)

                pow = self.diffbased_pow
                if len(diffbased.shape) > 2:
                    diffbased = np.sum(diffbased * diffbased, axis=2)  # L2 distance over color channels
                    pow = pow / 2

                diffbased = np.power(diffbased, pow)

                if np.sum(diffbased) > 1e-6:  # avoid failure case when there is little (no) difference
                    diffbased = diffbased / np.max(diffbased)  # normalize to [0-1]
                else:
                    diffbased = 0

            # add uniform weight
            sample_prob = self.centerbias_weight * centerbias + \
                          self.diffbased_weight * diffbased + \
                          self.uniform_weight

            # normalize probability; ensure that the total sum is equal 1
            sample_prob = sample_prob / np.sum(sample_prob)

            return stratified_grid_sampling(h, w, ho, wo, sample_prob=sample_prob, num_samples=num_samples)


def centerbias_prob(bias, h, w):
    # rescale to match image size
    centerbias = zoom(bias, (h / 1024, w / 1024), order=0, mode='nearest')  # original centerbias is 1024x1024
    # renormalize log density after performing zoom
    centerbias -= logsumexp(centerbias)
    # softmax for probabilities
    centerbias_p = np.exp(centerbias) / np.sum(np.exp(centerbias))
    return centerbias_p


def perturbed_grid(h, w, perturb_amount=.75):
    rsamples = perturb_amount * np.random.rand(2, h, w)
    gh, gw = np.meshgrid(np.arange(0., w, 1.), np.arange(0., h, 1.))
    gh += rsamples[0]
    gw += rsamples[1]
    return np.concatenate([np.atleast_3d(gh), np.atleast_3d(gw)], axis=2)


def halton_sequence(n, b):
    m, d = 0, 1
    samples = np.zeros(n)
    for i in range(n):
        x = d - m
        if x == 1:
            m = 1
            d *= b
        else:
            y = d // b
            while x <= y:
                y //= b
            m = (b + 1) * y - x
        samples[i] = m / d
    return samples


def halton_sequence_2d(n):
    haltonx = halton_sequence(n, 2)
    haltony = halton_sequence(n, 3)
    return np.concatenate([haltonx, haltony], axis=0).reshape(2, -1)


def stratified_grid_sampling(h, w, ho, wo, sample_prob=None, num_samples=1, num_samples_cell=4):
    __cells_r_min = 4  # minimum number of cells across max(h,w)
    __cellsize_ratio_min = 0.025  # minimum cell size: max(h,w) / ratio (in pixels); note: scales with image size

    aspect_ratio = max(h, w) / min(h, w)
    num_cells_r = max(__cells_r_min, num_samples / num_samples_cell / num_samples_cell / aspect_ratio)

    cell_size = int(max(max(h, w) * __cellsize_ratio_min, max(h, w) // num_cells_r))

    # step size in the original array
    sh = int(np.ceil((h - ho) / cell_size))
    sw = int(np.ceil((w - wo) / cell_size))

    # zero padded original probability array
    probs = np.zeros((cell_size * sh + ho, cell_size * sw + wo))
    probs[:h, :w] = sample_prob.reshape(h, w)

    probs = view_as_windows(probs, (cell_size + ho - 1, cell_size + wo - 1), (cell_size, cell_size))
    probs = np.sum(probs, axis=(2, 3))
    probs /= np.sum(probs)
    num_patches = np.ceil((probs * num_samples)).astype(int)

    # make sure total num patches doesnt exceed the queried number
    num_patches_shape = num_patches.shape
    num_patches = num_patches.flatten()
    while num_samples < num_patches.sum():
        ind_decr = (np.random.rand(num_patches.sum() - num_samples) * num_patches.shape[0]).astype(int)
        num_patches[ind_decr] = np.maximum(num_patches[ind_decr] - 1, 0)
    num_patches = num_patches.reshape(*num_patches_shape)

    halton_seq = halton_sequence_2d(num_samples)

    jmax = num_patches_shape[0] * cell_size + ho
    imax = num_patches_shape[1] * cell_size + wo

    jmax = num_patches_shape[0] - (jmax - h) / cell_size
    imax = num_patches_shape[1] - (imax - w) / cell_size

    num_cells = num_patches_shape[0] * num_patches_shape[1]
    cells_order = np.random.permutation(num_cells)

    patches_tot = 0
    samples = []
    for index in range(num_cells):
        index = cells_order[index]

        j = index // num_patches_shape[1]
        i = index % num_patches_shape[1]

        num_patches_c = num_patches[j, i]

        if num_patches_c < 1:
            continue

        halton_seq_h = halton_seq[0, patches_tot: patches_tot + num_patches_c]
        halton_seq_w = halton_seq[1, patches_tot: patches_tot + num_patches_c]

        patches = np.zeros((2, num_patches_c), np.int)
        patches[0] = np.minimum(j + halton_seq_h, jmax) * cell_size
        patches[1] = np.minimum(i + halton_seq_w, imax) * cell_size

        for k in range(num_patches_c):
            samples.append((patches[0, k], patches[1, k], ho, wo))

        patches_tot += num_patches_c

    return samples
