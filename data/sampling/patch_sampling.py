import math

import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp
from skimage.util.shape import view_as_windows

import torch
import torch.nn as nn

from matplotlib import pyplot as plt


def plt_show(title, img, draw=True, dpi=300):
    if draw:
        plt.figure(dpi=dpi)
        plt.title(title)
        plt.imshow(img)
        plt.colorbar()
        plt.show()


class PatchSampler(object):
    __centerbias_image_path__ = "modules/Attention/deepgaze2/centerbias.npy"

    def __init__(self,
                 centerbias_weight=0.,
                 diffbased_weight=0.,
                 uniform_weight=1.,
                 ):
        """
        class to generate i,j coordinates for sampling patches from a 2D image
        :param centerbias_weight:
        :param diffbased_weight:
        :param uniform_weight:
        """

        self.centerbias_weight = max(0., centerbias_weight)
        self.diffbased_weight = max(0., diffbased_weight)
        self.uniform_weight = max(0., uniform_weight)

        total_weight = self.uniform_weight + self.diffbased_weight + self.centerbias_weight
        assert 1e-6 < total_weight, "Must specify non-zero weights."

        if self.centerbias_weight > 0:
            self.centerbias_template = np.load(PatchSampler.__centerbias_image_path__)

    def __call__(self, h, w, ho, wo, diff=None, num_samples=1, debug=False):
        return self.get_sample_params(h, w, ho, wo, diff=diff, num_samples=num_samples, debug=debug)

    def get_sample_params(self, h, w, ho, wo, diff=None, num_samples=1, debug=False):
        if self.diffbased_weight == 0 and self.centerbias_weight == 0:
            if debug:
                print('Using uniform stratified sampling')
            # stratified grid with equal probability for each cell
            return stratified_grid_sampling(h, w, ho, wo, sample_prob=np.ones((h, w)), num_samples=num_samples,
                                            debug=debug)
        else:
            # compute centerbiased component
            centerbias = 0
            if self.centerbias_weight > 0:
                centerbias = centerbias_prob(self.centerbias_template, h, w)
                centerbias = centerbias / np.max(centerbias)

            diffbased = 0
            # compute diffbased component
            if self.diffbased_weight > 0:
                assert diff is not None, "PatchSampler: 'diff' input must be specified for difference-based sampling."

                diffbased = diff.copy()

                if len(diff.shape) > 2:
                    diffbased = np.sqrt(np.sum(diffbased * diffbased, axis=2))  # L2 distance over color channels

                if np.std(diffbased) > 1e-6:  # avoid failure case when there is little (no) difference
                    diffbased = diffbased / np.std(diffbased)  # normalize by std deviation
                else:
                    diffbased = 0

            sample_prob = self.centerbias_weight * centerbias + \
                          self.diffbased_weight * diffbased + \
                          self.uniform_weight

            # normalize probability; ensure that the total sum is equal 1
            sample_prob = sample_prob / np.sum(sample_prob)

            return stratified_grid_sampling(h, w, ho, wo, sample_prob=sample_prob, num_samples=num_samples, debug=debug)


def centerbias_prob(bias, h, w):
    # rescale to match image size
    centerbias = zoom(bias, (h / 1024, w / 1024), order=0, mode='nearest')  # original centerbias is 1024x1024
    # renormalize log density after performing zoom
    centerbias -= logsumexp(centerbias)
    # softmax for probabilities
    centerbias = np.exp(centerbias)
    centerbias_p = centerbias / np.sum(centerbias)
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


def stratified_grid_sampling(h, w, ho, wo, sample_prob, num_samples=1, randomize_cell_order=True, debug=False):
    __cellsize_ratio = 3
    __patchsize_ratio = 0.5
    __patch2image_ratio = 3

    clip_int = lambda x, a, b: int(max(a, min(b, x)))
    cell_size = clip_int(
        np.sqrt(h * w / num_samples * __cellsize_ratio),
        __patchsize_ratio * min(ho, wo),
        max(h, w) / max(ho, wo) * __patch2image_ratio,
    )

    if debug:
        print('stratified_grid_sampling, h, w, ho, wo, cell_size, num_samples', h, w, ho, wo, cell_size, num_samples)

    # step size in the original array
    sh = int(np.ceil((h - ho) / cell_size))
    sw = int(np.ceil((w - wo) / cell_size))

    # maximum cell indices
    jcell_dec = ((h - ho) / cell_size) % 1.
    icell_dec = ((w - wo) / cell_size) % 1.

    # zero padded original probability array to match full cell size and count
    probs = np.zeros((cell_size * sh + ho, cell_size * sw + wo))
    probs[:h, :w] = sample_prob.reshape(h, w)

    # if debug:
    #     plt_show(
    #         "{}x{}p; probability".format(ho, wo, h, w),
    #         probs.astype(float)
    #     )

    probs = view_as_windows(probs, (cell_size + ho - 1, cell_size + wo - 1), (cell_size, cell_size))
    probs = np.sum(probs, axis=(2, 3))

    # rescale edge cell probabilities to compensate for partial cells
    if 1e-3 < jcell_dec:
        probs[-1] *= jcell_dec
    if 1e-3 < icell_dec:
        probs[:, -1] *= icell_dec

    probs /= np.sum(probs)
    # probs[probs < 1e-6] = 0
    num_patches = np.round((probs * num_samples)).astype(int)

    # make sure total num patches doesnt exceed the queried number
    num_patches_shape = num_patches.shape
    num_patches = num_patches.flatten()
    while num_samples != num_patches.sum():
        diff = num_patches.sum() - num_samples
        ind_decr = (np.random.rand(abs(diff)) * num_patches.shape[0]).astype(int)
        incr = 1 if diff < 0 else -1
        num_patches[ind_decr] = np.maximum(num_patches[ind_decr] + incr, 0)
    num_patches = num_patches.reshape(*num_patches_shape)

    if debug:
        print('num_patches, min, max, sum', num_patches.min(), num_patches.max(), num_patches.sum())
        plt_show(
            "PROBS: {}x{}p; probability/cell".format(ho, wo, h, w),
            probs.astype(float)
        )
        plt_show(
            "PATCHES: {}x{}p; {}x{}i; {}x{}c; {}s; Np/cell".format(ho, wo, h, w, *num_patches.shape[:2], num_samples),
            num_patches.astype(float)
        )

    halton_seq = halton_sequence_2d(num_samples)

    num_cells = num_patches_shape[0] * num_patches_shape[1]

    if randomize_cell_order:
        cells_order = np.random.permutation(num_cells)  # randomize cell order to avoid repeating halton seq pattern
    else:
        cells_order = np.arange(num_cells)

    patches_tot = 0
    samples = []
    for index in range(num_cells):
        # current cell index
        index = cells_order[index]
        j = index // num_patches_shape[1]
        i = index % num_patches_shape[1]

        num_patches_c = num_patches[j, i]  # current cell num patches
        if num_patches_c < 1:
            continue  # zero patch case

        halton_seq_h = halton_seq[0, patches_tot: patches_tot + num_patches_c]
        halton_seq_w = halton_seq[1, patches_tot: patches_tot + num_patches_c]

        # rescale edge cell indices to compensate for partial cells
        if j == num_patches_shape[0] - 1 and 1e-3 < jcell_dec:
            halton_seq_h *= jcell_dec
        if i == num_patches_shape[1] - 1 and 1e-3 < icell_dec:
            halton_seq_w *= icell_dec

        patches = np.zeros((2, num_patches_c), np.int)
        patches[0] = (j + halton_seq_h) * cell_size
        patches[1] = (i + halton_seq_w) * cell_size

        for k in range(num_patches_c):
            samples.append((patches[0, k], patches[1, k]))

        patches_tot += num_patches_c

    return samples


def get_iqa_patches_old(imgs: tuple,
                        tensors: tuple,
                        patch_count,
                        patch_dim,
                        patch_sampler
                        ):
    """
    returns a tuple with patches and patch positions. Supports FR and NR IQA.
    :param imgs: tuple of images, FR-IQA uses 2 images (ref. and dist. images) else only 1 for NR-IQA
    :param patch_count: how many patches per images
    :param patch_dim: patch dimensions
    :param patch_sampler: patch sampler to use
    :return:
    """
    assert len(imgs) == len(tensors), "Image and Tensor counts should match."

    h, w = imgs[0].height, imgs[0].width

    diff = compute_diff(imgs)

    samples = patch_sampler.get_sample_params(
        h, w,
        patch_dim[0], patch_dim[1],
        diff=diff,  # for difference-based sampling
        num_samples=patch_count,
    )

    patches_pos = np.array(samples)

    # patches_pos = patches_pos / np.array([h - patch_dim[0], w - patch_dim[1]], float).reshape(1, 2)

    def tensor2patches(tensor, samples):
        patches = torch.zeros((patch_count, 3) + patch_dim)
        for patch_num, (i, j) in enumerate(samples):
            patches[patch_num] = tensor[:, i: i + patch_dim[0], j: j + patch_dim[1]]
        return patches

    # convert patches to tensors and add to output tuple
    patches_tuple = tuple()
    for tensor in tensors:
        patches_tuple += (tensor2patches(tensor, samples),)

    # add UV coords to output tuple
    patches_tuple += (patches_pos,)

    # output tuple format: (img1 patches, img2 patches, positions)
    return patches_tuple


def compute_diff(pil_imgs):
    def pil2np(img, prenormalize=True):
        im = np.array(img).astype(float)
        if prenormalize:
            im -= im.min()
            im /= im.max()
        return im

    imgs_np = [pil2np(img) for img in pil_imgs]

    # compute difference
    ref_img = imgs_np[0]
    diff = np.zeros_like(ref_img)
    for dist_img in imgs_np[1:]:
        diff += np.abs(ref_img - dist_img)

    return diff / (len(imgs_np) - 1)  # average the difference


def compute_patch_scales(patch_num_scales, h, w, ho, wo):
    patch_dim_m = max(ho, wo)
    if 1 < patch_num_scales:
        # determine how many scales are possible
        dim_max = min(h, w)
        patch_num_scales_max = 0
        while 1 < dim_max:
            patch_num_scales_max += 1
            dim_max = (dim_max - patch_dim_m) / 2
        patch_num_scales = min(patch_num_scales_max, patch_num_scales)
    else:
        patch_num_scales = 1
    return patch_num_scales


def compute_num_patches(patch_count, patch_num_scales, scale_num_samples_ratio):
    """
    computes patch counts per scale given a total number of scales and a total patch count.

    :param patch_count:
    :param patch_num_scales:
    :param scale_num_samples_ratio:
            ex: using total of 3 scales,
            scale 2: 16x16
            scale 1: 32x32
            scale 0: 64x64
    :return:
    """
    num_patches = 2 ** (scale_num_samples_ratio * np.arange(patch_num_scales))
    num_patches = np.ceil(num_patches * patch_count / np.sum(num_patches)).astype(int)
    cum_samples = np.cumsum(num_patches)
    for i in range(patch_num_scales):
        if patch_count <= cum_samples[i]:
            num_patches[i] -= cum_samples[i] - patch_count
            num_patches[i + 1:] = 0
            break
    return num_patches


def get_iqa_patches(imgs: tuple,
                    tensors: tuple,
                    patch_count,
                    patch_dim,
                    patch_sampler,
                    patch_num_scales,
                    scale_num_samples_ratio=1.75,
                    randomize_patch_scale_order=True,
                    debug=False
                    ):
    """
    returns a tuple with patch_data and patch positions. Supports FR and NR IQA.
    :param imgs: tuple of images, FR-IQA uses 2 images (ref. and dist. images) else only 1 for NR-IQA
    :param patch_count: how many patch_data per images
    :param patch_dim: patch dimensions
    :param patch_sampler: patch sampler to use
    :param patch_num_scales: how many different scales to extract
    :param scale_num_samples_ratio: constant r used to compute number of patches per scale:
        num_patches_for_scale = 2 ** (r * range(i)), where i is current scale index (ordered: 0, 1, 2, ...).
        higher values lead to fewer patches for large scales.
        example when ratio=2 corresponds to num_patches will be 1,4,16,64...
        example when using 2 scales -> index 0: 32x32 pixels, index 1: 16x16 pixels
        see function "compute_num_samples()"
    :param randomize_patch_scale_order: when multiple scales are used, patches by default are ordered by scale:
        ex: patch sequence with corresponding scales 0,0,...,0,1,1,...,1,2,2,...,2...
        when randomize_patch_scale_order is set to True, the final patch sequence will be randomly shuffled
    :return:
    """
    assert len(imgs) == len(tensors), "get_iqa_patches(): Image and Tensor counts should match."
    assert patch_num_scales <= patch_count, "get_iqa_patches(): number of patches must be at least number of scales."

    height, width = imgs[0].height, imgs[0].width

    # print('want patch_count', patch_count, 'patch_num_scales', patch_num_scales)
    # print('num_patches', num_patches)
    # exit()

    # precompute patch order
    if randomize_patch_scale_order:
        patch_indices = np.random.permutation(patch_count)
    else:
        patch_indices = np.arange(patch_count)

    diff = compute_diff(imgs)
    patch_num_scales = compute_patch_scales(patch_num_scales, height, width, patch_dim[0], patch_dim[1])
    num_patches = compute_num_patches(patch_count, patch_num_scales, scale_num_samples_ratio)

    if debug:
        print('num_patches', num_patches)

    patches = torch.zeros((len(imgs), patch_count, 3) + (patch_dim[0], patch_dim[1]))
    positions = torch.zeros((patch_count, 2))
    scales = torch.zeros((patch_count, 2))
    mean_pooler = nn.AvgPool2d(kernel_size=2)  # 2x downsampler
    num_samples_total = 0
    for scale in range(patch_num_scales):
        num_patches_s = num_patches[-scale - 1]

        h, w = diff.shape[:2]

        samples = patch_sampler.get_sample_params(
            h, w,
            patch_dim[0], patch_dim[1],
            diff=diff,
            num_samples=num_patches_s,
            # debug=debug
        )  # N x 2

        if debug:
            patches_pos = torch.as_tensor(np.array(samples)) * 2 ** scale  # 1. debug
        else:
            patches_pos = np.array(samples) + \
                          np.array([patch_dim[0] // 2, patch_dim[1] // 2], float).reshape(1, 2)  # 3. centers
            patches_pos = patches_pos / np.array([h - patch_dim[0] // 2, w - patch_dim[1] // 2], float).reshape(1, 2)  # 3. rescale to [0, 1]
            patches_pos = torch.clamp(torch.as_tensor(patches_pos), 0., 1. - 1e-6)  # 3.

        patches_scale = torch.zeros_like(patches_pos)
        patches_scale[:, 0] = scale  # 1. simply the scale order number
        patches_scale[:, 1] = scale  # 1.
        # patches_scale[:, 0] = patch_dim[0] / (h - patch_dim[0])  # 2. normalized value in range 0-1
        # patches_scale[:, 1] = patch_dim[1] / (w - patch_dim[1])  # 2.
        if debug:
            patches_scale[:, 0] = patch_dim[0] * 2 ** scale  # 3. debug; size of the current patch (ex: 16, 32, 64...)
            patches_scale[:, 1] = patch_dim[1] * 2 ** scale  # 3.

        # extract patches
        patch_indices_s = patch_indices[num_samples_total: num_samples_total + num_patches_s]

        # extract patch mini tensors
        def tensor2patches(k, tensor):
            for patch_num, (i, j) in enumerate(samples):
                patch_ind = patch_indices_s[patch_num]  # get randomized patch order
                patches[k, patch_ind] = tensor[:, i: i + patch_dim[0], j: j + patch_dim[1]]

        for k, tensor in enumerate(tensors):
            tensor2patches(k, tensor)

        for patch_num in range(len(samples)):
            patch_ind = patch_indices_s[patch_num]  # get randomized patch order
            positions[patch_ind] = patches_pos[patch_num]
            scales[patch_ind] = patches_scale[patch_num]

        # if debug:
        #     t_img = torch.permute(tensors[0], (1, 2, 0)).numpy()
        #     plt_show("scale-{}".format(scale), t_img)

        # downsample for the next scale
        tensors = [mean_pooler(tensor) for tensor in tensors]

        diff = torch.permute(mean_pooler(torch.permute(torch.as_tensor(diff), (2, 0, 1))), (1, 2, 0)).numpy()
        num_samples_total += num_patches_s

        # sanity check
        if patch_count <= num_samples_total:
            break

    # output tuple format: (p1, ..., pN, positions, scales)
    data_tuple = tuple()
    for k, tensor in enumerate(tensors):
        data_tuple += (patches[k],)
    data_tuple += (positions, scales)

    return data_tuple
