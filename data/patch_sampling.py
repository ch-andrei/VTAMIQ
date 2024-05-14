import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp
from skimage.util.shape import view_as_windows

import torch
import torch.nn as nn

from PIL import Image

from matplotlib import pyplot as plt

from utils.logging import log_warn
from utils.misc.temporary_numpy_seed import TemporaryNumpySeed

# constants must be different values
GRID_TYPE_HALTON = 0
GRID_TYPE_PERTURBED = 1
GRID_TYPE_PERTURBED_SIMPLE = 2

DIFF_TYPE_MAGNITUDE = 0
DIFF_TYPE_DARK = 1

DEFAULT_NUM_SAMPLES_RATIO = 1.7  # see get_iqa_patches() for more info
GRID_TYPE_PERTURBED_AMOUNT = 0.2  # 0.5 = half cell distance (possible overlap with neighbor); 1.0 = full cell distance


def plt_show(title, img, draw=True, dpi=300):
    if draw:
        plt.figure(dpi=dpi, )
        plt.title(title)
        plt.imshow(img)
        plt.colorbar()
        plt.show()


def pil2np(img, prenormalize=True):
    im = np.array(img).astype(float)
    if prenormalize:
        # inplace
        im -= im.min()
        im /= im.max()
    return im


class PatchSampler(object):
    __centerbias_image_path__ = "modules/Attention/deepgaze2/centerbias.npy"

    def __init__(
            self,
            centerbias_weight=0.0,
            diff_weight=0.0,
            uniform_weight=1.0,
            grid_type=GRID_TYPE_PERTURBED_SIMPLE,
            diff_type=DIFF_TYPE_MAGNITUDE,
            perturbed_amount=GRID_TYPE_PERTURBED_AMOUNT,
        ):
        """
        class to generate i,j coordinates for sampling patches from a 2D image
        :param centerbias_weight:
        :param diff_weight:
        :param uniform_weight:
        """

        if grid_type == GRID_TYPE_PERTURBED_SIMPLE:
            if 0 < centerbias_weight or 0 < diff_weight:
                log_warn("PatchSampler with GRID_TYPE_PERTURBED_SIMPLE and centerbias_weight and diff_weight are non-zero. Will use uniform sampling.")
            centerbias_weight = 0
            diff_weight = 0

        self.centerbias_weight = max(0., centerbias_weight)
        self.diff_weight = max(0., diff_weight)
        self.uniform_weight = max(0., uniform_weight)

        total_weight = self.uniform_weight + self.diff_weight + self.centerbias_weight
        if total_weight < 1e-6:
            raise ValueError("Total weight must be non-zero.")

        if self.centerbias_weight > 0:
            self.centerbias_template = np.load(PatchSampler.__centerbias_image_path__)

        self.grid_type = grid_type
        self.diff_type = diff_type
        self.perturbed_amount = perturbed_amount

    def __call__(self, h, w, ho, wo, diff=None, num_samples=1, debug=False):
        return self.get_sample_params(h, w, ho, wo, diff=diff, num_samples=num_samples, debug=debug)

    def get_sample_params(self, h, w, ho, wo, diff=None, num_samples=1, debug=False):
        if self.diff_weight == 0 and self.centerbias_weight == 0:
            if debug:
                print('Using uniform stratified sampling')
            # stratified grid with equal probability for each cell
            return stratified_grid_sampling(
                h, w, ho, wo, sample_prob=np.ones((h, w)),
                num_samples=num_samples,
                grid_function_type=self.grid_type,
                perturbed_amount=self.perturbed_amount,
                debug=debug
            )
        else:
            # compute centerbiased component
            centerbias = 0
            if self.centerbias_weight > 0:
                centerbias = self.centerbias_prob(h, w)
                centerbias = centerbias / np.max(centerbias)

            diffbased = 0
            # compute diffbased component
            if self.diff_weight > 0:
                if diff is None:
                    raise ValueError("PatchSampler: 'diff' input must be specified for difference-based sampling.")

                diffbased = diff.copy()

                if np.std(diffbased) > 1e-6:  # avoid failure case when there is little (no) difference
                    diffbased = diffbased / np.std(diffbased)  # normalize by std deviation
                else:
                    diffbased = 0

            sample_prob = self.centerbias_weight * centerbias + \
                          self.diff_weight * diffbased + \
                          self.uniform_weight

            # normalize probability; ensure that the total sum is equal 1
            sample_prob = sample_prob / np.sum(sample_prob)

            return stratified_grid_sampling(
                h, w, ho, wo, sample_prob=sample_prob,
                num_samples=num_samples,
                grid_function_type=self.grid_type,
                perturbed_amount=self.perturbed_amount,
                debug=debug
            )

    def compute_diff(self, pil_imgs, diff_pow=1.0):
        if self.diff_weight == 0:
            return None

        imgs_np = [pil2np(img) for img in pil_imgs]

        if self.diff_type == DIFF_TYPE_MAGNITUDE:
            # compute difference
            ref_img = imgs_np[0]

            diff = np.zeros_like(ref_img)

            for dist_img in imgs_np[1:]:
                diff += np.abs(ref_img - dist_img)

            diff = diff / (len(imgs_np) - 1)  # average over all images

        elif self.diff_type == DIFF_TYPE_DARK:
            diff = imgs_np[0]

            # inspired by normal distribution with max always at 1
            # low values have the highest weight
            sigma = 0.1
            diff = np.exp(-0.5 * (diff / sigma) ** 2.0)

        else:
            raise ValueError()

        # if RGB input
        if len(diff.shape) == 3:
            diff = np.sum(diff * diff, axis=2)  # L2 squared distance over color channels
            diff_pow /= 2.0  # p / 2 because also need to apply sqrt to the squared L2 distance

        diff = np.power(diff, diff_pow)

        return diff

    def centerbias_prob(self, h, w):
        # rescale to match image size
        ho, wo = 1024, 1024  # original centerbias image resolution is 1024x1024
        centerbias = zoom(self.centerbias_template, (h / ho, w / wo), order=0, mode='nearest')
        # renormalize log density after performing zoom
        centerbias -= logsumexp(centerbias)
        # softmax for probabilities
        centerbias = np.exp(centerbias)
        centerbias_p = centerbias / np.sum(centerbias)
        return centerbias_p


def grid_sequence(h, w=None):
    if w is None:
        w = h
    hh = np.arange(h, dtype=float)  # + 0.5  # 0.5 offset for tile centers not top left corner
    ww = np.arange(w, dtype=float)  # + 0.5  # 0.5 offset for tile centers not top left corner
    gh, gw = np.meshgrid(hh, ww)
    grid = np.stack([gh, gw])
    return grid


def halton_sequence_1d(n, b):
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


def halton_sequence_2d(n, indexing="xy"):
    haltonx = halton_sequence_1d(n, 2)
    haltony = halton_sequence_1d(n, 3)
    halton = np.concatenate([haltonx, haltony], axis=0).reshape(2, -1)
    if indexing == "xy":
        return halton
    elif indexing == "ij":
        return np.transpose(halton)
    else:
        raise ValueError("Indexing must be either 'xy' or 'ij'.")


def stratified_grid_sampling(
        h, w, ho, wo, sample_prob,
        num_samples=1,
        randomize_cell_order=True,
        grid_function_type=GRID_TYPE_PERTURBED,
        perturbed_amount=GRID_TYPE_PERTURBED_AMOUNT,
        debug=False
    ):
    __cellsize_ratio = 4.
    __patchsize_ratio = 0.75
    __patch2image_ratio = 3.

    if grid_function_type == GRID_TYPE_PERTURBED_SIMPLE:
        probs = np.ones((1, 1), float)

    else:
        cell_size_d = np.sqrt(h * w / num_samples * __cellsize_ratio)  # heuristic for optimal cell size
        cell_size_min = __patchsize_ratio * min(ho, wo)  # mininum
        cell_size_max = max(h, w) / max(ho, wo) * __patch2image_ratio  # maximum

        clip_int = lambda x, a, b: int(max(a, min(b, x)))
        cell_size = clip_int(cell_size_d, cell_size_min, cell_size_max)

        if debug:
            print('stratified_grid_sampling, h, w, ho, wo, num_samples', h, w, ho, wo, num_samples)
            print('cellsizes (final, desired, min, max): ', cell_size, cell_size_d, cell_size_min, cell_size_max)

        # step size in the original array
        sh = int(np.ceil((h - ho) / cell_size))
        sw = int(np.ceil((w - wo) / cell_size))

        # edge size modifiers due to partial cells
        icell_dec = ((w - wo) / cell_size) % 1.
        jcell_dec = ((h - ho) / cell_size) % 1.
        icell_dec = 1. if icell_dec < 1e-3 else icell_dec
        jcell_dec = 1. if jcell_dec < 1e-3 else jcell_dec

        # zero-padded (right/bottom) original probability array to match full cell size and count
        probs = np.zeros((cell_size * sh + ho, cell_size * sw + wo))
        probs[:h, :w] = sample_prob.reshape(h, w)

        # if debug:
        #     plt_show(
        #         "{}x{}p; probability".format(ho, wo, h, w),
        #         probs.astype(float)
        #     )

        probs = view_as_windows(probs, (cell_size + ho - 1, cell_size + wo - 1), (cell_size, cell_size))
        probs = np.sum(probs, axis=(2, 3))  # sum over the windows
        probs /= np.sum(probs)  # normalize

    num_patches_cells = np.ceil((probs * num_samples)).astype(int)

    # make sure total num patches doesnt exceed the queried number
    # "random dissolve" subtract
    num_patches_shape = num_patches_cells.shape  # original shape
    num_patches_cells = num_patches_cells.flatten()
    num_cells = len(num_patches_cells)
    while num_samples != num_patches_cells.sum():
        num_total = num_patches_cells.sum()
        num_adjust = num_total - num_samples  # mismatch between queried and current number of patches
        # p_adjust = probability to adjust; remove from low number of samples first (keep high points)
        p_adjust = num_patches_cells / num_total
        p_adjust = (p_adjust.max() + 1e-3) - p_adjust  # ~inverse probabilities
        p_adjust /= p_adjust.sum()
        # generate random indices for adjustment
        indices_adjust = np.random.choice(num_cells, abs(num_adjust), replace=True, p=p_adjust)
        value_adjust = 1 if num_adjust < 0 else -1  # add if too few and subtract if too many
        # Note: if attempting to adjust the same index multiple times, only one adjustment is applied due to indexing
        num_patches_cells[indices_adjust] = np.maximum(num_patches_cells[indices_adjust] + value_adjust, 0)
    num_patches_cells = num_patches_cells.reshape(*num_patches_shape)

    if debug:
        print("num_cells", num_cells)
        print('num_patches_cells, min, max, sum', num_patches_cells.min(), num_patches_cells.max(), num_patches_cells.sum())
        plt_show(
            "PROBS: {}x{}i; {}x{}c ({}); probability/cell".format(h, w, *num_patches_cells.shape[:2], cell_size),
            probs.astype(float)
        )
        plt_show(
            "PATCHES: {}x{}i; {}x{}c ({}); {}s; Np/cell".format(h, w, *num_patches_cells.shape[:2], cell_size, num_samples),
            num_patches_cells.astype(float)
        )

    num_patches_width = lambda num_patches, aspect=1.0: np.maximum(np.ceil(np.sqrt(num_patches / aspect)), 1.)

    # precompute random sample positions
    if grid_function_type == GRID_TYPE_HALTON:
        sample_pos = halton_sequence_2d(num_samples, indexing="xy")

    elif grid_function_type == GRID_TYPE_PERTURBED or grid_function_type == GRID_TYPE_PERTURBED_SIMPLE:

        if grid_function_type == GRID_TYPE_PERTURBED:
            widths = num_patches_width(num_patches_cells)
            max_width = int(widths.max())
            sample_pos = grid_sequence(max_width)

        else:  # grid_function_type == GRID_TYPE_PERTURBED_SIMPLE
            aspect_ratio = h / w
            widths = num_patches_width(num_patches_cells, aspect=aspect_ratio)
            heights = np.ceil(widths * aspect_ratio)
            sample_pos = grid_sequence(heights[0], widths[0])

        sample_rand = (2. * np.random.rand(2, num_samples) - 1.0) * 2. * perturbed_amount

    else:
        raise ValueError("Unsupported grid function type.")

    if randomize_cell_order:
        cells_order = np.random.permutation(num_cells)  # randomize cell order to avoid repeating halton seq pattern
    else:
        cells_order = np.arange(num_cells)

    patches_tot = 0
    samples = np.zeros((2, num_samples), float)
    for index in range(num_cells):
        # current cell index
        index = cells_order[index]
        j = index // num_patches_shape[1]
        i = index % num_patches_shape[1]

        num_patches_cell = num_patches_cells[j, i]  # current cell num patches
        if num_patches_cell < 1:
            continue  # zero patch case

        # get random sample positions for the current cell
        if grid_function_type == GRID_TYPE_HALTON:
            sample_pos_c = sample_pos[:, patches_tot: patches_tot + num_patches_cell]

        elif grid_function_type == GRID_TYPE_PERTURBED or grid_function_type == GRID_TYPE_PERTURBED_SIMPLE:
            samples_rand_c = sample_rand[:, patches_tot: patches_tot + num_patches_cell]

            if grid_function_type == GRID_TYPE_PERTURBED:
                width = int(widths[j, i])  # current width
                indices_rand = np.random.choice(width * width, size=num_patches_cell, replace=False)
                # select grid of appropriate size
                sample_pos_c = sample_pos[:, : width, : width].reshape(2, -1)[:, indices_rand]
                sample_pos_c = (sample_pos_c + samples_rand_c) / width
                sample_pos_c = np.clip(sample_pos_c + 1. / width / 2, 0., 1.)  # offset and clip

            else:
                height = int(heights[j, i])  # current width
                width = int(widths[j, i])  # current width
                height_width = np.array([height, width]).reshape(2, 1)
                indices_rand = np.random.choice(height * width, size=num_patches_cell, replace=False)
                sample_pos_c = sample_pos[:, : width, : height].reshape(2, -1)[:, indices_rand]
                sample_pos_c = (sample_pos_c + samples_rand_c) / height_width
                # sample_pos_c[1] = (sample_pos_c[1] + samples_rand_c[1]) / width
                sample_pos_c = np.clip(sample_pos_c + 1. / height_width / 2, 0., 1.)  # offset and clip
                # sample_pos_c[1] = np.clip(sample_pos_c[1] + 1. / width / 2, 0., 1.)  # offset and clip

            # print('width sample_pos_c', width, sample_pos_c.min(), sample_pos_c.max())
        else:
            raise ValueError("Unsupported grid function type.")

        if grid_function_type == GRID_TYPE_PERTURBED_SIMPLE:
            samples[0, patches_tot: patches_tot + num_patches_cell] = (j + sample_pos_c[0]) * (h - ho)
            samples[1, patches_tot: patches_tot + num_patches_cell] = (i + sample_pos_c[1]) * (w - wo)

        else:
            # rescale edge cell indices to compensate for partial edge cells
            if j == (num_patches_shape[0] - 1):
                sample_pos_c[0] *= jcell_dec
            if i == (num_patches_shape[1] - 1):
                sample_pos_c[1] *= icell_dec

            samples[0, patches_tot: patches_tot + num_patches_cell] = (j + sample_pos_c[0]) * cell_size
            samples[1, patches_tot: patches_tot + num_patches_cell] = (i + sample_pos_c[1]) * cell_size

        patches_tot += num_patches_cell

    return samples


def compute_patch_num_scales(patch_num_scales, h, w, ho, wo):
    patch_dim_m = max(ho, wo)

    if 1 < patch_num_scales:
        # determine how many scales are possible
        dim_max = min(h, w)
        patch_num_scales_max = 0
        while 1 < dim_max:
            patch_num_scales_max += 1
            dim_max = (dim_max - patch_dim_m) / 2
        return max(1, min(patch_num_scales_max - 1, patch_num_scales))

    else:
        return 1


def compute_num_pixels(patch_count, patch_num_scales, scale_num_samples_ratio=DEFAULT_NUM_SAMPLES_RATIO):
    num_patches = compute_num_patches_per_scale(patch_count, patch_num_scales, scale_num_samples_ratio)
    return compute_num_pixels_for_patches(num_patches)


def compute_num_pixels_for_patches(num_patches):
    patch_num_scales = len(num_patches)
    num_pixels = 2 ** (2 * np.flip(np.arange(patch_num_scales))) * 16 * 16  # pixels/patch for each scale
    num_pixels = num_pixels * num_patches  # pixels/patch x num_patches
    num_pixels = np.sum(num_pixels)  # total pixels
    return num_pixels


def compute_num_patches_per_scale(patch_count, patch_num_scales, scale_num_samples_ratio):
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


def get_iqa_patches(imgs: (tuple, list),
                    tensors: (tuple, list),
                    patch_count,
                    patch_dim,
                    patch_sampler: PatchSampler,
                    patch_num_scales,
                    scale_num_samples_ratio=DEFAULT_NUM_SAMPLES_RATIO,
                    use_aligned_patches=True,
                    randomize_patch_scale_order=False,
                    random_seed=None,
                    debug=False,
                    ):
    """
    applies random sampling to extract patches from the input images/tensors
    :param imgs: input images; FR-IQA uses 2 images (ref. and dist. images) else only 1 for NR-IQA
    :param tensors: input tensors (same count and shape as imgs)
    :param patch_count: how many patch_data per images
    :param patch_dim: patch dimensions
    :param patch_sampler: patch sampler to use
    :param patch_num_scales: how many different scales to extract
    :param scale_num_samples_ratio: constant r used to compute number of patches per scale:
        num_patches_for_scale = 2 ** (r * range(i)), where i is current scale index (ordered: 0, 1, 2, ...).
        higher values lead to fewer patches for large scales.
        example when ratio=2 corresponds to num_patches will be [1,4,16,64,...]
                for 2 scales -> index 0: 32x32 pixels, index 1: 16x16 pixels
        Note: higher number allocates more patches for larger patch scales
        see function "compute_num_patches_per_scale()"
    :param randomize_patch_scale_order: when multiple scales are used, patches by default are ordered by scale:
        ex: patch sequence with corresponding scales 0,0,...,0,1,1,...,1,2,2,...,2...
        when randomize_patch_scale_order is set to True, the final patch sequence will also be randomly shuffled
    :param use_aligned_patches: toogle to sample the same or a different set of patch positions for each tensor
    :param random_seed: allows to set the random seed of the sampling
    :return: tuple {(patch data, patch positions, patch scales)}
    """
    num_imgs = len(imgs)

    if num_imgs != len(tensors):
        raise ValueError("get_iqa_patches(): Image and Tensor counts should match.")
    if patch_count < patch_num_scales:
        raise ValueError("get_iqa_patches(): number of patches larger than the number of scales.")

    # does not modify seed if random_seed=None
    with TemporaryNumpySeed(random_seed):
        img_ref = imgs[0]
        if isinstance(img_ref, Image.Image):
            height, width = img_ref.height, img_ref.width
        elif isinstance(img_ref, np.ndarray):
            height, width = img_ref.shape[:2]
        else:
            raise TypeError("Unsupported input")

        # print('want patch_count', patch_count, 'patch_num_scales', patch_num_scales)
        # print('num_patches', num_patches)
        # exit()

        # precompute patch order if necessary
        patch_indices = None
        if randomize_patch_scale_order:
            patch_indices = np.random.permutation(patch_count)

        diff = patch_sampler.compute_diff(imgs)
        patch_num_scales = compute_patch_num_scales(patch_num_scales, height, width, patch_dim, patch_dim)
        num_patches = compute_num_patches_per_scale(patch_count, patch_num_scales, scale_num_samples_ratio)

        use_scales = 1 < patch_num_scales

        # if debug:
        #     print('num_patches', num_patches)
        #     print('use_aligned_patches', use_aligned_patches)

        patch_dims = np.array([patch_dim // 2, patch_dim // 2], np.float32).reshape(1, 2)

        def compute_patch_sampling(h, w, num_patches, diff):
            return patch_sampler.get_sample_params(
                h, w, patch_dim, patch_dim, diff=diff, num_samples=num_patches,
                # debug=debug
            )

        # extract patch mini tensors
        def get_tensors_patches(k, tensor, patch_indices, _pos, _scales):
            _samples = samples[0] if use_aligned_patches else samples[k]
            _offset = 0 if use_aligned_patches else [num_patches_s * i for i in range(num_imgs)][k]

            # write patch positions and scales
            pos[k, patch_indices] = _pos[_offset: _offset + num_patches_s]
            if use_scales:
                scales[k, patch_indices] = _scales[_offset: _offset + num_patches_s]

            # write patch tensor pixels
            indices = np.arange(patch_dim)  # 0-P indices
            indices = np.array(np.meshgrid(indices, indices, indexing="ij"))  # P x P pixel meshgrid of indices
            # compute indices of pixels for each sample position; reshape samples and indices so that dims broadcast
            # samples.shape = (2,N,1,1)
            # indices.shape = (2,1,16,16)
            _samples = _samples.reshape((2, -1, 1, 1)) + indices.reshape((2, 1, patch_dim, patch_dim))
            patches[k, patch_indices] = torch.permute(tensor[:, _samples[0], _samples[1]], dims=(1, 0, 2, 3))

        # prepare output tensor patches
        tensors = torch.stack(tensors, dim=0)
        patches = torch.zeros((num_imgs, patch_count, 3) + (patch_dim, patch_dim), dtype=torch.float32)
        pos = torch.zeros((num_imgs, patch_count, 2), dtype=torch.float32)
        scales = torch.zeros((num_imgs, patch_count), dtype=torch.int) if use_scales else None
        mean_pooler = nn.AvgPool2d(kernel_size=2)  # 2x downsampler
        num_samples_total = 0
        for scale in range(patch_num_scales):
            num_patches_s = num_patches[-scale - 1]

            h, w = tensors[0].shape[1:3]

            hw2patch_ratio = np.array([h - patch_dim // 2, w - patch_dim // 2], np.float32).reshape(1, 2)

            num_resamples = 1 if use_aligned_patches else len(imgs)
            samples = [compute_patch_sampling(h, w, num_patches_s, diff) for _ in range(num_resamples)]

            # prepare position values
            _pos = torch.from_numpy(np.concatenate(samples, axis=1))
            _pos = torch.permute(_pos, (1, 0))  # 3. num_tensors x N x 2
            _pos = (_pos + patch_dims) / hw2patch_ratio  # 3. centers, rescaled to [0, 1]
            _pos = torch.clamp(_pos, 0., 1. - 1e-6)  # 3.

            if debug:
                _pos = torch.from_numpy(np.concatenate(samples, axis=1))  # 2 x N
                _pos = torch.permute(_pos, (1, 0))  # N x 2
                _pos = _pos * (2 ** scale)  # 1. debug

            _scales = None
            if use_scales:
                # prepare scale values
                _scales = torch.zeros(_pos.shape[0], dtype=torch.int)
                _scales[:] = scale  # 1. simply the scale order number
                # patches_scale[:, 0] = patch_dim / (h - patch_dim)  # 2. normalized value in range 0-1
                # patches_scale[:, 1] = patch_dim / (w - patch_dim)  # 2.

                if debug:
                    _scales[:] = patch_dim * 2 ** scale  # 3. debug; size of the current patch (ex: 16, 32, 64...)

            # extract patches
            if patch_indices is None:
                patch_indices_s = slice(num_samples_total, num_samples_total + num_patches_s)
            else:
                patch_indices_s = patch_indices[num_samples_total: num_samples_total + num_patches_s]

            for k, tensor in enumerate(tensors):
                get_tensors_patches(k, tensor, patch_indices_s, _pos, _scales)

            # if debug:
            #     t_img = torch.permute(tensors[0], (1, 2, 0)).numpy()
            #     plt_show("scale-{}".format(scale), t_img)W

            # downsample for the next scale
            tensors = mean_pooler(tensors)
            if diff is not None:
                diff = mean_pooler(torch.as_tensor(diff).view(1, 1, *diff.shape)).squeeze().numpy()
            num_samples_total += num_patches_s

            # sanity check
            if patch_count <= num_samples_total:
                break

    # patches = patches.view(num_imgs * patch_count, 3, patch_dim, patch_dim)
    # pos = pos.view(num_imgs * patch_count, 2)
    # scales = scales.view(num_imgs * patch_count)

    return patches, pos, scales
