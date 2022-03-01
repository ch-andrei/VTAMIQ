import torch
import torch.nn as nn
import torchvision.transforms.functional as functional

import numpy as np


def transform(img,
              crop_params=None,
              r_flip=None,
              norm_params=None,
              grayscale=False,
              ):
    """
    transforms a PIL image to tensor while applying several data augmentation operations
    :param img: PIL image
    :param crop_params:
        i, j, h, w: top left corner coord [i, j] and [height, width] to extract
    :param r_flip:
        bool, toggle for horizontal flip
    :param norm_params:
        None or tuple [norm_std, norm_mean] norm_std and norm_mean should have 3 channels each
    :param grayscale:
        Convert to grayscale and stack in 3 channels
    :return:
    """

    # crop to output dimension
    if crop_params is not None:
        i, j, h, w = crop_params
        img = functional.crop(img, i, j, h, w)

    # random flip
    if r_flip:
        img = functional.hflip(img)

    if grayscale:
        img = functional.rgb_to_grayscale(img, num_output_channels=3)

    # Transform to tensor
    tensor = functional.to_tensor(img)  # 3xHxW

    # normalize
    if norm_params is not None:
        norm_mean, norm_std = norm_params
        tensor = functional.normalize(tensor, norm_mean, norm_std)

    return tensor


def get_transform_params(img1, img2,
                         patch_sampler,
                         patch_dim=None,
                         r_flip=None
                         ):
    """
    :param img1:
    :param img2:
        PIL images
    :param patch_sampler:
    :param patch_dim:
        None (no resizing) or tuple holding image dimensions h x w for resizing the image
    :return:
    """
    if patch_dim is None:
        crop_params = None
    else:
        crop_params = patch_sampler.get_sample_params(img1, img2, patch_dim[0], patch_dim[1])

    if r_flip is None:
        r_flip = torch.rand(1) <= 0.5

    return crop_params, r_flip


def transform_img(img,
                  patch_sampler,
                  patch_dim=None,
                  norm_params=None
                  ):
    crop_params, r_flip = get_transform_params(img, img, patch_sampler, patch_dim)
    tensor = transform(img, crop_params, r_flip, norm_params)
    return tensor


def transform_img_pair(img1, img2,
                       patch_sampler,
                       patch_dim=None,
                       norm_params=None
                       ):
    crop_params, r_flip = get_transform_params(img1, img2, patch_sampler, patch_dim)
    tensor1 = transform(img1, crop_params, r_flip, norm_params)
    tensor2 = transform(img2, crop_params, r_flip, norm_params)
    return tensor1, tensor2


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

    def pil2np(img, prenormalize=True):
        im = np.array(img).astype(float)
        # normalizing the difference seems to capture visual disparity more efficiently
        if prenormalize:
            im -= im.min()
            im /= im.max()
        return im

    imgs_np = [pil2np(img) for img in imgs]

    # compute difference
    ref_img = imgs_np[0]
    diff = np.zeros_like(ref_img)
    for dist_img in imgs_np[1:]:
        diff += np.abs(ref_img - dist_img)
    diff = diff / (len(imgs_np) - 1)  # average the difference

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
        patches_tuple += (tensor2patches(tensor, samples), )

    # add UV coords to output tuple
    patches_tuple += (patches_pos, )

    # output tuple format: (img1 patches, img2 patches, positions)
    return patches_tuple


def get_iqa_patches(imgs: tuple,
                    tensors: tuple,
                    patch_count,
                    patch_dim,
                    patch_sampler,
                    patch_num_scales,
                    scale_num_samples_ratio=1.5
                    ):
    """
    returns a tuple with patch_data and patch positions. Supports FR and NR IQA.
    :param imgs: tuple of images, FR-IQA uses 2 images (ref. and dist. images) else only 1 for NR-IQA
    :param patch_count: how many patch_data per images
    :param patch_dim: patch dimensions
    :param patch_sampler: patch sampler to use
    :param patch_num_scales: how many different scales to extract
    :param scale_num_samples_ratio: used to compute number of patches per scale, higher values lead to less patches for large scales
    :return:
    """
    assert len(imgs) == len(tensors), "get_iqa_patches(): Image and Tensor counts should match."
    assert patch_num_scales <= patch_count, "get_iqa_patches(): number of patches must be at least number of scales."

    # timers = [Timer() for _ in range(6)]
    # timers[0].start()

    height, width = imgs[0].height, imgs[0].width

    def compute_diff():
        def pil2np(img, prenormalize=True):
            im = np.array(img).astype(float)
            if prenormalize:
                im -= im.min()
                im /= im.max()
            return im

        imgs_np = [pil2np(img) for img in imgs]

        # compute difference
        ref_img = imgs_np[0]
        diff = np.zeros_like(ref_img)
        for dist_img in imgs_np[1:]:
            diff += np.abs(ref_img - dist_img)
        return diff / (len(imgs_np) - 1)  # average the difference

    diff = compute_diff()

    def compute_patch_scales(patch_num_scales):
        patch_dim_m = max(patch_dim[0], patch_dim[1])
        if 1 < patch_num_scales:
            # determine how many scales are possible
            dim_max = min(height, width)
            patch_num_scales_max = 1
            while True:
                dim_max = (dim_max - patch_dim_m) / 2
                if dim_max <= 1:  # stop at one less than max number
                    break
                patch_num_scales_max += 1
            patch_num_scales = min(patch_num_scales_max, patch_num_scales)
        else:
            patch_num_scales = 1
        return patch_num_scales

    patch_num_scales = compute_patch_scales(patch_num_scales)

    # print('want patch_count', patch_count, 'patch_num_scales', patch_num_scales)
    #

    def compute_num_samples():
        num_samples = 2 ** (scale_num_samples_ratio * np.arange(patch_num_scales))
        num_samples = np.ceil(num_samples * patch_count / np.sum(num_samples)).astype(int)
        cum_samples = np.cumsum(num_samples)
        for i in range(patch_num_scales):
            if patch_count <= cum_samples[i]:
                num_samples[i] -= cum_samples[i] - patch_count
                num_samples[i+1:] = 0
                break
        return num_samples

    num_samples = compute_num_samples()

    # print('num_samples', num_samples)
    # exit()

    mean_pooler = nn.AvgPool2d(kernel_size=2)

    # precompute patch order
    patch_indices = np.arange(patch_count)
    np.random.shuffle(patch_indices)

    patches = torch.zeros((len(imgs), patch_count, 3) + (patch_dim[0], patch_dim[1]))
    positions = torch.zeros((patch_count, 2))
    scales = torch.zeros((patch_count, 2))

    # timers[0].stop()
    # timers[1].start()

    num_samples_total = 0
    for scale in range(patch_num_scales):
        # timers[2].start()

        num_samples_s = num_samples[-scale-1]

        h, w = diff.shape[:2]

        # print('h, w, num_samples_s', h, w, num_samples_s)

        # timers[3].start()

        samples = patch_sampler.get_sample_params(
            h, w,
            patch_dim[0], patch_dim[1],
            diff=diff,
            num_samples=num_samples_s,
        )  # N x 2

        # timers[3].stop()

        # patches_pos = torch.as_tensor(np.array(samples)) * 2 ** scale  # 1. debug
        patches_pos = np.array(samples) + np.array([patch_dim[0]//2, patch_dim[1]//2], float).reshape(1, 2)  # 3. centers
        patches_pos = patches_pos / np.array([h - patch_dim[0]//2, w - patch_dim[1]//2], float).reshape(1, 2)  # 3. rescale to [0, 1]
        patches_pos = torch.clamp(torch.as_tensor(patches_pos), 0., 1. - 1e-6)  # 3.

        patches_scale = torch.zeros_like(patches_pos)
        patches_scale[:, 0] = scale  # 1. simply the scale number
        patches_scale[:, 1] = scale  # 1.
        # patches_scale[:, 0] = patch_dim[0] / (h - patch_dim[0])  # 2. relative 0-1
        # patches_scale[:, 1] = patch_dim[1] / (w - patch_dim[1])  # 2. relative 0-1
        # patches_scale[:, 0] = patch_dim[0] * 2 ** scale  # 3. debug
        # patches_scale[:, 1] = patch_dim[1] * 2 ** scale  # 3. debug

        # timers[4].start()
        # extract patches
        patch_indices_s = patch_indices[num_samples_total: num_samples_total + num_samples_s]

        def tensor2patches(k):
            for patch_num, (i, j) in enumerate(samples):
                patch_num = patch_indices_s[patch_num]  # get randomized patch order
                patches[k, patch_num] = tensor[:, i: i + patch_dim[0], j: j + patch_dim[1]]

        # convert patch_data to tensors and add to output tuple
        for k, tensor in enumerate(tensors):
            tensor2patches(k)
        # timers[4].stop()

        positions[num_samples_total: num_samples_total + num_samples_s] = patches_pos
        scales[num_samples_total: num_samples_total + num_samples_s] = patches_scale

        # timers[5].start()

        # downsample for the next scale
        tensors = [mean_pooler(tensor) for tensor in tensors]
        diff = torch.permute(mean_pooler(torch.permute(torch.as_tensor(diff), (2, 0, 1))), (1, 2, 0)).numpy()
        num_samples_total += num_samples_s

        # sanity check
        if patch_count <= num_samples_total:
            break

    # timers[5].stop()
    # timers[2].stop()

    # timers[1].stop()

    # print(
    #     'preloop:', timers[0].delta_avg, '\n'
    #     'inloop', timers[2].delta_avg, '\n'
    #     "totalloop", timers[1].delta_avg, '\n'
    #     "sampling", timers[3].total(), timers[3].delta_avg, timers[3].deltas, '\n'
    #     "tensorextract", timers[4].total(), timers[4].delta_avg, timers[4].deltas, '\n'
    #     "downsample", timers[5].total(), timers[5].delta_avg, timers[5].deltas, '\n'
    #     "\n"
    # )

    # output tuple format: (p1, ..., pN, positions, scales)
    data_tuple = tuple()
    for k, tensor in enumerate(tensors):
        data_tuple += (patches[k], )
    data_tuple += (positions, scales)

    return data_tuple