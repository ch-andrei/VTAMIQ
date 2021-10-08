import torch
import torchvision.transforms.functional as functional

from data.patch_datasets import PatchDataset


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


def get_iqa_patches(imgs: tuple,
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

    samples = patch_sampler.get_sample_params(
        imgs[0].height, imgs[0].width,
        patch_dim[0], patch_dim[1],
        imgs=imgs,  # for difference-based sampling
        num_samples=patch_count,
    )

    def tensor2patches(tensor, samples):
        patches = torch.zeros((patch_count, 3) + patch_dim)
        for patch_num, (i, j, ho, wo) in enumerate(samples):
            patches[patch_num] = tensor[:, i: i + ho, j: j + wo]
        return patches

    # constant for normalizing crop parameters (for uv)
    hw = PatchDataset.get_height_width_factor(imgs[0], patch_dim)

    patches_pos = torch.zeros((patch_count, 2))
    for patch_num, sample in enumerate(samples):
        patches_pos[patch_num] = PatchDataset.crop_params_pos(sample[0], sample[1], hw)

    patches_tuple = tuple()
    for tensor in tensors:
        patches_tuple += (tensor2patches(tensor, samples), )
    patches_tuple += (patches_pos, )

    return patches_tuple
