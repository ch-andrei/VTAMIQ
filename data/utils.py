import torch
import torchvision.transforms.functional as functional


def transform(img,
              crop_params=None,
              h_flip=False,
              v_flip=False,
              norm_params=None,
              grayscale=False,
              ):
    """
    transforms a PIL image to tensor while applying several data augmentation operations
    :param img: PIL image
    :param crop_params:
        i, j, h, w: top left corner coord [i, j] and [height, width] to extract
    :param h_flip, v_flip:
        bool, toggle for horizontal/vertical flip
    :param norm_params:
        None or tuple [norm_std, norm_mean] norm_std and norm_mean should have 3 channels each
    :param grayscale:
        Convert to grayscale and stack in 3 channels
    :param display_model:
        display model simulation and PU-encoding
    :return:
    """

    # crop to output dimension
    if crop_params is not None:
        i, j, h, w = crop_params
        img = functional.crop(img, i, j, h, w)

    # random flips
    if h_flip:
        img = functional.hflip(img)
    if v_flip:
        img = functional.vflip(img)

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
                         h_flip=None,
                         v_flip=None,
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

    rsamples = torch.rand(2)

    if h_flip is None:
        h_flip = rsamples[0] <= 0.5

    if v_flip is None:
        v_flip = rsamples[1] <= 0.5

    return crop_params, h_flip, v_flip


def transform_img(img,
                  patch_sampler,
                  patch_dim=None,
                  norm_params=None
                  ):
    crop_params, h_flip, v_flip = get_transform_params(img, img, patch_sampler, patch_dim)
    tensor = transform(img, crop_params, h_flip, norm_params)
    return tensor


def transform_img_pair(img1, img2,
                       patch_sampler,
                       patch_dim=None,
                       norm_params=None
                       ):
    crop_params, h_flip, v_flip = get_transform_params(img1, img2, patch_sampler, patch_dim)
    tensor1 = transform(img1, crop_params, h_flip, v_flip, norm_params)
    tensor2 = transform(img2, crop_params, h_flip, v_flip, norm_params)
    return tensor1, tensor2
