import imageio
import numpy as np
import torch
import torchvision.transforms.functional as functional
from torchvision.models import VGG16_Weights

from PIL import Image

import os


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # enable use of OpenEXR; must be set before 'import cv2'
import cv2


def get_imagenet_transfor_params():
    imagenet_params = VGG16_Weights.IMAGENET1K_V1
    imagenet_transforms = imagenet_params.transforms()
    return imagenet_transforms.mean, imagenet_transforms.std


def normalize_values(v, normalize, normalize_mean_std, vmin=None, vmax=None, vmean=None, vstd=None, inplace=True):
    if not inplace:
        v = v.copy()
    if normalize:
        v -= v.min() if vmin is None else vmin
        v /= v.max() if vmax is None else (vmax - vmin)
    if normalize_mean_std:
        v -= v.mean() if vmean is None else vmean
        v /= v.std() if vstd is None else vstd
    return v


def reverse_values(v, reverse, vmin=None, vmax=None):
    if reverse:
        v = (v.min() if vmin is None else vmin) + (v.max() if vmax is None else vmax) - v
    return v


def imread(path, is_hdr=False):
    if is_hdr:
        # pass cv2.IMREAD_ANYDEPTH to imread() to prevent rescaling to 0-255; read raw EXR data instead
        img = cv2.imread(path, flags=cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
        return img
    else:
        return Image.open(path).convert('RGB')  # RGB


def transform_img(
        img: (Image, np.ndarray),
        crop_params=None,
        h_flip=False,
        v_flip=False,
        norm_mean=None,
        norm_std=None,
        grayscale=False,
):
    """
    transforms a PIL image to tensor while applying several data augmentation operations
    :param img: PIL image
    :param crop_params:
        i, j, h, w: top left corner coord [i, j] and [height, width] to extract
    :param h_flip:
    :param v_flip:
        bool, toggles for horizontal/vertical flip
    :param norm_mean:
    :param norm_std:
        None if normalization disabled; else should have 3 channels each
    :param grayscale:
        Convert to grayscale and stack in 3 channels
    :return:
    """

    # Transform to tensor
    tensor = functional.to_tensor(img).to(dtype=torch.float32)  # 3xHxW

    # crop to output dimension
    if crop_params is not None:
        i, j, h, w = crop_params
        tensor = functional.crop(tensor, i, j, h, w)

    # random flips
    if h_flip:
        tensor = functional.hflip(tensor)
    if v_flip:
        tensor = functional.vflip(tensor)

    if grayscale:
        tensor = functional.rgb_to_grayscale(tensor, num_output_channels=3)

    # normalize
    if norm_mean is not None and norm_std is not None:
        tensor = functional.normalize(tensor, norm_mean, norm_std)

    return tensor
