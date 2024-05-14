import cv2
import numpy as np

from utils.logging import log_warn


# simple TMO (Schlick 1994)
def desaturated_color_to_lum_ratios(rgb, L, L_p, s=0.6, eps=1e-6):
    h, w = L.shape
    L = np.repeat(L.flatten(), 3).reshape(h, w, 3)
    L_p = np.repeat(L_p.flatten(), 3).reshape(h, w, 3)
    rgb_out = np.zeros_like(rgb)
    rgb_out[L > eps] = np.power(rgb[L > eps] / L[L > eps], s) * L_p[L > eps]
    return rgb_out


def normalize_array(a):
    b = a - np.min(a)
    if abs(b.max()) > 1e-6:
        b /= b.max()
    return b


def resize_keep_aspect_ratio(img, resolution=(1080, 1920), zoom=False):
    h1, w1 = resolution
    h, w = img.shape[:2]
    has_color = len(img.shape) > 2

    resize_factor_h = h1 / h
    resize_factor_w = w1 / w

    by_h = (resize_factor_w > resize_factor_h) if not zoom else (resize_factor_w < resize_factor_h)
    resize_w = int(w * resize_factor_h) if by_h else int(w1)
    resize_h = int(h1) if by_h else int(h * resize_factor_w)

    img_r_aspect = resize(img, resize_w, resize_h)
    img_r_aspect = normalize_array(img_r_aspect)

    h_r, w_r = img_r_aspect.shape[:2]

    h_offset = max(0, int(((h1 - h_r) if not zoom else (h_r - h1)) / 2))
    w_offset = max(0, int(((w1 - w_r) if not zoom else (w_r - w1)) / 2))

    if zoom:
        return img_r_aspect[h_offset: h1 + h_offset, w_offset: w1 + w_offset]
    else:
        img_r = np.zeros((h1, w1, 3) if has_color else (h1, w1))
        img_r[h_offset: h_offset + h_r, w_offset: w_offset + w_r] = img_r_aspect
        return img_r


def crop_img(img, top_left_coord=None, bottom_right_coord=None, crop_ratio=None):
    h, w = img.shape[:2]

    if top_left_coord is None:
        top_left_coord = (0, 0)

    if bottom_right_coord is None:
        bottom_right_coord = (h, w)

    if crop_ratio is not None:
        crop_ratio = max(0, min(1, crop_ratio))

        if crop_ratio == 0:
            top_left_coord = (0, 0)
            bottom_right_coord = (h, w)
        else:
            h_e = crop_ratio * h
            w_e = crop_ratio * w

            h_o = int((h - h_e) / 2)
            w_o = int((w - w_e) / 2)

            top_left_coord = np.array((h_o, w_o), np.int)
            bottom_right_coord = np.array((h_o + h_e, w_o + w_e), np.int)

    return img[top_left_coord[0]: bottom_right_coord[0], top_left_coord[1]: bottom_right_coord[1]]


def resize_if_bigger_than(img, max_megapixels=1920*1080):
    x, y = img.shape[:2]
    scale_ratio = 1. * max_megapixels / x / y
    if scale_ratio < 1.0:
        img = resize(img, scale_ratio)
    return img


def resize_resolution(img, resolution):
    return cv2.resize(img, resolution, interpolation=cv2.INTER_CUBIC)


def resize(img, rescale_p1, rescale_p2=None):
    if rescale_p2 is None:
        return cv2.resize(img, (0, 0), fx=rescale_p1, fy=rescale_p1, interpolation=cv2.INTER_CUBIC)
    return cv2.resize(img, (rescale_p1, rescale_p2), interpolation=cv2.INTER_CUBIC)


def file_path(name, path):
    return "{}/{}".format(path, name)


def imwrite(name, path, img, isfloat=True):
    img_u = (np.clip(img, 0, 1) * 255).astype(np.uint8) if isfloat else img
    filepath = file_path(name, path)
    print("imwrite() writing to", filepath)
    cv2.imwrite(filepath, img_u)


# read image from a folder by filename alone (without knowing its file extension)
def imread_unknown_extension(img_name_no_ext, img_path, extensions=None,
                             rescale=1.0, format_float=False,
                             clip_negative=True,
                             input_is_grayscale=False,
                             convert_to_luminance=False,
                             return_filename=False,
                             rescale_if_too_big=True):
    if len(img_name_no_ext) > 3 and img_name_no_ext[-3] == '.':
        log_warn("input file name [{}] is expected to have no extension.".format(img_name_no_ext))

    if extensions is None:
        # default extensions
        extensions = ['png', 'jpg', 'bmp', 'exr']

    read_fail_count = 0
    for extension in extensions:
        filename = '{}.{}'.format(img_name_no_ext, extension)
        try:
            img = imread(filename, img_path,
                         rescale=rescale,
                         format_float=format_float,
                         clip_negative=clip_negative,
                         input_is_grayscale=input_is_grayscale,
                         convert_to_luminance=convert_to_luminance,
                         rescale_if_too_big=rescale_if_too_big)
            if return_filename:
                return img, filename
            else:
                return img
        except FileNotFoundError:
            read_fail_count += 1
            continue
    if read_fail_count >= len(extensions):
        raise FileNotFoundError(
            'Could not read file {}/{} with possible extensions {}'.format(img_path, img_name_no_ext, extensions))


# read image from a folder
def imread(img_name, img_path,
           rescale=1.0,
           format_float=False,
           clip_negative=True,
           input_is_grayscale=False,
           convert_to_luminance=False,
           rescale_if_too_big=True):
    if input_is_grayscale:
        convert_to_luminance = False  # force off if input is already grayscale
    img_name = img_name.lower()
    path = file_path(img_name, img_path)
    flags = cv2.IMREAD_GRAYSCALE if input_is_grayscale else cv2.IMREAD_COLOR
    if '.exr' in img_name or '.hdr' in img_name:
        flags |= cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, flags)
    if img is None:
        raise(FileNotFoundError('Could not read file {}'.format(path)))
    if not rescale == 1.:
        img = resize(img, rescale)
    if rescale_if_too_big:
        img = resize_if_bigger_than(img)
    # img = np.array(img, np.float)  # necessary for opencv > 4.0
    if clip_negative:
        img = np.maximum(img, 0)
    if convert_to_luminance or format_float:
        img = img.astype(float) / 255.
    if convert_to_luminance:
        from utils.image_processing.color_spaces import srgb2lum  # important at runtime to avoid circular imports
        return srgb2lum(img)
    return img


# converts an image to an array of color values of shape 3xN where N is the number of pixels
def img2array(img, c=3, to_float=True):
    w, h = img.shape[:2]
    ar = img.flatten().reshape((w * h, c)).transpose()
    return ar.astype(np.float32) if to_float else ar


# returns a (win_radius*2+1) by (win_radius*2+1) square window around point i, j
# zero pads areas that fall outside the array
def get_window(values, i, j, win_radius):
    h, w = values.shape[:2]

    size = win_radius * 2 + 1
    if len(values.shape) < 3:
        win = np.zeros((size, size))
    else:
        win = np.zeros((size, size, values.shape[2]))

    i_min = max(i - win_radius, 0)
    j_min = max(j - win_radius, 0)
    i_max = min(i + win_radius, h)
    j_max = min(j + win_radius, w)

    i_min_w = win_radius - (i - i_min)
    j_min_w = win_radius - (j - j_min)
    i_max_w = win_radius + (i_max - i)
    j_max_w = win_radius + (j_max - j)

    win[i_min_w: i_max_w, j_min_w: j_max_w] = values[i_min: i_max, j_min: j_max]

    return win


def loop_display(img_dict, ind_max, delay=1000):
    ind = 0
    while True:
        for img_tag in img_dict:
            img = img_dict[img_tag][ind]
            cv2.imshow(img_tag, img)
        ind = (ind + 1) % ind_max
        cv2.waitKey(delay)


def ensure3d(ar):
    ar = np.array(ar)
    if len(ar.shape) == 2:
        ar = np.atleast_3d(ar)
        ar = np.repeat(ar, 3, axis=2)
    return ar
