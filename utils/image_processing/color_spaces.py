import cv2
import numpy as np

from utils.image_processing.image_tools import img2array, imread


# sources:
# https://en.wikipedia.org/wiki/Relative_luminance
# https://www.cl.cam.ac.uk/~rkm38/pdfs/mantiuk2016perceptual_display.pdf
def srgb2rgb(rgb, gamma=2.4):
    # convert to linear rgb
    c = 0.04045
    a = 0.055
    rgb_lin = np.power((rgb + a) / (1.0 + a), gamma)
    rgb_lin[rgb < c] = rgb[rgb < c] / 12.92
    return rgb_lin


def rgb2srgb(rgb_lin, gamma=2.4):
    c = 0.0031308
    a = 0.055
    srgb = (1 + a) * np.power(rgb_lin, 1 / gamma) - a
    srgb[rgb_lin < c] = rgb_lin[rgb_lin < c] * 12.92
    return srgb


def srgb2lum(rgb, gamma=2.4):
    # converts to linear RGB then to luminance
    rgb_lin = srgb2rgb(rgb, gamma)
    return rgb2gray_matlab(rgb_lin)


def rgb_lum_weighted(rgb):
    lum_w = np.zeros_like(rgb)
    lum_w[..., 0] = rgb[..., 0] * 0.2126
    lum_w[..., 1] = rgb[..., 1] * 0.7152
    lum_w[..., 2] = rgb[..., 2] * 0.0722
    return lum_w


def rgb_lum_unweighted(rgb):
    lum_w = np.zeros_like(rgb)
    lum_w[..., 0] = rgb[..., 0] / 0.2126
    lum_w[..., 1] = rgb[..., 1] / 0.7152
    lum_w[..., 2] = rgb[..., 2] / 0.0722
    return lum_w


def rgb2lum(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def rgb2gray_matlab(rgb):
    return rgb[..., 0] * 0.2989 + rgb[..., 1] * 0.5870 + rgb[..., 2] * 0.1140


# converts image from RGB to XYZ color space
# http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
def rgb2xyz(img, shape_as_input=False, format_float=True):
    h, w, c = img.shape

    rgb = img2array(img)

    if format_float:
        rgb /= 255

    rgb = srgb2rgb(rgb)

    m = np.array(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]], np.float32)

    xyz = m.dot(rgb)

    return xyz.transpose().reshape(h, w, 3) if shape_as_input else xyz


def xyz2rgb(img, shape_as_input=False, cast_uint8=True):
    h, w, c = img.shape

    xyz = img2array(img)

    m = np.array([
        [3.2404542 , -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [0.0556434 , -0.2040259,  1.0572252]], np.float32)

    rgb = m.dot(xyz)

    if cast_uint8:
        rgb = (rgb * 255).astype(np.uint8)

    return rgb.transpose().reshape(h, w, 3) if shape_as_input else rgb


# conversion from RGB to Cie L*a*b* color space is done as per
# https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=cvtcolor#cvtcolor
# reference illuminant: d65
def rgb2CieLab(rgb, format_float=False):
    xyz = rgb2xyz(rgb, shape_as_input=False, format_float=format_float)

    X, Y, Z = (xyz[0], xyz[1], xyz[2])
    # d65
    X /= 0.950456
    Z /= 1.088754

    def f_L(t, thresh=0.008856):
        L = np.zeros_like(t)
        inds = t > thresh
        ninds = (np.logical_not(inds))
        if inds.any():
            L[inds] = 116.0 * np.power(t[inds], 1.0 / 3.0) - 16.0
        if ninds.any():
            L[ninds] = 903.3 * t[ninds]
        return L

    def f_ab(t, thresh=0.008856):
        ab = np.zeros_like(t)
        inds = t > thresh
        ninds = np.logical_not(inds)
        if inds.any():
            ab[inds] = np.power(t[inds], 1.0 / 3.0)
        if ninds.any():
            ab[ninds] = 7.787 * t[ninds] + 16.0 / 116.0
        return ab

    lab = np.zeros((3, xyz.shape[1]), np.float32)
    lab[0] = f_L(Y) / 100
    lab[1] = (500.0 * (f_ab(X) - f_ab(Y)) + 127) / 255
    lab[2] = (200.0 * (f_ab(Y) - f_ab(Z)) + 127) / 255

    return lab.astype(np.float32)

# test
if __name__ == "__main__":
    imgpath = 'footage/renders'
    imgname = '25000_image_1_1920_1080.png'

    img = imread(imgname, imgpath)
    h, w, c = img.shape
    xyz = rgb2xyz(img, shape_as_input=True)
    rgb = xyz2rgb(xyz, shape_as_input=True)
    diff = (img == rgb)
    print("lossless for {}/{}={}% of all values".format(diff.sum(), h * w * c, diff.sum() / (h * w * c)))
    cv2.imshow("img", img)
    cv2.imshow("rgb", rgb)
    cv2.waitKey()


