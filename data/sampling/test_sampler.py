import numpy as np
import torch
import cv2

from utils.misc.timer import Timer
from data.sampling.patch_sampling import PatchSampler, centerbias_prob, stratified_grid_sampling

from matplotlib import pyplot as plt
from PIL import Image


def imread(path):
    im = Image.open(path).convert("RGB")
    return np.array(im, float) / 255


def plt_show(title, img, dpi=300):
    plt.figure(dpi=dpi)
    plt.title(title)
    plt.imshow(img)
    plt.colorbar()
    plt.show()


def alignImage(img_ref_sdr, img_test_hdr):
    """
    align img_ref_sdr to img_test_hdr, return aligned img_ref_sdr
    :param img_ref_sdr:
    :param img_test_hdr:
    :return:
    """

    im1 = (img_ref_sdr * 255).astype(np.uint8)
    im2 = (img_test_hdr * 255).astype(np.uint8)

    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.1

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # # Draw top matches
    # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    # cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1_aligned = cv2.warpPerspective(img_ref_sdr, h, (width, height))

    return im1_aligned


def get_renault_images():
    scene_8h15pm = "8h15pm"
    scene_Studio = "Studio"
    scene_8h15am = "8h15am"
    scene_1h15pm = "1h15pm"
    scene_6h15pm = "6h15pm"
    scene = scene_8h15pm

    path = "I:/Datasets/IRYStec-Renault-Simulations/new_1_1-25-2022"
    name_ref = "OriginalHMI.png"

    from utils.image_processing.image_tools import imread

    name_beauty = scene + "_local_Beauty.exr"
    img_ref = imread(name_ref, path, format_float=True)
    img_beauty = imread(name_beauty, path, format_float=False)  # file is already float

    # align reference image with test image
    # first convert HDR test image to SDR
    im2 = img_beauty / img_beauty.max()  # normalize to 0-1
    im2 = im2 ** (1./2.2)  # undo gamma
    im1 = alignImage(img_ref, im2)

    # bgr to rgb
    im1 = np.flip(im1, axis=2)
    im2 = np.flip(im2, axis=2)

    return im1, im2


def test_rsamples():
    path = "I:/Datasets/tid2013"
    im1 = imread(path + "/reference_images/I01.BMP")
    # im2 = imread(path + "/distorted_images/i01_01_4.BMP")  # random noise
    # im2 = imread(path + "/distorted_images/i01_18_5.BMP")  # color diff
    im2 = imread(path + "/distorted_images/i01_15_5.BMP")
    # im2 = imread(path + "/distorted_images/i01_07_5.BMP")
    # im2 = imread(path + "/distorted_images/i01_08_5.BMP")
    ps = PatchSampler(
        centerbias_weight=2,
        diffbased_weight=2,
        uniform_weight=0.5,
    )
    h, w = im1.shape[0], im1.shape[1]
    out_dim = (16, 16)
    hits = np.zeros((h, w))
    num_samples = 1000
    samples = ps.get_sample_params(h, w, out_dim[0], out_dim[1], diff=(im1-im2), num_samples=num_samples)
    for sample in samples:
        x, y = sample[:2]
        # hx, hy = sample[:2]
        hits[x: x + out_dim[0], y: y + out_dim[1]] += 1
        hit = hits[x: x + out_dim[0], y: y + out_dim[1]]
        shape = hit.shape
        if shape[0] < out_dim[0] or shape[1] < out_dim[1]:
            print("NOT A GOOD SQUARE {}".format(shape))

    # plt_show("im", im1)
    diff = im2 - im1
    plt_show("MSE", np.sqrt(np.sum(diff * diff, axis=2)))
    plt_show("Patches", hits)
    plt_show("Image 1", im1)
    plt_show("Image 2", im2)


def test_gsamples():
    path = "I:/Datasets/tid2013"
    im1 = imread(path + "/reference_images/I01.BMP")
    # im2 = imread(path + "/distorted_images/i01_01_4.BMP")  # random noise
    im2 = imread(path + "/distorted_images/i01_18_5.BMP")  # color diff
    # im2 = imread(path + "/distorted_images/i01_15_5.BMP")  # block error
    # im2 = imread(path + "/distorted_images/i01_07_5.BMP")
    # im2 = imread(path + "/distorted_images/i01_08_5.BMP")

    scale_patches = False
    r = 1
    from utils.image_processing.image_tools import resize_keep_aspect_ratio, resize
    im1 = resize(im1, r)
    im2 = resize(im2, r)
    print(im1.shape)

    ps = PatchSampler(
        centerbias_weight=2.,
        diffbased_weight=2.,
        uniform_weight=0.5,
    )
    patch_size = 16
    patch_size = patch_size * r if scale_patches else patch_size
    h, w = im1.shape[0], im1.shape[1]
    out_dim = (patch_size, patch_size)
    num_samples = 256

    timer = Timer()
    num_runs = 12
    for i in range(num_runs):
        timer.start()
        samples = ps.get_sample_params(h, w, out_dim[0], out_dim[1], diff=(im1-im2), num_samples=num_samples)
        timer.stop()

        hits = np.zeros((h, w))
        for sample in samples:
            x, y = sample[:2]
            hits[x: x + out_dim[0], y: y + out_dim[1]] += 1
            hit = hits[x: x + out_dim[0], y: y + out_dim[1]]
            shape = hit.shape
            if shape[0] < out_dim[0] or shape[1] < out_dim[1]:
                print("NOT A GOOD SQUARE {}".format(shape))
        plt_show("Patches", hits)

    print('avg runtime:', timer.delta_avg)

    # plt_show("im", im1)
    diff = im2 - im1
    plt_show("Image 1", im1, dpi=1000)
    plt_show("Image 2", im2, dpi=1000)
    plt_show("MSE", np.sqrt(np.sum(diff * diff, axis=2)), dpi=1000)


def test_util_patches():
    from data.sampling.patch_sampling import get_iqa_patches
    from data.utils import transform as transform_pil

    # path = "I:/Datasets/tid2013"
    # im1 = imread(path + "/reference_images/I01.BMP")
    # im2 = imread(path + "/distorted_images/i01_01_4.BMP")  # random noise
    # im2 = imread(path + "/distorted_images/i01_18_5.BMP")  # color diff
    # im2 = imread(path + "/distorted_images/i01_15_5.BMP")  # block error
    # im2 = imread(path + "/distorted_images/i01_07_5.BMP")  # quantization
    # im2 = imread(path + "/distorted_images/i01_08_5.BMP")

    # path = "I:/Datasets/PIPAL"
    # im1 = imread(path + "/Test_Ref/A0000.BMP")
    # im2 = imread(path + "/Test_Dist/A0000_10_08.BMP")
    # im2 = imread(path + "/Test_Dist/A0000_10_09.BMP")
    # im2 = imread(path + "/Test_Dist/A0000_10_15.BMP")

    im1, im2 = get_renault_images()

    scale_patches = False
    r = 1.
    from utils.image_processing.image_tools import resize
    im1 = resize(im1, r)
    im2 = resize(im2, r)

    print('input shapes', im1.shape, im2.shape)

    im1 = Image.fromarray((im1 * 255).astype(np.uint8))
    im2 = Image.fromarray((im2 * 255).astype(np.uint8))

    t1 = transform_pil(im1)
    t2 = transform_pil(im2)

    patch_num_scales = 5
    patch_size = 16
    patch_size = patch_size * r if scale_patches else patch_size
    out_dim = (patch_size, patch_size)

    num_samples = 2000

    ps = PatchSampler(
        centerbias_weight=0.1,
        diffbased_weight=1,
        uniform_weight=0.1,
    )
    h, w = im1.height, im1.width

    with torch.no_grad():
        timer = Timer()
        num_runs = 1
        for i in range(num_runs):
            timer.start()
            # p1, p2, pos = get_iqa_patches((im1, im2), (t1, t2), num_samples, out_dim, ps)
            # scale = (torch.as_tensor([out_dim[0], out_dim[1]]).expand(pos.shape[0], 2)).numpy().astype(int)
            p1, p2, pos, scale = get_iqa_patches((im1, im2), (t1, t2), num_samples, out_dim, ps, patch_num_scales,
                                                 debug=True
                                                 )
            timer.stop()

            hits = np.zeros((h, w))
            for i in range(p1.shape[0]):
                x, y = int(pos[i, 0]), int(pos[i, 1])
                sx, sy = int(scale[i, 0]), int(scale[i, 1])
                hits[x: x + sx, y: y + sy] += 1
                hit = hits[x: x + sx, y: y + sy]
                shape = hit.shape
                if shape[0] < sx or shape[1] < sy:
                    print("NOT A GOOD SQUARE {}".format(shape))

        print('avg runtime:', timer.delta_avg)

    im1 = np.array(im1, float)
    im2 = np.array(im2, float)

    diff = im2 - im1
    # plt_show("Image 1", im1 / 255)
    # plt_show("Image 2", im2 / 255)
    plt_show("RMSE", np.sqrt(np.sum(diff * diff, axis=2)))
    plt_show("Patches", hits)


if __name__ == '__main__':
    # test_gsamples()
    test_util_patches()
