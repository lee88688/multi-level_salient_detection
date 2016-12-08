# -*- coding: utf-8 -*-

from multiprocessing import Process, freeze_support, Queue
from scipy.spatial.distance import cdist
from skimage.segmentation import slic, mark_boundaries
from skimage.color import rgb2lab, rgb2grey
from skimage.feature import canny
from skimage.future import graph
from scipy.ndimage.morphology import grey_dilation
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from skimage import io, color
import find_pictures_use_region as fp
import numpy as np
import cv2
import os
import random
from time import time
import train_saliency

# features_path = os.getcwd() + os.sep + "features"
# test_out_path = os.getcwd() + os.sep + "tests"

# original_img_dir = r"G:\Project\paper2\other_image\MSRA-1000_images"
# binary_img_dir = r"G:\Project\paper2\other_image\binarymasks"
# general300_cache_out_dir = r'G:\Project\paper2\cache\cache_general_kmeans_300\feature_cache'
# cache_out_dir = r'G:\Project\paper2\cache\cache_features_local_surround_300\feature_cache'
#
# dut_original_img_dir = r"F:\lee\DUT\DUT-OMRON-image"
# dut_binary_img_dir = r"F:\lee\DUT\pixelwiseGT-new-PNG"
# dut_general300_cache_out_dir = r'F:\lee\cache\cache_dut_general_kmeans_300\feature_cache'
# dut_cache_out_dir = r'F:\lee\cache\cache_dut_features_local_surround_300\feature_cache'
#
# saliency_img_out_dir = r"F:\lee\saliency_map\out"
# general_cache_out_dir = r'F:\lee\cache\cache_general_kmeans\feature_cache'


def extract_features(img, binary_masks=None, segments_number=500):
    """
    :type binary_masks: numpy.array
    :param img:
    :param binary_masks: bool array the same size of img, if a pixsel is a saliency pixsel, binary_masks's position
     corresponding to img's array is true.
    :return: features
    """

    # useful functions
    normalize = lambda s: (s - s.min()) / (s.max() - s.min())
    normalize_zero = lambda s: (s - s.min()) / (s.max() - s.min() + 1)
    d_fun = lambda d_c, d_p: d_c / (5 * (1 + d_p))
    get_filter_kernel = lambda x, y: cv2.mulTransposed(cv2.getGaussianKernel(x, y), False)

    # prepare variables
    img_lab = rgb2lab(img)
    segments = slic(img_lab, n_segments=segments_number, compactness=30.0, convert2lab=False)
    max_segments = segments.max() + 1

    # saliency_super_pixels
    if binary_masks is not None:
        saliency_super_pixels = np.zeros((max_segments, 1), dtype=np.float64)
        for i in xrange(max_segments):
            # saliency_super_pixels
            saliency_super_pixels[i] = binary_masks[segments == i].mean()
    else:
        saliency_super_pixels = None

    # create x,y feather
    shape = img.shape
    size = img.size
    a = shape[0]
    b = shape[1]
    x_axis = np.linspace(0, b - 1, num=b)
    y_axis = np.linspace(0, a - 1, num=a)

    x_coordinate = np.tile(x_axis, (a, 1,))  # 创建X轴的坐标表
    y_coordinate = np.tile(y_axis, (b, 1,))  # 创建y轴的坐标表
    y_coordinate = np.transpose(y_coordinate)

    coordinate_segments_mean = np.zeros((max_segments, 2))

    # create lab feather
    img_l = img_lab[:, :, 0]
    img_a = img_lab[:, :, 1]
    img_b = img_lab[:, :, 2]

    img_segments_mean = np.zeros((max_segments, 3))

    # FT feature
    # gaussian_kernel = cv2.mulTransposed(cv2.getGaussianKernel(3, 3), False)  # create gaussianKernel
    blur_img_lab = cv2.filter2D(img_lab, -1, get_filter_kernel(5, 5))
    blur_lm = blur_img_lab[:, :, 0].mean()
    blur_am = blur_img_lab[:, :, 1].mean()
    blur_bm = blur_img_lab[:, :, 2].mean()
    blur_sm = np.sqrt((blur_img_lab[:, :, 0] - blur_lm) ** 2 + (blur_img_lab[:, :, 1] - blur_am) ** 2 + (
        blur_img_lab[:, :, 2] - blur_bm) ** 2)
    ft_feature = np.zeros((max_segments, 1))

    # size feature
    size_feature = np.zeros((max_segments, 1))

    # color center feature
    w_sum = np.sum(blur_sm)
    x_center = np.sum(blur_sm * x_coordinate) / w_sum
    y_center = np.sum(blur_sm * y_coordinate) / w_sum
    center_color_map = np.exp(- (np.abs(x_coordinate - x_center) + np.abs(y_coordinate - y_center)) / 250)
    center_color_feature = np.zeros((max_segments, 1))

    # edge feature
    edge_img = grey_dilation(canny(cv2.filter2D(rgb2grey(img), -1, get_filter_kernel(10, 5))), size=(5, 5))
    edge_feature = np.zeros((max_segments, 1))

    for i in xrange(max_segments):
        segments_i = segments == i

        coordinate_segments_mean[i, 0] = x_coordinate[segments_i].mean()
        coordinate_segments_mean[i, 1] = y_coordinate[segments_i].mean()

        img_segments_mean[i, 0] = img_l[segments_i].mean()
        img_segments_mean[i, 1] = img_a[segments_i].mean()
        img_segments_mean[i, 2] = img_b[segments_i].mean()

        ft_feature[i] = blur_sm[segments_i].mean()

        size_feature[i] = blur_sm[segments_i].size / float(size)

        center_color_feature[i] = center_color_map[segments_i].mean()

        edge_feature[i] = edge_img[segments_i].sum()

    # CA feature
    ca_feature = np.sum(d_fun(cdist(img_segments_mean, img_segments_mean),
                              cdist(coordinate_segments_mean, coordinate_segments_mean)), axis=1)
    ca_feature = np.array([ca_feature]).T  # transpose it to column vector

    # element distribution
    wc_ij = np.exp(-cdist(img_segments_mean, img_segments_mean)**2 / (2 * 20 ** 2))
    wc_ij = wc_ij / wc_ij.sum(axis=1)[:, None]
    mu_i = np.dot(wc_ij, coordinate_segments_mean)
    distribution = np.dot(wc_ij, np.linalg.norm(coordinate_segments_mean - mu_i, axis=1)**2)
    distribution = normalize(distribution)
    distribution = np.array([distribution]).T

    # uniqueness plus
    wp_ij = np.exp(cdist(coordinate_segments_mean, coordinate_segments_mean)**2 / (2 * 100 ** 2))
    wp_ij = wp_ij / wp_ij.sum(axis=1)[:, None]
    mu_i_c = np.dot(wp_ij, img_segments_mean)
    # uniqueness_plus = np.dot(wp_ij, np.linalg.norm(img_segments_mean - mu_i_c, axis=1)**2)
    uniqueness_plus = np.sum(cdist(img_segments_mean, mu_i_c)**2 * wp_ij, axis=1)
    uniqueness_plus = normalize(uniqueness_plus)
    uniqueness_plus = np.array([uniqueness_plus]).T

    # backup
    coordinate_segments_mean_copy = coordinate_segments_mean.copy()
    img_segments_mean_copy = img_segments_mean.copy()

    # edge feature
    # edge_feature_list.append([edge_feature.max(), edge_feature.mean()])
    if edge_feature.max() < 10:
        # print "find it!"
        edge_img = grey_dilation(canny(rgb2grey(img)), size=(5, 5))
        for i in xrange(max_segments):
            edge_feature[i] = edge_img[segments == i].sum()

    # normalize features
    img_segments_mean[:, 0] = normalize(img_segments_mean[:, 0])
    img_segments_mean[:, 1] = normalize(img_segments_mean[:, 1])
    img_segments_mean[:, 2] = normalize(img_segments_mean[:, 2])
    coordinate_segments_mean[:, 0] = normalize(coordinate_segments_mean[:, 0])
    coordinate_segments_mean[:, 1] = normalize(coordinate_segments_mean[:, 1])
    ft_feature = normalize(ft_feature)
    ca_feature = 1 - np.exp(- ca_feature * 1.5 / float(max_segments))
    ca_feature = normalize(ca_feature)
    size_feature = normalize(size_feature)
    edge_feature = normalize_zero(edge_feature)

    # region level features
    sigma1 = 50
    sigma2 = 20
    # todo: change region_segment params
    segments_region, labels = region_segment(np.concatenate((img_segments_mean, coordinate_segments_mean,
                                                             uniqueness_plus, distribution), axis=1), segments)
    max_region_segments = segments_region.max() + 1
    coordinate_region_segments_mean = np.zeros((max_region_segments, 2))
    img_region_segments_mean = np.zeros((max_region_segments, 3))
    for i in xrange(max_region_segments):
        segments_i = segments_region == i

        coordinate_region_segments_mean[i, 0] = x_coordinate[segments_i].mean()
        coordinate_region_segments_mean[i, 1] = y_coordinate[segments_i].mean()

        img_region_segments_mean[i, 0] = img_l[segments_i].mean()
        img_region_segments_mean[i, 1] = img_a[segments_i].mean()
        img_region_segments_mean[i, 2] = img_b[segments_i].mean()

    C_i_R = np.apply_along_axis(lambda x: img_region_segments_mean[x[0], :], 1, np.array([labels]).T)
    D = cdist(img_segments_mean_copy, img_region_segments_mean)**2 - (np.linalg.norm(img_segments_mean_copy - C_i_R, axis=1)**2)[:, None]

    w_ij = np.exp(cdist(coordinate_segments_mean_copy, coordinate_region_segments_mean)**2 / (2 * sigma1**2))
    w_ij = w_ij / w_ij.sum(axis=1)[:, None]

    region_conlor_contrast = np.sum(w_ij * D, axis=1)
    region_conlor_contrast = normalize(region_conlor_contrast)

    wd_ij = np.exp(cdist(img_segments_mean_copy, img_region_segments_mean)**2 / (2 * sigma2**2))
    wd_ij = wd_ij / wd_ij.sum(axis=1)[:, None]
    mu_i = np.dot(wd_ij, coordinate_region_segments_mean)
    DR = np.sum(cdist(mu_i, coordinate_region_segments_mean) * wd_ij, axis=1)
    # DR = np.dot(wd_ij, np.linalg.norm(coordinate_segments_mean - mu_i, axis=1)**2)
    DR = normalize(DR)

    if binary_masks is not None:
        return np.concatenate((img_segments_mean, coordinate_segments_mean, uniqueness_plus, distribution,
                               center_color_feature, ft_feature, ca_feature, size_feature, edge_feature,
                               region_conlor_contrast[:, None], DR[:, None],
                               saliency_super_pixels), axis=1), segments, np.concatenate((img_segments_mean_copy, coordinate_segments_mean_copy), axis=1), img_lab
    else:
        return np.concatenate((img_segments_mean, coordinate_segments_mean, uniqueness_plus, distribution,
                               center_color_feature, ft_feature, ca_feature, size_feature, edge_feature,
                               region_conlor_contrast[:, None], DR[:, None],
                               ), axis=1), segments, np.concatenate((img_segments_mean_copy, coordinate_segments_mean_copy), axis=1), img_lab


def extract_general_features(img, binary_masks=None, segments_number=500, th=0.001):
    # useful functions
    normalize = lambda s: (s - s.min()) / (s.max() - s.min())

    # prepare variables
    img_lab = rgb2lab(img)
    segments = slic(img_lab, n_segments=segments_number, compactness=30.0, convert2lab=False)
    max_segments = segments.max() + 1

    # saliency_super_pixels
    if binary_masks is not None:
        saliency_super_pixels = np.zeros((max_segments, 1), dtype=np.float64)
        for i in xrange(max_segments):
            # saliency_super_pixels
            saliency_super_pixels[i] = binary_masks[segments == i].mean()
    else:
        saliency_super_pixels = None

    # create x,y feather
    shape = img.shape
    a = shape[0]
    b = shape[1]
    x_axis = np.linspace(0, b - 1, num=b)
    y_axis = np.linspace(0, a - 1, num=a)

    x_coordinate = np.tile(x_axis, (a, 1,))  # 创建X轴的坐标表
    y_coordinate = np.tile(y_axis, (b, 1,))  # 创建y轴的坐标表
    y_coordinate = np.transpose(y_coordinate)

    coordinate_segments_mean = np.zeros((max_segments, 2))

    # create lab feather
    img_l = img_lab[:, :, 0]
    img_a = img_lab[:, :, 1]
    img_b = img_lab[:, :, 2]

    img_segments_mean = np.zeros((max_segments, 3))

    for i in xrange(max_segments):
        segments_i = segments == i

        coordinate_segments_mean[i, 0] = x_coordinate[segments_i].mean()
        coordinate_segments_mean[i, 1] = y_coordinate[segments_i].mean()

        img_segments_mean[i, 0] = img_l[segments_i].mean()
        img_segments_mean[i, 1] = img_a[segments_i].mean()
        img_segments_mean[i, 2] = img_b[segments_i].mean()

    segments_region, labels = region_segment(img, segments, th=th)

    max_region_segments = segments_region.max() + 1
    coordinate_region_segments_mean = np.zeros((max_region_segments, 2))
    img_region_segments_mean = np.zeros((max_region_segments, 3))
    for i in xrange(max_region_segments):
        segments_i = segments_region == i

        coordinate_region_segments_mean[i, 0] = x_coordinate[segments_i].mean()
        coordinate_region_segments_mean[i, 1] = y_coordinate[segments_i].mean()

        img_region_segments_mean[i, 0] = img_l[segments_i].mean()
        img_region_segments_mean[i, 1] = img_a[segments_i].mean()
        img_region_segments_mean[i, 2] = img_b[segments_i].mean()

    if binary_masks is not None:
        return np.concatenate((img_segments_mean, coordinate_segments_mean,
                               saliency_super_pixels), axis=1), \
               np.concatenate((img_region_segments_mean, coordinate_region_segments_mean), axis=1), \
               segments, segments_region, labels
    else:
        return np.concatenate((img_segments_mean, coordinate_segments_mean,
                               ), axis=1), \
               np.concatenate((img_region_segments_mean, coordinate_region_segments_mean), axis=1), \
               segments, segments_region, labels


def extract_general_features_kmeans(img, binary_masks=None, segments_number=500):
    # useful functions
    normalize = lambda s: (s - s.min()) / (s.max() - s.min())

    # prepare variables
    img_lab = rgb2lab(img)
    segments = slic(img_lab, n_segments=segments_number, compactness=30.0, convert2lab=False)
    max_segments = segments.max() + 1

    # saliency_super_pixels
    if binary_masks is not None:
        saliency_super_pixels = np.zeros((max_segments, 1), dtype=np.float64)
        for i in xrange(max_segments):
            # saliency_super_pixels
            saliency_super_pixels[i] = binary_masks[segments == i].mean()
    else:
        saliency_super_pixels = None

    # create x,y feather
    shape = img.shape
    a = shape[0]
    b = shape[1]
    x_axis = np.linspace(0, b - 1, num=b)
    y_axis = np.linspace(0, a - 1, num=a)

    x_coordinate = np.tile(x_axis, (a, 1,))  # 创建X轴的坐标表
    y_coordinate = np.tile(y_axis, (b, 1,))  # 创建y轴的坐标表
    y_coordinate = np.transpose(y_coordinate)

    coordinate_segments_mean = np.zeros((max_segments, 2))

    # create lab feather
    img_l = img_lab[:, :, 0]
    img_a = img_lab[:, :, 1]
    img_b = img_lab[:, :, 2]

    img_segments_mean = np.zeros((max_segments, 3))

    for i in xrange(max_segments):
        segments_i = segments == i

        coordinate_segments_mean[i, 0] = x_coordinate[segments_i].mean()
        coordinate_segments_mean[i, 1] = y_coordinate[segments_i].mean()

        img_segments_mean[i, 0] = img_l[segments_i].mean()
        img_segments_mean[i, 1] = img_a[segments_i].mean()
        img_segments_mean[i, 2] = img_b[segments_i].mean()

    segments_region, labels = train_saliency.region_segment(np.concatenate((img_segments_mean, coordinate_segments_mean),
                               axis=1), segments)

    max_region_segments = segments_region.max() + 1
    coordinate_region_segments_mean = np.zeros((max_region_segments, 2))
    img_region_segments_mean = np.zeros((max_region_segments, 3))
    for i in xrange(max_region_segments):
        segments_i = segments_region == i

        coordinate_region_segments_mean[i, 0] = x_coordinate[segments_i].mean()
        coordinate_region_segments_mean[i, 1] = y_coordinate[segments_i].mean()

        img_region_segments_mean[i, 0] = img_l[segments_i].mean()
        img_region_segments_mean[i, 1] = img_a[segments_i].mean()
        img_region_segments_mean[i, 2] = img_b[segments_i].mean()

    if binary_masks is not None:
        return np.concatenate((img_segments_mean, coordinate_segments_mean,
                               saliency_super_pixels), axis=1), \
               np.concatenate((img_region_segments_mean, coordinate_region_segments_mean), axis=1), \
               segments, segments_region, labels
    else:
        return np.concatenate((img_segments_mean, coordinate_segments_mean,
                               ), axis=1), \
               np.concatenate((img_region_segments_mean, coordinate_region_segments_mean), axis=1), \
               segments, segments_region, labels


def extract_general_features_kmeans_rgb(img, binary_masks, segments_number=500):
    # prepare variables
    img_lab = rgb2lab(img)
    segments = slic(img_lab, n_segments=segments_number, compactness=30.0, convert2lab=False)
    max_segments = segments.max() + 1

    # saliency_super_pixels
    saliency_super_pixels = np.zeros((max_segments, 1), dtype=np.float64)
    # for i in xrange(max_segments):
    #     # saliency_super_pixels
    #     saliency_super_pixels[i] = binary_masks[segments == i].mean()

    # create x,y feather
    shape = img.shape
    a = shape[0]
    b = shape[1]
    x_axis = np.linspace(0, b - 1, num=b)
    y_axis = np.linspace(0, a - 1, num=a)

    x_coordinate = np.tile(x_axis, (a, 1,))  # 创建X轴的坐标表
    y_coordinate = np.tile(y_axis, (b, 1,))  # 创建y轴的坐标表
    y_coordinate = np.transpose(y_coordinate)

    coordinate_segments_mean = np.zeros((max_segments, 2))

    # create lab feature
    img_l = img_lab[:, :, 0]
    img_a = img_lab[:, :, 1]
    img_b = img_lab[:, :, 2]

    # create rgb feature
    img_r = img[:, :, 0]
    img_g = img[:, :, 1]
    img_b = img[:, :, 2]

    img_segments_mean = np.zeros((max_segments, 3))
    img_rgb_segments_mean = np.zeros((max_segments, 3))

    for i in xrange(max_segments):
        segments_i = segments == i

        coordinate_segments_mean[i, 0] = x_coordinate[segments_i].mean()
        coordinate_segments_mean[i, 1] = y_coordinate[segments_i].mean()

        img_segments_mean[i, 0] = img_l[segments_i].mean()
        img_segments_mean[i, 1] = img_a[segments_i].mean()
        img_segments_mean[i, 2] = img_b[segments_i].mean()

        img_rgb_segments_mean[i, 0] = img_r[segments_i].mean()
        img_rgb_segments_mean[i, 1] = img_g[segments_i].mean()
        img_rgb_segments_mean[i, 2] = img_b[segments_i].mean()

        saliency_super_pixels[i] = binary_masks[segments_i].mean()

    segments_region, labels = train_saliency.region_segment(np.concatenate((img_segments_mean, coordinate_segments_mean),
                               axis=1), segments)

    max_region_segments = segments_region.max() + 1
    coordinate_region_segments_mean = np.zeros((max_region_segments, 2))
    img_region_segments_mean = np.zeros((max_region_segments, 3))
    img_rgb_region_segments_mean = np.zeros((max_region_segments, 3))
    for i in xrange(max_region_segments):
        segments_i = segments_region == i

        coordinate_region_segments_mean[i, 0] = x_coordinate[segments_i].mean()
        coordinate_region_segments_mean[i, 1] = y_coordinate[segments_i].mean()

        img_region_segments_mean[i, 0] = img_l[segments_i].mean()
        img_region_segments_mean[i, 1] = img_a[segments_i].mean()
        img_region_segments_mean[i, 2] = img_b[segments_i].mean()

        img_rgb_region_segments_mean[i, 0] = img_r[segments_i].mean()
        img_rgb_region_segments_mean[i, 1] = img_g[segments_i].mean()
        img_rgb_region_segments_mean[i, 2] = img_b[segments_i].mean()

    return np.concatenate((img_segments_mean, coordinate_segments_mean, img_rgb_segments_mean,
                           ), axis=1), \
           np.concatenate((img_region_segments_mean, coordinate_region_segments_mean, img_rgb_region_segments_mean), axis=1), \
           segments, segments_region, labels


def extract_features_from_cache(img_name, original_img_dir, cache_dir):
    if not os.path.exists(cache_dir):
        raise NameError("Not such cache dir!")
    region_features_dir = cache_dir + "_region"
    segments_dir = cache_dir + "_segments"
    region_labels_dir = cache_dir + "_region_labels"
    img_npy_name = img_name.split('.')[0] + ".npy"

    normalize = lambda s: (s - s.min()) / (s.max() - s.min())
    normalize_zero = lambda s: (s - s.min()) / (s.max() - s.min() + 1)
    d_fun = lambda d_c, d_p: d_c / (5 * (1 + d_p))
    get_filter_kernel = lambda x, y: cv2.mulTransposed(cv2.getGaussianKernel(x, y), False)
    sigma1 = 77.4125539537
    sigma2 = 21.56378688
    sigma3 = 50
    sigma4 = 23.8579998503

    img = io.imread(original_img_dir + os.sep + img_name)
    feature = np.load(cache_dir + os.sep + img_npy_name)
    segments = np.load(segments_dir + os.sep + img_npy_name)
    region_feature = np.load(region_features_dir + os.sep + img_npy_name)
    region_labels = np.load(region_labels_dir + os.sep + img_npy_name)

    img_segments_mean = feature[:, 0:3]
    coordinate_segments_mean = feature[:, 3:5]
    saliency_super_pixels = feature[:, -1]
    img_region_segments_mean = region_feature[:, 0:3]
    coordinate_region_segments_mean = region_feature[:, 3:5]

    img_lab = rgb2lab(img)
    max_segments = segments.max() + 1

    # create x,y feather
    shape = img.shape
    size = img.size
    a = shape[0]
    b = shape[1]
    x_axis = np.linspace(0, b - 1, num=b)
    y_axis = np.linspace(0, a - 1, num=a)

    x_coordinate = np.tile(x_axis, (a, 1,))  # 创建X轴的坐标表
    y_coordinate = np.tile(y_axis, (b, 1,))  # 创建y轴的坐标表
    y_coordinate = np.transpose(y_coordinate)

    # FT feature
    blur_img_lab = cv2.filter2D(img_lab, -1, get_filter_kernel(5, 5))
    blur_lm = blur_img_lab[:, :, 0].mean()
    blur_am = blur_img_lab[:, :, 1].mean()
    blur_bm = blur_img_lab[:, :, 2].mean()
    blur_sm = np.sqrt((blur_img_lab[:, :, 0] - blur_lm) ** 2 + (blur_img_lab[:, :, 1] - blur_am) ** 2 + (
        blur_img_lab[:, :, 2] - blur_bm) ** 2)
    ft_feature = np.zeros((max_segments, 1))

    # size feature
    size_feature = np.zeros((max_segments, 1))

    # color center feature
    w_sum = np.sum(blur_sm)
    x_center = np.sum(blur_sm * x_coordinate) / w_sum
    y_center = np.sum(blur_sm * y_coordinate) / w_sum
    center_color_map = np.exp(- (np.abs(x_coordinate - x_center) + np.abs(y_coordinate - y_center)) / 250)
    center_color_feature = np.zeros((max_segments, 1))

    # edge feature
    edge_img = grey_dilation(canny(cv2.filter2D(rgb2grey(img), -1, get_filter_kernel(10, 5))), size=(5, 5))
    edge_feature = np.zeros((max_segments, 1))

    for i in xrange(max_segments):
        segments_i = segments == i

        ft_feature[i] = blur_sm[segments_i].mean()

        size_feature[i] = blur_sm[segments_i].size / float(size)

        center_color_feature[i] = center_color_map[segments_i].mean()

        edge_feature[i] = edge_img[segments_i].sum()

    # CA feature
    ca_feature = np.sum(d_fun(cdist(img_segments_mean, img_segments_mean),
                              cdist(coordinate_segments_mean, coordinate_segments_mean)), axis=1)
    ca_feature = np.array([ca_feature]).T  # transpose it to column vector

    # uniqueness plus
    w_ij = 1 - np.exp(-cdist(coordinate_segments_mean, coordinate_segments_mean) * sigma1)
    w_ij = w_ij / w_ij.sum(axis=1)[:, None]
    # mu_i_c = np.dot(wp_ij, img_segments_mean)
    uniqueness_plus = np.sum(cdist(img_segments_mean, img_segments_mean) * w_ij, axis=1)
    uniqueness_plus = normalize(uniqueness_plus)
    uniqueness_plus = np.array([uniqueness_plus]).T

    # distribution
    wc_ij = np.exp(-cdist(img_segments_mean, img_segments_mean) ** 2 / (2 * sigma2 ** 2))
    wc_ij = wc_ij / wc_ij.sum(axis=1)[:, None]
    mu_i = np.dot(wc_ij, coordinate_segments_mean)
    distribution = np.dot(wc_ij, np.linalg.norm(coordinate_segments_mean - mu_i, axis=1) ** 2)
    distribution = normalize(distribution)
    distribution = np.array([distribution]).T

    r = np.unique(region_labels, return_counts=True)
    size = r[1]*1.0/region_labels.size

    D = cdist(img_segments_mean, img_region_segments_mean)

    w_ij = 1 - np.exp(
        -cdist(coordinate_segments_mean, coordinate_region_segments_mean) * sigma3)
    w_ij = w_ij / w_ij.sum(axis=1)[:, None]

    region_conlor_contrast = np.sum(w_ij * D * size, axis=1)
    region_conlor_contrast = normalize(region_conlor_contrast)
    region_conlor_contrast = region_conlor_contrast[:, None]

    # wd_ij = np.exp(-cdist(img_segments_mean, img_region_segments_mean) / self.__region_sigma2)
    wd_ij = np.exp(-cdist(img_segments_mean, img_region_segments_mean) ** 2 / (2 * sigma4 ** 2))
    wd_ij = wd_ij / wd_ij.sum(axis=1)[:, None]
    mu_i = np.dot(wd_ij, coordinate_region_segments_mean)
    DR = np.sum(cdist(mu_i, coordinate_region_segments_mean) * wd_ij * size, axis=1)
    DR = normalize(DR)
    DR = DR[:, None]

    # normalize features
    img_segments_mean[:, 0] = normalize(img_segments_mean[:, 0])
    img_segments_mean[:, 1] = normalize(img_segments_mean[:, 1])
    img_segments_mean[:, 2] = normalize(img_segments_mean[:, 2])
    coordinate_segments_mean[:, 0] = normalize(coordinate_segments_mean[:, 0])
    coordinate_segments_mean[:, 1] = normalize(coordinate_segments_mean[:, 1])
    ft_feature = normalize(ft_feature)
    ca_feature = 1 - np.exp(- ca_feature * 1.5 / float(max_segments))
    ca_feature = normalize(ca_feature)
    size_feature = normalize(size_feature)
    edge_feature = normalize_zero(edge_feature)

    return np.concatenate((img_segments_mean, coordinate_segments_mean, uniqueness_plus, distribution,
                           center_color_feature, ft_feature, ca_feature, size_feature, edge_feature,
                           region_conlor_contrast, DR,
                           saliency_super_pixels[:, None]), axis=1), segments


def extract_features_from_cache2(img_name, original_img_dir, cache_dir):
    '''
    this function extract local features with regions, local include the superpixel itself
    :param img_name:
    :param original_img_dir:
    :param cache_dir:
    :return:
    '''
    if not os.path.exists(cache_dir):
        raise NameError("Not such cache dir!")
    region_features_dir = cache_dir + "_region"
    segments_dir = cache_dir + "_segments"
    region_labels_dir = cache_dir + "_region_labels"
    img_npy_name = img_name.split('.')[0] + ".npy"

    normalize = lambda s: (s - s.min()) / (s.max() - s.min())
    normalize_zero = lambda s: (s - s.min()) / (s.max() - s.min() + 1)
    d_fun = lambda d_c, d_p: d_c / (5 * (1 + d_p))
    get_filter_kernel = lambda x, y: cv2.mulTransposed(cv2.getGaussianKernel(x, y), False)
    sigma1 = 10
    sigma2 = 21.10405218
    sigma3 = 50
    sigma4 = 23.3557741048

    img = io.imread(original_img_dir + os.sep + img_name)
    feature = np.load(cache_dir + os.sep + img_npy_name)
    segments = np.load(segments_dir + os.sep + img_npy_name)
    region_feature = np.load(region_features_dir + os.sep + img_npy_name)
    region_labels = np.load(region_labels_dir + os.sep + img_npy_name)

    img_segments_mean = feature[:, 0:3]
    coordinate_segments_mean = feature[:, 3:5]
    saliency_super_pixels = feature[:, -1]
    img_region_segments_mean = region_feature[:, 0:3]
    coordinate_region_segments_mean = region_feature[:, 3:5]

    img_lab = rgb2lab(img)
    max_segments = segments.max() + 1

    # create x,y feather
    shape = img.shape
    size = img.size
    a = shape[0]
    b = shape[1]
    x_axis = np.linspace(0, b - 1, num=b)
    y_axis = np.linspace(0, a - 1, num=a)

    x_coordinate = np.tile(x_axis, (a, 1,))  # 创建X轴的坐标表
    y_coordinate = np.tile(y_axis, (b, 1,))  # 创建y轴的坐标表
    y_coordinate = np.transpose(y_coordinate)

    # FT feature
    blur_img_lab = cv2.filter2D(img_lab, -1, get_filter_kernel(5, 5))
    blur_lm = blur_img_lab[:, :, 0].mean()
    blur_am = blur_img_lab[:, :, 1].mean()
    blur_bm = blur_img_lab[:, :, 2].mean()
    blur_sm = np.sqrt((blur_img_lab[:, :, 0] - blur_lm) ** 2 + (blur_img_lab[:, :, 1] - blur_am) ** 2 + (
        blur_img_lab[:, :, 2] - blur_bm) ** 2)
    ft_feature = np.zeros((max_segments, 1))

    # color center feature
    w_sum = np.sum(blur_sm)
    x_center = np.sum(blur_sm * x_coordinate) / w_sum
    y_center = np.sum(blur_sm * y_coordinate) / w_sum
    center_color_map = np.exp(- (np.abs(x_coordinate - x_center) + np.abs(y_coordinate - y_center)) / 250)
    center_color_feature = np.zeros((max_segments, 1))

    # edge feature
    edge_img = grey_dilation(canny(cv2.filter2D(rgb2grey(img), -1, get_filter_kernel(10, 5))), size=(5, 5))
    edge_feature = np.zeros((max_segments, 1))

    # local element
    # left and right
    local_map1 = np.concatenate([np.diff(segments, axis=1), np.zeros([a, 1])], axis=1)
    local_map2 = np.fliplr(np.concatenate([np.diff(np.fliplr(segments), axis=1), np.zeros([a, 1])], axis=1))
    # up and down
    local_map3 = np.concatenate([np.diff(segments, axis=0), np.zeros([1, b])], axis=0)
    local_map4 = np.flipud(np.concatenate([np.diff(np.flipud(segments), axis=0), np.zeros([1, b])], axis=0))

    local_map1 = local_map1.astype(np.bool)
    local_map2 = local_map2.astype(np.bool)
    local_map3 = local_map3.astype(np.bool)
    local_map4 = local_map4.astype(np.bool)

    x_coordinate_map = x_coordinate.copy()
    x_coordinate_tmp = x_coordinate.copy()
    y_coordinate_map = y_coordinate.copy()
    y_coordinate_tmp = y_coordinate.copy()
    x_coordinate_map[local_map1] += 1
    x_coordinate_map[local_map2] -= 1
    y_coordinate_map[local_map3] += 1
    y_coordinate_map[local_map4] -= 1

    x_coordinate_map[~(local_map1 | local_map2)] = 0
    y_coordinate_tmp[~(local_map1 | local_map2)] = 0

    y_coordinate_map[~(local_map3 | local_map4)] = 0
    x_coordinate_tmp[~(local_map3 | local_map4)] = 0

    coordinate_r = np.zeros([a, b, 2], dtype=np.int32)
    coordinate_c = np.zeros([a, b, 2], dtype=np.int32)
    # column
    coordinate_c[:, :, 0] = y_coordinate_tmp
    coordinate_c[:, :, 1] = x_coordinate_map
    # row
    coordinate_r[:, :, 0] = y_coordinate_map
    coordinate_r[:, :, 1] = x_coordinate_tmp

    neighbors = np.eye(max_segments, dtype=np.bool)
    func = lambda a: segments[a[0], a[1]]

    for i in xrange(max_segments):
        segments_i = segments == i

        ft_feature[i] = blur_sm[segments_i].mean()

        center_color_feature[i] = center_color_map[segments_i].mean()

        edge_feature[i] = edge_img[segments_i].sum()

        # column
        c = coordinate_c[segments_i]
        d = np.unique(np.apply_along_axis(func, axis=1, arr=c[(c[:, 0] > 0) | (c[:, 1] > 0), :]))
        neighbors[i, d] = True
        # row
        c = coordinate_r[segments_i]
        d = np.unique(np.apply_along_axis(func, axis=1, arr=c[(c[:, 0] > 0) | (c[:, 1] > 0), :]))
        neighbors[i, d] = True

    # uniqueness plus
    w_ij = 1 - np.exp(-cdist(coordinate_segments_mean, coordinate_segments_mean) * sigma1)
    w_ij = w_ij / w_ij.sum(axis=1)[:, None]
    # mu_i_c = np.dot(wp_ij, img_segments_mean)
    uniqueness_plus = np.sum(cdist(img_segments_mean, img_segments_mean) * w_ij, axis=1)
    uniqueness_plus = np.array([uniqueness_plus]).T

    # distribution
    wc_ij = np.exp(-cdist(img_segments_mean, img_segments_mean) ** 2 / (2 * sigma2 ** 2))
    wc_ij = wc_ij / wc_ij.sum(axis=1)[:, None]
    mu_i = np.dot(wc_ij, coordinate_segments_mean)
    distribution = np.dot(wc_ij, np.linalg.norm(coordinate_segments_mean - mu_i, axis=1) ** 2)
    distribution = np.array([distribution]).T

    r = np.unique(region_labels, return_counts=True)
    size = r[1]*1.0/region_labels.size

    D = cdist(img_segments_mean, img_region_segments_mean)

    w_ij = 1 - np.exp(
        -cdist(coordinate_segments_mean, coordinate_region_segments_mean) * sigma3)
    w_ij = w_ij / w_ij.sum(axis=1)[:, None]

    region_conlor_contrast = np.sum(w_ij * D * size, axis=1)
    region_conlor_contrast = region_conlor_contrast[:, None]

    wd_ij = np.exp(-cdist(img_segments_mean, img_region_segments_mean) ** 2 / (2 * sigma4 ** 2))
    wd_ij = wd_ij / wd_ij.sum(axis=1)[:, None]
    mu_i = np.dot(wd_ij, coordinate_region_segments_mean)
    DR = np.sum(cdist(mu_i, coordinate_region_segments_mean) * wd_ij * size, axis=1)
    DR = DR[:, None]

    # local features
    local_img_segments_mean = np.zeros_like(img_segments_mean)
    local_coordinate_segments_mean = np.zeros_like(coordinate_segments_mean)
    local_ft_feature = np.zeros_like(ft_feature)
    local_edge_feature = np.zeros_like(edge_feature)
    local_distribution = np.zeros_like(distribution)
    local_uniqueness_plus = np.zeros_like(uniqueness_plus)
    local_region_conlor_contrast = np.zeros_like(region_conlor_contrast)
    local_DR = np.zeros_like(DR)
    for i in xrange(max_segments):
        local_img_segments_mean[i, :] = np.mean(img_segments_mean[neighbors[i, :], :], axis=0)
        local_coordinate_segments_mean[i, :] = np.mean(coordinate_segments_mean[neighbors[i, :], :], axis=0)
        local_ft_feature[i, :] = np.mean(ft_feature[neighbors[i, :], :], axis=0)
        local_edge_feature[i, :] = np.mean(edge_feature[neighbors[i, :], :], axis=0)
        local_distribution[i, :] = np.mean(distribution[neighbors[i, :], :], axis=0)
        local_uniqueness_plus[i, :] = np.mean(uniqueness_plus[neighbors[i, :], :], axis=0)
        local_region_conlor_contrast[i, :] = np.mean(region_conlor_contrast[neighbors[i, :], :], axis=0)
        local_DR[i, :] = np.mean(DR[neighbors[i, :], :], axis=0)

    # normalize features
    img_segments_mean[:, 0] = normalize(img_segments_mean[:, 0])
    img_segments_mean[:, 1] = normalize(img_segments_mean[:, 1])
    img_segments_mean[:, 2] = normalize(img_segments_mean[:, 2])
    coordinate_segments_mean[:, 0] = normalize(coordinate_segments_mean[:, 0])
    coordinate_segments_mean[:, 1] = normalize(coordinate_segments_mean[:, 1])
    ft_feature = normalize(ft_feature)
    edge_feature = normalize_zero(edge_feature)
    uniqueness_plus = normalize(uniqueness_plus)
    distribution = normalize(distribution)
    # region features
    region_conlor_contrast = normalize(region_conlor_contrast)
    DR = normalize(DR)
    # local features
    local_img_segments_mean[:, 0] = normalize(local_img_segments_mean[:, 0])
    local_img_segments_mean[:, 1] = normalize(local_img_segments_mean[:, 1])
    local_img_segments_mean[:, 2] = normalize(local_img_segments_mean[:, 2])
    local_coordinate_segments_mean[:, 0] = normalize(local_coordinate_segments_mean[:, 0])
    local_coordinate_segments_mean[:, 1] = normalize(local_coordinate_segments_mean[:, 1])
    local_ft_feature = normalize(local_ft_feature)
    local_edge_feature = normalize_zero(local_edge_feature)
    local_distribution = normalize(local_distribution)
    local_uniqueness_plus = normalize(local_uniqueness_plus)
    local_region_conlor_contrast = normalize(local_region_conlor_contrast)
    local_DR = normalize(local_DR)

    return np.concatenate((img_segments_mean, coordinate_segments_mean, uniqueness_plus, distribution,
                           center_color_feature, ft_feature, edge_feature, region_conlor_contrast, DR,
                           local_img_segments_mean, local_coordinate_segments_mean, local_ft_feature,
                           local_edge_feature, local_distribution, local_uniqueness_plus,
                           local_region_conlor_contrast, local_DR,
                           saliency_super_pixels[:, None]), axis=1), segments


def extract_features_from_cache3(img_name, original_img_dir, cache_dir):
    '''
    this function extract local features with regions, local not include the superpixel itself
    :param img_name:
    :param original_img_dir:
    :param cache_dir:
    :return:
    '''
    if not os.path.exists(cache_dir):
        raise NameError("Not such cache dir!")
    region_features_dir = cache_dir + "_region"
    segments_dir = cache_dir + "_segments"
    region_labels_dir = cache_dir + "_region_labels"
    img_npy_name = img_name.split('.')[0] + ".npy"

    normalize = lambda s: (s - s.min()) / (s.max() - s.min())
    normalize_zero = lambda s: (s - s.min()) / (s.max() - s.min() + 1)
    d_fun = lambda d_c, d_p: d_c / (5 * (1 + d_p))
    get_filter_kernel = lambda x, y: cv2.mulTransposed(cv2.getGaussianKernel(x, y), False)
    sigma1 = 10
    sigma2 = 21.10405218
    sigma3 = 50
    sigma4 = 23.3557741048

    img = io.imread(original_img_dir + os.sep + img_name)
    feature = np.load(cache_dir + os.sep + img_npy_name)
    segments = np.load(segments_dir + os.sep + img_npy_name)
    region_feature = np.load(region_features_dir + os.sep + img_npy_name)
    region_labels = np.load(region_labels_dir + os.sep + img_npy_name)

    img_segments_mean = feature[:, 0:3]
    coordinate_segments_mean = feature[:, 3:5]
    saliency_super_pixels = feature[:, -1]
    img_region_segments_mean = region_feature[:, 0:3]
    coordinate_region_segments_mean = region_feature[:, 3:5]

    img_lab = rgb2lab(img)
    max_segments = segments.max() + 1

    # create x,y feather
    shape = img.shape
    size = img.size
    a = shape[0]
    b = shape[1]
    x_axis = np.linspace(0, b - 1, num=b)
    y_axis = np.linspace(0, a - 1, num=a)

    x_coordinate = np.tile(x_axis, (a, 1,))  # 创建X轴的坐标表
    y_coordinate = np.tile(y_axis, (b, 1,))  # 创建y轴的坐标表
    y_coordinate = np.transpose(y_coordinate)

    # FT feature
    blur_img_lab = cv2.filter2D(img_lab, -1, get_filter_kernel(5, 5))
    blur_lm = blur_img_lab[:, :, 0].mean()
    blur_am = blur_img_lab[:, :, 1].mean()
    blur_bm = blur_img_lab[:, :, 2].mean()
    blur_sm = np.sqrt((blur_img_lab[:, :, 0] - blur_lm) ** 2 + (blur_img_lab[:, :, 1] - blur_am) ** 2 + (
        blur_img_lab[:, :, 2] - blur_bm) ** 2)
    ft_feature = np.zeros((max_segments, 1))

    # color center feature
    w_sum = np.sum(blur_sm)
    x_center = np.sum(blur_sm * x_coordinate) / w_sum
    y_center = np.sum(blur_sm * y_coordinate) / w_sum
    center_color_map = np.exp(- (np.abs(x_coordinate - x_center) + np.abs(y_coordinate - y_center)) / 250)
    center_color_feature = np.zeros((max_segments, 1))

    # edge feature
    edge_img = grey_dilation(canny(cv2.filter2D(rgb2grey(img), -1, get_filter_kernel(10, 5))), size=(5, 5))
    edge_feature = np.zeros((max_segments, 1))

    # local element
    # left and right
    local_map1 = np.concatenate([np.diff(segments, axis=1), np.zeros([a, 1])], axis=1)
    local_map2 = np.fliplr(np.concatenate([np.diff(np.fliplr(segments), axis=1), np.zeros([a, 1])], axis=1))
    # up and down
    local_map3 = np.concatenate([np.diff(segments, axis=0), np.zeros([1, b])], axis=0)
    local_map4 = np.flipud(np.concatenate([np.diff(np.flipud(segments), axis=0), np.zeros([1, b])], axis=0))

    local_map1 = local_map1.astype(np.bool)
    local_map2 = local_map2.astype(np.bool)
    local_map3 = local_map3.astype(np.bool)
    local_map4 = local_map4.astype(np.bool)

    x_coordinate_map = x_coordinate.copy()
    x_coordinate_tmp = x_coordinate.copy()
    y_coordinate_map = y_coordinate.copy()
    y_coordinate_tmp = y_coordinate.copy()
    x_coordinate_map[local_map1] += 1
    x_coordinate_map[local_map2] -= 1
    y_coordinate_map[local_map3] += 1
    y_coordinate_map[local_map4] -= 1

    x_coordinate_map[~(local_map1 | local_map2)] = 0
    y_coordinate_tmp[~(local_map1 | local_map2)] = 0

    y_coordinate_map[~(local_map3 | local_map4)] = 0
    x_coordinate_tmp[~(local_map3 | local_map4)] = 0

    coordinate_r = np.zeros([a, b, 2], dtype=np.int32)
    coordinate_c = np.zeros([a, b, 2], dtype=np.int32)
    # column
    coordinate_c[:, :, 0] = y_coordinate_tmp
    coordinate_c[:, :, 1] = x_coordinate_map
    # row
    coordinate_r[:, :, 0] = y_coordinate_map
    coordinate_r[:, :, 1] = x_coordinate_tmp

    neighbors = np.eye(max_segments, dtype=np.bool)
    func = lambda a: segments[a[0], a[1]]

    for i in xrange(max_segments):
        segments_i = segments == i

        ft_feature[i] = blur_sm[segments_i].mean()

        center_color_feature[i] = center_color_map[segments_i].mean()

        edge_feature[i] = edge_img[segments_i].sum()

        # column
        c = coordinate_c[segments_i]
        d = np.unique(np.apply_along_axis(func, axis=1, arr=c[(c[:, 0] > 0) | (c[:, 1] > 0), :]))
        neighbors[i, d] = True
        # row
        c = coordinate_r[segments_i]
        d = np.unique(np.apply_along_axis(func, axis=1, arr=c[(c[:, 0] > 0) | (c[:, 1] > 0), :]))
        neighbors[i, d] = True

    # uniqueness plus
    w_ij = 1 - np.exp(-cdist(coordinate_segments_mean, coordinate_segments_mean) * sigma1)
    w_ij = w_ij / w_ij.sum(axis=1)[:, None]
    # mu_i_c = np.dot(wp_ij, img_segments_mean)
    uniqueness_plus = np.sum(cdist(img_segments_mean, img_segments_mean) * w_ij, axis=1)
    uniqueness_plus = np.array([uniqueness_plus]).T

    # distribution
    wc_ij = np.exp(-cdist(img_segments_mean, img_segments_mean) ** 2 / (2 * sigma2 ** 2))
    wc_ij = wc_ij / wc_ij.sum(axis=1)[:, None]
    mu_i = np.dot(wc_ij, coordinate_segments_mean)
    distribution = np.dot(wc_ij, np.linalg.norm(coordinate_segments_mean - mu_i, axis=1) ** 2)
    distribution = np.array([distribution]).T

    r = np.unique(region_labels, return_counts=True)
    size = r[1]*1.0/region_labels.size

    D = cdist(img_segments_mean, img_region_segments_mean)

    w_ij = 1 - np.exp(
        -cdist(coordinate_segments_mean, coordinate_region_segments_mean) * sigma3)
    w_ij = w_ij / w_ij.sum(axis=1)[:, None]

    region_conlor_contrast = np.sum(w_ij * D * size, axis=1)
    region_conlor_contrast = region_conlor_contrast[:, None]

    wd_ij = np.exp(-cdist(img_segments_mean, img_region_segments_mean) ** 2 / (2 * sigma4 ** 2))
    wd_ij = wd_ij / wd_ij.sum(axis=1)[:, None]
    mu_i = np.dot(wd_ij, coordinate_region_segments_mean)
    DR = np.sum(cdist(mu_i, coordinate_region_segments_mean) * wd_ij * size, axis=1)
    DR = DR[:, None]

    # local features
    local_img_segments_mean = np.zeros_like(img_segments_mean)
    local_coordinate_segments_mean = np.zeros_like(coordinate_segments_mean)
    local_ft_feature = np.zeros_like(ft_feature)
    local_edge_feature = np.zeros_like(edge_feature)
    local_distribution = np.zeros_like(distribution)
    local_uniqueness_plus = np.zeros_like(uniqueness_plus)
    local_region_conlor_contrast = np.zeros_like(region_conlor_contrast)
    local_DR = np.zeros_like(DR)
    for i in xrange(max_segments):
        neighbors[i, i] = False
        local_img_segments_mean[i, :] = np.mean(img_segments_mean[neighbors[i, :], :], axis=0)
        local_coordinate_segments_mean[i, :] = np.mean(coordinate_segments_mean[neighbors[i, :], :], axis=0)
        local_ft_feature[i, :] = np.mean(ft_feature[neighbors[i, :], :], axis=0)
        local_edge_feature[i, :] = np.mean(edge_feature[neighbors[i, :], :], axis=0)
        local_distribution[i, :] = np.mean(distribution[neighbors[i, :], :], axis=0)
        local_uniqueness_plus[i, :] = np.mean(uniqueness_plus[neighbors[i, :], :], axis=0)
        local_region_conlor_contrast[i, :] = np.mean(region_conlor_contrast[neighbors[i, :], :], axis=0)
        local_DR[i, :] = np.mean(DR[neighbors[i, :], :], axis=0)

    # normalize features
    img_segments_mean[:, 0] = normalize(img_segments_mean[:, 0])
    img_segments_mean[:, 1] = normalize(img_segments_mean[:, 1])
    img_segments_mean[:, 2] = normalize(img_segments_mean[:, 2])
    coordinate_segments_mean[:, 0] = normalize(coordinate_segments_mean[:, 0])
    coordinate_segments_mean[:, 1] = normalize(coordinate_segments_mean[:, 1])
    ft_feature = normalize(ft_feature)
    edge_feature = normalize_zero(edge_feature)
    uniqueness_plus = normalize(uniqueness_plus)
    distribution = normalize(distribution)
    # region features
    region_conlor_contrast = normalize(region_conlor_contrast)
    DR = normalize(DR)
    # local features
    local_img_segments_mean[:, 0] = normalize(local_img_segments_mean[:, 0])
    local_img_segments_mean[:, 1] = normalize(local_img_segments_mean[:, 1])
    local_img_segments_mean[:, 2] = normalize(local_img_segments_mean[:, 2])
    local_coordinate_segments_mean[:, 0] = normalize(local_coordinate_segments_mean[:, 0])
    local_coordinate_segments_mean[:, 1] = normalize(local_coordinate_segments_mean[:, 1])
    local_ft_feature = normalize(local_ft_feature)
    local_edge_feature = normalize_zero(local_edge_feature)
    local_distribution = normalize(local_distribution)
    local_uniqueness_plus = normalize(local_uniqueness_plus)
    local_region_conlor_contrast = normalize(local_region_conlor_contrast)
    local_DR = normalize(local_DR)

    return np.concatenate((img_segments_mean, coordinate_segments_mean, uniqueness_plus, distribution,
                           center_color_feature, ft_feature, edge_feature, region_conlor_contrast, DR,
                           local_img_segments_mean, local_coordinate_segments_mean, local_ft_feature,
                           local_edge_feature, local_distribution, local_uniqueness_plus,
                           local_region_conlor_contrast, local_DR,
                           saliency_super_pixels[:, None]), axis=1), segments


def extract_features_from_cache4(img_name, original_img_dir, cache_dir):
    '''
    this function extract local features with regions, local not include the superpixel itself
    :param img_name:
    :param original_img_dir:
    :param cache_dir: general cache dir
    :return:
    '''
    if not os.path.exists(cache_dir):
        raise NameError("Not such cache dir!")
    region_features_dir = cache_dir + "_region"
    segments_dir = cache_dir + "_segments"
    region_labels_dir = cache_dir + "_region_labels"
    img_npy_name = img_name.split('.')[0] + ".npy"

    normalize = lambda s: (s - s.min()) / (s.max() - s.min())
    normalize_zero = lambda s: (s - s.min()) / (s.max() - s.min() + 1)
    d_fun = lambda d_c, d_p: d_c / (5 * (1 + d_p))
    get_filter_kernel = lambda x, y: cv2.mulTransposed(cv2.getGaussianKernel(x, y), False)
    sigma1 = 10
    sigma2 = 21.10405218
    sigma3 = 50
    sigma4 = 23.3557741048

    img = io.imread(original_img_dir + os.sep + img_name)
    feature = np.load(cache_dir + os.sep + img_npy_name)
    segments = np.load(segments_dir + os.sep + img_npy_name)
    region_feature = np.load(region_features_dir + os.sep + img_npy_name)
    region_labels = np.load(region_labels_dir + os.sep + img_npy_name)

    img_segments_mean = feature[:, 0:3]
    coordinate_segments_mean = feature[:, 3:5]
    saliency_super_pixels = feature[:, -1]
    img_region_segments_mean = region_feature[:, 0:3]
    coordinate_region_segments_mean = region_feature[:, 3:5]

    img_lab = rgb2lab(img)
    max_segments = segments.max() + 1

    # create x,y feather
    shape = img.shape
    size = img.size
    a = shape[0]
    b = shape[1]
    x_axis = np.linspace(0, b - 1, num=b)
    y_axis = np.linspace(0, a - 1, num=a)

    x_coordinate = np.tile(x_axis, (a, 1,))  # 创建X轴的坐标表
    y_coordinate = np.tile(y_axis, (b, 1,))  # 创建y轴的坐标表
    y_coordinate = np.transpose(y_coordinate)

    # border mask
    border_mask = (x_coordinate < 16) | (x_coordinate > b - 17) | (y_coordinate < 16) | (y_coordinate > a - 17)
    r = np.unique(segments[border_mask], return_counts=True)
    border_feature = np.zeros((max_segments, 1))
    border_feature[r[0], 0] = r[1]

    # FT feature
    blur_img_lab = cv2.filter2D(img_lab, -1, get_filter_kernel(5, 5))
    blur_lm = blur_img_lab[:, :, 0].mean()
    blur_am = blur_img_lab[:, :, 1].mean()
    blur_bm = blur_img_lab[:, :, 2].mean()
    blur_sm = np.sqrt((blur_img_lab[:, :, 0] - blur_lm) ** 2 + (blur_img_lab[:, :, 1] - blur_am) ** 2 + (
        blur_img_lab[:, :, 2] - blur_bm) ** 2)
    ft_feature = np.zeros((max_segments, 1))

    # color center feature
    w_sum = np.sum(blur_sm)
    x_center = np.sum(blur_sm * x_coordinate) / w_sum
    y_center = np.sum(blur_sm * y_coordinate) / w_sum
    center_color_map = np.exp(- (np.abs(x_coordinate - x_center) + np.abs(y_coordinate - y_center)) / 250)
    center_color_feature = np.zeros((max_segments, 1))

    # edge feature
    edge_img = grey_dilation(canny(cv2.filter2D(rgb2grey(img), -1, get_filter_kernel(10, 5))), size=(5, 5))
    edge_feature = np.zeros((max_segments, 1))

    # local element
    # left and right
    local_map1 = np.concatenate([np.diff(segments, axis=1), np.zeros([a, 1])], axis=1)
    local_map2 = np.fliplr(np.concatenate([np.diff(np.fliplr(segments), axis=1), np.zeros([a, 1])], axis=1))
    # up and down
    local_map3 = np.concatenate([np.diff(segments, axis=0), np.zeros([1, b])], axis=0)
    local_map4 = np.flipud(np.concatenate([np.diff(np.flipud(segments), axis=0), np.zeros([1, b])], axis=0))

    local_map1 = local_map1.astype(np.bool)
    local_map2 = local_map2.astype(np.bool)
    local_map3 = local_map3.astype(np.bool)
    local_map4 = local_map4.astype(np.bool)

    x_coordinate_map = x_coordinate.copy()
    x_coordinate_tmp = x_coordinate.copy()
    y_coordinate_map = y_coordinate.copy()
    y_coordinate_tmp = y_coordinate.copy()
    x_coordinate_map[local_map1] += 1
    x_coordinate_map[local_map2] -= 1
    y_coordinate_map[local_map3] += 1
    y_coordinate_map[local_map4] -= 1

    x_coordinate_map[~(local_map1 | local_map2)] = 0
    y_coordinate_tmp[~(local_map1 | local_map2)] = 0

    y_coordinate_map[~(local_map3 | local_map4)] = 0
    x_coordinate_tmp[~(local_map3 | local_map4)] = 0

    coordinate_r = np.zeros([a, b, 2], dtype=np.int32)
    coordinate_c = np.zeros([a, b, 2], dtype=np.int32)
    # column
    coordinate_c[:, :, 0] = y_coordinate_tmp
    coordinate_c[:, :, 1] = x_coordinate_map
    # row
    coordinate_r[:, :, 0] = y_coordinate_map
    coordinate_r[:, :, 1] = x_coordinate_tmp

    neighbors = np.eye(max_segments, dtype=np.bool)
    func = lambda a: segments[a[0], a[1]]

    for i in xrange(max_segments):
        segments_i = segments == i

        ft_feature[i] = blur_sm[segments_i].mean()

        center_color_feature[i] = center_color_map[segments_i].mean()

        edge_feature[i] = edge_img[segments_i].sum()

        # column
        c = coordinate_c[segments_i]
        d = np.unique(np.apply_along_axis(func, axis=1, arr=c[(c[:, 0] > 0) | (c[:, 1] > 0), :]))
        neighbors[i, d] = True
        # row
        c = coordinate_r[segments_i]
        d = np.unique(np.apply_along_axis(func, axis=1, arr=c[(c[:, 0] > 0) | (c[:, 1] > 0), :]))
        neighbors[i, d] = True

    # uniqueness plus
    w_ij = 1 - np.exp(-cdist(coordinate_segments_mean, coordinate_segments_mean) * sigma1)
    w_ij = w_ij / w_ij.sum(axis=1)[:, None]
    # mu_i_c = np.dot(wp_ij, img_segments_mean)
    uniqueness_plus = np.sum(cdist(img_segments_mean, img_segments_mean) * w_ij, axis=1)
    uniqueness_plus = np.array([uniqueness_plus]).T

    # distribution
    wc_ij = np.exp(-cdist(img_segments_mean, img_segments_mean) ** 2 / (2 * sigma2 ** 2))
    wc_ij = wc_ij / wc_ij.sum(axis=1)[:, None]
    mu_i = np.dot(wc_ij, coordinate_segments_mean)
    distribution = np.dot(wc_ij, np.linalg.norm(coordinate_segments_mean - mu_i, axis=1) ** 2)
    distribution = np.array([distribution]).T

    r = np.unique(region_labels, return_counts=True)
    size = r[1]*1.0/region_labels.size

    D = cdist(img_segments_mean, img_region_segments_mean)

    w_ij = 1 - np.exp(
        -cdist(coordinate_segments_mean, coordinate_region_segments_mean) * sigma3)
    w_ij = w_ij / w_ij.sum(axis=1)[:, None]

    region_conlor_contrast = np.sum(w_ij * D * size, axis=1)
    region_conlor_contrast = region_conlor_contrast[:, None]

    wd_ij = np.exp(-cdist(img_segments_mean, img_region_segments_mean) ** 2 / (2 * sigma4 ** 2))
    wd_ij = wd_ij / wd_ij.sum(axis=1)[:, None]
    mu_i = np.dot(wd_ij, coordinate_region_segments_mean)
    DR = np.sum(cdist(mu_i, coordinate_region_segments_mean) * wd_ij * size, axis=1)
    DR = DR[:, None]

    # local features
    local_img_segments_mean = np.zeros_like(img_segments_mean)
    local_coordinate_segments_mean = np.zeros_like(coordinate_segments_mean)
    local_ft_feature = np.zeros_like(ft_feature)
    local_edge_feature = np.zeros_like(edge_feature)
    local_distribution = np.zeros_like(distribution)
    local_uniqueness_plus = np.zeros_like(uniqueness_plus)
    local_region_conlor_contrast = np.zeros_like(region_conlor_contrast)
    local_DR = np.zeros_like(DR)
    for i in xrange(max_segments):
        neighbors[i, i] = False
        local_img_segments_mean[i, :] = np.mean(img_segments_mean[neighbors[i, :], :], axis=0)
        local_coordinate_segments_mean[i, :] = np.mean(coordinate_segments_mean[neighbors[i, :], :], axis=0)
        local_ft_feature[i, :] = np.mean(ft_feature[neighbors[i, :], :], axis=0)
        local_edge_feature[i, :] = np.mean(edge_feature[neighbors[i, :], :], axis=0)
        local_distribution[i, :] = np.mean(distribution[neighbors[i, :], :], axis=0)
        local_uniqueness_plus[i, :] = np.mean(uniqueness_plus[neighbors[i, :], :], axis=0)
        local_region_conlor_contrast[i, :] = np.mean(region_conlor_contrast[neighbors[i, :], :], axis=0)
        local_DR[i, :] = np.mean(DR[neighbors[i, :], :], axis=0)

    # normalize features
    img_segments_mean[:, 0] = normalize(img_segments_mean[:, 0])
    img_segments_mean[:, 1] = normalize(img_segments_mean[:, 1])
    img_segments_mean[:, 2] = normalize(img_segments_mean[:, 2])
    coordinate_segments_mean[:, 0] = normalize(coordinate_segments_mean[:, 0])
    coordinate_segments_mean[:, 1] = normalize(coordinate_segments_mean[:, 1])
    border_feature = normalize(border_feature)
    center_color_feature = normalize(center_color_feature)
    ft_feature = normalize(ft_feature)
    edge_feature = normalize_zero(edge_feature)
    uniqueness_plus = normalize(uniqueness_plus)
    distribution = normalize(distribution)
    # region features
    region_conlor_contrast = normalize(region_conlor_contrast)
    DR = normalize(DR)
    # local features
    local_img_segments_mean[:, 0] = normalize(local_img_segments_mean[:, 0])
    local_img_segments_mean[:, 1] = normalize(local_img_segments_mean[:, 1])
    local_img_segments_mean[:, 2] = normalize(local_img_segments_mean[:, 2])
    local_coordinate_segments_mean[:, 0] = normalize(local_coordinate_segments_mean[:, 0])
    local_coordinate_segments_mean[:, 1] = normalize(local_coordinate_segments_mean[:, 1])
    local_ft_feature = normalize(local_ft_feature)
    local_edge_feature = normalize_zero(local_edge_feature)
    local_distribution = normalize(local_distribution)
    local_uniqueness_plus = normalize(local_uniqueness_plus)
    local_region_conlor_contrast = normalize(local_region_conlor_contrast)
    local_DR = normalize(local_DR)

    return np.concatenate((img_segments_mean, coordinate_segments_mean, border_feature, uniqueness_plus, distribution,
                           center_color_feature, ft_feature, edge_feature, region_conlor_contrast, DR,
                           local_img_segments_mean, local_coordinate_segments_mean, local_ft_feature,
                           local_edge_feature, local_distribution, local_uniqueness_plus,
                           local_region_conlor_contrast, local_DR,
                           saliency_super_pixels[:, None]), axis=1), segments, neighbors


def extract_features_from_cache5(img_name, original_img_dir, cache_dir):
    '''
    this function extract local features with regions, local not include the superpixel itself
    :param img_name:
    :param original_img_dir:
    :param cache_dir: general cache dir
    :return:
    '''
    if not os.path.exists(cache_dir):
        raise NameError("Not such cache dir!")
    region_features_dir = cache_dir + "_region"
    segments_dir = cache_dir + "_segments"
    region_labels_dir = cache_dir + "_region_labels"
    img_npy_name = img_name.split('.')[0] + ".npy"

    normalize = lambda s: (s - s.min()) / (s.max() - s.min())
    normalize_zero = lambda s: (s - s.min()) / (s.max() - s.min() + 1)
    d_fun = lambda d_c, d_p: d_c / (5 * (1 + d_p))
    get_filter_kernel = lambda x, y: cv2.mulTransposed(cv2.getGaussianKernel(x, y), False)
    sigma1 = 10
    sigma2 = 21.10405218
    sigma3 = 50
    sigma4 = 23.3557741048

    img = io.imread(original_img_dir + os.sep + img_name)
    feature = np.load(cache_dir + os.sep + img_npy_name)
    segments = np.load(segments_dir + os.sep + img_npy_name)
    region_feature = np.load(region_features_dir + os.sep + img_npy_name)
    region_labels = np.load(region_labels_dir + os.sep + img_npy_name)

    img_segments_mean = feature[:, 0:3]
    coordinate_segments_mean = feature[:, 3:5]
    saliency_super_pixels = feature[:, -1]
    img_region_segments_mean = region_feature[:, 0:3]
    coordinate_region_segments_mean = region_feature[:, 3:5]

    img_lab = rgb2lab(img)
    max_segments = segments.max() + 1

    # create x,y feather
    shape = img.shape
    size = img.size
    a = shape[0]
    b = shape[1]
    x_axis = np.linspace(0, b - 1, num=b)
    y_axis = np.linspace(0, a - 1, num=a)

    x_coordinate = np.tile(x_axis, (a, 1,))  # 创建X轴的坐标表
    y_coordinate = np.tile(y_axis, (b, 1,))  # 创建y轴的坐标表
    y_coordinate = np.transpose(y_coordinate)

    # border mask
    border_mask = (x_coordinate < 16) | (x_coordinate > b - 17) | (y_coordinate < 16) | (y_coordinate > a - 17)
    r = np.unique(segments[border_mask], return_counts=True)
    border_feature = np.zeros((max_segments, 1))
    border_feature[r[0], 0] = r[1]

    # FT feature
    blur_img_lab = cv2.filter2D(img_lab, -1, get_filter_kernel(5, 5))
    blur_lm = blur_img_lab[:, :, 0].mean()
    blur_am = blur_img_lab[:, :, 1].mean()
    blur_bm = blur_img_lab[:, :, 2].mean()
    blur_sm = np.sqrt((blur_img_lab[:, :, 0] - blur_lm) ** 2 + (blur_img_lab[:, :, 1] - blur_am) ** 2 + (
        blur_img_lab[:, :, 2] - blur_bm) ** 2)
    ft_feature = np.zeros((max_segments, 1))

    # color center feature
    w_sum = np.sum(blur_sm)
    x_center = np.sum(blur_sm * x_coordinate) / w_sum
    y_center = np.sum(blur_sm * y_coordinate) / w_sum
    center_color_map = np.exp(- (np.abs(x_coordinate - x_center) + np.abs(y_coordinate - y_center)) / 250)
    center_color_feature = np.zeros((max_segments, 1))

    # edge feature
    edge_img = grey_dilation(canny(cv2.filter2D(rgb2grey(img), -1, get_filter_kernel(10, 5))), size=(5, 5))
    edge_feature = np.zeros((max_segments, 1))

    # local element
    # left and right
    local_map1 = np.concatenate([np.diff(segments, axis=1), np.zeros([a, 1])], axis=1)
    local_map2 = np.fliplr(np.concatenate([np.diff(np.fliplr(segments), axis=1), np.zeros([a, 1])], axis=1))
    # up and down
    local_map3 = np.concatenate([np.diff(segments, axis=0), np.zeros([1, b])], axis=0)
    local_map4 = np.flipud(np.concatenate([np.diff(np.flipud(segments), axis=0), np.zeros([1, b])], axis=0))

    local_map1 = local_map1.astype(np.bool)
    local_map2 = local_map2.astype(np.bool)
    local_map3 = local_map3.astype(np.bool)
    local_map4 = local_map4.astype(np.bool)

    x_coordinate_map = x_coordinate.copy()
    x_coordinate_tmp = x_coordinate.copy()
    y_coordinate_map = y_coordinate.copy()
    y_coordinate_tmp = y_coordinate.copy()
    x_coordinate_map[local_map1] += 1
    x_coordinate_map[local_map2] -= 1
    y_coordinate_map[local_map3] += 1
    y_coordinate_map[local_map4] -= 1

    x_coordinate_map[~(local_map1 | local_map2)] = 0
    y_coordinate_tmp[~(local_map1 | local_map2)] = 0

    y_coordinate_map[~(local_map3 | local_map4)] = 0
    x_coordinate_tmp[~(local_map3 | local_map4)] = 0

    coordinate_r = np.zeros([a, b, 2], dtype=np.int32)
    coordinate_c = np.zeros([a, b, 2], dtype=np.int32)
    # column
    coordinate_c[:, :, 0] = y_coordinate_tmp
    coordinate_c[:, :, 1] = x_coordinate_map
    # row
    coordinate_r[:, :, 0] = y_coordinate_map
    coordinate_r[:, :, 1] = x_coordinate_tmp

    neighbors = np.eye(max_segments, dtype=np.bool)
    func = lambda a: segments[a[0], a[1]]

    for i in xrange(max_segments):
        segments_i = segments == i

        ft_feature[i] = blur_sm[segments_i].mean()

        center_color_feature[i] = center_color_map[segments_i].mean()

        edge_feature[i] = edge_img[segments_i].sum()

        # column
        c = coordinate_c[segments_i]
        d = np.unique(np.apply_along_axis(func, axis=1, arr=c[(c[:, 0] > 0) | (c[:, 1] > 0), :]))
        neighbors[i, d] = True
        # row
        c = coordinate_r[segments_i]
        d = np.unique(np.apply_along_axis(func, axis=1, arr=c[(c[:, 0] > 0) | (c[:, 1] > 0), :]))
        neighbors[i, d] = True

    # uniqueness plus
    w_ij = 1 - np.exp(-cdist(coordinate_segments_mean, coordinate_segments_mean) * sigma1)
    w_ij = w_ij / w_ij.sum(axis=1)[:, None]
    # mu_i_c = np.dot(wp_ij, img_segments_mean)
    uniqueness_plus = np.sum(cdist(img_segments_mean, img_segments_mean) * w_ij, axis=1)
    uniqueness_plus = np.array([uniqueness_plus]).T

    # distribution
    wc_ij = np.exp(-cdist(img_segments_mean, img_segments_mean) ** 2 / (2 * sigma2 ** 2))
    wc_ij = wc_ij / wc_ij.sum(axis=1)[:, None]
    mu_i = np.dot(wc_ij, coordinate_segments_mean)
    distribution = np.dot(wc_ij, np.linalg.norm(coordinate_segments_mean - mu_i, axis=1) ** 2)
    distribution = np.array([distribution]).T

    r = np.unique(region_labels, return_counts=True)
    size = r[1]*1.0/region_labels.size

    D = cdist(img_segments_mean, img_region_segments_mean)

    w_ij = 1 - np.exp(
        -cdist(coordinate_segments_mean, coordinate_region_segments_mean) * sigma3)
    w_ij = w_ij / w_ij.sum(axis=1)[:, None]

    region_conlor_contrast = np.sum(w_ij * D * size, axis=1)
    region_conlor_contrast = region_conlor_contrast[:, None]

    wd_ij = np.exp(-cdist(img_segments_mean, img_region_segments_mean) ** 2 / (2 * sigma4 ** 2))
    wd_ij = wd_ij / wd_ij.sum(axis=1)[:, None]
    mu_i = np.dot(wd_ij, coordinate_region_segments_mean)
    DR = np.sum(cdist(mu_i, coordinate_region_segments_mean) * wd_ij * size, axis=1)
    DR = DR[:, None]

    # local features
    local_img_segments_mean = np.zeros_like(img_segments_mean)
    local_coordinate_segments_mean = np.zeros_like(coordinate_segments_mean)
    local_ft_feature = np.zeros_like(ft_feature)
    local_edge_feature = np.zeros_like(edge_feature)
    for i in xrange(max_segments):
        neighbors[i, i] = False
        local_img_segments_mean[i, :] = np.mean(img_segments_mean[neighbors[i, :], :], axis=0)
        local_coordinate_segments_mean[i, :] = np.mean(coordinate_segments_mean[neighbors[i, :], :], axis=0)
        local_ft_feature[i, :] = np.mean(ft_feature[neighbors[i, :], :], axis=0)
        local_edge_feature[i, :] = np.mean(edge_feature[neighbors[i, :], :], axis=0)

    # normalize features
    img_segments_mean[:, 0] = normalize(img_segments_mean[:, 0])
    img_segments_mean[:, 1] = normalize(img_segments_mean[:, 1])
    img_segments_mean[:, 2] = normalize(img_segments_mean[:, 2])
    coordinate_segments_mean[:, 0] = normalize(coordinate_segments_mean[:, 0])
    coordinate_segments_mean[:, 1] = normalize(coordinate_segments_mean[:, 1])
    border_feature = normalize(border_feature)
    center_color_feature = normalize(center_color_feature)
    ft_feature = normalize(ft_feature)
    edge_feature = normalize_zero(edge_feature)
    uniqueness_plus = normalize(uniqueness_plus)
    distribution = normalize(distribution)
    # region features
    region_conlor_contrast = normalize(region_conlor_contrast)
    DR = normalize(DR)
    # local features
    local_img_segments_mean[:, 0] = normalize(local_img_segments_mean[:, 0])
    local_img_segments_mean[:, 1] = normalize(local_img_segments_mean[:, 1])
    local_img_segments_mean[:, 2] = normalize(local_img_segments_mean[:, 2])
    local_coordinate_segments_mean[:, 0] = normalize(local_coordinate_segments_mean[:, 0])
    local_coordinate_segments_mean[:, 1] = normalize(local_coordinate_segments_mean[:, 1])
    local_ft_feature = normalize(local_ft_feature)
    local_edge_feature = normalize_zero(local_edge_feature)

    return np.concatenate((img_segments_mean, coordinate_segments_mean, border_feature, uniqueness_plus, distribution,
                           center_color_feature, ft_feature, edge_feature, region_conlor_contrast, DR,
                           local_img_segments_mean, local_coordinate_segments_mean, local_ft_feature,
                           local_edge_feature,
                           saliency_super_pixels[:, None]), axis=1), segments, neighbors


def extract_features_from_cache6(img_name, original_img_dir, cache_dir):
    '''
    add Manifold Ranking features
    :param img_name:
    :param original_img_dir:
    :param cache_dir: general cache dir
    :return:
    '''
    if not os.path.exists(cache_dir):
        raise NameError("Not such cache dir!")
    region_features_dir = cache_dir + "_region"
    segments_dir = cache_dir + "_segments"
    region_labels_dir = cache_dir + "_region_labels"
    frame_info_dir = cache_dir + "_frame_info"
    img_npy_name = img_name.split('.')[0] + ".npy"

    normalize = lambda s: (s - s.min()) / (s.max() - s.min())
    normalize_zero = lambda s: (s - s.min()) / (s.max() - s.min() + 1)
    d_fun = lambda d_c, d_p: d_c / (5 * (1 + d_p))
    get_filter_kernel = lambda x, y: cv2.mulTransposed(cv2.getGaussianKernel(x, y), False)
    sigma1 = 10
    sigma2 = 21.10405218
    sigma3 = 50
    sigma4 = 23.3557741048

    img = io.imread(original_img_dir + os.sep + img_name)
    frame = np.load(frame_info_dir + os.sep + img_npy_name)
    feature = np.load(cache_dir + os.sep + img_npy_name)
    segments = np.load(segments_dir + os.sep + img_npy_name)
    region_feature = np.load(region_features_dir + os.sep + img_npy_name)
    region_labels = np.load(region_labels_dir + os.sep + img_npy_name)
    # remove frame
    img = img[frame[2]:frame[3], frame[4]:frame[5]]

    img_segments_mean = feature[:, 0:3]
    coordinate_segments_mean = feature[:, 3:5]
    saliency_super_pixels = feature[:, -1]
    img_region_segments_mean = region_feature[:, 0:3]
    coordinate_region_segments_mean = region_feature[:, 3:5]
    img_segments_mean_copy = img_segments_mean.copy()  # backup for manifold ranking feature

    img_lab = rgb2lab(img)
    max_segments = segments.max() + 1

    # create x,y feather
    shape = img.shape
    size = img.size
    a = shape[0]
    b = shape[1]
    x_axis = np.linspace(0, b - 1, num=b)
    y_axis = np.linspace(0, a - 1, num=a)

    x_coordinate = np.tile(x_axis, (a, 1,))  # 创建X轴的坐标表
    y_coordinate = np.tile(y_axis, (b, 1,))  # 创建y轴的坐标表
    y_coordinate = np.transpose(y_coordinate)

    # border mask
    border_mask = (x_coordinate < 16) | (x_coordinate > b - 17) | (y_coordinate < 16) | (y_coordinate > a - 17)
    r = np.unique(segments[border_mask], return_counts=True)
    border_feature = np.zeros((max_segments, 1))
    border_feature[r[0], 0] = r[1]

    # FT feature
    blur_img_lab = cv2.filter2D(img_lab, -1, get_filter_kernel(5, 5))
    blur_lm = blur_img_lab[:, :, 0].mean()
    blur_am = blur_img_lab[:, :, 1].mean()
    blur_bm = blur_img_lab[:, :, 2].mean()
    blur_sm = np.sqrt((blur_img_lab[:, :, 0] - blur_lm) ** 2 + (blur_img_lab[:, :, 1] - blur_am) ** 2 + (
        blur_img_lab[:, :, 2] - blur_bm) ** 2)
    ft_feature = np.zeros((max_segments, 1))

    # color center feature
    w_sum = np.sum(blur_sm)
    x_center = np.sum(blur_sm * x_coordinate) / w_sum
    y_center = np.sum(blur_sm * y_coordinate) / w_sum
    center_color_map = np.exp(- (np.abs(x_coordinate - x_center) + np.abs(y_coordinate - y_center)) / 250)
    center_color_feature = np.zeros((max_segments, 1))

    # edge feature
    edge_img = grey_dilation(canny(cv2.filter2D(rgb2grey(img), -1, get_filter_kernel(10, 5))), size=(5, 5))
    edge_feature = np.zeros((max_segments, 1))

    # local element
    # left and right
    local_map1 = np.concatenate([np.diff(segments, axis=1), np.zeros([a, 1])], axis=1)
    local_map2 = np.fliplr(np.concatenate([np.diff(np.fliplr(segments), axis=1), np.zeros([a, 1])], axis=1))
    # up and down
    local_map3 = np.concatenate([np.diff(segments, axis=0), np.zeros([1, b])], axis=0)
    local_map4 = np.flipud(np.concatenate([np.diff(np.flipud(segments), axis=0), np.zeros([1, b])], axis=0))

    local_map1 = local_map1.astype(np.bool)
    local_map2 = local_map2.astype(np.bool)
    local_map3 = local_map3.astype(np.bool)
    local_map4 = local_map4.astype(np.bool)

    x_coordinate_map = x_coordinate.copy()
    x_coordinate_tmp = x_coordinate.copy()
    y_coordinate_map = y_coordinate.copy()
    y_coordinate_tmp = y_coordinate.copy()
    x_coordinate_map[local_map1] += 1
    x_coordinate_map[local_map2] -= 1
    y_coordinate_map[local_map3] += 1
    y_coordinate_map[local_map4] -= 1

    x_coordinate_map[~(local_map1 | local_map2)] = 0
    y_coordinate_tmp[~(local_map1 | local_map2)] = 0

    y_coordinate_map[~(local_map3 | local_map4)] = 0
    x_coordinate_tmp[~(local_map3 | local_map4)] = 0

    coordinate_r = np.zeros([a, b, 2], dtype=np.int32)
    coordinate_c = np.zeros([a, b, 2], dtype=np.int32)
    # column
    coordinate_c[:, :, 0] = y_coordinate_tmp
    coordinate_c[:, :, 1] = x_coordinate_map
    # row
    coordinate_r[:, :, 0] = y_coordinate_map
    coordinate_r[:, :, 1] = x_coordinate_tmp

    neighbors = np.eye(max_segments, dtype=np.bool)
    func = lambda a: segments[a[0], a[1]]

    for i in xrange(max_segments):
        segments_i = segments == i

        ft_feature[i] = blur_sm[segments_i].mean()

        center_color_feature[i] = center_color_map[segments_i].mean()

        edge_feature[i] = edge_img[segments_i].sum()

        # column
        c = coordinate_c[segments_i]
        d = np.unique(np.apply_along_axis(func, axis=1, arr=c[(c[:, 0] > 0) | (c[:, 1] > 0), :]))
        neighbors[i, d] = True
        # row
        c = coordinate_r[segments_i]
        d = np.unique(np.apply_along_axis(func, axis=1, arr=c[(c[:, 0] > 0) | (c[:, 1] > 0), :]))
        neighbors[i, d] = True

    # uniqueness plus
    w_ij = 1 - np.exp(-cdist(coordinate_segments_mean, coordinate_segments_mean) * sigma1)
    w_ij = w_ij / w_ij.sum(axis=1)[:, None]
    # mu_i_c = np.dot(wp_ij, img_segments_mean)
    uniqueness_plus = np.sum(cdist(img_segments_mean, img_segments_mean) * w_ij, axis=1)
    uniqueness_plus = np.array([uniqueness_plus]).T

    # distribution
    wc_ij = np.exp(-cdist(img_segments_mean, img_segments_mean) ** 2 / (2 * sigma2 ** 2))
    wc_ij = wc_ij / wc_ij.sum(axis=1)[:, None]
    mu_i = np.dot(wc_ij, coordinate_segments_mean)
    distribution = np.dot(wc_ij, np.linalg.norm(coordinate_segments_mean - mu_i, axis=1) ** 2)
    distribution = np.array([distribution]).T

    r = np.unique(region_labels, return_counts=True)
    size = r[1]*1.0/region_labels.size

    D = cdist(img_segments_mean, img_region_segments_mean)

    w_ij = 1 - np.exp(
        -cdist(coordinate_segments_mean, coordinate_region_segments_mean) * sigma3)
    w_ij = w_ij / w_ij.sum(axis=1)[:, None]

    region_conlor_contrast = np.sum(w_ij * D * size, axis=1)
    region_conlor_contrast = region_conlor_contrast[:, None]

    wd_ij = np.exp(-cdist(img_segments_mean, img_region_segments_mean) ** 2 / (2 * sigma4 ** 2))
    wd_ij = wd_ij / wd_ij.sum(axis=1)[:, None]
    mu_i = np.dot(wd_ij, coordinate_region_segments_mean)
    DR = np.sum(cdist(mu_i, coordinate_region_segments_mean) * wd_ij * size, axis=1)
    DR = DR[:, None]

    # local features
    local_img_segments_mean = np.zeros_like(img_segments_mean)
    local_coordinate_segments_mean = np.zeros_like(coordinate_segments_mean)
    local_ft_feature = np.zeros_like(ft_feature)
    local_edge_feature = np.zeros_like(edge_feature)
    for i in xrange(max_segments):
        neighbors[i, i] = False
        local_img_segments_mean[i, :] = np.mean(img_segments_mean[neighbors[i, :], :], axis=0)
        local_coordinate_segments_mean[i, :] = np.mean(coordinate_segments_mean[neighbors[i, :], :], axis=0)
        local_ft_feature[i, :] = np.mean(ft_feature[neighbors[i, :], :], axis=0)
        local_edge_feature[i, :] = np.mean(edge_feature[neighbors[i, :], :], axis=0)


    Aff = fp.manifold_ranking_aff(img_segments_mean_copy, segments, neighbors.copy())
    # top
    salt = np.zeros((max_segments, 1))
    salt[np.unique(segments[0, :])] = 1
    salt = normalize(np.dot(Aff, salt))
    # down
    sald = np.zeros((max_segments, 1))
    sald[np.unique(segments[segments.shape[0] - 1, :])] = 1
    sald = normalize(np.dot(Aff, sald))
    # left
    sall = np.zeros((max_segments, 1))
    sall[np.unique(segments[:, 0])] = 1
    sall = normalize(np.dot(Aff, sall))
    # right
    salr = np.zeros((max_segments, 1))
    salr[np.unique(segments[:, segments.shape[1] - 1])] = 1
    salr = normalize(np.dot(Aff, salr))

    # normalize features
    img_segments_mean[:, 0] = normalize(img_segments_mean[:, 0])
    img_segments_mean[:, 1] = normalize(img_segments_mean[:, 1])
    img_segments_mean[:, 2] = normalize(img_segments_mean[:, 2])
    coordinate_segments_mean[:, 0] = normalize(coordinate_segments_mean[:, 0])
    coordinate_segments_mean[:, 1] = normalize(coordinate_segments_mean[:, 1])
    border_feature = normalize(border_feature)
    center_color_feature = normalize(center_color_feature)
    ft_feature = normalize(ft_feature)
    edge_feature = normalize_zero(edge_feature)
    uniqueness_plus = normalize(uniqueness_plus)
    distribution = normalize(distribution)
    # region features
    region_conlor_contrast = normalize(region_conlor_contrast)
    DR = normalize(DR)
    # local features
    local_img_segments_mean[:, 0] = normalize(local_img_segments_mean[:, 0])
    local_img_segments_mean[:, 1] = normalize(local_img_segments_mean[:, 1])
    local_img_segments_mean[:, 2] = normalize(local_img_segments_mean[:, 2])
    local_coordinate_segments_mean[:, 0] = normalize(local_coordinate_segments_mean[:, 0])
    local_coordinate_segments_mean[:, 1] = normalize(local_coordinate_segments_mean[:, 1])
    local_ft_feature = normalize(local_ft_feature)
    local_edge_feature = normalize_zero(local_edge_feature)

    return np.concatenate((img_segments_mean, coordinate_segments_mean, border_feature, uniqueness_plus, distribution,
                           center_color_feature, ft_feature, edge_feature, region_conlor_contrast, DR,
                           local_img_segments_mean, local_coordinate_segments_mean, local_ft_feature,
                           local_edge_feature, salt, sald, sall, salr,
                           saliency_super_pixels[:, None]), axis=1), segments, neighbors


def extract_features_from_cache7(img_name, original_img_dir, cache_dir):
    '''
    add Manifold Ranking features, region Manifold Ranking features
    :param img_name:
    :param original_img_dir:
    :param cache_dir: general cache dir
    :return:
    '''
    if not os.path.exists(cache_dir):
        raise NameError("Not such cache dir!")
    region_features_dir = cache_dir + "_region"
    segments_dir = cache_dir + "_segments"
    region_labels_dir = cache_dir + "_region_labels"
    frame_info_dir = cache_dir + "_frame_info"
    img_npy_name = img_name.split('.')[0] + ".npy"

    normalize = lambda s: (s - s.min()) / (s.max() - s.min())
    normalize_zero = lambda s: (s - s.min()) / (s.max() - s.min() + 1)
    d_fun = lambda d_c, d_p: d_c / (5 * (1 + d_p))
    get_filter_kernel = lambda x, y: cv2.mulTransposed(cv2.getGaussianKernel(x, y), False)
    sigma1 = 10
    sigma2 = 21.10405218
    sigma3 = 50
    sigma4 = 23.3557741048

    img = io.imread(original_img_dir + os.sep + img_name)
    frame = np.load(frame_info_dir + os.sep + img_npy_name)
    feature = np.load(cache_dir + os.sep + img_npy_name)
    segments = np.load(segments_dir + os.sep + img_npy_name)
    region_feature = np.load(region_features_dir + os.sep + img_npy_name)
    region_labels = np.load(region_labels_dir + os.sep + img_npy_name)
    # remove frame
    img = img[frame[2]:frame[3], frame[4]:frame[5]]

    img_segments_mean = feature[:, 0:3]
    coordinate_segments_mean = feature[:, 3:5]
    saliency_super_pixels = feature[:, -1]
    img_region_segments_mean = region_feature[:, 0:3]
    coordinate_region_segments_mean = region_feature[:, 3:5]
    img_segments_mean_copy = img_segments_mean.copy()  # backup for manifold ranking feature

    img_lab = rgb2lab(img)
    max_segments = segments.max() + 1

    # create x,y feather
    shape = img.shape
    a = shape[0]
    b = shape[1]
    x_axis = np.linspace(0, b - 1, num=b)
    y_axis = np.linspace(0, a - 1, num=a)

    x_coordinate = np.tile(x_axis, (a, 1,))  # 创建X轴的坐标表
    y_coordinate = np.tile(y_axis, (b, 1,))  # 创建y轴的坐标表
    y_coordinate = np.transpose(y_coordinate)

    # border mask
    # border_mask = (x_coordinate < 16) | (x_coordinate > b - 17) | (y_coordinate < 16) | (y_coordinate > a - 17)
    # r = np.unique(segments[border_mask], return_counts=True)
    # border_feature = np.zeros((max_segments, 1))
    # border_feature[r[0], 0] = r[1]

    # FT feature
    blur_img_lab = cv2.filter2D(img_lab, -1, get_filter_kernel(5, 5))
    blur_lm = blur_img_lab[:, :, 0].mean()
    blur_am = blur_img_lab[:, :, 1].mean()
    blur_bm = blur_img_lab[:, :, 2].mean()
    blur_sm = np.sqrt((blur_img_lab[:, :, 0] - blur_lm) ** 2 + (blur_img_lab[:, :, 1] - blur_am) ** 2 + (
        blur_img_lab[:, :, 2] - blur_bm) ** 2)
    ft_feature = np.zeros((max_segments, 1))

    # color center feature
    w_sum = np.sum(blur_sm)
    x_center = np.sum(blur_sm * x_coordinate) / w_sum
    y_center = np.sum(blur_sm * y_coordinate) / w_sum
    center_color_map = np.exp(- (np.abs(x_coordinate - x_center) + np.abs(y_coordinate - y_center)) / 250)
    center_color_feature = np.zeros((max_segments, 1))

    # edge feature
    edge_img = grey_dilation(canny(cv2.filter2D(rgb2grey(img), -1, get_filter_kernel(10, 5))), size=(5, 5))
    edge_feature = np.zeros((max_segments, 1))

    # local element
    # left and right
    local_map1 = np.concatenate([np.diff(segments, axis=1), np.zeros([a, 1])], axis=1)
    local_map2 = np.fliplr(np.concatenate([np.diff(np.fliplr(segments), axis=1), np.zeros([a, 1])], axis=1))
    # up and down
    local_map3 = np.concatenate([np.diff(segments, axis=0), np.zeros([1, b])], axis=0)
    local_map4 = np.flipud(np.concatenate([np.diff(np.flipud(segments), axis=0), np.zeros([1, b])], axis=0))

    local_map1 = local_map1.astype(np.bool)
    local_map2 = local_map2.astype(np.bool)
    local_map3 = local_map3.astype(np.bool)
    local_map4 = local_map4.astype(np.bool)

    x_coordinate_map = x_coordinate.copy()
    x_coordinate_tmp = x_coordinate.copy()
    y_coordinate_map = y_coordinate.copy()
    y_coordinate_tmp = y_coordinate.copy()
    x_coordinate_map[local_map1] += 1
    x_coordinate_map[local_map2] -= 1
    y_coordinate_map[local_map3] += 1
    y_coordinate_map[local_map4] -= 1

    x_coordinate_map[~(local_map1 | local_map2)] = 0
    y_coordinate_tmp[~(local_map1 | local_map2)] = 0

    y_coordinate_map[~(local_map3 | local_map4)] = 0
    x_coordinate_tmp[~(local_map3 | local_map4)] = 0

    coordinate_r = np.zeros([a, b, 2], dtype=np.int32)
    coordinate_c = np.zeros([a, b, 2], dtype=np.int32)
    # column
    coordinate_c[:, :, 0] = y_coordinate_tmp
    coordinate_c[:, :, 1] = x_coordinate_map
    # row
    coordinate_r[:, :, 0] = y_coordinate_map
    coordinate_r[:, :, 1] = x_coordinate_tmp

    neighbors = np.eye(max_segments, dtype=np.bool)
    func = lambda a: segments[a[0], a[1]]

    for i in xrange(max_segments):
        segments_i = segments == i

        ft_feature[i] = blur_sm[segments_i].mean()

        center_color_feature[i] = center_color_map[segments_i].mean()

        edge_feature[i] = edge_img[segments_i].sum()

        # column
        c = coordinate_c[segments_i]
        d = np.unique(np.apply_along_axis(func, axis=1, arr=c[(c[:, 0] > 0) | (c[:, 1] > 0), :]))
        neighbors[i, d] = True
        # row
        c = coordinate_r[segments_i]
        d = np.unique(np.apply_along_axis(func, axis=1, arr=c[(c[:, 0] > 0) | (c[:, 1] > 0), :]))
        neighbors[i, d] = True

    # uniqueness plus
    w_ij = 1 - np.exp(-cdist(coordinate_segments_mean, coordinate_segments_mean) * sigma1)
    w_ij = w_ij / w_ij.sum(axis=1)[:, None]
    # mu_i_c = np.dot(wp_ij, img_segments_mean)
    uniqueness_plus = np.sum(cdist(img_segments_mean, img_segments_mean) * w_ij, axis=1)
    uniqueness_plus = np.array([uniqueness_plus]).T

    # distribution
    wc_ij = np.exp(-cdist(img_segments_mean, img_segments_mean) ** 2 / (2 * sigma2 ** 2))
    wc_ij = wc_ij / wc_ij.sum(axis=1)[:, None]
    mu_i = np.dot(wc_ij, coordinate_segments_mean)
    distribution = np.dot(wc_ij, np.linalg.norm(coordinate_segments_mean - mu_i, axis=1) ** 2)
    distribution = np.array([distribution]).T

    r = np.unique(region_labels, return_counts=True)
    size = r[1]*1.0/region_labels.size

    D = cdist(img_segments_mean, img_region_segments_mean)

    w_ij = 1 - np.exp(
        -cdist(coordinate_segments_mean, coordinate_region_segments_mean) * sigma3)
    w_ij = w_ij / w_ij.sum(axis=1)[:, None]

    region_conlor_contrast = np.sum(w_ij * D * size, axis=1)
    region_conlor_contrast = region_conlor_contrast[:, None]

    wd_ij = np.exp(-cdist(img_segments_mean, img_region_segments_mean) ** 2 / (2 * sigma4 ** 2))
    wd_ij = wd_ij / wd_ij.sum(axis=1)[:, None]
    mu_i = np.dot(wd_ij, coordinate_region_segments_mean)
    DR = np.sum(cdist(mu_i, coordinate_region_segments_mean) * wd_ij * size, axis=1)
    DR = DR[:, None]

    # local features
    local_img_segments_mean = np.zeros_like(img_segments_mean)
    local_coordinate_segments_mean = np.zeros_like(coordinate_segments_mean)
    local_ft_feature = np.zeros_like(ft_feature)
    local_edge_feature = np.zeros_like(edge_feature)
    for i in xrange(max_segments):
        neighbors[i, i] = False
        local_img_segments_mean[i, :] = np.mean(img_segments_mean[neighbors[i, :], :], axis=0)
        local_coordinate_segments_mean[i, :] = np.mean(coordinate_segments_mean[neighbors[i, :], :], axis=0)
        local_ft_feature[i, :] = np.mean(ft_feature[neighbors[i, :], :], axis=0)
        local_edge_feature[i, :] = np.mean(edge_feature[neighbors[i, :], :], axis=0)


    Aff = fp.manifold_ranking_aff(img_segments_mean_copy, segments, neighbors.copy())
    # top
    salt = np.zeros((max_segments, 1))
    salt[np.unique(segments[0, :])] = 1
    salt = normalize(np.dot(Aff, salt))
    # down
    sald = np.zeros((max_segments, 1))
    sald[np.unique(segments[segments.shape[0] - 1, :])] = 1
    sald = normalize(np.dot(Aff, sald))
    # left
    sall = np.zeros((max_segments, 1))
    sall[np.unique(segments[:, 0])] = 1
    sall = normalize(np.dot(Aff, sall))
    # right
    salr = np.zeros((max_segments, 1))
    salr[np.unique(segments[:, segments.shape[1] - 1])] = 1
    salr = normalize(np.dot(Aff, salr))

    #region manifold ranking features
    region_mr_feature = []
    for i in xrange(max(region_labels) + 1):
        mr_feature = np.zeros((max_segments, 1))
        mr_feature[region_labels == i] = 1
        mr_feature = normalize(np.dot(Aff, mr_feature))
        region_mr_feature.append(mr_feature)

    # normalize features
    img_segments_mean[:, 0] = normalize(img_segments_mean[:, 0])
    img_segments_mean[:, 1] = normalize(img_segments_mean[:, 1])
    img_segments_mean[:, 2] = normalize(img_segments_mean[:, 2])
    coordinate_segments_mean[:, 0] = normalize(coordinate_segments_mean[:, 0])
    coordinate_segments_mean[:, 1] = normalize(coordinate_segments_mean[:, 1])
    center_color_feature = normalize(center_color_feature)
    ft_feature = normalize(ft_feature)
    edge_feature = normalize_zero(edge_feature)
    uniqueness_plus = normalize(uniqueness_plus)
    distribution = normalize(distribution)
    # region features
    region_conlor_contrast = normalize(region_conlor_contrast)
    DR = normalize(DR)
    # local features
    local_img_segments_mean[:, 0] = normalize(local_img_segments_mean[:, 0])
    local_img_segments_mean[:, 1] = normalize(local_img_segments_mean[:, 1])
    local_img_segments_mean[:, 2] = normalize(local_img_segments_mean[:, 2])
    local_coordinate_segments_mean[:, 0] = normalize(local_coordinate_segments_mean[:, 0])
    local_coordinate_segments_mean[:, 1] = normalize(local_coordinate_segments_mean[:, 1])
    local_ft_feature = normalize(local_ft_feature)
    local_edge_feature = normalize_zero(local_edge_feature)

    return np.concatenate([img_segments_mean, coordinate_segments_mean, uniqueness_plus, distribution,
                           center_color_feature, ft_feature, edge_feature, region_conlor_contrast, DR,
                           local_img_segments_mean, local_coordinate_segments_mean, local_ft_feature,
                           local_edge_feature, salt, sald, sall, salr] + region_mr_feature +
                           [saliency_super_pixels[:, None]], axis=1), segments, neighbors


def region_segment(img, segments, th=0.001):
    g = graph.rag_mean_color(img, segments, mode="similarity")
    normalize_segments = graph.cut_normalized(segments, g, thresh=th)

    n_max = normalize_segments.max() + 1
    segment_list = []
    labels = np.zeros(segments.max() + 1, dtype=np.int32)
    for i in xrange(n_max):
        segments_i = normalize_segments == i
        if segments_i.sum() != 0:
            segment_list.append(segments_i)
            labels[np.unique(segments[segments_i])] = len(segment_list) - 1

    segments_region = np.zeros_like(segments)
    for i, segments_i in enumerate(segment_list):
        segments_region[segments_i] = i

    return segments_region, labels


def save_features(original_img_dir, binary_img_dir, cache_dir):
    if (not os.path.exists(original_img_dir)) and not (os.path.exists(binary_img_dir)):
        raise NameError("Path does not exits, check out!")
    # check out features_dir
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    segments_dir = cache_dir + "_segments"
    if not os.path.exists(segments_dir):
        os.mkdir(segments_dir)
    img_lab_dir = cache_dir + "_img_lab"
    if not os.path.exists(img_lab_dir):
        os.mkdir(img_lab_dir)
    segments_mean_dir = cache_dir + "_segments_mean"
    if not os.path.exists(segments_mean_dir):
        os.mkdir(segments_mean_dir)

    list_features_dir = os.listdir(original_img_dir)
    list_features_dir = filter(lambda f: os.path.splitext(f)[1] == '.jpg', list_features_dir)

    for f in list_features_dir:
        img_path_name = original_img_dir + os.sep + f
        binary_img_path_name = binary_img_dir + os.sep + os.path.splitext(f)[0] + '.bmp'
        img = io.imread(img_path_name)
        binary_img = io.imread(binary_img_path_name)
        features, segments, segments_mean, img_lab = extract_features(img, binary_img[:, :] > 0)
        np.save(cache_dir + os.sep + os.path.splitext(f)[0] + '.npy', features)
        np.save(segments_dir + os.sep + os.path.splitext(f)[0] + '.npy', segments)
        np.save(segments_mean_dir + os.sep + os.path.splitext(f)[0] + '.npy', segments_mean)
        np.save(img_lab_dir + os.sep + os.path.splitext(f)[0] + '.npy', img_lab)


def save_general_features(original_img_dir, binary_img_dir, cache_dir):
    if (not os.path.exists(original_img_dir)) and not (os.path.exists(binary_img_dir)):
        raise NameError("Path does not exits, check out!")
    # check out features_dir
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    region_features_dir = cache_dir + "_region"
    if not os.path.exists(region_features_dir):
        os.mkdir(region_features_dir)
    segments_dir = cache_dir + "_segments"
    if not os.path.exists(segments_dir):
        os.mkdir(segments_dir)
    region_segments_dir = cache_dir + "_region_segments"
    if not os.path.exists(region_segments_dir):
        os.mkdir(region_segments_dir)
    region_labels_dir = cache_dir + "_region_labels"
    if not os.path.exists(region_labels_dir):
        os.mkdir(region_labels_dir)

    list_features_dir = os.listdir(original_img_dir)
    list_features_dir = filter(lambda f: os.path.splitext(f)[1] == '.jpg', list_features_dir)

    for f in list_features_dir:
        img_path_name = original_img_dir + os.sep + f
        binary_img_path_name = binary_img_dir + os.sep + os.path.splitext(f)[0] + '.bmp'
        img = io.imread(img_path_name)
        binary_img = io.imread(binary_img_path_name)
        features, region_features, segments, region_segments, labels = extract_general_features(img, binary_img[:, :] > 0)
        np.save(cache_dir + os.sep + os.path.splitext(f)[0] + '.npy', features.astype(np.float32))
        np.save(region_features_dir + os.sep + os.path.splitext(f)[0] + '.npy', region_features.astype(np.float32))
        np.save(segments_dir + os.sep + os.path.splitext(f)[0] + '.npy', segments.astype(np.int16))
        np.save(region_segments_dir + os.sep + os.path.splitext(f)[0] + '.npy', region_segments.astype(np.int16))
        np.save(region_labels_dir + os.sep + os.path.splitext(f)[0] + '.npy', labels.astype(np.int16))


def preprocess_binary_image(binary_image):
    '''
    this function deals with binary image array
    :param binary_image: binary image array, 2-d or 3-d array
    :return: binary array, 2-d array
    '''
    # arr = None
    if len(binary_image.shape) > 2:
        arr = color.rgb2gray(binary_image)
    else:
        arr = binary_image
    return arr > (arr.max() / 2)


def preprocess_binary_image_frame(binary_image, frame):
    '''
    this function deals with binary image array
    :param binary_image:
    :param frame: frame array
    :return:
    '''
    # arr = None
    if len(binary_image.shape) > 2:
        arr = color.rgb2gray(binary_image)
    else:
        arr = binary_image
    return (arr > (arr.max() / 2))[frame[2]:frame[3], frame[4]:frame[5]]


def save_general_features_kmeans(original_img_dir, binary_img_dir, cache_dir, segments_number=300, original_img_ext='jpg', binary_img_ext='bmp'):
    if (not os.path.exists(original_img_dir)) and not (os.path.exists(binary_img_dir)):
        raise NameError("Path does not exits, check out!")
    # check out features_dir
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    region_features_dir = cache_dir + "_region"
    if not os.path.exists(region_features_dir):
        os.mkdir(region_features_dir)
    segments_dir = cache_dir + "_segments"
    if not os.path.exists(segments_dir):
        os.mkdir(segments_dir)
    region_segments_dir = cache_dir + "_region_segments"
    if not os.path.exists(region_segments_dir):
        os.mkdir(region_segments_dir)
    region_labels_dir = cache_dir + "_region_labels"
    if not os.path.exists(region_labels_dir):
        os.mkdir(region_labels_dir)

    list_features_dir = os.listdir(original_img_dir)
    list_features_dir = filter(lambda f: f.split('.')[-1] == original_img_ext, list_features_dir)

    for f in list_features_dir:
        img_path_name = original_img_dir + os.sep + f
        binary_img_path_name = binary_img_dir + os.sep + os.path.splitext(f)[0] + '.' + binary_img_ext
        img = io.imread(img_path_name)
        binary_arr = preprocess_binary_image(io.imread(binary_img_path_name))
        features, region_features, segments, region_segments, labels = extract_general_features_kmeans(img, binary_arr, segments_number=segments_number)
        np.save(cache_dir + os.sep + os.path.splitext(f)[0] + '.npy', features.astype(np.float32))
        np.save(region_features_dir + os.sep + os.path.splitext(f)[0] + '.npy', region_features.astype(np.float32))
        np.save(segments_dir + os.sep + os.path.splitext(f)[0] + '.npy', segments.astype(np.int16))
        np.save(region_segments_dir + os.sep + os.path.splitext(f)[0] + '.npy', region_segments.astype(np.int16))
        np.save(region_labels_dir + os.sep + os.path.splitext(f)[0] + '.npy', labels.astype(np.int16))


def save_general_features_kmeans2(original_img_dir, binary_img_dir, cache_dir, segments_number=300, original_img_ext='jpg', binary_img_ext='bmp'):
    '''
    this function add remove_frame function, and save the frame info to general cache dir
    :param original_img_dir:
    :param binary_img_dir:
    :param cache_dir:
    :param segments_number:
    :param original_img_ext:
    :param binary_img_ext:
    :return:
    '''
    if (not os.path.exists(original_img_dir)) and not (os.path.exists(binary_img_dir)):
        raise NameError("Path does not exits, check out!")
    # check out features_dir
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    region_features_dir = cache_dir + "_region"
    if not os.path.exists(region_features_dir):
        os.mkdir(region_features_dir)
    segments_dir = cache_dir + "_segments"
    if not os.path.exists(segments_dir):
        os.mkdir(segments_dir)
    region_segments_dir = cache_dir + "_region_segments"
    if not os.path.exists(region_segments_dir):
        os.mkdir(region_segments_dir)
    region_labels_dir = cache_dir + "_region_labels"
    if not os.path.exists(region_labels_dir):
        os.mkdir(region_labels_dir)
    frame_info_dir = cache_dir + "_frame_info"
    if not os.path.exists(frame_info_dir):
        os.mkdir(frame_info_dir)

    list_features_dir = os.listdir(original_img_dir)
    list_features_dir = filter(lambda f: f.split('.')[-1] == original_img_ext, list_features_dir)

    for f in list_features_dir:
        img_path_name = original_img_dir + os.sep + f
        binary_img_path_name = binary_img_dir + os.sep + os.path.splitext(f)[0] + '.' + binary_img_ext
        img = io.imread(img_path_name)
        img, frame_info = remove_frame(img)
        binary_arr = preprocess_binary_image_frame(io.imread(binary_img_path_name), frame_info)
        features, region_features, segments, region_segments, labels = extract_general_features_kmeans(img, binary_arr, segments_number=segments_number)
        np.save(cache_dir + os.sep + os.path.splitext(f)[0] + '.npy', features.astype(np.float32))
        np.save(frame_info_dir + os.sep + os.path.splitext(f)[0] + '.npy', frame_info.astype(np.uint8))
        np.save(region_features_dir + os.sep + os.path.splitext(f)[0] + '.npy', region_features.astype(np.float32))
        np.save(segments_dir + os.sep + os.path.splitext(f)[0] + '.npy', segments.astype(np.int16))
        np.save(region_segments_dir + os.sep + os.path.splitext(f)[0] + '.npy', region_segments.astype(np.int16))
        np.save(region_labels_dir + os.sep + os.path.splitext(f)[0] + '.npy', labels.astype(np.int16))


def save_general_features_multiprocess(pics, original_img_dir, binary_img_dir, cache_dir, segments_number=300, original_img_ext='jpg', binary_img_ext='bmp'):
    '''
    use for multiprocess
    :param pic: list of the pictures' name
    :param original_img_dir:
    :param binary_img_dir:
    :param cache_dir:
    :param segments_number:
    :param original_img_ext:
    :param binary_img_ext:
    :return:
    '''
    if (not os.path.exists(original_img_dir)) and not (os.path.exists(binary_img_dir)):
        raise NameError("Path does not exits, check out!")
    # check out features_dir
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    region_features_dir = cache_dir + "_region"
    if not os.path.exists(region_features_dir):
        os.mkdir(region_features_dir)
    segments_dir = cache_dir + "_segments"
    if not os.path.exists(segments_dir):
        os.mkdir(segments_dir)
    region_segments_dir = cache_dir + "_region_segments"
    if not os.path.exists(region_segments_dir):
        os.mkdir(region_segments_dir)
    region_labels_dir = cache_dir + "_region_labels"
    if not os.path.exists(region_labels_dir):
        os.mkdir(region_labels_dir)
    frame_info_dir = cache_dir + "_frame_info"
    if not os.path.exists(frame_info_dir):
        os.mkdir(frame_info_dir)

    list_features_dir = filter(lambda f: f.split('.')[-1] == original_img_ext, pics)

    for f in list_features_dir:
        img_path_name = original_img_dir + os.sep + f
        binary_img_path_name = binary_img_dir + os.sep + os.path.splitext(f)[0] + '.' + binary_img_ext
        img = io.imread(img_path_name)
        img, frame_info = remove_frame(img)
        binary_arr = preprocess_binary_image_frame(io.imread(binary_img_path_name), frame_info)
        features, region_features, segments, region_segments, labels = extract_general_features_kmeans(img, binary_arr, segments_number=segments_number)
        np.save(cache_dir + os.sep + os.path.splitext(f)[0] + '.npy', features.astype(np.float32))
        np.save(frame_info_dir + os.sep + os.path.splitext(f)[0] + '.npy', frame_info.astype(np.int16))
        np.save(region_features_dir + os.sep + os.path.splitext(f)[0] + '.npy', region_features.astype(np.float32))
        np.save(segments_dir + os.sep + os.path.splitext(f)[0] + '.npy', segments.astype(np.int16))
        np.save(region_segments_dir + os.sep + os.path.splitext(f)[0] + '.npy', region_segments.astype(np.int16))
        np.save(region_labels_dir + os.sep + os.path.splitext(f)[0] + '.npy', labels.astype(np.int16))


def save_general_features_kmeans_rgb(original_img_dir, binary_img_dir, cache_dir, segments_number=300):
    if (not os.path.exists(original_img_dir)) and not (os.path.exists(binary_img_dir)):
        raise NameError("Path does not exits, check out!")
    # check out features_dir
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    region_features_dir = cache_dir + "_region"
    if not os.path.exists(region_features_dir):
        os.mkdir(region_features_dir)
    segments_dir = cache_dir + "_segments"
    if not os.path.exists(segments_dir):
        os.mkdir(segments_dir)
    region_segments_dir = cache_dir + "_region_segments"
    if not os.path.exists(region_segments_dir):
        os.mkdir(region_segments_dir)
    region_labels_dir = cache_dir + "_region_labels"
    if not os.path.exists(region_labels_dir):
        os.mkdir(region_labels_dir)

    list_features_dir = os.listdir(original_img_dir)
    list_features_dir = filter(lambda f: os.path.splitext(f)[1] == '.jpg', list_features_dir)

    for f in list_features_dir:
        img_path_name = original_img_dir + os.sep + f
        binary_img_path_name = binary_img_dir + os.sep + os.path.splitext(f)[0] + '.bmp'
        img = io.imread(img_path_name)
        binary_img = io.imread(binary_img_path_name)
        features, region_features, segments, region_segments, labels = extract_general_features_kmeans_rgb(img, binary_img[:, :] > 0, segments_number=segments_number)
        np.save(cache_dir + os.sep + os.path.splitext(f)[0] + '.npy', features.astype(np.float32))
        np.save(region_features_dir + os.sep + os.path.splitext(f)[0] + '.npy', region_features.astype(np.float32))
        np.save(segments_dir + os.sep + os.path.splitext(f)[0] + '.npy', segments.astype(np.int16))
        np.save(region_segments_dir + os.sep + os.path.splitext(f)[0] + '.npy', region_segments.astype(np.int16))
        np.save(region_labels_dir + os.sep + os.path.splitext(f)[0] + '.npy', labels.astype(np.int16))


def save_features_from_general_cache(original_img_dir, general_cache_dir, cache_dir):
    if not os.path.exists(general_cache_dir):
        raise NameError("general cache dir not exits!")

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    segments_dir = cache_dir + "_segments"
    if not os.path.exists(segments_dir):
        os.mkdir(segments_dir)

    list_dir = filter(lambda s: s.split(".")[-1] == "jpg", os.listdir(original_img_dir))
    for f in list_dir:
        features, segments = extract_features_from_cache(f, original_img_dir, general_cache_dir)
        np.save(cache_dir + os.sep + os.path.splitext(f)[0] + '.npy', features)
        np.save(segments_dir + os.sep + os.path.splitext(f)[0] + '.npy', segments.astype(np.int16))


def save_features_from_general_cache2(original_img_dir, general_cache_dir, cache_dir):
    if not os.path.exists(general_cache_dir):
        raise NameError("general cache dir not exits!")

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    segments_dir = cache_dir + "_segments"
    if not os.path.exists(segments_dir):
        os.mkdir(segments_dir)

    list_dir = filter(lambda s: s.split(".")[-1] == "jpg", os.listdir(original_img_dir))
    for f in list_dir:
        features, segments = extract_features_from_cache2(f, original_img_dir, general_cache_dir)
        np.save(cache_dir + os.sep + os.path.splitext(f)[0] + '.npy', features)
        np.save(segments_dir + os.sep + os.path.splitext(f)[0] + '.npy', segments.astype(np.int16))


def save_features_from_general_cache3(original_img_dir, general_cache_dir, cache_dir, original_img_ext='jpg'):
    if not os.path.exists(general_cache_dir):
        raise NameError("general cache dir not exits!")

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    segments_dir = cache_dir + "_segments"
    if not os.path.exists(segments_dir):
        os.mkdir(segments_dir)

    list_dir = filter(lambda s: s.split(".")[-1] == original_img_ext, os.listdir(original_img_dir))
    for f in list_dir:
        features, segments = extract_features_from_cache3(f, original_img_dir, general_cache_dir)
        np.save(cache_dir + os.sep + os.path.splitext(f)[0] + '.npy', features)
        np.save(segments_dir + os.sep + os.path.splitext(f)[0] + '.npy', segments.astype(np.int16))


def save_features_from_general_cache4(original_img_dir, general_cache_dir, cache_dir, original_img_ext='jpg'):
    if not os.path.exists(general_cache_dir):
        raise NameError("general cache dir not exits!")

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    segments_dir = cache_dir + "_segments"
    if not os.path.exists(segments_dir):
        os.mkdir(segments_dir)
    neighbor_dir = cache_dir + "_neighbor"
    if not os.path.exists(neighbor_dir):
        os.mkdir(neighbor_dir)

    list_dir = filter(lambda s: s.split(".")[-1] == original_img_ext, os.listdir(original_img_dir))
    for f in list_dir:
        features, segments, neighbor = extract_features_from_cache4(f, original_img_dir, general_cache_dir)
        np.save(cache_dir + os.sep + os.path.splitext(f)[0] + '.npy', features)
        np.save(segments_dir + os.sep + os.path.splitext(f)[0] + '.npy', segments.astype(np.int16))
        np.save(neighbor_dir + os.sep + os.path.splitext(f)[0] + '.npy', neighbor)


def save_features_from_general_cache5(original_img_dir, general_cache_dir, cache_dir, original_img_ext='jpg'):
    if not os.path.exists(general_cache_dir):
        raise NameError("general cache dir not exits!")

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    segments_dir = cache_dir + "_segments"
    if not os.path.exists(segments_dir):
        os.mkdir(segments_dir)
    neighbor_dir = cache_dir + "_neighbor"
    if not os.path.exists(neighbor_dir):
        os.mkdir(neighbor_dir)

    list_dir = filter(lambda s: s.split(".")[-1] == original_img_ext, os.listdir(original_img_dir))
    for f in list_dir:
        features, segments, neighbor = extract_features_from_cache5(f, original_img_dir, general_cache_dir)
        np.save(cache_dir + os.sep + os.path.splitext(f)[0] + '.npy', features)
        np.save(segments_dir + os.sep + os.path.splitext(f)[0] + '.npy', segments.astype(np.int16))
        np.save(neighbor_dir + os.sep + os.path.splitext(f)[0] + '.npy', neighbor)


def save_features_from_general_cache_multiprocess(pics, original_img_dir, general_cache_dir, cache_dir, original_img_ext='jpg'):
    if not os.path.exists(general_cache_dir):
        raise NameError("general cache dir not exits!")

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    segments_dir = cache_dir + "_segments"
    if not os.path.exists(segments_dir):
        os.mkdir(segments_dir)
    neighbor_dir = cache_dir + "_neighbor"
    if not os.path.exists(neighbor_dir):
        os.mkdir(neighbor_dir)

    list_dir = filter(lambda s: s.split(".")[-1] == original_img_ext, pics)
    for f in list_dir:
        features, segments, neighbor = extract_features_from_cache7(f, original_img_dir, general_cache_dir)
        np.save(cache_dir + os.sep + os.path.splitext(f)[0] + '.npy', features)
        np.save(segments_dir + os.sep + os.path.splitext(f)[0] + '.npy', segments.astype(np.int16))
        np.save(neighbor_dir + os.sep + os.path.splitext(f)[0] + '.npy', neighbor)


def find_abnormal_result(cache_dir):
    region_labels_dir = cache_dir + "_region_labels"
    if not os.path.exists(region_labels_dir):
        raise NameError(region_labels_dir + " does not exits!")

    list_dir = filter(lambda s: s.split(".")[-1] == "npy", os.listdir(region_labels_dir))
    abnormal_list = []
    for f in list_dir:
        labels = np.load(region_labels_dir + os.sep + f)
        if(np.unique(labels).size <= 5):
            abnormal_list.append(f)

    return abnormal_list


def recreate_cache_file(name_list, original_img_dir, binary_img_dir, cache_dir):
    if (not os.path.exists(original_img_dir)) and not (os.path.exists(binary_img_dir)):
        raise NameError("Path does not exits, check out!")
    # check out features_dir
    region_features_dir = cache_dir + "_region"
    segments_dir = cache_dir + "_segments"
    region_segments_dir = cache_dir + "_region_segments"
    region_labels_dir = cache_dir + "_region_labels"

    for f in name_list:
        img_path_name = original_img_dir + os.sep + f
        binary_img_path_name = binary_img_dir + os.sep + os.path.splitext(f)[0] + '.bmp'
        img = io.imread(img_path_name)
        binary_img = io.imread(binary_img_path_name)
        th = 0.002
        features, region_features, segments, region_segments, labels = extract_general_features(img, binary_img[:, :] > 0, th=th)
        while np.unique(labels).size <= 5:
            th *= 2
            features, region_features, segments, region_segments, labels = extract_general_features(img, binary_img[:, :] > 0, th=th)
        np.save(cache_dir + os.sep + os.path.splitext(f)[0] + '.npy', features.astype(np.float32))
        np.save(region_features_dir + os.sep + os.path.splitext(f)[0] + '.npy', region_features.astype(np.float32))
        np.save(segments_dir + os.sep + os.path.splitext(f)[0] + '.npy', segments.astype(np.int16))
        np.save(region_segments_dir + os.sep + os.path.splitext(f)[0] + '.npy', region_segments.astype(np.int16))
        np.save(region_labels_dir + os.sep + os.path.splitext(f)[0] + '.npy', labels.astype(np.int16))


def remove_frame(img):
    edge_map = canny(color.rgb2grey(img))
    m, n = edge_map.shape
    flagt = 0; flagd = 0; flagl = 0; flagr = 0;
    t = 1; d = 1; l = 1; r = 1;  # represent the width of the frame
    threshold = 0.6

    for i in xrange(30):
        pbt = edge_map[i, :].mean()
        pbd = edge_map[m - i - 1, :].mean()
        pbl = edge_map[:, i].mean()
        pbr = edge_map[:, n - i - 1].mean()
        if pbt > threshold: flagt = 1; t = i + 1;
        if pbd > threshold: flagd = 1; d = i + 1;
        if pbl > threshold: flagl = 1; l = i + 1;
        if pbr > threshold: flagr = 1; r = i + 1;

    flagm = flagt + flagd + flagl + flagr
    # we assume that there exists a frame when one more lines parallel to the image side are detected
    if flagm > 1:
        max_width = max([t, d, l, r])
        if t == 1: t = max_width
        if d == 1: d = max_width
        if l == 1: l = max_width
        if r == 1: r = max_width
        r_img = img[t:(m - d), l:(n - r)]
        f = [m, n, t, (m - d), l, (n - r)]
    else:
        r_img = img
        f = [m, n, 0, m, 0, n]

    return r_img, np.array(f)


def feature_to_image(feature, segments, frame=None, enhance=False, binary=False):
    normalize = lambda s: (s - s.min()) / (s.max() - s.min())

    max_segments = segments.max() + 1
    if max_segments != feature.size:
        raise NameError("feature size does not match segments.")

    if enhance and (feature > 0).sum() > 5 and (feature == feature.max()).sum() < 5:
        mask = feature < 0
        while np.count_nonzero(feature == feature.max()) < 10:
            mask = mask | (feature == feature.max())
            feature[mask] = 0
            feature[mask] = feature.max()
        feature = normalize(feature)

    if binary:
        feature = feature > feature.mean()

    feature_img = np.zeros_like(segments, dtype=np.float64)

    for i in xrange(max_segments):
        segments_i = segments == i
        feature_img[segments_i] = feature[i]

    if frame is None:
        return feature_img
    else:
        saliency_img = np.zeros(frame[0:2], np.float64)
        saliency_img[frame[2]:frame[3], frame[4]:frame[5]] = feature_img
        return saliency_img


def predicts_to_images(predicts_dir, images_dir, cache_dir):
    '''
    this function is used for pridicts to images
    :param features_dir: pridicts
    :param images_dir:
    :param cache_dir:
    :return:
    '''
    if not os.path.exists(predicts_dir):
        raise NameError("feature dir not exits!")
    segments_dir = cache_dir + "_segments"
    if not os.path.exists(segments_dir):
        raise NameError("segments_dir not exits!")

    if not os.path.exists(images_dir):
        os.mkdir(images_dir)

    list_dir = filter(lambda s: s.split('.')[-1] == 'npy', os.listdir(predicts_dir))
    for f in list_dir:
        predict = np.load(predicts_dir + os.sep + f)
        segments = np.load(segments_dir + os.sep + f)
        img = feature_to_image(predict, segments)
        io.imsave(images_dir + os.sep + f.split('.')[0] + '.png', img)



