# -*- coding: utf-8 -*-

from multiprocessing import Process, freeze_support, Value
from scipy.spatial.distance import cdist
from skimage.segmentation import slic
from skimage.color import rgb2lab, rgb2grey
from skimage.feature import canny
from scipy.ndimage.morphology import grey_dilation
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from skimage import io
import numpy as np
import cv2
import os
import random
import pickle
from time import time
from train_saliency import region_segment, extract_region_features, up_sample
from save_features import feature_to_image
from path import saliency_img_out_dir, saliency_feature_out_dir

# features_path = os.getcwd() + os.sep + "features"
# test_out_path = os.getcwd() + os.sep + "tests"
# original_img_dir = r"G:\Project\paper2\other_image\MSRA-1000_images"
# binary_img_dir = r"G:\Project\paper2\other_image\binarymasks"
# saliency_img_out_dir = r"G:\Project\paper2\out"
# cache_out_dir = r'G:\Project\paper2\cache\cache_features_local_surround_300\feature_cache'


def extract_features(img, binary_masks=None, segments_number=600):
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
    segments = slic(img_lab, n_segments=segments_number, sigma=5, convert2lab=False)
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
                               saliency_super_pixels), axis=1), segments
    else:
        return np.concatenate((img_segments_mean, coordinate_segments_mean, uniqueness_plus, distribution,
                               center_color_feature, ft_feature, ca_feature, size_feature, edge_feature,
                               region_conlor_contrast[:, None], DR[:, None],
                               ), axis=1), segments


def get_features(original_img_dir, binary_img_dir, sample_picture_number=20, segments_number=300, pic_list=None):
    if (not os.path.exists(original_img_dir)) and not (os.path.exists(binary_img_dir)):
        raise NameError("Path does not exits, check out!")

    if pic_list is None:
        list_features_dir = os.listdir(original_img_dir)
        list_features_dir = filter(lambda f: os.path.splitext(f)[1] == '.jpg', list_features_dir)
        random.shuffle(list_features_dir)
        # choose first 16 picture as feature files
        features_img = list_features_dir[0:sample_picture_number]
    features_img = pic_list
    features_list = []
    # original scale
    for f in features_img:
        img_path_name = original_img_dir + os.sep + f
        binary_img_path_name = binary_img_dir + os.sep + os.path.splitext(f)[0] + '.bmp'
        img = io.imread(img_path_name)
        binary_img = io.imread(binary_img_path_name)
        features, _ = extract_features(img, binary_img[:, :] > 0, segments_number=segments_number)
        features_list.append(features)
    final_array = np.concatenate(features_list, axis=0)
    feature = final_array[:, 0:(final_array.shape[1] - 1)]
    label = final_array[:, -1]
    return feature, label


def get_features_use_cache(cache_dir, pic_list):
    if not os.path.exists(cache_dir):
        raise NameError("cache dir does not exist!")

    features_list = []
    for f in pic_list:
        npy_name = f.split('.')[0] + ".npy"
        features = np.load(cache_dir + os.sep + npy_name)
        features_list.append(features)
    final_array = np.concatenate(features_list, axis=0)
    feature = final_array[:, 0:(final_array.shape[1] - 1)]
    label = final_array[:, -1]
    return feature, label


def load_features(features_path, name_list=None, features_dict=None):
    if not os.path.exists(features_path):
        raise NameError("Path does not exits, check out!")

    if name_list is None:
        features_name_list = os.listdir(features_path)
        features_name_list = filter(lambda f: os.path.splitext(f)[1] == '.npy', features_name_list)
    else:
        name_list = filter(lambda s: os.path.splitext(s)[1] == '.npy', name_list)
        if name_list == []:
            raise NameError("extension must be .npy!")
        features_name_list = name_list

    features_list = []
    if features_dict is None:
        for f in features_name_list:
            f = features_path + os.sep + f
            features_list.append(np.load(f))
    else:
        for f in features_name_list:
            name = os.path.splitext(f)[0]
            features_list.append(features_dict[name])

    final_array = np.concatenate(features_list, axis=0)

    feature = final_array[:, 0:(final_array.shape[1] - 1)]
    label = final_array[:, -1]
    return feature, label


def load_features_as_dict(features_path):
    if not os.path.exists(features_path):
        raise NameError("Path does not exits, check out!")

    features_name_list = os.listdir(features_path)
    features_name_list = filter(lambda f: os.path.splitext(f)[1] == '.npy', features_name_list)
    features_dict = {}
    for f in features_name_list:
        name = os.path.splitext(f)[0]
        f = features_path + os.sep + f
        features_dict[name] = (np.load(f))

    return features_dict


def train_features(feature, label, C=1):
    clf = svm.SVC(C=C, probability=True)
    clf.fit(feature, label)
    return clf


def train_features_lg(feature, label, C=1):
    clf = LogisticRegression(C=C)
    clf.fit(feature, label)
    return clf


def test_image(clf, img):
    features, segments = extract_features(img)
    predict_result = clf.predict_proba(features)
    saliency_img = np.zeros(img.shape[0:2], dtype=np.float64)
    max_segments = segments.max() + 1
    for i in xrange(max_segments):
        saliency_img[segments == i] = predict_result[i, 1]
    return saliency_img


def test_image_use_cache(clf, img_name, cache_dir):
    segments_dir = cache_dir + "_segments"
    if (not os.path.exists(cache_dir)) and (not os.path.exists(segments_dir)):
        raise NameError("path does not exist!")
    features = np.load(cache_dir + os.sep + img_name.split('.')[0] + '.npy')
    feature = features[:, 0:(features.shape[1] - 1)]
    segments = np.load(segments_dir + os.sep + img_name.split('.')[0] + '.npy')
    predict_result = clf.predict_proba(feature)
    saliency_img = np.zeros(segments.shape[0:2], dtype=np.float64)
    max_segments = int(segments.max()) + 1
    for i in xrange(max_segments):
        saliency_img[segments == i] = predict_result[i, 1]
    return saliency_img


def test_feature_use_cache(clf, img_name, cache_dir):
    segments_dir = cache_dir + "_segments"
    if (not os.path.exists(cache_dir)) and (not os.path.exists(segments_dir)):
        raise NameError("path does not exist!")
    features = np.load(cache_dir + os.sep + img_name.split('.')[0] + '.npy')
    feature = features[:, 0:(features.shape[1] - 1)]
    predict_result = clf.predict_proba(feature)
    return predict_result[:, 1]


def test_image_use_region(clf, img_name, cache_dir, original_img_dir):
    normalize = lambda s: (s - s.min()) / (s.max() - s.min())

    segments_dir = cache_dir + "_segments"
    if (not os.path.exists(cache_dir)) and (not os.path.exists(segments_dir)):
        raise NameError("path does not exist!")
    features = np.load(cache_dir + os.sep + img_name.split('.')[0] + '.npy')
    feature = features[:, 0:(features.shape[1] - 1)]
    segments = np.load(segments_dir + os.sep + img_name.split('.')[0] + '.npy')
    predict_result = clf.predict_proba(feature)

    img = io.imread(original_img_dir + os.sep + img_name)
    segments_region, labels = region_segment(features[:, 0:5], segments)
    region_feature = extract_region_features(img, segments, segments_region, labels)

    saliency_img =  predict_result[:, 1] + region_feature[:, 2]
    saliency_img = normalize(saliency_img)
    return feature_to_image(saliency_img, segments)



def test_score(clf, original_img_dir, binary_img_dir):
    list_dir = os.listdir(original_img_dir)
    list_dir = filter(lambda f: os.path.splitext(f)[1] == '.jpg', list_dir)
    score_array = np.zeros(len(list_dir))
    i = 0
    for f in list_dir:
        img = io.imread(original_img_dir + os.sep + f)
        binary_img = io.imread(binary_img_dir + os.sep + os.path.splitext(f)[0] + '.bmp')
        test_array, _ = extract_features(img, binary_img[:, :, 0] > 0)
        feature = test_array[:, 0:(test_array.shape[1] - 1)]
        label = test_array[:, -1] > 0.9
        predict_result = clf.predict(feature)
        score_array[i] = f1_score(label, predict_result, average="binary")
        i = i + 1
    return score_array


def find_max_score(cache_dir, max_interation=100, pic_num=5, C=10):
    if not os.path.exists(cache_dir):
        raise NameError("Path does not exits, check out!")
    list_features_dir = os.listdir(cache_dir)
    list_features_dir = filter(lambda f: os.path.splitext(f)[1] == '.npy', list_features_dir)
    feature_array, label_array = load_features(cache_dir)
    label_array = label_array > 0.9
    score_array_list = []
    features_img_list = []
    for i in xrange(max_interation):
        random.shuffle(list_features_dir)
        # choose first 16 picture as feature files
        features_img = list_features_dir[0:pic_num]
        features_img_list.append(features_img)
        # exact features
        feature, label = load_features(cache_dir, features_img)
        # train model
        clf = train_features(feature, label > 0.9, C)
        predict_result = clf.predict(feature_array)
        score_array = f1_score(label_array, predict_result, average="binary")
        score_array_list.append(score_array)
    return features_img_list, np.array(score_array_list)


def print_max_score(cache_dir, max_interation=100, pic_num=5, C=10, pic_list=None):
    if not os.path.exists(cache_dir):
        raise NameError("Path does not exits, check out!")
    list_features_dir = os.listdir(cache_dir)
    list_features_dir = filter(lambda f: os.path.splitext(f)[1] == '.npy', list_features_dir)
    feature_array, label_array = load_features(cache_dir)
    features_dict = load_features_as_dict(cache_dir)
    label_array = label_array > 0.9
    max_score = 0
    max_score_image_list = []
    # stand alone image test
    if pic_list is not None:
        if not isinstance(pic_list, list):
            raise NameError("pic_list is not a list!")
        max_interation = 1
        pic_num = len(pic_list)
        list_features_dir = pic_list

    for i in xrange(max_interation):
        if pic_num != 1:
            random.shuffle(list_features_dir)
            features_img = list_features_dir[0:pic_num]
        else:
            features_img = [list_features_dir[i]]
        # exact features
        feature, label = load_features(cache_dir, features_img, features_dict)
        # train model
        label_binary = label > 0.9
        if label_binary.max() != True:
            label_binary = label > 0.6
        clf = train_features(feature, label_binary, C)
        predict_result = clf.predict(feature_array)
        score = f1_score(label_array, predict_result, average="binary")
        if score > max_score:
            max_score = score
            max_score_image_list = features_img
            print "max score is: " + str(max_score) + " picture list: " + str(max_score_image_list) + " i:" +str(i)
        # print "score is: " + str(score) + " picture list: " + str(features_img)
    if pic_list is not None:
        return
    return max_score_image_list, max_score


def print_max_score_multiprocess(max_score, cache_dir, max_interation=100, pic_num=5, C=10):
    if not os.path.exists(cache_dir):
        raise NameError("Path does not exits, check out!")
    list_features_dir = os.listdir(cache_dir)
    list_features_dir = filter(lambda f: os.path.splitext(f)[1] == '.npy', list_features_dir)
    feature_array, label_array = load_features(cache_dir)
    features_dict = load_features_as_dict(cache_dir)
    label_array = label_array > 0.9

    for i in xrange(max_interation):
        if pic_num != 1:
            random.shuffle(list_features_dir)
            features_img = list_features_dir[0:pic_num]
        else:
            features_img = [list_features_dir[i]]
        # exact features
        feature, label = load_features(cache_dir, features_img, features_dict)
        # train model
        label_binary = label > 0.9
        if label_binary.max() != True:
            label_binary = label > 0.6
        clf = train_features(feature, label_binary, C)
        predict_result = clf.predict(feature_array)
        score = f1_score(label_array, predict_result, average="binary")
        if score > max_score.value:
            max_score.value = score
            max_score_image_list = features_img
            print "max score is: " + str(max_score.value) + " picture list: " + str(max_score_image_list) + " i:" +str(i)
        # print "score is: " + str(score) + " picture list: " + str(features_img)


def print_all_score(cache_dir, C=10, iter=None):
    if not os.path.exists(cache_dir):
        raise NameError("Path does not exits, check out!")
    list_features_dir = os.listdir(cache_dir)
    list_features_dir = filter(lambda f: os.path.splitext(f)[1] == '.npy', list_features_dir)
    feature_array, label_array = load_features(cache_dir)
    features_dict = load_features_as_dict(cache_dir)
    label_array = label_array > 0.9
    max_score = 0
    max_score_image_list = []
    max_interation = len(list_features_dir)

    for i in xrange(max_interation - 1):
        for f in list_features_dir[i+1:]:
            features_img = [list_features_dir[i], f]
            if (iter is not None) and (i == iter[0]) and (list_features_dir.index(f) == iter[1]):
                iter = None
            if iter is not None:
                continue
            print "iteration is: " + str(i) + ", sub-iteration is: " + str(list_features_dir.index(f))
            # exact features
            feature, label = load_features(cache_dir, features_img, features_dict)
            # train model
            label_binary = label > 0.9
            if label_binary.max() != True:
                label_binary = label > 0.6
            clf = train_features(feature, label_binary, C)
            predict_result = clf.predict(feature_array)
            score = f1_score(label_array, predict_result, average="binary")
            if score > max_score:
                max_score = score
                max_score_image_list = features_img
                print "max score is: " + str(max_score) + " picture list: " + str(max_score_image_list)
            # print "score is: " + str(score) + " picture list: " + str(features_img)
    # return max_score_image_list, max_score


def product_saliency_map(original_img_dir, binary_img_dir, test_img_dir, max_interation, pic_num, c, extra=0):
    """
    random generate parameter
    :param original_img_dir:
    :param binary_img_dir:
    :param test_img_dir:
    :param max_interation:
    :param pic_num:
    :param c:
    :return: None
    """
    a, b = find_max_score(original_img_dir, binary_img_dir, test_img_dir, max_interation, pic_num, c)
    with open('a' + str(pic_num) + '_' + str(c + extra) + '.pkl', 'w') as f:
        pickle.dump(a, f)
    np.save('b' + str(pic_num) + '_' + str(c + extra) + '.npy', b)
    r = b.argmax()
    pic_list = a[r]
    pic_list = map(lambda s: os.path.splitext(s)[0] + '.jpg', pic_list)
    product_saliency_image(original_img_dir, binary_img_dir, pic_list, c, 'max')


def product_saliency_image(original_img_dir, binary_img_dir, pic_list, c, extra):
    pic_num = len(pic_list)
    feature, label = get_features(original_img_dir, binary_img_dir, pic_list=pic_list)
    # train feature
    clf = train_features(feature, label > 0.9, C=c)
    # product saliency map
    list_dir = os.listdir(original_img_dir)
    list_dir = filter(lambda f: os.path.splitext(f)[1] == '.jpg', list_dir)
    out_dir = saliency_img_out_dir + str(pic_num) + '_' + str(c) + '_' + str(extra)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for f in list_dir:
        img = io.imread(original_img_dir + os.sep + f)
        saliency_img = test_image(clf, img)
        io.imsave(out_dir + os.sep + f.split('.')[0] + '.png', saliency_img)


def product_saliency_image_use_cache(train_cache_dir, cache_dir, pic_list, c, extra):
    pic_num = len(pic_list)
    # feature, label = get_features(original_img_dir, binary_img_dir, pic_list=pic_list)
    feature, label = get_features_use_cache(train_cache_dir, pic_list)
    # train feature
    clf = train_features(feature, label > 0.9, C=c)
    # product saliency map
    list_dir = os.listdir(cache_dir)
    list_dir = filter(lambda f: os.path.splitext(f)[1] == '.npy', list_dir)
    out_dir = saliency_img_out_dir + str(pic_num) + '_' + str(c) + '_' + str(extra)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for f in list_dir:
        saliency_img = test_image_use_cache(clf, f, cache_dir)
        # saliency_img = test_image_use_region(clf, f, cache_dir, original_img_dir)
        io.imsave(out_dir + os.sep + f.split(".")[0] + ".png", saliency_img)


def product_saliency_feature_use_cache(train_cache_dir, cache_dir, pic_list, c, extra):
    pic_num = len(pic_list)
    # feature, label = get_features(original_img_dir, binary_img_dir, pic_list=pic_list)
    feature, label = get_features_use_cache(train_cache_dir, pic_list)
    # train feature
    clf = train_features(feature, label > 0.9, C=c)
    # product saliency map
    list_dir = os.listdir(cache_dir)
    list_dir = filter(lambda f: os.path.splitext(f)[1] == '.npy', list_dir)
    out_dir = saliency_feature_out_dir + str(pic_num) + '_' + str(c) + '_' + str(extra)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for f in list_dir:
        saliency_feature = test_feature_use_cache(clf, f, cache_dir)
        np.save(out_dir + os.sep + f.split(".")[0] + ".npy", saliency_feature)


def product_saliency_image_use_cache_upsample(original_img_dir, binary_img_dir, cache_dir, pic_list, c, extra):
    pic_num = len(pic_list)
    feature, label = get_features(original_img_dir, binary_img_dir, pic_list=pic_list)
    # train feature
    clf = train_features(feature, label > 0.9, C=c)
    # product saliency map
    list_dir = os.listdir(cache_dir)
    list_dir = filter(lambda f: os.path.splitext(f)[1] == '.npy', list_dir)
    out_dir = saliency_img_out_dir + str(pic_num) + '_' + str(c) + '_' + str(extra)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    segments_dir = cache_dir + "_segments"
    img_lab_dir = cache_dir + "_img_lab"
    segments_mean_dir = cache_dir + "_segments_mean"
    if not os.path.exists(cache_dir) or not os.path.exists(segments_dir):
        raise NameError('cache dir cannot be found!')

    for f in list_dir:
        features = np.load(cache_out_dir + os.sep + f.split('.')[0] + '.npy')
        # segments = np.load(segments_dir + os.sep + f.split('.')[0] + '.npy')
        segments_mean = np.load(segments_mean_dir + os.sep + f.split('.')[0] + '.npy')
        img_lab = np.load(img_lab_dir + os.sep + f.split('.')[0] + '.npy')

        feature = features[:, 0:(features.shape[1] - 1)]
        predict_result = clf.predict_proba(feature)
        saliency_img = up_sample(img_lab, predict_result[:, 1], segments_mean[:, 0:3], segments_mean[:, 3:5])

        io.imsave(out_dir + os.sep + f.split(".")[0] + ".png", saliency_img)


def product_saliency_image_use_cache_upsample2(train_cache_dir, general_cache_dir, cache_dir, img_dir, pic_list, c, extra):
    pic_num = len(pic_list)
    out_dir = saliency_img_out_dir + str(pic_num) + '_' + str(c) + '_' + str(extra)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    segments_dir = cache_dir + "_segments"
    segments_mean_dir = general_cache_dir
    if not os.path.exists(cache_dir) or not os.path.exists(segments_dir):
        raise NameError('cache dir cannot be found!')

    feature, label = get_features_use_cache(train_cache_dir, pic_list)
    # train feature
    clf = train_features(feature, label > 0.9, C=c)
    # product saliency map
    list_dir = os.listdir(cache_dir)
    list_dir = filter(lambda f: os.path.splitext(f)[1] == '.npy', list_dir)

    for f in list_dir:
        features = np.load(cache_dir + os.sep + f.split('.')[0] + '.npy')
        segments_mean = np.load(segments_mean_dir + os.sep + f.split('.')[0] + '.npy')
        img_lab = rgb2lab(io.imread(img_dir + os.sep + f.split('.')[0] + '.jpg'))

        feature = features[:, 0:(features.shape[1] - 1)]
        predict_result = clf.predict_proba(feature)
        saliency_img = up_sample(img_lab, predict_result[:, 1], segments_mean[:, 0:3], segments_mean[:, 3:5])

        io.imsave(out_dir + os.sep + f.split(".")[0] + ".png", saliency_img)


def manifold_ranking_saliency(predicts, features, segments, neighbors):
    normalize = lambda s: (s - s.min()) / (s.max() - s.min())

    # get the surroundings of the surroundings of the superpixel
    x = np.arange(neighbors.shape[0])
    n = neighbors.copy()
    for i in xrange(neighbors.shape[0]):
        mask = np.any(neighbors[x[neighbors[i, :]], :], axis=0)
        n[i, :] |= mask
    neighbors = n
    border_sp = np.unique(np.concatenate([segments[0, :], segments[segments.shape[0]-1, :], segments[:, 0], segments[:, segments.shape[1] - 1]]))
    neighbors[np.eye(neighbors.shape[0], dtype=np.bool)] = True
    neighbors[:, border_sp] = True

    img_segments_mean = features[:, 0:3]
    W = cdist(img_segments_mean, img_segments_mean)
    W_max = W[neighbors].max()
    W_min = W[neighbors].min()
    W = np.exp(-(W - W_min) / ((W_max - W_min) * 0.05))
    # W = np.exp(-W / (W_max * 0.1))
    W[~neighbors] = 0
    D = np.diag(W.sum(axis=1))
    Aff = np.linalg.inv(D - 0.99 * W)
    Aff[np.eye(Aff.shape[0], dtype=np.bool)] = 0

    # predicts = normalize(predicts)
    mr_saliency = np.zeros_like(predicts)
    mr_saliency[predicts > predicts.mean()] = 2

    return normalize(np.dot(Aff, mr_saliency))


def manifold_ranking_aff(features, segments, neighbors):
    # get the surroundings of the surroundings of the superpixel
    x = np.arange(neighbors.shape[0])
    n = neighbors.copy()
    for i in xrange(neighbors.shape[0]):
        mask = np.any(neighbors[x[neighbors[i, :]], :], axis=0)
        n[i, :] |= mask
    neighbors = n
    border_sp = np.unique(np.concatenate([segments[0, :], segments[segments.shape[0]-1, :], segments[:, 0], segments[:, segments.shape[1] - 1]]))
    neighbors[np.eye(neighbors.shape[0], dtype=np.bool)] = True
    neighbors[:, border_sp] = True

    img_segments_mean = features[:, 0:3]
    W = cdist(img_segments_mean, img_segments_mean)
    W_max = W[neighbors].max()
    W_min = W[neighbors].min()
    W = np.exp(-(W - W_min) / ((W_max - W_min) * 0.05))
    # W = np.exp(-W / (W_max * 0.1))
    W[~neighbors] = 0
    D = np.diag(W.sum(axis=1))
    Aff = np.linalg.inv(D - 0.99 * W)
    Aff[np.eye(Aff.shape[0], dtype=np.bool)] = 0

    return Aff



def manifold_ranking_saliency2(predicts, features, segments, neighbors, region_labels):
    '''
    use kmeans region to generate W
    :param predicts:
    :param features:
    :param segments:
    :param neighbors:
    :return:
    '''
    normalize = lambda s: (s - s.min()) / (s.max() - s.min())

    # get the kmeans region superpixels
    for i in xrange(neighbors.shape[0]):
        mask = region_labels == region_labels[i]
        neighbors[i, mask] = True
    border_sp = np.unique(np.concatenate([segments[0, :], segments[segments.shape[0]-1, :], segments[:, 0], segments[:, segments.shape[1] - 1]]))
    neighbors[np.eye(neighbors.shape[0], dtype=np.bool)] = True
    neighbors[:, border_sp] = True

    # img_segments_mean = features[:, 0:3]
    img_segments_mean = features
    W = cdist(img_segments_mean, img_segments_mean)
    W_max = W[neighbors].max()
    W_min = W[neighbors].min()
    W = np.exp(-(W - W_min) / ((W_max - W_min) * 0.05))
    # W = np.exp(-W / (W_max * 0.1))
    W[~neighbors] = 0
    D = np.diag(W.sum(axis=1))
    Aff = np.linalg.inv(D - 0.99 * W)
    Aff[np.eye(Aff.shape[0], dtype=np.bool)] = 0

    # predicts = normalize(predicts)
    mr_saliency = np.zeros_like(predicts)
    mr_saliency[predicts > predicts.mean()] = 1

    return normalize(np.dot(Aff, mr_saliency))


def manifold_ranking_saliency3(predicts, features):
    '''
    use all W
    :param predicts:
    :param features:
    :param segments:
    :param neighbors:
    :return:
    '''
    normalize = lambda s: (s - s.min()) / (s.max() - s.min())

    img_segments_mean = features[:, 0:3]
    W = cdist(img_segments_mean, img_segments_mean)
    W_max = W.max()
    W_min = W.min()
    W = np.exp(-(W - W_min) / ((W_max - W_min) * 0.05))
    # W = np.exp(-W / (W_max * 0.1))
    D = np.diag(W.sum(axis=1))
    Aff = np.linalg.inv(D - 0.99 * W)
    Aff[np.eye(Aff.shape[0], dtype=np.bool)] = 0

    # predicts = normalize(predicts)
    mr_saliency = np.zeros_like(predicts)
    mr_saliency[predicts > predicts.mean()] = 1

    return normalize(np.dot(Aff, mr_saliency))