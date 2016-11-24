# -*- coding: utf-8 -*-

from multiprocessing import Process, freeze_support, Queue
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
from time import time
from train_saliency import region_segment, feature_to_image, extract_region_features

features_path = os.getcwd() + os.sep + "features"
test_out_path = os.getcwd() + os.sep + "tests"
original_img_dir = r"G:\Project\paper2\other_image\MSRA-1000_images"
binary_img_dir = r"G:\Project\paper2\other_image\binarymasks"
saliency_img_out_dir = r"G:\Project\paper2\out"
cache_out_dir = 'G:\\Project\\paper2\\feature_cache'


def extract_features(img, binary_masks=None, segments_number=300):
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
    # ft_feature = np.zeros((max_segments, 1))

    # size feature
    # size_feature = np.zeros((max_segments, 1))

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

        # ft_feature[i] = blur_sm[segments_i].mean()

        # size_feature[i] = blur_sm[segments_i].size / float(size)

        center_color_feature[i] = center_color_map[segments_i].mean()

        edge_feature[i] = edge_img[segments_i].sum()

    # CA feature
    # ca_feature = np.sum(d_fun(cdist(img_segments_mean, img_segments_mean),
    #                           cdist(coordinate_segments_mean, coordinate_segments_mean)), axis=1)
    # ca_feature = np.array([ca_feature]).T  # transpose it to column vector

    # element distribution
    wc_ij = np.exp(-cdist(img_segments_mean, img_segments_mean)**2 / (2 * 20 ** 2))
    wc_ij = wc_ij / wc_ij.sum(axis=1)[:, None]
    mu_i = np.dot(wc_ij, coordinate_segments_mean)
    distribution = np.dot(wc_ij, np.linalg.norm(coordinate_segments_mean - mu_i, axis=1)**2)
    distribution = normalize(distribution)
    distribution = np.array([distribution]).T

    # element uniqueness plus feature
    wp_ij = np.exp(cdist(coordinate_segments_mean, coordinate_segments_mean)**2 / (2 * 100 ** 2))
    wp_ij = wp_ij / wp_ij.sum(axis=1)[:, None]
    mu_i_c = np.dot(wp_ij, img_segments_mean)
    # uniqueness_plus = np.dot(wp_ij, np.linalg.norm(img_segments_mean - mu_i_c, axis=1)**2)
    uniqueness_plus = np.sum(cdist(img_segments_mean, mu_i_c)**2 * wp_ij, axis=1)
    uniqueness_plus = normalize(uniqueness_plus)
    uniqueness_plus = np.array([uniqueness_plus]).T

    # uniqueness and distribution feature
    # ud_feature = uniqueness * np.exp(-6*distribution)

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
    # ft_feature = normalize(ft_feature)
    # ca_feature = 1 - np.exp(- ca_feature * 1.5 / float(max_segments))
    # ca_feature = normalize(ca_feature)
    # size_feature = normalize(size_feature)
    edge_feature = normalize_zero(edge_feature)

    if binary_masks is not None:
        return np.concatenate((img_segments_mean, coordinate_segments_mean, uniqueness_plus, distribution,
                               center_color_feature, edge_feature,
                               saliency_super_pixels), axis=1), segments
    else:
        return np.concatenate((img_segments_mean, coordinate_segments_mean, uniqueness_plus, distribution,
                               center_color_feature, edge_feature,
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


def save_features(original_img_dir, binary_img_dir, cache_dir):
    if (not os.path.exists(original_img_dir)) and not (os.path.exists(binary_img_dir)):
        raise NameError("Path does not exits, check out!")
    # check out features_dir
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    segments_dir = cache_dir + "_segments"
    if not os.path.exists(segments_dir):
        os.mkdir(segments_dir)

    list_features_dir = os.listdir(original_img_dir)
    list_features_dir = filter(lambda f: os.path.splitext(f)[1] == '.jpg', list_features_dir)

    for f in list_features_dir:
        img_path_name = original_img_dir + os.sep + f
        binary_img_path_name = binary_img_dir + os.sep + os.path.splitext(f)[0] + '.bmp'
        img = io.imread(img_path_name)
        binary_img = io.imread(binary_img_path_name)
        features, segments = extract_features(img, binary_img[:, :] > 0)
        np.save(cache_dir + os.sep + os.path.splitext(f)[0] + '.npy', features)
        np.save(segments_dir + os.sep + os.path.splitext(f)[0] + '.npy', segments)


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


def analysis_score_array(b):
    return b.argmax()


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
    r = analysis_score_array(b)
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


def product_saliency_image_use_cache(original_img_dir, binary_img_dir, cache_dir, pic_list, c, extra):
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
        # saliency_img = test_image_use_cache(clf, f, cache_dir)
        saliency_img = test_image_use_region(clf, f, cache_dir, original_img_dir)
        io.imsave(out_dir + os.sep + f.split(".")[0] + ".png", saliency_img)


def parameter_saliency_map(original_img_dir, binary_img_dir, a, b, pic_num, c):
    pic_list = a[analysis_score_array(b)]
    feature, label = get_features(original_img_dir, binary_img_dir, pic_list=pic_list)
    # train feature
    clf = train_features(feature, label > 0.9, C=c)
    # product saliency map
    list_dir = os.listdir(original_img_dir)
    list_dir = filter(lambda f: os.path.splitext(f)[1] == '.jpg', list_dir)
    out_dir = saliency_img_out_dir + str(pic_num) + '_' + str(c)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for f in list_dir:
        img = io.imread(original_img_dir + os.sep + f)
        saliency_img = test_image(clf, img)
        io.imsave(out_dir + os.sep + f, saliency_img)


def image_saliency_map(original_img_dir, binary_img_dir, pic_list, pic_num, c, extra):
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
        io.imsave(out_dir + os.sep + f, saliency_img)


# %%
import pickle
from multiprocessing import Process, freeze_support

if __name__ == "__main__":
    freeze_support()
    start = time()
    pic_list = ['2_68_68401.npy', '0_17_17275.npy', '0_4_4328.npy', '0_1_1004.npy', '2_77_77109.npy']
    pic_list = map(lambda s: s.split('.')[0] + '.jpg', pic_list);
    feature, label = get_features(original_img_dir, binary_img_dir, pic_list=pic_list)
    clf = train_features(feature, label > 0.8, C=1)

    # iter = [639, 712]
    # save_features(original_img_dir, binary_img_dir, cache_out_dir)
    # print_all_score(cache_out_dir, C=1, iter=iter)
    product_saliency_image_use_cache(original_img_dir, binary_img_dir, cache_out_dir, pic_list, 1, "region_no_normalize")
    # product_saliency_image(original_img_dir, binary_img_dir, pic_list, 10, "withft")
    print_max_score(cache_out_dir, 5000, 5, C=1)
    # print_all_score(cache_out_dir, C=1)
    # iamge_name_list, score = print_max_score(cache_out_dir, 100, 2)
    # p1 = Process(target=print_max_score, args=(cache_out_dir, 10000, 2))
    # p2 = Process(target=print_max_score, args=(cache_out_dir, 10000, 5))
    # p1.start()
    # p2.start()
    # p1.join()
    # p2.join()
    stop = time()
    print "total time is: " + str(stop - start)
    # save_features(original_img_dir, binary_img_dir, cache_out_dir)

# if __name__ == "__main__":
#     freeze_support()
#     start = time()
#     # process 1
#     with open('a' + str(5) + '_' + str(100) + '.pkl', 'r') as f:
#         a = pickle.load(f)
#     b5_100 = np.load('b' + str(5) + '_' + str(100) + '.npy')
#     b_min = b5_100.min(axis=1)
#     a1 = np.array(a)
#     a1 = a1[b_min > 0.5, :]
#
#     with open('a' + str(5) + '_' + str(1) + '.pkl', 'r') as f:
#         a = pickle.load(f)
#     b5_1 = np.load('b' + str(5) + '_' + str(1) + '.npy')
#     b_min = b5_1.min(axis=1)
#     a2 = np.array(a)
#     a2 = a2[b_min > 0.5, :]
#
#     # a1
#     for i in xrange(0, a1.shape[0], 2):
#
#         p1 = Process(target=image_saliency_map, args=(original_img_dir, binary_img_dir, list(a1[i, :]), 5, 100, i))
#
#         # process 2
#         if i + 1 < a1.shape[0]:
#             p2 = Process(target=image_saliency_map, args=(original_img_dir, binary_img_dir, list(a1[i+1, :]), 5, 100, i+1))
#
#         p1.start()
#         p2.start()
#         p1.join()
#         p2.join()
#         stop = time()
#         print "total time is %f" % ((stop - start) / 60 / 60)
#
#     # a2
#     for i in xrange(0, a2.shape[0], 2):
#
#         p1 = Process(target=image_saliency_map, args=(original_img_dir, binary_img_dir, list(a2[i, :]), 5, 1, i))
#
#         # process 2
#         if i + 1 < a2.shape[0]:
#             p2 = Process(target=image_saliency_map, args=(original_img_dir, binary_img_dir, list(a2[i+1, :]), 5, 1, i+1))
#
#         p1.start()
#         p2.start()
#         p1.join()
#         p2.join()
#         stop = time()
#         print "total time is %f" % ((stop - start) / 60 / 60)


# if __name__ == "__main__":
#     freeze_support()
#     pic_num = [5, 10, 15]
#     for i in pic_num:
#         start = time()
#         c_num = [1, 1000]
#         # process 1
#         with open('a' + str(i) + '_' + str(c_num[0]) + '.pkl', 'r') as f:
#             a = pickle.load(f)
#         b = np.load('b' + str(i) + '_' + str(c_num[0]) + '.npy')
#         p1 = Process(target=parameter_saliency_map, args=(original_img_dir, binary_img_dir, a, b, i, c_num[1]))
#
#         # process 2
#         with open('a' + str(i) + '_' + str(c_num[1]) + '.pkl', 'r') as f:
#             a = pickle.load(f)
#         b = np.load('b' + str(i) + '_' + str(c_num[1]) + '.npy')
#         p2 = Process(target=parameter_saliency_map, args=(original_img_dir, binary_img_dir, a, b, i, c_num[0]))
#
#         p1.start()
#         p2.start()
#         p1.join()
#         p2.join()
#         stop = time()
#         print "total time is %f" % ((stop - start) / 60 / 60)


# if __name__ == "__main__":
#     freeze_support()
#     p1 = Process(target=product_saliency_map, args=(original_img_dir, binary_img_dir, os.getcwd(), 500, 2, 10))
#     p2 = Process(target=product_saliency_map, args=(original_img_dir, binary_img_dir, os.getcwd(), 500, 5, 10))
#     start = time()
#     p1.start()
#     p2.start()
#     p1.join()
#     p2.join()
#     stop = time()
#     print "total time is %f" % ((stop - start) / 60 / 60)


# if __name__ == "__main__":
#     freeze_support()
#     pic_num = [5, 10, 15]
#     for i in pic_num:
#         start = time()
#         c_num = [1, 1000]
#         p1 = Process(target=product_saliency_map, args=(original_img_dir, binary_img_dir, os.getcwd(), 100, i, c_num[0]))
#         p2 = Process(target=product_saliency_map, args=(original_img_dir, binary_img_dir, os.getcwd(), 100, i, c_num[1]))
#         p1.start()
#         p2.start()
#         p1.join()
#         p2.join()
#         stop = time()
#         print "total time is %f" % ((stop - start) / 60 / 60)
