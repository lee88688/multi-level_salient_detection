# -*- coding: utf-8 -*-

from scipy.spatial.distance import cdist
from skimage.segmentation import slic, mark_boundaries
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

start = time()
features_path = os.getcwd() + os.sep + "features"
test_out_path = os.getcwd() + os.sep + "tests"
original_img_dir = "G:\\Project\\paper2\\images"
binary_img_dir = "G:\\Project\\paper2\\binarymasks2"
saliency_img_out_dir = "G:\\Project\\paper2\\out"


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

    # element uniqueness feature
    wp_ij = np.exp(cdist(coordinate_segments_mean, coordinate_segments_mean)**2 / (2 * 100 ** 2))
    wp_ij = wp_ij / wp_ij.sum(axis=1)[:, None]
    # w_ij = wp_ij * wc_ij
    # w_ij = w_ij / w_ij.sum(axis=1)[:, None]
    # uniqueness = np.sum(cdist(img_segments_mean, img_segments_mean)**2 * wp_ij, axis=1)
    uniqueness = np.sum(cdist(img_segments_mean, img_segments_mean)**2 * wp_ij, axis=1)
    uniqueness = normalize(uniqueness)
    uniqueness = np.array([uniqueness]).T

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
    ft_feature = normalize(ft_feature)
    ca_feature = 1 - np.exp(- ca_feature * 1.5 / float(max_segments))
    ca_feature = normalize(ca_feature)
    size_feature = normalize(size_feature)
    edge_feature = normalize_zero(edge_feature)

    if binary_masks is not None:
        return np.concatenate((img_segments_mean, coordinate_segments_mean, uniqueness_plus, distribution, uniqueness,
                               center_color_feature, ft_feature, ca_feature, size_feature, edge_feature,
                               saliency_super_pixels), axis=1), segments
    else:
        return np.concatenate((img_segments_mean, coordinate_segments_mean, uniqueness_plus, distribution, uniqueness,
                               center_color_feature, ft_feature, ca_feature, size_feature, edge_feature,
                               ), axis=1), segments


def save_features(original_img_dir, binary_img_dir, sample_picture_number=20, multi_scale=False, segments_number=300,
                  pic_list=None):
    if (not os.path.exists(original_img_dir)) and not (os.path.exists(binary_img_dir)):
        raise NameError("Path does not exits, check out!")
    # check out features_dir
    if not os.path.exists(features_path):
        os.mkdir(features_path)

    if pic_list is None:
        list_features_dir = os.listdir(original_img_dir)
        list_features_dir = filter(lambda f: os.path.splitext(f)[1] == '.jpg', list_features_dir)
        random.shuffle(list_features_dir)
        # choose first 16 picture as feature files
        features_img = list_features_dir[0:sample_picture_number]
    features_img = pic_list

    # start to extract feature and save
    if multi_scale == True:
        # smaller scale, 1/2 of original
        for f in features_img:
            img_path_name = original_img_dir + os.sep + f
            binary_img_path_name = binary_img_dir + os.sep + os.path.splitext(f)[0] + '.bmp'
            img = io.imread(img_path_name)
            binary_img = io.imread(binary_img_path_name)
            features, _ = extract_features(img, binary_img[:, :, 0] > 0, segments_number=segments_number / 2)
            np.save(features_path + os.sep + os.path.splitext(f)[0] + "-" + str(segments_number / 2) + '.npy', features)

    # original scale
    for f in features_img:
        img_path_name = original_img_dir + os.sep + f
        binary_img_path_name = binary_img_dir + os.sep + os.path.splitext(f)[0] + '.bmp'
        img = io.imread(img_path_name)
        binary_img = io.imread(binary_img_path_name)
        features, _ = extract_features(img, binary_img[:, :, 0] > 0, segments_number=segments_number)
        np.save(features_path + os.sep + os.path.splitext(f)[0] + "-" + str(segments_number) + '.npy', features)


def load_features(features_path):
    if not os.path.exists(features_path):
        raise NameError("Path does not exits, check out!")
    features_name_list = os.listdir(features_path)
    features_name_list = filter(lambda f: os.path.splitext(f)[1] == '.npy', features_name_list)
    features_list = []
    for f in features_name_list:
        f = features_path + os.sep + f
        features_list.append(np.load(f))

    final_array = np.concatenate(features_list, axis=0)

    feature = final_array[:, 0:(final_array.shape[1] - 1)]
    label = final_array[:, -1]
    return feature, label


def train_features(feature, label, C=1):
    clf = svm.SVC(C=C, probability=True)
    clf.fit(feature, label)
    return clf


def train_features_lg(feature, label):
    clf = LogisticRegression()
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


def find_max_score(original_img_dir, binary_img_dir, test_img_dir, max_interation=100):
    if (not os.path.exists(original_img_dir)) and not (os.path.exists(binary_img_dir)):
        raise NameError("Path does not exits, check out!")
    list_features_dir = os.listdir(original_img_dir)
    list_features_dir = filter(lambda f: os.path.splitext(f)[1] == '.jpg', list_features_dir)
    score_array_list = []
    features_img_list = []
    for i in xrange(max_interation):
        random.shuffle(list_features_dir)
        # choose first 16 picture as feature files
        features_img = list_features_dir[0:20]
        features_img_list.append(features_img)
        # exact features
        features_list = []
        for f in features_img:
            img_path_name = original_img_dir + os.sep + f
            binary_img_path_name = binary_img_dir + os.sep + os.path.splitext(f)[0] + '.bmp'
            img = io.imread(img_path_name)
            binary_img = io.imread(binary_img_path_name)
            features, _ = extract_features(img, binary_img[:, :, 0] > 0)
            features_list.append(features)
        final_array = np.concatenate(features_list, axis=0)
        feature = final_array[:, 0:(final_array.shape[1] - 1)]
        label = final_array[:, -1]
        # train model
        clf = train_features(feature, label > 0.9)
        score_array = test_score(clf, test_img_dir, binary_img_dir)
        score_array_list.append(score_array)
    return features_img_list, np.array(score_array_list)


def analysis_score_array(b):
    b_max = b.max(axis=1)
    b_min = b.min(axis=1)
    b_mean = b.mean(axis=1)

    index = b_mean.argmax()
    print "mean max: max: %f, min: %f, mean: %f, index: %d" % (b_max[index], b_min[index], b_mean[index], index)
    index = b_max.argmax()
    print "max max: max: %f, min: %f, mean: %f, index: %d" % (b_max[index], b_min[index], b_mean[index], index)
    index = b_min.argmax()
    print "min max: max: %f, min: %f, mean: %f, index: %d" % (b_max[index], b_min[index], b_mean[index], index)
    b_a = b_mean/(b_max - b_min)
    index = b_a.argmax()
    print "score max: max: %f, min: %f, mean: %f, index: %d" % (b_max[index], b_min[index], b_mean[index], index)


# %% save feature
# import pickle
# with open('pic_list5_1.pkl', 'r') as f:
#     pic_list = pickle.load(f)
# edge_feature_list = []
# save_features(original_img_dir, binary_img_dir, pic_list=pic_list)
# feature, label = load_features(features_path)
# clf = train_features(feature, label > 0.9, C=1)

# %% show result
# list_dir = os.listdir(original_img_dir)
# list_dir = filter(lambda f: os.path.splitext(f)[1] == '.jpg', list_dir)
# for f in list_dir:
#    img = io.imread(original_img_dir + os.sep + f)
#    saliency_img = test_image(clf, img)
#    io.imsave(saliency_img_out_dir + os.sep + f, saliency_img)


# %% region
from sklearn.cluster import KMeans


def feature_to_image(feature, segments):
    max_segments = segments.max() + 1
    if max_segments != feature.size:
        raise NameError("feature size does not match segments.")

    feature_img = np.zeros_like(segments, dtype=np.float64)

    for i in xrange(max_segments):
        segments_i = segments == i
        feature_img[segments_i] = feature[i]

    return feature_img


def up_sample(img_lab, saliency, img_segments_mean, coordinate_segments_mean):
    size = img_lab.size/3
    shape = img_lab.shape
    # size = img.size
    a = shape[0]
    b = shape[1]
    x_axis = np.linspace(0, b - 1, num=b)
    y_axis = np.linspace(0, a - 1, num=a)

    x_coordinate = np.tile(x_axis, (a, 1,))  # 创建X轴的坐标表
    y_coordinate = np.tile(y_axis, (b, 1,))  # 创建y轴的坐标表
    y_coordinate = np.transpose(y_coordinate)

    c_i = np.concatenate((img_lab[:, :, 0].reshape(size, 1), img_lab[:, :, 1].reshape(size, 1), img_lab[:, :, 2].reshape(size, 1)), axis=1)
    p_i = np.concatenate((x_coordinate.reshape(size, 1), y_coordinate.reshape(size, 1)), axis=1)
    w_ij = np.exp(-1.0/(2*30)*(cdist(c_i, img_segments_mean)**2 + cdist(p_i, coordinate_segments_mean)**2))
    w_ij = w_ij / w_ij.sum(axis=1)[:, None]
    if len(saliency.shape) != 2 or saliency.shape[1] != 1:
        saliency = saliency[:, None]
    saliency_pixel = np.dot(w_ij, saliency)
    return saliency_pixel.reshape(shape[0:2])


def region_segment(features, segments):
    samples = features  # use lab color, uniqueness and distribution feature
    # samples = features

    # Unsupervised learning
    # bandwidth = estimate_bandwidth(samples, quantile=0.2)
    # ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # lables = ms.fit_predict(samples)
    labels = KMeans(n_clusters=20).fit_predict(samples)

    # devide regions
    n_labels = labels.max() + 1
    segments_i_list = []
    for i in xrange(n_labels):
        segments_i_list.append(np.zeros_like(segments, dtype=np.bool))

    for index, v in np.ndenumerate(labels):
        segments_i_list[v] = segments_i_list[v] | (segments == index[0])

    segments_region = np.zeros_like(segments)
    for i, segments_i in enumerate(segments_i_list):
        segments_region[segments_i] = i

    return segments_region, labels


def extract_region_features(img, segments, segments_region, labels):
    sigma1 = 50
    sigma2 = 20
    normalize = lambda s: (s - s.min()) / (s.max() - s.min())

    img_lab = rgb2lab(img)
    max_segments = segments.max() + 1
    max_region_segments = segments_region.max() + 1

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

    coordinate_region_segments_mean = np.zeros((max_region_segments, 2))
    img_region_segments_mean = np.zeros((max_region_segments, 3))
    for i in xrange(max_region_segments):
        segments_i = segments_region == i

        coordinate_region_segments_mean[i, 0] = x_coordinate[segments_i].mean()
        coordinate_region_segments_mean[i, 1] = y_coordinate[segments_i].mean()

        img_region_segments_mean[i, 0] = img_l[segments_i].mean()
        img_region_segments_mean[i, 1] = img_a[segments_i].mean()
        img_region_segments_mean[i, 2] = img_b[segments_i].mean()

    # coordinate_global_segments_mean = np.array([a/2.0, b/2.0])
    # img_global_segments_mean = np.array([img_l.sum()/(a*b), img_a.sum()/(a*b), img_b.sum()/(a*b)])

    # every superpixel corresponds to its region mean value
    C_i_R = np.apply_along_axis(lambda x: img_region_segments_mean[x[0], :], 1, np.array([labels]).T)
    D = cdist(img_segments_mean, img_region_segments_mean)**2 - (np.linalg.norm(img_segments_mean - C_i_R, axis=1)**2)[:, None]

    w_ij = np.exp(cdist(coordinate_segments_mean, coordinate_region_segments_mean)**2 / (2 * sigma1**2))
    w_ij = w_ij / w_ij.sum(axis=1)[:, None]

    region_conlor_contrast = np.sum(w_ij * D, axis=1)
    region_conlor_contrast = normalize(region_conlor_contrast)

    wd_ij = np.exp(cdist(img_segments_mean, img_region_segments_mean)**2 / (2 * sigma2**2))
    wd_ij = wd_ij / wd_ij.sum(axis=1)[:, None]
    mu_i = np.dot(wd_ij, coordinate_region_segments_mean)
    DR = np.sum(cdist(mu_i, coordinate_region_segments_mean) * wd_ij, axis=1)
    # DR = np.dot(wd_ij, np.linalg.norm(coordinate_segments_mean - mu_i, axis=1)**2)
    DR = normalize(DR)
    
    fuse = DR*region_conlor_contrast
    # fuse = normalize(fuse)

    return np.concatenate((region_conlor_contrast[:, None], DR[:, None], fuse[:, None]), axis=1)


# %% test
if __name__ == "__main__":
    list_dir = os.listdir(original_img_dir)
    list_dir = filter(lambda f: os.path.splitext(f)[1] == '.jpg', list_dir)
    random.shuffle(list_dir)
    # f = '(655).jpg', '(75).jpg'
    # f = list_dir[0]
    # f = '(111).jpg'
    for f in [list_dir[0]]:
        print f
        img = io.imread(r'G:\Project\paper2\images\(1).jpg')
        # img = io.imread(original_img_dir + os.sep + f)
        features, segments = extract_features(img)

        segments_region, labels = region_segment(features[:, 0:5], segments)
        region_feature = extract_region_features(img, segments, segments_region, labels)

        # io.imshow(img)
        # io.show()
        # io.imshow(mark_boundaries(img, segments_region))
        # io.show()
        # binary_img = io.imread(binary_img_dir + os.sep + f.split(".")[0] + ".bmp", )
        # io.imshow(binary_img)
        # io.show()
        io.imshow(feature_to_image(region_feature[:, 1], segments))
        io.show()
        io.imshow(feature_to_image(region_feature[:, 0], segments))
        io.show()
        # io.imshow(feature_to_image(region_feature[:, 2], segments))
        # io.show()
    # feature_img = feature_to_image(features[:, 5], segments)
    # io.imshow(feature_img)
    # io.show()
    # feature_img = feature_to_image(features[:, 7], segments)
    # io.imshow(feature_img)
    # io.show()

# f_list = list_dir[100:110]
# for f in f_list:
#     img = io.imread(original_img_dir + os.sep + f)
#     features, segments = extract_features(img)
#     io.imshow(mark_boundaries(img, segments))
#     io.show()
#     feature_img = feature_to_image(features[:, 6], segments)
#     io.imshow(feature_img)
#     io.show()
