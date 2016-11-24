# -*- coding: utf-8 -*-

from scipy.spatial.distance import cdist
from scipy import optimize
from skimage.segmentation import slic, mark_boundaries
from skimage.color import rgb2lab, rgb2grey
from skimage.feature import canny
from scipy.ndimage.morphology import grey_dilation
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score, mean_absolute_error
from sklearn.cluster import KMeans
from skimage import io
import numpy as np
import cv2
import os
import random
from time import time

start = time()
features_path = os.getcwd() + os.sep + "features"
test_out_path = os.getcwd() + os.sep + "tests"
original_img_dir = r"G:\Project\paper2\other_image\MSRA-1000_images"
binary_img_dir = r"G:\Project\paper2\other_image\binarymasks"
saliency_img_out_dir = r"G:\Project\paper2\out_sf_plus4"
cache_out_dir = r'G:\Project\paper2\cache\cache_general_kmeans_rgb_300\feature_cache'


class SFPlus:
    """sf method implement"""

    def __init__(self):
        self.__global_sigma1 = 10
        self.__global_sigma2 = 10
        self.__global_sigma3 = 21.10405218

        self.__region_sigma1 = 50
        self.__region_sigma2 = 23.3557741048

        self.__global_fuse_c = 3

        self.normalize = lambda s: (s - s.min()) / (s.max() - s.min())

    def set_global_sigma(self, sigma1, sigma2, sigma3):
        self.__global_sigma1 = sigma1
        self.__global_sigma2 = sigma2
        self.__global_sigma3 = sigma3

    def set_region_sigma(self, sigma1, sigma2):
        self.__region_sigma1 = sigma1
        self.__region_sigma2 = sigma2

    def set_global_fuse_c(self, c):
        self.__global_fuse_c = c

    def extract_features(self, img, binary_masks=None, segments_number=300):
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

        # element distribution
        wc_ij = np.exp(-cdist(img_segments_mean, img_segments_mean) ** 2 / (2 * self.__global_sigma1 ** 2))
        wc_ij = wc_ij / wc_ij.sum(axis=1)[:, None]
        mu_i = np.dot(wc_ij, coordinate_segments_mean)
        distribution = np.dot(wc_ij, np.linalg.norm(coordinate_segments_mean - mu_i, axis=1) ** 2)
        distribution = normalize(distribution)
        distribution = np.array([distribution]).T

        # element uniqueness feature
        wp_ij = np.exp(
            -cdist(coordinate_segments_mean, coordinate_segments_mean) ** 2 / (2 * self.__global_sigma2 ** 2))
        wp_ij = wp_ij / wp_ij.sum(axis=1)[:, None]
        uniqueness = np.sum(cdist(img_segments_mean, img_segments_mean) ** 2 * wp_ij, axis=1)
        uniqueness = normalize(uniqueness)
        uniqueness = np.array([uniqueness]).T

        # uniqueness plus
        # w_ij = np.exp(cdist(coordinate_segments_mean, coordinate_segments_mean)**2 / (2 * 60 ** 2))
        w_ij = 1 - np.exp(
            cdist(coordinate_segments_mean, coordinate_segments_mean) ** 2 / (2 * self.__global_sigma3 ** 2))
        w_ij = w_ij / w_ij.sum(axis=1)[:, None]
        mu_i_c = np.dot(wp_ij, img_segments_mean)
        # uniqueness_plus = np.dot(wp_ij, np.linalg.norm(img_segments_mean - mu_i_c, axis=1)**2)
        uniqueness_plus = np.sum(cdist(img_segments_mean, mu_i_c) ** 2 * w_ij, axis=1)
        uniqueness_plus = normalize(uniqueness_plus)
        uniqueness_plus = np.array([uniqueness_plus]).T

        if binary_masks is not None:
            return np.concatenate(
                (uniqueness_plus, distribution, uniqueness, img_segments_mean, coordinate_segments_mean,
                 saliency_super_pixels), axis=1), segments
        else:
            return np.concatenate(
                (uniqueness_plus, distribution, uniqueness, img_segments_mean, coordinate_segments_mean,
                 ), axis=1), segments

    def feature_to_image(self, feature, segments, enhance=False):
        max_segments = segments.max() + 1
        if max_segments != feature.size:
            raise NameError("feature size does not match segments.")

        if enhance and (feature > 0).sum() > 5 and (feature == feature.max()).sum() < 5:
            mask = feature < 0
            while np.count_nonzero(feature == feature.max()) < 10:
                mask = mask | (feature == feature.max())
                feature[mask] = 0
                feature[mask] = feature.max()
            feature = self.normalize(feature)

        feature_img = np.zeros_like(segments, dtype=np.float64)

        for i in xrange(max_segments):
            segments_i = segments == i
            feature_img[segments_i] = feature[i]

        return feature_img

    def up_sample(self, img_lab, saliency, img_segments_mean, coordinate_segments_mean):
        size = img_lab.size / 3
        shape = img_lab.shape
        # size = img.size
        a = shape[0]
        b = shape[1]
        x_axis = np.linspace(0, b - 1, num=b)
        y_axis = np.linspace(0, a - 1, num=a)

        x_coordinate = np.tile(x_axis, (a, 1,))  # 创建X轴的坐标表
        y_coordinate = np.tile(y_axis, (b, 1,))  # 创建y轴的坐标表
        y_coordinate = np.transpose(y_coordinate)

        c_i = np.concatenate(
            (img_lab[:, :, 0].reshape(size, 1), img_lab[:, :, 1].reshape(size, 1), img_lab[:, :, 2].reshape(size, 1)),
            axis=1)
        p_i = np.concatenate((x_coordinate.reshape(size, 1), y_coordinate.reshape(size, 1)), axis=1)
        w_ij = np.exp(
            -1.0 / (2 * 30) * (cdist(c_i, img_segments_mean) ** 2 + cdist(p_i, coordinate_segments_mean) ** 2))
        w_ij = w_ij / w_ij.sum(axis=1)[:, None]
        if len(saliency.shape) != 2 or saliency.shape[1] != 1:
            saliency = saliency[:, None]
        saliency_pixel = np.dot(w_ij, saliency)
        return saliency_pixel.reshape(shape[0:2])

    def region_segment(self, features, segments):
        samples = features  # use lab color, uniqueness and distribution feature
        # samples = features

        # Unsupervised learning
        # bandwidth = estimate_bandwidth(samples, quantile=0.2)
        # ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        # lables = ms.fit_predict(samples)
        labels = KMeans(n_clusters=8).fit_predict(samples)

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

    def extract_region_features(self, img, segments, segments_region, labels):
        normalize = lambda s: (s - s.min()) / (s.max() - s.min())

        img_lab = img
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
        D = cdist(img_segments_mean, img_region_segments_mean) ** 2 - (np.linalg.norm(img_segments_mean - C_i_R,
                                                                                      axis=1) ** 2)[:, None]

        w_ij = 1 - np.exp(
            -cdist(coordinate_segments_mean, coordinate_region_segments_mean) ** 2 / (2 * self.__region_sigma1 ** 2))
        w_ij = w_ij / w_ij.sum(axis=1)[:, None]

        region_conlor_contrast = np.sum(w_ij * D, axis=1)
        region_conlor_contrast = normalize(region_conlor_contrast)

        wd_ij = np.exp(-cdist(img_segments_mean, img_region_segments_mean) ** 2 / (2 * self.__region_sigma2 ** 2))
        wd_ij = wd_ij / wd_ij.sum(axis=1)[:, None]
        mu_i = np.dot(wd_ij, coordinate_region_segments_mean)
        DR = np.sum(cdist(mu_i, coordinate_region_segments_mean) * wd_ij, axis=1)
        # DR = np.dot(wd_ij, np.linalg.norm(coordinate_segments_mean - mu_i, axis=1)**2)
        DR = normalize(DR)

        # fuse = DR * region_conlor_contrast
        # fuse = normalize(fuse)

        return np.concatenate((region_conlor_contrast[:, None], DR[:, None]), axis=1)

    def extract_region_features2(self, features, region_features, labels):
        normalize = lambda s: (s - s.min()) / (s.max() - s.min())

        # for lab color
        # img_segments_mean = features[:, 0:3]
        # coordinate_segments_mean = features[:, 3:5]
        # img_region_segments_mean = region_features[:, 0:3]
        # coordinate_region_segments_mean = region_features[:, 3:5]

        # for rgb color
        img_segments_mean = features[:, 5:8]
        coordinate_segments_mean = features[:, 3:5]
        img_region_segments_mean = region_features[:, 5:8]
        coordinate_region_segments_mean = region_features[:, 3:5]

        r = np.unique(labels, return_counts=True)
        size = r[1]*1.0/labels.size

        # every superpixel corresponds to its region mean value
        # C_i_R = np.apply_along_axis(lambda x: img_region_segments_mean[x[0], :], 1, np.array([labels]).T)
        # D = cdist(img_segments_mean, img_region_segments_mean) - (np.linalg.norm(img_segments_mean - C_i_R,
        #                                                                               axis=1))[:, None]
        D = cdist(img_segments_mean, img_region_segments_mean)

        w_ij = 1 - np.exp(
            -cdist(coordinate_segments_mean, coordinate_region_segments_mean) * self.__region_sigma1)
        w_ij = w_ij / w_ij.sum(axis=1)[:, None]

        region_conlor_contrast = np.sum(w_ij * D * size, axis=1)
        region_conlor_contrast = normalize(region_conlor_contrast)

        # wd_ij = np.exp(-cdist(img_segments_mean, img_region_segments_mean) / self.__region_sigma2)
        wd_ij = np.exp(-cdist(img_segments_mean, img_region_segments_mean) ** 2 / (2 * self.__region_sigma2 ** 2))
        wd_ij = wd_ij / wd_ij.sum(axis=1)[:, None]
        mu_i = np.dot(wd_ij, coordinate_region_segments_mean)
        DR = np.sum(cdist(mu_i, coordinate_region_segments_mean) * wd_ij * size, axis=1)
        # DR = np.dot(wd_ij, np.linalg.norm(coordinate_segments_mean - mu_i, axis=1)**2)
        DR = normalize(DR)

        return np.concatenate((region_conlor_contrast[:, None], DR[:, None]), axis=1)

    def product_saliency_map(self, original_img_dir, saliency_img_out_dir):
        normalize = lambda s: (s - s.min()) / (s.max() - s.min())
        if not os.path.exists(saliency_img_out_dir):
            os.mkdir(saliency_img_out_dir)
        list_dir = filter(lambda s: s.split('.')[-1] == 'jpg', os.listdir(original_img_dir))
        for f in list_dir:
            img = io.imread(original_img_dir + os.sep + f)
            features, segments = self.extract_features(img)
            io.imsave(saliency_img_out_dir + os.sep + f.split('.')[0] + '.png',
                      self.feature_to_image(normalize(features[:, 0] * np.exp(-3 * features[:, 1])), segments))

    def product_saliency_map_use_cache(self, func, cache_dir, saliency_img_out_dir):
        if not os.path.exists(saliency_img_out_dir):
            os.mkdir(saliency_img_out_dir)
        segments_dir = cache_dir + "_segments"
        if not os.path.exists(cache_dir) or not os.path.exists(segments_dir):
            raise NameError("cache directory not found!")
        list_dir = filter(lambda s: s.split('.')[-1] == 'npy', os.listdir(cache_dir))
        for f in list_dir:
            feature = np.load(cache_dir + os.sep + f)
            segments = np.load(segments_dir + os.sep + f)
            img_segments_mean = feature[:, 0:3]
            coordinate_segments_mean = feature[:, 3:5]
            distribution = self.distribution_feature(img_segments_mean, coordinate_segments_mean)
            uniqueness = self.uniqueness_feature(img_segments_mean, coordinate_segments_mean)
            saliency_map = self.feature_to_image(self.normalize(func(uniqueness, distribution)), segments)
            io.imsave(saliency_img_out_dir + os.sep + f.split(".")[0] + '.png', saliency_map)

    def show_saliency_map(self, func, cache_dir):
        segments_dir = cache_dir + "_segments"
        if not os.path.exists(cache_dir) or not os.path.exists(segments_dir):
            raise NameError("cache directory not found!")
        list_dir = filter(lambda s: s.split('.')[-1] == 'npy', os.listdir(cache_dir))
        for f in list_dir:
            feature = np.load(cache_dir + os.sep + f)
            segments = np.load(segments_dir + os.sep + f)
            img_segments_mean = feature[:, 0:3]
            coordinate_segments_mean = feature[:, 3:5]
            distribution = self.distribution_feature(img_segments_mean, coordinate_segments_mean)
            uniqueness = self.uniqueness_feature(img_segments_mean, coordinate_segments_mean)
            io.imshow(self.feature_to_image(self.normalize(func(uniqueness, distribution)), segments))
            io.show()

    def show_region_saliency_map(self, cache_dir):
        segments_dir = cache_dir + "_segments"
        img_lab_dir = cache_dir + "_img_lab"
        if not os.path.exists(cache_dir) or not os.path.exists(segments_dir) or not os.path.exists(img_lab_dir):
            raise NameError("cache directory not found!")
        list_dir = filter(lambda s: s.split('.')[-1] == 'npy', os.listdir(cache_dir))
        for f in list_dir:
            print f
            feature = np.load(cache_dir + os.sep + f)
            segments = np.load(segments_dir + os.sep + f)
            img = np.load(img_lab_dir + os.sep + f)
            segmensts_region, region_labels = self.region_segment(feature[:, 0:5], segments)
            region_feature = self.extract_region_features(img, segments, segmensts_region, region_labels)
            io.imshow(self.feature_to_image(region_feature[:, 0], segments))
            io.show()
            io.imshow(self.feature_to_image(region_feature[:, 1], segments))
            io.show()

    def distribution_feature(self, img_segments_mean, coordinate_segments_mean):
        wc_ij = np.exp(-cdist(img_segments_mean, img_segments_mean) ** 2 / (2 * self.__global_sigma3 ** 2))
        wc_ij = wc_ij / wc_ij.sum(axis=1)[:, None]
        mu_i = np.dot(wc_ij, coordinate_segments_mean)
        distribution = np.dot(wc_ij, np.linalg.norm(coordinate_segments_mean - mu_i, axis=1) ** 2)
        distribution = self.normalize(distribution)
        distribution = np.array([distribution]).T
        return distribution

    def uniqueness_feature(self, img_segments_mean, coordinate_segments_mean):
        wp_ij = np.exp(
            -cdist(coordinate_segments_mean, coordinate_segments_mean) ** 2 / (2 * self.__global_sigma1 ** 2))
        wp_ij = wp_ij / wp_ij.sum(axis=1)[:, None]

        # uniqueness plus
        w_ij = 1 - np.exp(
            -cdist(coordinate_segments_mean, coordinate_segments_mean) ** 2 / (2 * self.__global_sigma2 ** 2))
        w_ij = w_ij / w_ij.sum(axis=1)[:, None]
        mu_i_c = np.dot(wp_ij, img_segments_mean)
        uniqueness_plus = np.sum(cdist(img_segments_mean, mu_i_c) ** 2 * w_ij, axis=1)
        uniqueness_plus = self.normalize(uniqueness_plus)
        uniqueness_plus = np.array([uniqueness_plus]).T
        return uniqueness_plus

    def uniqueness_feature2(self, img_segments_mean, coordinate_segments_mean):
        # wp_ij = np.exp(
        #     -cdist(coordinate_segments_mean, coordinate_segments_mean) * self.__global_sigma1)
        # wp_ij = wp_ij / wp_ij.sum(axis=1)[:, None]

        # uniqueness plus
        w_ij = 1 - np.exp(-cdist(coordinate_segments_mean, coordinate_segments_mean) * self.__global_sigma2)
        w_ij = w_ij / w_ij.sum(axis=1)[:, None]
        # mu_i_c = np.dot(wp_ij, img_segments_mean)
        uniqueness_plus = np.sum(cdist(img_segments_mean, img_segments_mean) * w_ij, axis=1)
        uniqueness_plus = self.normalize(uniqueness_plus)
        uniqueness_plus = np.array([uniqueness_plus]).T
        return uniqueness_plus

    def uniqueness_original_feature(self, img_segments_mean, coordinate_segments_mean):
        wp_ij = np.exp(
            -cdist(coordinate_segments_mean, coordinate_segments_mean) ** 2 / (2 * self.__global_sigma2 ** 2))
        wp_ij = wp_ij / wp_ij.sum(axis=1)[:, None]

        uniqueness = np.sum(cdist(img_segments_mean, img_segments_mean)**2 * wp_ij, axis=1);
        uniqueness = self.normalize(uniqueness)

        return uniqueness

    def find_max_method(self, func, cache_dir):
        segments_dir = cache_dir + "_segments"
        img_lab_dir = cache_dir + "_img_lab"
        if not os.path.exists(cache_dir) or not os.path.exists(segments_dir) or not os.path.exists(img_lab_dir):
            raise NameError("some dir not found!" + cache_dir)

        list_dir = filter(lambda s: s.split('.')[-1] == 'npy', os.listdir(cache_dir))
        feature_list = []
        for f in list_dir:
            feature_list.append(np.load(cache_dir + os.sep + f))
        feature_array = np.concatenate(feature_list, axis=0)
        labels = feature_array[:, -1] > 0.9

        max_score = 0
        for sigma1 in xrange(10, 101):
            for sigma3 in xrange(10, 101):
                self.__global_sigma1 = sigma1
                self.__global_sigma3 = sigma3
                predict_list = []
                for feature in feature_list:
                    img_segments_mean = feature[:, 0:3]
                    coordinate_segments_mean = feature[:, 3:5]
                    distribution = self.distribution_feature(img_segments_mean, coordinate_segments_mean)
                    uniqueness = self.uniqueness_feature(img_segments_mean, coordinate_segments_mean)
                    predict_list.append(self.normalize(func(uniqueness, distribution)))
                score = average_precision_score(labels, np.concatenate(predict_list))
                if score > max_score:
                    max_score = score
                    print "sigma1:{0}, sigma3:{1}, max_score:{2}".format(sigma1, sigma3, max_score)

    def optimize_params(self, cache_dir):
        segments_dir = cache_dir + "_segments"
        img_lab_dir = cache_dir + "_img_lab"
        if not os.path.exists(cache_dir) or not os.path.exists(segments_dir) or not os.path.exists(img_lab_dir):
            raise NameError("some dir not found!" + cache_dir)

        list_dir = filter(lambda s: s.split('.')[-1] == 'npy', os.listdir(cache_dir))
        feature_list = []
        for f in list_dir:
            feature_list.append(np.load(cache_dir + os.sep + f))

        x0 = [50, 50, 50]

        def callback(xk):
            print xk
            return xk

        res = optimize.fmin_powell(self.value_params, x0, args=(feature_list,), ftol=1e-6, callback=callback, disp=True)
        return res

    def get_params_res(self, params, cache_dir):
        segments_dir = cache_dir + "_segments"
        img_lab_dir = cache_dir + "_img_lab"
        if not os.path.exists(cache_dir) or not os.path.exists(segments_dir) or not os.path.exists(img_lab_dir):
            raise NameError("some dir not found!" + cache_dir)

        list_dir = filter(lambda s: s.split('.')[-1] == 'npy', os.listdir(cache_dir))
        feature_list = []
        for f in list_dir:
            feature_list.append(np.load(cache_dir + os.sep + f))
        res = self.value_params(params, feature_list)
        return res

    def value_params(self, x, *args):
        """
        this function is for optimize
        :param x: sigma1 --- sigma3
        :param args: (func, feature_list, labels,)
        :return: -score, because of optimize function is for minimize
        """
        self.__global_sigma1, self.__global_sigma2, self.__global_sigma3 = x
        feature_list = args[0]

        func = lambda u, d: u * np.exp(-self.__global_fuse_c * d)
        score_array = np.zeros(len(feature_list))
        for i, feature in enumerate(feature_list):
            img_segments_mean = feature[:, 0:3]
            coordinate_segments_mean = feature[:, 3:5]
            label = feature[:, -1]
            distribution = self.distribution_feature(img_segments_mean, coordinate_segments_mean)
            uniqueness = self.uniqueness_feature(img_segments_mean, coordinate_segments_mean)
            predict = self.normalize(func(uniqueness, distribution))
            score_array[i] = mean_absolute_error(label, predict)

        score = score_array.mean()
        return score

    def value_params_uniqueness(self, x, *args):
        self.__global_sigma1, self.__global_sigma2 = x
        feature_list = args[0]

        score_array = np.zeros(len(feature_list))
        for i, feature in enumerate(feature_list):
            img_segments_mean = feature[:, 0:3]
            coordinate_segments_mean = feature[:, 3:5]
            label = feature[:, -1]
            uniqueness = self.uniqueness_feature(img_segments_mean, coordinate_segments_mean)
            predict = uniqueness
            score_array[i] = mean_absolute_error(label, predict)

        score = score_array.mean()
        return score

    def value_params_uniqueness2(self, x, *args):
        self.__global_sigma2 = x
        feature_list = args[0]

        score_array = np.zeros(len(feature_list))
        for i, feature in enumerate(feature_list):
            img_segments_mean = feature[:, 0:3]
            coordinate_segments_mean = feature[:, 3:5]
            label = feature[:, -1]
            uniqueness = self.uniqueness_feature2(img_segments_mean, coordinate_segments_mean)
            predict = uniqueness
            score_array[i] = mean_absolute_error(label, predict)

        score = score_array.mean()
        print "score: {0}, x:{1}".format(score, x)
        return score


    def value_params_original_uniqueness(self, x, *args):
        self.__global_sigma2 = x
        feature_list = args[0]

        score_array = np.zeros(len(feature_list))
        for i, feature in enumerate(feature_list):
            img_segments_mean = feature[:, 0:3]
            coordinate_segments_mean = feature[:, 3:5]
            label = feature[:, -1]
            uniqueness = self.uniqueness_original_feature(img_segments_mean, coordinate_segments_mean)
            predict = uniqueness
            score_array[i] = mean_absolute_error(label, predict)

        score = score_array.mean()
        print "score: {0}, x:{1}".format(score, x)
        return score

    def value_params_distribution(self, x, *args):
        self.__global_sigma3 = x
        feature_list = args[0]

        score_array = np.zeros(len(feature_list))
        for i, feature in enumerate(feature_list):
            img_segments_mean = feature[:, 0:3]
            coordinate_segments_mean = feature[:, 3:5]
            label = feature[:, -1]
            distribution = self.distribution_feature(img_segments_mean, coordinate_segments_mean)
            predict = (1 - distribution)
            score_array[i] = mean_absolute_error(label, predict)

        score = score_array.mean()
        print "score: {0}, x:{1}".format(score, x)
        return score

    def value_params_region_color(self, x, *args):
        self.__region_sigma1 = x
        feature_list = args[0]
        segments_list = args[1]
        img_list = args[2]

        score_array = np.zeros(len(feature_list))
        for i, feature in enumerate(feature_list):
            label = feature[:, -1]
            segments_region, region_label = self.region_segment(feature[:, 0:5], segments_list[i])
            region_feature = self.extract_region_features(img_list[i], segments_list[i], segments_region, region_label)
            predict = region_feature[:, 0]
            score_array[i] = mean_absolute_error(label, predict)

        score = score_array.mean()
        return score

    def value_params_region_color2(self, x, *args):
        if np.abs(x) < 1e-3:
            self.__region_sigma1 = x + 1e-3
        else:
            self.__region_sigma1 = x
        feature_list = args[0]
        region_feature_list = args[1]
        labels_list = args[2]

        score_array = np.zeros(len(feature_list))
        for i, feature in enumerate(feature_list):
            label = feature[:, -1]
            region_feature = self.extract_region_features2(feature, region_feature_list[i], labels_list[i])
            predict = region_feature[:, 0]
            score_array[i] = mean_absolute_error(label, predict)

        score = score_array.mean()
        print "score: {0}, x:{1}".format(score, x)
        return score

    def value_params_region_distribution(self, x, *args):
        if np.abs(x) < 1e-3:
            self.__region_sigma2 = x + 1e-3
        else:
            self.__region_sigma2 = x
        feature_list = args[0]
        segments_list = args[1]
        img_list = args[2]

        score_array = np.zeros(len(feature_list))
        for i, feature in enumerate(feature_list):
            label = feature[:, -1]
            segments_region, region_label = self.region_segment(feature[:, 0:5], segments_list[i])
            region_feature = self.extract_region_features2(img_list[i], segments_list[i], segments_region, region_label)
            predict = (1 - region_feature[:, 1])
            score_array[i] = mean_absolute_error(label, predict)

        score = score_array.mean()
        return score

    def value_params_region_distribution2(self, x, *args):
        self.__region_sigma2 = x
        feature_list = args[0]
        region_feature_list = args[1]
        labels_list = args[2]

        score_array = np.zeros(len(feature_list))
        for i, feature in enumerate(feature_list):
            label = feature[:, -1]
            region_feature = self.extract_region_features2(feature, region_feature_list[i], labels_list[i])
            predict = (1 - region_feature[:, 1])
            score_array[i] = mean_absolute_error(label, predict)

        score = score_array.mean()
        print "score: {0}, x:{1}".format(score, x)
        return score

    def optimize_params_region(self, cache_dir, color_feature=True):
        segments_dir = cache_dir + "_segments"
        img_lab_dir = cache_dir + "_img_lab"
        if not os.path.exists(cache_dir) or not os.path.exists(segments_dir) or not os.path.exists(img_lab_dir):
            raise NameError("some dir not found!" + cache_dir)

        list_dir = filter(lambda s: s.split('.')[-1] == 'npy', os.listdir(cache_dir))
        feature_list = []
        img_list = []
        segments_list = []
        for f in list_dir:
            feature_list.append(np.load(cache_dir + os.sep + f))
            img_list.append(np.load(img_lab_dir + os.sep + f))
            segments_list.append(np.load(segments_dir + os.sep + f))

        x0 = 150

        def callback(xk):
            print xk
            return xk

        if color_feature:
            res = optimize.fmin_powell(self.value_params_region_color, x0, args=(feature_list, segments_list, img_list),
                                       ftol=1e-3, callback=callback, disp=True)
        else:
            res = optimize.fmin_powell(self.value_params_region_distribution, x0,
                                       args=(feature_list, segments_list, img_list), ftol=1e-3, callback=callback,
                                       disp=True)
        return res

    def optimize_params_region2(self, cache_dir, color_feature=True):
        region_features_dir = cache_dir + "_region"
        region_labels_dir = cache_dir + "_region_labels"

        if not os.path.exists(cache_dir):
            raise NameError("some dir not found!" + cache_dir)

        list_dir = filter(lambda s: s.split('.')[-1] == 'npy', os.listdir(cache_dir))
        feature_list = []
        region_features_list = []
        region_labels_list = []
        for f in list_dir:
            feature_list.append(np.load(cache_dir + os.sep + f))
            region_features_list.append(np.load(region_features_dir + os.sep + f))
            region_labels_list.append(np.load(region_labels_dir + os.sep + f))

        x0 = 50

        def callback(xk):
            print xk
            return xk

        if color_feature:
            res = optimize.fmin_powell(self.value_params_region_color2, x0,
                                       args=(feature_list, region_features_list, region_labels_list), ftol=1e-3,
                                       callback=callback, disp=True)
        else:
            res = optimize.fmin_powell(self.value_params_region_distribution2, x0,
                                       args=(feature_list, region_features_list, region_labels_list), ftol=1e-3,
                                       callback=callback, disp=True)
        return res

    def optimize_params_global(self, cache_dir, is_uniqueness=True, x0=None):
        if not os.path.exists(cache_dir):
            raise NameError("some dir not found!" + cache_dir)

        list_dir = filter(lambda s: s.split('.')[-1] == 'npy', os.listdir(cache_dir))
        feature_list = []
        for f in list_dir:
            feature_list.append(np.load(cache_dir + os.sep + f))

        def callback(xk):
            print xk
            return xk

        if x0 is None: x0 = 10

        if is_uniqueness:
            res = optimize.fmin_powell(self.value_params_uniqueness2, x0, args=(feature_list,), ftol=1e-3,
                                       callback=callback, disp=True)
        else:
            res = optimize.fmin_powell(self.value_params_distribution, x0, args=(feature_list,), ftol=1e-3,
                                       callback=callback, disp=True)
        return res

    def optimize_params_uniqueness(self, cache_dir):
        segments_dir = cache_dir + "_segments"
        if not os.path.exists(cache_dir) or not os.path.exists(segments_dir):
            raise NameError("some dir not found!" + cache_dir)

        list_dir = filter(lambda s: s.split('.')[-1] == 'npy', os.listdir(cache_dir))
        feature_list = []
        for f in list_dir:
            feature_list.append(np.load(cache_dir + os.sep + f))

        def callback(xk):
            print xk
            return xk

        x0 = 200
        res = optimize.fmin_powell(self.value_params_original_uniqueness, x0, args=(feature_list,), ftol=1e-3,
                                   callback=callback, disp=True)
        return res


# %% test
if __name__ == "__main__":
    # normalize = lambda s: (s - s.min()) / (s.max() - s.min())
    # get_filter_kernel = lambda x, y: cv2.mulTransposed(cv2.getGaussianKernel(x, y), False)
    # list_dir = os.listdir(cache_out_dir)
    # list_dir = filter(lambda f: os.path.splitext(f)[-1] == '.npy', list_dir)
    # segments_dir = cache_out_dir + "_segments"
    # img_lab_dir = cache_out_dir + "_img_lab"
    # segments_mean_dir = cache_out_dir + "_segments_mean"
    # if not os.path.exists(cache_out_dir) or not os.path.exists(segments_dir):
    #     raise NameError('cache dir cannot be found!')
    if not os.path.exists(saliency_img_out_dir):
        os.mkdir(saliency_img_out_dir)
    # for f in list_dir:
    #     features = np.load(cache_out_dir + os.sep + f.split('.')[0] + '.npy')
    #     segments = np.load(segments_dir + os.sep + f.split('.')[0] + '.npy')
    #     segments_mean = np.load(segments_mean_dir + os.sep + f.split('.')[0] + '.npy')
    #     img_lab = np.load(img_lab_dir + os.sep + f.split('.')[0] + '.npy')
    #
    #     sf = normalize(features[:, 5]*np.exp(-features[:, 6]))
    #     saliency_pixel = up_sample(img_lab, sf, segments_mean[:, 0:3], segments_mean[:, 3:5])
    #     io.imsave(saliency_img_out_dir + os.sep + f.split('.')[0] + '.png', saliency_pixel)

    sf1 = SFPlus()
    # sf1.find_max_method(lambda u,d: u*d, cache_out_dir)
    sigma = [16.23815252, 11.40926778, 6.36241358]  # value 0.147060, lambda u,d: u*(1-d)
    sigma = [1.31658874e+01, 1.42586799e+01, 3.00303736e+06,
             4.27952145e+00]  # value 0.142705865285, lambda u,d: u*np.exp(-sigma[3]*d)
    simga = [13.83741607, 10.10849983, 6.48653194]  # value 0.145809, lambda u,d: u*np.exp(-3*d)
    sigma_region = [50.48809074, 3992.48658874]  # x0 = 50
    # sigma = [10, 20, 60]
    # print sf1.get_params_res(sigma, lambda u,d: u*(1-d), cache_out_dir)
    # print sf1.optimize_params(cache_out_dir)
    # sf1.show_saliency_map(lambda u,d: u*d, cache_out_dir)
    # sf1.set_global_sigma(sigma[0], sigma[1], sigma[2])
    # sf1.product_saliency_map_use_cache(lambda u,d: u*np.exp(-3*d), cache_out_dir, saliency_img_out_dir)
    # sf1.show_region_saliency_map(cache_out_dir)
    # sf1.optimize_params_region(cache_out_dir)
    # sf1.optimize_params_region(cache_out_dir, False)
    # sf1.set_global_fuse_c(3)
    # sf1.optimize_params(cache_out_dir)
    # print "region color"
    # print sf1.optimize_params_region2(cache_out_dir)
    # print "region distribution"
    # print sf1.optimize_params_region2(cache_out_dir, False)
    # print "uniqueness"
    # print sf1.optimize_params_global(cache_out_dir)
    # print "distribution"
    # print sf1.optimize_params_global(cache_out_dir, False)
    # print "original uniqueness"
    # print sf1.optimize_params_uniqueness(cache_out_dir)
    # print "uniqueness_plus"
    # for i in xrange(10, 110, 10):
    #     for j in xrange(10, 110, 10):
    #         print "i: {0}, j: {1}".format(i, j)
    #         print sf1.optimize_params_global(cache_out_dir, x0=[i, j])

    region_features_dir = cache_out_dir + "_region"
    region_labels_dir = cache_out_dir + "_region_labels"
    segments_dir = cache_out_dir + "_segments"
    list_dir = filter(lambda s: s.split('.')[-1] == 'npy', os.listdir(cache_out_dir))
    feature_list = []
    region_features_list = []
    region_labels_list = []
    sf1.set_region_sigma(50, 23)
    for f in list_dir[10:]:
        # print f
        feature = np.load(cache_out_dir + os.sep + f)
        region_feature = np.load(region_features_dir + os.sep + f)
        region_labels = np.load(region_labels_dir + os.sep + f)
        segments = np.load(segments_dir + os.sep + f)

        # img_segments_mean = feature[:, 5:8]
        # coordinate_segments_mean = feature[:, 3:5]
        # u = sf1.uniqueness_feature2(img_segments_mean, coordinate_segments_mean)
        # d = sf1.distribution_feature(img_segments_mean, coordinate_segments_mean)
        # io.imshow(sf1.feature_to_image(u, segments))
        # io.show()
        # io.imshow(sf1.feature_to_image(d, segments))
        # io.show()

        r = sf1.extract_region_features2(feature, region_feature, region_labels)
        io.imshow(sf1.feature_to_image(r[:, 0], segments))
        io.show()
        io.imshow(sf1.feature_to_image(1-r[:, 1], segments))
        io.show()

