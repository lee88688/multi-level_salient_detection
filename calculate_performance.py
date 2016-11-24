# -*- coding: utf-8 -*-

import numpy as np
from skimage import color, io
import os
import cv2


normalize = lambda s: (s - s.min()) / (s.max() - s.min()) * 255

def performance(input_image, gt_image):
    '''
    this function calculates performance between the saliency map and ground truth map.
    :param input_image: saliency map
    :param gt_image: corresponding ground truth map
    :return:
    '''

    if len(gt_image.shape) > 2:
        gt_image = (color.rgb2grey(gt_image) * 255) > 128
    if len(input_image.shape) > 2:
        input_image = (color.rgb2grey(input_image) * 255).astype(np.uint8)

    assert input_image.shape == gt_image.shape

    if gt_image.dtype is np.dtype(np.bool):
        gt_binary = gt_image
    else:
        gt_binary = gt_image > 128

    result = {}

    tp = np.histogram(input_image[gt_binary], np.arange(0, 257))
    fp = np.histogram(input_image[~gt_binary], np.arange(0, 257))

    total_tp = tp[0].sum()

    tp = (np.cumsum(tp[0][::-1]))[::-1]
    fp = (np.cumsum(fp[0][::-1]))[::-1]
    fn = total_tp - tp;

    P = tp / (tp + fp + 1e-3);
    R = tp / (tp + fn + 1e-3);

    AUC = -(np.diff(R) * (P[0:-1] + P[1:]) / 2).sum()
    AUC = AUC + R[-1] * P[-1];

    result['truePositives'] = tp
    result['falsePositives'] = fp
    result['falseNegatives'] = fn
    result['precision'] = P
    result['recall'] = R
    result['areaUnderCurve'] = AUC
    result['meanAbsoluteError'] = np.abs((input_image / 255.0) - gt_binary.astype(np.float32)).mean()

    beta_square = 0.3
    Ta = 2 * input_image.mean()
    segment_in = input_image > Ta
    r1, _ = np.histogram(segment_in[gt_binary], 2)
    r2, _ = np.histogram(segment_in[~gt_binary], 2)
    tp = r1[1]
    fn = r1[0]
    fp = r2[1]
    P = tp/(tp + fp + 1e-3);
    R = tp/(tp + fn + 1e-3);
    F1 = (1 + beta_square) * P * R / (beta_square * P + R + 1e-3);

    result['F1'] = F1
    result['P'] = P
    result['R'] = R

    return result


def translate_list2dic(performace_list):
    '''
    translate the performance list to performance dic
    :param performace_list: performance list containing performance dic
    :return: performance dic that contains every performance parameter array
    '''
    assert type(performace_list) is list
    assert type(performace_list[0]) is dict

    performance_dic = {k: [] for k in performace_list[0].keys()}

    for p in performace_list:
        for k in p.keys():
            performance_dic[k].append(p[k])

    for k in performance_dic.keys():
        performance_dic[k] = np.array(performance_dic[k])

    return performance_dic



def performance_results(saliency_map_dir, binary_map_dir, saliency_map_ext='png', binary_map_ext='bmp'):
    '''
    this function is used for calculating p-r curve, AUC, MAE, F1, Precision and Recall score.
    :param saliency_map_dir: the saliency map directory
    :param binary_map_dir: ground truth directory
    :param saliency_map_ext: saliency image extension
    :param binary_map_ext: binary image extension
    :return: performance dictionary with average of parameters
    '''

    if not os.path.exists(saliency_map_dir) or not os.path.exists(binary_map_dir):
        raise NameError("'{}' dir or '{}' dir not exist, check out!")
    if saliency_map_ext is None or binary_map_ext is None:
        raise NameError("can't give a None parameters!")

    file_list = filter(lambda s: s.split('.')[-1] == saliency_map_ext, os.listdir(saliency_map_dir))
    performance_list = []
    for f in file_list:
        # input_image = io.imread(saliency_map_dir + os.sep + f)
        input_image = cv2.imread(saliency_map_dir + os.sep + f, flags=cv2.CV_LOAD_IMAGE_GRAYSCALE)
        gt_image = io.imread(binary_map_dir + os.sep + f.split('.')[0] + '.' + binary_map_ext)
        performance_list.append(performance(input_image, gt_image))

    performance_dic = translate_list2dic(performance_list)
    result_dic = {}
    for k in performance_dic.keys():
        result_dic[k] = performance_dic[k].mean(axis=0)

    return result_dic, performance_dic


def test_performance():
    import scipy.io as sio
    # i = cv2.imread(r"G:\Project\paper2\out5_5_region_300_local_surround\0_0_77.png", flags=cv2.CV_LOAD_IMAGE_GRAYSCALE)
    i = io.imread(r"G:\Project\paper2\out5_5_region_300_local_surround\0_0_77.png")
    i = normalize(i.astype(np.float)).astype(np.uint8)
    b = io.imread(r"G:\Project\paper2\other_image\binarymasks\0_0_77.bmp")

    p = performance(i, b)
    p_m = sio.loadmat(r'G:\Project\paper2\matlab\matlab.mat')['performance']

    for name in p_m.dtype.names:
        print "{} performance difference is {}".format(name, np.abs(p[name] - p_m[name][0][0]).mean())

    return p, p_m


if __name__ == "__main__":
    # cc = test_performance()
    r = performance_results(r'G:\Project\paper2\out5_5_region_300_local_surround_30k',
                                          r'G:\Project\paper2\other_image\binarymasks')
