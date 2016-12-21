# -*- coding: utf-8 -*-

import os


if os.path.exists("G:/"):
    original_img_dir = r"G:\Project\paper2\other_image\MSRA-1000_images"
    binary_img_dir = r"G:\Project\paper2\other_image\binarymasks"
    general300_cache_out_dir = r'G:\Project\paper2\cache\cache_general_kmeans_300\feature_cache'
    cache_out_dir1 = r'G:\Project\paper2\cache\cache_features_local_surround_300\feature_cache'
    cache_out_dir5 = r'G:\Project\paper2\cache\cache_features_local_surround_300_5\feature_cache'
    cache_out_dir4 = r'G:\Project\paper2\cache\cache_features_local_surround_300_4\feature_cache'
    cache_out_dir8 = r'G:\Project\paper2\cache\cache_features_local_surround_300_8\feature_cache'
    cache_out_dir9 = r'G:\Project\paper2\cache\cache_features_local_surround_300_9\feature_cache'

    dut_original_img_dir = r"G:\Project\paper2\other_image\DUT\DUT-OMRON-image"
    dut_binary_img_dir = r"G:\Project\paper2\other_image\DUT\pixelwiseGT-new-PNG"
    dut_general300_cache_out_dir = r'G:\Project\paper2\cache\cache_dut_general_kmeans_300\feature_cache'
    dut_cache_out_dir = r'G:\Project\paper2\cache\cache_dut_features_local_surround_300\feature_cache'

    saliency_img_out_dir = r"G:\Project\paper2\out\image\out"
    saliency_feature_out_dir = r"G:\Project\paper2\out\feature\out"
    general_cache_out_dir = general300_cache_out_dir
else:
    original_img_dir = r"F:\lee\MSRA-1000_images"
    binary_img_dir = r"F:\lee\binarymasks"
    general300_cache_out_dir = r'F:\lee\cache\cache_general_kmeans_300\feature_cache'
    cache_out_dir1 = r'F:\lee\cache\cache_features_local_surround_300\feature_cache'
    cache_out_dir5 = r'F:\lee\cache\cache_features_local_surround_300_5\feature_cache'
    cache_out_dir4 = r'F:\lee\cache\cache_features_local_surround_300_4\feature_cache'
    cache_out_dir8 = r'F:\lee\cache\cache_features_local_surround_300_8\feature_cache'
    cache_out_dir9 = r'F:\lee\cache\cache_features_local_surround_300_9\feature_cache'

    dut_original_img_dir = r"F:\lee\DUT\DUT-OMRON-image"
    dut_binary_img_dir = r"F:\lee\DUT\pixelwiseGT-new-PNG"
    dut_general300_cache_out_dir = r'F:\lee\cache\cache_dut_general_kmeans_300\feature_cache'
    dut_cache_out_dir = r'F:\lee\cache\cache_dut_features_local_surround_300\feature_cache'

    saliency_img_out_dir = r"F:\lee\saliency_map\out"
    saliency_feature_out_dir = r"F:\lee\predict_result"
    general_cache_out_dir = general300_cache_out_dir

features_path = os.getcwd() + os.sep + "features"
test_out_path = os.getcwd() + os.sep + "tests"