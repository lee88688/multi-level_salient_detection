# -*- coding: utf-8 -*-

from skimage.future import graph
from skimage import data, io, segmentation, color
import os
import numpy as np
from rpca import robust_pca
from multiprocessing import Process, Value, freeze_support

d = r"G:\Project\paper2\images"
original_img_dir = r"G:\Project\paper2\other_image\MSRA-1000_images"

cache_dir = r'G:\Project\paper2\cache\cache_features\feature_cache'
segments_dir = cache_dir + "_segments"

general_cache_dir = r'G:\Project\paper2\cache\cache_general_kmeans\feature_cache'
general_region_features_dir = general_cache_dir + "_region"
general_segments_dir = general_cache_dir + "_segments"
general_region_segments_dir = general_cache_dir + "_region_segments"
general_region_labels_dir = general_cache_dir + "_region_labels"

normalize = lambda s: (s - s.min()) / (s.max() - s.min())

def group_segments(img_name, c=30.0):
    img = io.imread(img_name)
    label1 = segmentation.slic(img, compactness=c, n_segments=500)
    g = graph.rag_mean_color(img, label1, mode="similarity")
    label2 = graph.cut_normalized(label1, g)
    return color.label2rgb(label2, img, kind="avg")


def mark_slic(img_name, c=30.0):
    img = io.imread(img_name)
    segments = segmentation.slic(img, n_segments=500, compactness=c)
    return segmentation.mark_boundaries(img, segments)


def feature_to_image(feature, segments, enhance=False):
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

    feature_img = np.zeros_like(segments, dtype=np.float64)

    for i in xrange(max_segments):
        segments_i = segments == i
        feature_img[segments_i] = feature[i]

    return feature_img

#%% lab region segment
# img_name = d + os.sep + "(37).jpg"
# img = io.imread(img_name)
# label1 = segmentation.slic(img, compactness=30.0, n_segments=500)
# img_lab = color.rgb2lab(img)
# g = graph.rag_mean_color(img_lab, label1, mode="similarity", sigma=200)
# label2 = graph.cut_normalized(label1, g)
# io.imshow(color.label2rgb(label2, img, kind="avg"))
# print label2.max()

#%% region segment test
# list_dir = filter(lambda s: s.split(".")[-1] == "jpg", os.listdir(original_img_dir))
# for f in list_dir[:10]:
#     f = "0_13_13515.jpg"
#     img = io.imread(original_img_dir + os.sep + f)
#     segments = np.load(segments_dir + os.sep + f.split(".")[0] + ".npy")
#     region_segments = np.load(region_segments_dir + os.sep + f.split(".")[0] + ".npy")
#     io.imshow(segmentation.mark_boundaries(img, segments))
#     io.show()
#     io.imshow(segmentation.mark_boundaries(img, region_segments))
#     io.show()
#     break

# %% robust pca
# list_dir = filter(lambda s: s.split(".")[-1] == "npy", os.listdir(cache_dir))
# for f in list_dir[0:100]:
#     feature_array = np.load(cache_dir + os.sep + f)
#     segments = np.load(segments_dir + os.sep + f)
#     features = np.concatenate((feature_array[:, 5][:, None], (1-feature_array[:, 6])[:, None], feature_array[:, 7:10],
#                                feature_array[:, 12][:, None], 1-feature_array[:, 13][:, None]), axis=1)
#     label = feature_array[:, -1]
#     L, S = robust_pca(features)
#     # for i in range(S.shape[1]):
#     #     S[:, i] = normalize(S[:, i])
#     io.imshow(feature_to_image(normalize(np.linalg.norm(L, ord=None, axis=1)), segments))
#     io.show()

# %% region feature
# list_dir = filter(lambda s: s.split(".")[-1] == "npy", os.listdir(cache_dir))
# for f in list_dir[0:1]:
#     feature_array = np.load(cache_dir + os.sep + f)
#     general_features = np.load(general_cache_dir + os.sep + f)
#     general_region_features = np.load(general_region_features_dir + os.sep + f)
