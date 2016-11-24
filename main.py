# -*- coding: utf-8 -*-

from multiprocessing import Process, freeze_support, Value
from sendmail import send_mail
from save_features import *
from find_pictures_use_region import *
from path import *
from time import time


def find_pictures():
    start = time()

    max_score = Value('d', 0.0)
    # print_max_score_multiprocess(max_score, dut_cache_out_dir, 10000, 5, 10)
    p1 = Process(target=print_max_score_multiprocess, args=(max_score, cache_out_dir5, 10000, 5, 1))
    p2 = Process(target=print_max_score_multiprocess, args=(max_score, cache_out_dir5, 10000, 5, 1))
    p3 = Process(target=print_max_score_multiprocess, args=(max_score, cache_out_dir5, 10000, 5, 1))
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
    stop = time()
    msg = "the work has been done.\n" + "total time is: " + str((stop - start) / 3600)
    send_mail(msg)
    print msg


def save_feature():
    start = time()

    # save_general_features_kmeans(original_img_dir, binary_img_dir, general300_cache_out_dir, 300, 'jpg', 'bmp')
    # save_general_features_kmeans(dut_original_img_dir, dut_binary_img_dir, dut_general300_cache_out_dir, 300, 'jpg', 'png')

    # save_features_from_general_cache3(original_img_dir, general300_cache_out_dir, cache_out_dir, 'jpg')
    # save_features_from_general_cache3(dut_original_img_dir, dut_general300_cache_out_dir, dut_cache_out_dir, 'jpg')

    # save_features_from_general_cache4(original_img_dir, general300_cache_out_dir, cache_out_dir, 'jpg')
    # save_features_from_general_cache4(dut_original_img_dir, dut_general300_cache_out_dir, dut_cache_out_dir, 'jpg')

    save_features_from_general_cache5(original_img_dir, general300_cache_out_dir, cache_out_dir5, 'jpg')

    stop = time()
    print "total time is: " + str(stop - start)
    # send_mail("the cache file production has completed, the total time is {0} hours.".format((stop - start) / 3600.0))


def product_pictures():
    pic_list = ['0_24_24918.npy', '0_11_11830.npy', '3_110_110864.npy', '0_22_22047.npy', '0_15_15859.npy']
    pic_list = map(lambda s: s.split('.')[0] + '.jpg', pic_list);
    # product_saliency_image_use_cache(cache_out_dir, cache_out_dir, pic_list, 5, "region_300_local_surround_30k_2")
    product_saliency_feature_use_cache(cache_out_dir, cache_out_dir, pic_list, 1, "region_300_local_surround")


def mr_saliency():
    from random import shuffle
    predicts_dir = r'G:\Project\paper2\out\feature\out5_1_region_300_local_surround'
    images_dir = r'G:\Project\paper2\out\image\out'
    predicts_to_images(predicts_dir, images_dir, cache_out_dir)

    features_dir = general_cache_out_dir
    segments_dir = general_cache_out_dir + "_segments"
    neighbor_dir = cache_out_dir + "_neighbor"

    mr_images_dir = images_dir + "_mr"
    if not os.path.exists(mr_images_dir):
        os.mkdir(mr_images_dir)
    list_dir = filter(lambda s: s.split('.')[-1] == 'npy', os.listdir(predicts_dir))
    # shuffle(list_dir)
    for f in list_dir:
        predict = np.load(predicts_dir + os.sep + f)
        feature = np.load(features_dir + os.sep + f)
        segments = np.load(segments_dir + os.sep + f)
        neighbor = np.load(neighbor_dir + os.sep + f)
        io.imshow(feature_to_image(predict, segments))
        io.show()
        io.imshow(feature_to_image(manifold_ranking_saliency(predict, feature, segments, neighbor), segments))
        io.show()


def mr_saliency_save():
    predicts_dir = r'G:\Project\paper2\out\feature\out5_1_region_300_local_surround'
    images_dir = r'G:\Project\paper2\out\image\out'
    # predicts_to_images(predicts_dir, images_dir, cache_out_dir)

    features_dir = general_cache_out_dir
    segments_dir = general_cache_out_dir + "_segments"
    neighbor_dir = cache_out_dir + "_neighbor"

    mr_images_dir = images_dir + "_mr5"
    if not os.path.exists(mr_images_dir):
        os.mkdir(mr_images_dir)
    list_dir = filter(lambda s: s.split('.')[-1] == 'npy', os.listdir(predicts_dir))
    # shuffle(list_dir)
    for f in list_dir:
        predict = np.load(predicts_dir + os.sep + f)
        feature = np.load(features_dir + os.sep + f)
        segments = np.load(segments_dir + os.sep + f)
        neighbor = np.load(neighbor_dir + os.sep + f)
        io.imsave(mr_images_dir + os.sep + f.split('.')[0] + ".png", feature_to_image(manifold_ranking_saliency(predict, feature, segments, neighbor), segments))


if __name__ == "__main__":
    save_feature()



