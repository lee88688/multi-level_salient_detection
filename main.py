# -*- coding: utf-8 -*-

from multiprocessing import Process, freeze_support, Value, Pool
from sendmail import send_mail
import find_pictures_use_region as fp
import save_features as sf
import numpy as np
from path import *
from time import time, sleep
from skimage import io
from sklearn.ensemble import RandomForestClassifier

cache_out_dir = cache_out_dir9

def find_pictures():
    start = time()
    max_score = Value('d', 0.0)
    # print_max_score_multiprocess(max_score, dut_cache_out_dir, 10000, 5, 10)
    p1 = Process(target=fp.print_max_score_multiprocess, args=(max_score, cache_out_dir, 10000, 5, 1))
    p2 = Process(target=fp.print_max_score_multiprocess, args=(max_score, cache_out_dir, 10000, 5, 1))
    p3 = Process(target=fp.print_max_score_multiprocess, args=(max_score, cache_out_dir, 10000, 5, 1))
    p4 = Process(target=fp.print_max_score_multiprocess, args=(max_score, cache_out_dir, 10000, 5, 1))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
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

    sf.save_features_from_general_cache5(original_img_dir, general300_cache_out_dir, cache_out_dir, 'jpg')

    stop = time()
    print "total time is: " + str(stop - start)
    # send_mail("the cache file production has completed, the total time is {0} hours.".format((stop - start) / 3600.0))


def save_feature_multiprocess():
    start = time()

    # list_features_dir = os.listdir(original_img_dir)
    # pics = filter(lambda f: f.split('.')[-1] == "jpg", list_features_dir)
    # p1 = Process(target=sf.save_general_features_multiprocess, args=(pics[0:len(pics)/4], original_img_dir, binary_img_dir, general300_cache_out_dir, 300, 'jpg', 'bmp'))
    # p2 = Process(target=sf.save_general_features_multiprocess, args=(pics[len(pics)/4:len(pics)*2/4], original_img_dir, binary_img_dir, general300_cache_out_dir, 300, 'jpg', 'bmp'))
    # p3 = Process(target=sf.save_general_features_multiprocess, args=(pics[len(pics)*2/4:len(pics)*3/4],original_img_dir, binary_img_dir, general300_cache_out_dir, 300, 'jpg', 'bmp'))
    # p4 = Process(target=sf.save_general_features_multiprocess, args=(pics[len(pics)*3/4:], original_img_dir, binary_img_dir, general300_cache_out_dir, 300, 'jpg', 'bmp'))
    # p1.start()
    # p2.start()
    # p3.start()
    # p4.start()
    # p1.join()
    # p2.join()
    # p3.join()
    # p4.join()

    list_features_dir = os.listdir(original_img_dir)
    pics = filter(lambda f: f.split('.')[-1] == "jpg", list_features_dir)
    p1 = Process(target=sf.save_features_from_general_cache_multiprocess, args=(pics[0:len(pics)/4], original_img_dir, general300_cache_out_dir, cache_out_dir, 'jpg'))
    p2 = Process(target=sf.save_features_from_general_cache_multiprocess, args=(pics[len(pics)/4:len(pics)*2/4],original_img_dir, general300_cache_out_dir, cache_out_dir, 'jpg'))
    p3 = Process(target=sf.save_features_from_general_cache_multiprocess, args=(pics[len(pics)*2/4:len(pics)*3/4],original_img_dir, general300_cache_out_dir, cache_out_dir, 'jpg'))
    p4 = Process(target=sf.save_features_from_general_cache_multiprocess, args=(pics[len(pics)*3/4:],original_img_dir, general300_cache_out_dir, cache_out_dir, 'jpg'))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()

    stop = time()
    print "total time is: " + str(stop - start)


def product_pictures():
    pic_list = ['4_143_143199.npy', '0_23_23666.npy', '4_141_141591.npy', '2_77_77109.npy', '2_68_68932.npy']
    pic_list = map(lambda s: s.split('.')[0] + '.jpg', pic_list)
    # fp.product_saliency_image_use_cache(cache_out_dir, cache_out_dir, pic_list, 1, "mr")
    fp.product_saliency_image_use_selected_features(cache_out_dir, cache_out_dir, general_cache_out_dir, pic_list, 1, None, "reduce1")
    # fp.product_saliency_feature_use_cache(cache_out_dir, cache_out_dir, pic_list, 1, "mr")


def product_pictures_upsample():
    pic_list = ['0_24_24918.npy', '0_11_11830.npy', '3_110_110864.npy', '0_22_22047.npy', '0_15_15859.npy']
    pic_list = map(lambda s: s.split('.')[0] + '.jpg', pic_list);
    fp.product_saliency_image_use_cache_upsample2(cache_out_dir, general_cache_out_dir, cache_out_dir,
                                               original_img_dir, pic_list, 1, "upsample")


def mr_saliency():
    from random import shuffle
    predicts_dir = r'G:\Project\paper2\out\feature\out5_1_ext5'
    images_dir = r'G:\Project\paper2\out\image\out'
    # predicts_to_images(predicts_dir, images_dir, cache_out_dir)

    features_dir = cache_out_dir
    segments_dir = general_cache_out_dir + "_segments"
    neighbor_dir = cache_out_dir + "_neighbor"
    region_labels_dir = general_cache_out_dir + "_region_labels"

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
        region_labels = np.load(region_labels_dir + os.sep + f)
        io.imshow(fp.feature_to_image(predict, segments))
        io.show()
        # io.imshow(feature_to_image(predict > predict.mean(), segments))
        # io.show()
        # io.imshow(feature_to_image(manifold_ranking_saliency(predict, feature, segments, neighbor), segments))
        # io.show()
        # io.imshow(feature_to_image(manifold_ranking_saliency2(predict, feature, segments, neighbor, region_labels), segments))
        # io.show()
        io.imshow(fp.feature_to_image(fp.manifold_ranking_saliency3(predict, feature[:, 0:-1], segments, neighbor), segments))
        io.show()


def mr_saliency_save():
    predicts_dir = r'G:\Project\paper2\out\feature\out5_1_mr'
    images_dir = r'G:\Project\paper2\out\image\out'
    # predicts_to_images(predicts_dir, images_dir, cache_out_dir)

    features_dir = general_cache_out_dir
    segments_dir = general_cache_out_dir + "_segments"
    neighbor_dir = cache_out_dir + "_neighbor"
    region_labels_dir = general_cache_out_dir + "_region_labels"
    frame_info_dir = general_cache_out_dir + "_frame_info"

    mr_images_dir = images_dir + "_mr2"
    if not os.path.exists(mr_images_dir):
        os.mkdir(mr_images_dir)
    list_dir = filter(lambda s: s.split('.')[-1] == 'npy', os.listdir(predicts_dir))
    # shuffle(list_dir)
    for f in list_dir:
        predict = np.load(predicts_dir + os.sep + f)
        feature = np.load(features_dir + os.sep + f)
        segments = np.load(segments_dir + os.sep + f)
        neighbor = np.load(neighbor_dir + os.sep + f)
        region_labels = np.load(region_labels_dir + os.sep + f)
        frame = np.load(frame_info_dir + os.sep + f)
        mr_feature = fp.manifold_ranking_saliency(predict, feature[:, 0:3], segments, neighbor)
        img = sf.feature_to_image(mr_feature, segments, frame=frame)
        io.imsave(mr_images_dir + os.sep + f.split('.')[0] + ".png", img)


def mr_original():
    normalize = lambda s: (s - s.min()) / (s.max() - s.min())

    predicts_dir = r'G:\Project\paper2\out\feature\out5_1_region_300_local_surround'
    images_dir = r'G:\Project\paper2\out\image\out'
    # predicts_to_images(predicts_dir, images_dir, cache_out_dir)

    features_dir = general_cache_out_dir
    segments_dir = general_cache_out_dir + "_segments"
    neighbor_dir = cache_out_dir + "_neighbor"
    region_labels_dir = general_cache_out_dir + "_region_labels"

    mr_images_dir = images_dir + "_mr5"
    if not os.path.exists(mr_images_dir):
        os.mkdir(mr_images_dir)
    list_dir = filter(lambda s: s.split('.')[-1] == 'npy', os.listdir(predicts_dir))
    for f in list_dir:
        predict = np.load(predicts_dir + os.sep + f)
        feature = np.load(features_dir + os.sep + f)
        segments = np.load(segments_dir + os.sep + f)
        neighbor = np.load(neighbor_dir + os.sep + f)
        region_labels = np.load(region_labels_dir + os.sep + f)

        Aff = fp.manifold_ranking_aff(feature[:, 0:3], segments, neighbor.copy())

        salt = np.zeros(neighbor.shape[0])
        salt[np.unique(segments[0, :])] = 1
        salt = 1 - normalize(np.dot(Aff, salt))

        sald = np.zeros(neighbor.shape[0])
        sald[np.unique(segments[segments.shape[0] - 1, :])] = 1
        sald = 1 - normalize(np.dot(Aff, sald))

        sall = np.zeros(neighbor.shape[0])
        sall[np.unique(segments[:, 0])] = 1
        sall = 1 - normalize(np.dot(Aff, sall))

        salr = np.zeros(neighbor.shape[0])
        salr[np.unique(segments[:, segments.shape[1] - 1])] = 1
        salr = 1 - normalize(np.dot(Aff, salr))

        sal = salt * sald * sall * salr
        s = np.zeros(neighbor.shape[0])
        s[sal > sal.mean()] = 1
        s = normalize(np.dot(Aff, s))

        io.imsave(mr_images_dir + os.sep + f.split('.')[0] + '.png', fp.feature_to_image(s, segments))
        # return l


def RF_saliency():
    pic_list = ['1_46_46443.npy', '4_140_140686.npy', '2_68_68592.npy', '2_69_69370.npy', '2_68_68619.npy']
    feature, label = fp.get_features_use_cache(cache_out_dir, pic_list)
    clf = RandomForestClassifier(n_estimators=1000, n_jobs=4)
    clf.fit(feature, label > 0.9)
    return clf

    list_dir = filter(lambda s: s.split('.')[-1] == 'npy', os.listdir(cache_out_dir))
    frame_info_dir = general_cache_out_dir + "_frame_info"
    segments_dir = cache_out_dir + "_segments"
    out_dir = saliency_img_out_dir + "_RF"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for f in list_dir:
        feature = np.load(cache_out_dir + os.sep + f)
        feature = feature[:, 0:-1]
        segments = np.load(segments_dir + os.sep + f)
        predict_result = clf.predict_proba(feature)
        saliency_feature = predict_result[:, 1]
        if os.path.exists(frame_info_dir):
            frame = np.load(frame_info_dir + os.sep + f)
        else:
            frame = None
        saliency_img = sf.feature_to_image(saliency_feature, segments, frame)
        io.imsave(out_dir + os.sep + f.split(".")[0] + ".png", saliency_img)


if __name__ == "__main__":
    a = find_pictures()



