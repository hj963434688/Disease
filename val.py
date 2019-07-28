import tensorflow as tf
import train as tr
import net
import data_genertor
import os
import time
import cv2
import json
import numpy as np
import heapq as he

validate_path = '../dataset/AgriculDis/dataset/AgriculturalDisease_validationset/images/'
validate_Jsonpath= '../dataset/AgriculDis/dataset/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations.json'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
save_result = False
crop_num = 4
with open(validate_Jsonpath, 'r') as load_f:
    val_dict = json.load(load_f)

dirName = ['apple', 'cherry', 'corn', 'grape', 'citrus', 'peach', 'pepper', 'potato', 'strawberry', 'tomato']


def switchClass(id_class):
    if id_class>=0 and id_class<6:
        return 0, 0, 6
    if id_class>=6 and id_class<9:
        return 1, 6, 9
    if id_class>=9 and id_class<17:
        return 2, 9, 17
    if id_class>=17 and id_class<24:
        return 3, 17, 24
    if id_class>=24 and id_class<27:
        return 4, 24, 27
    if id_class>=27 and id_class<30:
        return 5, 27, 30
    if id_class>=30 and id_class<33:
        return 6, 30, 33
    if id_class>=33 and id_class<37:
        return 7, 33, 37
    if id_class>=37 and id_class<41:
        return 8, 37, 41
    if id_class>=41 and id_class<61:
        return 9, 41, 61


def validation():
    with tf.get_default_graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, tr.crop_size, tr.crop_size, 3], name='input_images')
        y = net.model(x, is_training=False, tag=tr.tag, num_class=tr.num_class)

        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt = tf.train.latest_checkpoint(tr.checkpoint_path)
            if ckpt:
                saver.restore(sess, ckpt)
                print('Restore from {}'.format(ckpt))

                images_file = data_genertor.get_images(validate_path)
                # with open(validate_Jsonpath, 'r') as load_f:
                #     val_dict = json.load(load_f)
                num = len(images_file)
                correct_num = 0  #top1
                correct_num_top5 = 0 #top5
                correct_num_class = 0  # 物种分类
                correct_num_assign_class = 0 #指定物种下分类病

                start = time.time()
                for i in range(len(images_file)):
                # for i in range(1):
                    img = cv2.imread(images_file[i])
                    if img is None:
                        num -= 1
                    img = img[:, :, ::-1]
                    # print(img.shape)
                    img_list = []
                    for _ in range(crop_num):
                        img_list.append(data_genertor.random_crop_resize(
                            img, rate=0.95, crop_size=tr.crop_size))

                    label = data_genertor.find_id(images_file[i], val_dict, tr.num_class)

                    ys = sess.run([y], feed_dict={x: img_list})
                    y_ = np.mean(ys, axis=0)[0]

                    flag = [False, False, False, False]
                    label_class = np.argmax(label)
                    y_class = np.argmax(y_)
                    label_info = switchClass(label_class)
                    y_info = switchClass(y_class)
                    y_assign_class = np.argmax(y_[label_info[1]:label_info[2]]) + label_info[1]
                    # print(y_class)
                    # print(y_assign_class)
                    # print(label_class)
                    # print(label_info)
                    if label_class == y_class:  # top1分类是否正确
                        correct_num += 1
                        flag = [True] * 4
                        # print(i, 'top1_right')
                    else:

                        y_top5_class = he.nlargest(3, range(len(y_)), y_.__getitem__)
                        if y_class in y_top5_class:  # top5 分类是否正确
                            correct_num_top5 += 1
                            flag[1] = True
                        if label_info[0] == y_info[0]:  # 物种分类是否正确
                            correct_num_class += 1
                            flag[2] = True
                        if label_class == y_assign_class:  # 指定物种,分类是否正确
                            correct_num_assign_class += 1
                            flag[3] = True
                            # print(i, 'top1_wrong, top5_right')

                    print(i, flag)

                correct_num_top5 = correct_num_top5 + correct_num
                correct_num_class = correct_num_class + correct_num
                correct_num_assign_class = correct_num_assign_class + correct_num

                print('total files {}, top1_acc {}, top5_acc {}, class_acc {}, assign_class_acc {},  fps {}'.format(
                    len(images_file), correct_num / num, correct_num_top5 / num, correct_num_class / num,
                    correct_num_assign_class / num, num / (time.time()-start)))

class NNservice:
    def __init__(self, model):
        with tf.get_default_graph().as_default():
            self.x = tf.placeholder(tf.float32, shape=[None, tr.crop_size, tr.crop_size, 3], name='input_images')
            self.y = net.model(self.x, is_training=False, tag=tr.tag, num_class=tr.num_class)
            self.saver = tf.train.Saver()
            self.sess = tf.Session()
            self.ckpt = tf.train.latest_checkpoint(model)
            self.saver.restore(self.sess, self.ckpt)
            print('Restore from {}'.format(self.ckpt))
            # self.sess.close()

    def hand(self, img_file):
        img = cv2.imread(img_file)
        if img is None:
            return
        img = img[:, :, ::-1]
        img_list = []
        for _ in range(crop_num):
            img_list.append(data_genertor.random_crop_resize(
                img, rate=0.95, crop_size=tr.crop_size))
        label = data_genertor.find_id(img_file, val_dict, tr.num_class)
        ys = self.sess.run([self.y], feed_dict={self.x: img_list})
        y_ = np.mean(ys, axis=0)[0]
        if np.argmax(label) == np.argmax(y_):
            print(img_file, 'right')
        else:
            print(img_file, 'wrong')
        return np.argmax(y_)


# def session_gene():
#     with tf.get_default_graph().as_default():
#         x = tf.placeholder(tf.float32, shape=[None, tr.crop_size, tr.crop_size, 3], name='input_images')
#
#         y = net.model(x, is_training=False, tag=tr.tag, num_class=tr.num_class)
#
#         saver = tf.train.Saver()
#         with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#             ckpt = tf.train.latest_checkpoint(tr.checkpoint_path)
#             if ckpt:
#                 saver.restore(sess, ckpt)
#                 print('Restore from {}'.format(ckpt))
#     return sess, y
#
#
# def validation_one(sess, y,  img_name):
#
#     start = time.time()
#     images_file = os.path.join(validate_path, img_name)
#     with open(validate_Jsonpath, 'r') as load_f:
#         val_dict = json.load(load_f)
#
#     # while(img)
#     # for i in range(len(images_file)):
#     # while
#     img = cv2.imread(images_file)
#     if img is None:
#         num -= 1
#     img = img[:, :, ::-1]
#     # print(img.shape)
#     img_list = []
#     for _ in range(crop_num):
#         img_list.append(data_genertor.random_crop_resize(
#             img, rate=0.95, crop_size=tr.crop_size))
#
#     label = data_genertor.find_id(images_file, val_dict, tr.num_class)
#
#     ys = sess.run([y], feed_dict={x: img_list})
#     y_ = np.mean(ys, axis=0)[0]
#     if np.argmax(label) == np.argmax(y_):
#         correct_num += 1
#         print('right')
#     else:
#         print('wrong')
#                 # print('total files {}, error file {}, val file {}, correct num{}, acc {}, time {}'.format(
#                 #     len(images_file), len(images_file) - num, num, correct_num, correct_num / num, time.time()-start))
#                 # print(correct_num / num, 'total time', (time.time()-start)/len(images_file))


if __name__ == '__main__':
    # validation()
    # validation_one('0a1e2ed0-619c-43da-8c47-f8000a252954___UF.GRC_YLCV_Lab 03060.JPG')
    # sess, y = session_gene()
    #
    # 1 img
    # validation_one(sess, y, '0a1e2ed0-619c-43da-8c47-f8000a252954___UF.GRC_YLCV_Lab 03060.JPG')
    # se = NNservice(tr.checkpoint_path)
    # print(se.hand(validate_path + '0a1e2ed0-619c-43da-8c47-f8000a252954___UF.GRC_YLCV_Lab 03060.JPG'))
    validation()