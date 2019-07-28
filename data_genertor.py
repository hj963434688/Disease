import tensorflow as tf
import cv2
import os
import glob
import time
import numpy as np
import threading
import multiprocessing
import scipy.misc
import json
import random


try:
    import queue
except ImportError:
    import Queue as queue

train_path = '../dataset/AgriculDis/dataset/AgriculturalDisease_trainingset/images/'
validate_path = '../dataset/AgriculDis/dataset/AgriculturalDisease_validationset/images/'
dirName = ['apple', 'cherry', 'corn', 'grape', 'citrus', 'peach', 'pepper', 'potato', 'strawberry', 'tomato']
train_jsonPath = "../dataset/AgriculDis/dataset/AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations.json"
validate_Jsonpath= '../dataset/AgriculDis/dataset/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations.json'

image_format = ['jpg', 'jpeg', 'JPG', 'png']

# image_format = ['png']


class GeneratorEnqueuer():
    """Builds a queue out of a data generator.

    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

    # Arguments
        generator: a generator function which endlessly yields data
        use_multiprocessing: use multiprocessing if True, otherwise threading
        wait_time: time to sleep in-between calls to `put()`
        random_seed: Initial seed for workers,
            will be incremented by one for each workers.
    """

    def __init__(self, generator,
                 use_multiprocessing=False,
                 wait_time=0.05,
                 random_seed=None):
        self.wait_time = wait_time
        self._generator = generator
        self._use_multiprocessing = use_multiprocessing
        self._threads = []
        self._stop_event = None
        self.queue = None
        self.random_seed = random_seed

    def start(self, workers=1, max_queue_size=10):
        """Kicks off threads which add data from the generator into the queue.

        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, threads could block on `put()`)
        """

        def data_generator_task():
            while not self._stop_event.is_set():
                try:
                    if self._use_multiprocessing or self.queue.qsize() < max_queue_size:
                        generator_output = next(self._generator)
                        self.queue.put(generator_output)
                    else:
                        time.sleep(self.wait_time)
                except Exception:
                    self._stop_event.set()
                    raise

        try:
            if self._use_multiprocessing:
                self.queue = multiprocessing.Queue(maxsize=max_queue_size)
                self._stop_event = multiprocessing.Event()
            else:
                self.queue = queue.Queue()
                self._stop_event = threading.Event()

            for _ in range(workers):
                if self._use_multiprocessing:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.seed(self.random_seed)
                    thread = multiprocessing.Process(target=data_generator_task)
                    thread.daemon = True
                    if self.random_seed is not None:
                        self.random_seed += 1
                else:
                    thread = threading.Thread(target=data_generator_task)
                self._threads.append(thread)
                thread.start()
        except:
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        """Stops running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called `start()`.

        # Arguments
            timeout: maximum time to wait on `thread.join()`.
        """
        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            if thread.is_alive():
                if self._use_multiprocessing:
                    thread.terminate()
                else:
                    thread.join(timeout)

        if self._use_multiprocessing:
            if self.queue is not None:
                self.queue.close()

        self._threads = []
        self._stop_event = None
        self.queue = None

    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        # Returns
            A generator
        """
        while self.is_running():
            if not self.queue.empty():
                inputs = self.queue.get()
                if inputs is not None:
                    yield inputs
            else:
                time.sleep(self.wait_time)

## 获取指定格式文件， 返回文件列表
def get_images(path):
    input_img = []
    for ext in image_format:
        input_img.extend(glob.glob(
            os.path.join(path, '*.{}'.format(ext))))
    return input_img

## 裁剪图片 224
def random_crop_resize(img, crop_size, rate=0.9):
    h, w, _ = img.shape
    size = int(min(h, w) * rate)
    hmax = h - size
    wmax = w - size
    h1 = random.randint(0, hmax)
    w1 = random.randint(0, wmax)
    img = img[h1:h1 + size, w1:w1 + size, :]
    img = cv2.resize(img, (crop_size, crop_size))
    return img


def random_flip(img):
    ax = random.randint(-1, 3)
    if ax != 2:
        img = cv2.flip(img, ax)
    return img


def find_id(image, image_dict, class_num):
    id = image.split('/')[-1]
    label = np.zeros(class_num)
    for item in image_dict:
        if item['image_id'] == id:
            label[item['disease_class']] = 1.0
            return label
    print('can not find id')


def generator(batch_size=32, crop_size=224, class_num=6):
    image_input_list = np.array(get_images(train_path))
    print('{} training images in {}'.format(
        image_input_list.shape[0], train_path))
    index = np.arange(0, image_input_list.shape[0])
    with open(train_jsonPath, 'r') as load_f:
        image_dict = json.load(load_f)
    while True:
        np.random.shuffle(index)
        image_batch = []
        label_batch = []

        for i in index:
            try:
                img = cv2.imread(image_input_list[i])[:, :, ::-1]
                # print(image_input_list[i])
                # cv2.imshow('img', img)
                # cv2.waitKey(0)
                img = random_crop_resize(img, rate=0.9, crop_size=crop_size)
                # cv2.imshow('img', img)
                # cv2.waitKey(0)

                img = random_flip(img)

                label = find_id(image_input_list[i], image_dict, class_num)
                # print(label)
                # print(img.shape)

                image_batch.append(img.astype(np.float32))
                label_batch.append(label.astype(np.float32))

                if len(image_batch) == batch_size:
                    # print('read64-------------')
                    yield image_batch, label_batch
                    image_batch = []
                    label_batch = []

            except Exception as e:
                print(image_input_list[i])
                import traceback
                traceback.print_exc()
                continue


def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        print('Generator use 10 batches for buffering, this may take a while, you can tune this yourself.')
        enqueuer.start(max_queue_size=10, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


def get_val_batch(crop_size=224, batch_size=64, class_num=61):
    image_input_list = np.array(get_images(validate_path))
    # print(image_input_list)
    # print('{} training images in {}'.format(
    #     image_input_list.shape[0], train_path))
    index = np.arange(0, image_input_list.shape[0])
    with open(validate_Jsonpath, 'r') as load_f:
        image_dict = json.load(load_f)
        np.random.shuffle(index)
        image_batch = []
        label_batch = []
        for i in index:
            try:
                img = cv2.imread(image_input_list[i])[:, :, ::-1]

                img = random_crop_resize(img, rate=0.9, crop_size=crop_size)

                img = random_flip(img)

                label = find_id(image_input_list[i], image_dict, class_num)

                image_batch.append(img.astype(np.float32))
                label_batch.append(label.astype(np.float32))

                if len(image_batch) == batch_size:
                    # print('read64-------------')
                    return image_batch, label_batch
            except Exception as e:
                print(image_input_list[i])
                import traceback
                traceback.print_exc()
                continue

if __name__ == '__main__':

    # img_generator = get_batch(num_workers=2, batch_size=64, class_num=61)
    # for i in range(1000):
    # data = next(img_generator)
    #     print(np.argmax(data[1]), i)
    data = get_val_batch()
    # print(len(data))
    print(len(data[0]))
    print(len(data[0][0]))
    # im = cv2.imread('../dataset/AgriculDis/dataset/AgriculturalDisease_trainingset/images/481e71af72f600f027078ae19a8ac8c6.jpg')
    # print(im)
    # cv2.imshow('a', im)
    # cv2.waitKey(0)
    # l = get_images(train_path + dirName[0])
    # for i in :
    # print(len(l))