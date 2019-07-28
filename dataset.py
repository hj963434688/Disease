import os
import shutil
import re
import json
import cv2
import numpy as np

imagesPath = "../dataset/AgriculDis/dataset/AgriculturalDisease_trainingset/test"
jsonPath = "../dataset/AgriculDis/dataset/AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations.json"
newDirPath = "../dataset/AgriculDis/dataset/AgriculturalDisease_trainingset/"
dirName = ['apple', 'cherry', 'corn', 'grape', 'citrus', 'peach', 'pepper', 'potato', 'strawberry', 'tomato']


def switchClass(id_class):
    if id_class>=0 and id_class<6:
        return dirName[0]
    if id_class>=6 and id_class<9:
        return dirName[1]
    if id_class>=9 and id_class<17:
        return dirName[2]
    if id_class>=17 and id_class<24:
        return dirName[3]
    if id_class>=24 and id_class<27:
        return dirName[4]
    if id_class>=27 and id_class<30:
        return dirName[5]
    if id_class>=30 and id_class<33:
        return dirName[6]
    if id_class>=33 and id_class<37:
        return dirName[7]
    if id_class>=37 and id_class<41:
        return dirName[8]
    if id_class>=41 and id_class<61:
        return dirName[9]


def image_class():
    with open(jsonPath, 'r') as load_f:
        disease_dict = json.load(load_f)
    images = os.listdir(imagesPath)
    for i in range(len(disease_dict)):
        image_id = disease_dict[i]['image_id']
        dc = disease_dict[i]['disease_class']
        className = switchClass(dc)
        newPath = newDirPath + "/" + className
        if not os.path.exists(newPath):
            os.mkdir(newPath)
        filePath = imagesPath + image_id
        shutil.copy(filePath, newPath)
        print(i, dc)

    print('end')


def check():
    with open(jsonPath, 'r') as load_f:
        disease_dict = json.load(load_f)
    images = os.listdir(imagesPath)
    print(len(disease_dict))
    print(len(images))
    print(images[0])
    for i in range(len(images)):
        flag= False
        for k in range(len(disease_dict)):
            image_id = disease_dict[k]['image_id']
            if image_id == images[i]:
                print(image_id, images[i])
                flag = True
        if flag:
            print(i)
        else:
            print(i, flag)

    print('end')


if __name__ == '__main__':
    # # image_class()
    # label = np.zeros(6)
    # print(label)
    check()