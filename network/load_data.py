import cv2
import os
from tqdm import tqdm
import numpy as np
from random import shuffle


training_dir = "/home/huseyin/Downloads/datas/training"
test_dir = "/home/huseyin/Downloads/datas/test"


def get_label(img):
    for counter, i in enumerate(tqdm(os.listdir(training_dir))):
        if img == i:
            label = counter
    return label


def get_training_data():
    training_data = []
    for img in tqdm(os.listdir(training_dir)):
        for img_second in os.listdir(os.path.join(training_dir, img)):
            for img_third in os.listdir(os.path.join(os.path.join(training_dir, img), img_second)):
                label = get_label(img)
                path = os.path.join(os.path.join(os.path.join(training_dir, img), img_second), img_third)
                img_third = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (28, 28))
                training_data.append([np.array(img_third), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def get_testing_data():
    test_data = []
    for img in tqdm(os.listdir(test_dir)):
        for i in os.listdir(os.path.join(test_dir, img)):
            label = get_label(img)
            path = os.path.join(os.path.join(test_dir, img), i)
            i = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (28, 28))
            test_data.append([np.array(i), np.array(label)])
    shuffle(test_data)
    np.save('test_data.npy', test_data)
    return test_data

get_testing_data()
get_training_data()
