from network.neuralnetwork import CNN
from keras.optimizers import Adam
import keras
import os
import numpy as np

train_labels = []
test_labels = []
test_path = "/home/huseyin/Downloads/datas/test"
train_path = "/home/huseyin/Downloads/datas/training"
train_data = np.load("/home/huseyin/PycharmProjects/Find Characters/network/train_data.npy")
test_data = np.load("/home/huseyin/PycharmProjects/Find Characters/network/test_data.npy")
classes = []
for directory in os.listdir(test_path):
    classes.append(directory)
train_images = np.array([i[0] for i in train_data]).reshape(-1, 28, 28, 1)
train_labels = [i[1] for i in train_data]
train_labels = keras.utils.to_categorical(train_labels, 62)
test_images = np.array([i[0] for i in test_data]).reshape(-1, 28, 28, 1)
test_labels = [i[1] for i in test_data]
test_labels = keras.utils.to_categorical(test_labels, 62)


depth = 28
height = 28
width = 1
model_class = CNN()
model = model_class.build(depth, height, width)


def train_cnn(model):
    model.compile(Adam(lr=.001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_images, train_labels, batch_size=20, epochs=10, verbose=1, validation_data=(test_images, test_labels))

    model.save("second_model_for_ocr.h5")

train_cnn(model)

