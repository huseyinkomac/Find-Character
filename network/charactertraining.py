from neuralnetwork import CNN
import keras
import numpy as np

train_labels = []
test_labels = []
train_data = np.load("/home/huseyin/PycharmProjects/FindCharacters/network/train_data.npy")
test_data = np.load("/home/huseyin/PycharmProjects/FindCharacters/network/test_data.npy")
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
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_images, train_labels, batch_size=20, epochs=5, verbose=1, validation_data=(test_images, test_labels))

    model.save("second_model_for_ocr.h5")

train_cnn(model)

