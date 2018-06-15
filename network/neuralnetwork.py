from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense


class CNN:
    @staticmethod
    def build(depth, height):

        model = Sequential()

        model.add(Conv2D(20, (3, 3), padding="same", input_shape=(depth, height)))
        model.add(Activation("relu"))
        model.add(Conv2D(20, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Second CONV => RELU => POOL Layer
        model.add(Conv2D(50, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(50, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(100, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(100, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(100, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(100, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # FC => RELU layers
        #  Fully Connected Layer -> ReLU Activation Function
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # Using Softmax Classifier for Linear Classification
        model.add(Dense(62))
        model.add(Activation("softmax"))

        return model
