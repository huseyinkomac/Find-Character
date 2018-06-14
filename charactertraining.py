from network.neuralnetwork import CNN
import keras
import numpy as np

depth = 128
height = 128
width = 1


model_class = CNN()
model = model_class.build(depth, height, width)



