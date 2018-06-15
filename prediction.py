from keras.models import load_model
import cv2
from sklearn.preprocessing import MinMaxScaler
import numpy as np


scaler = MinMaxScaler()
image = cv2.imread("7.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (28, 28))
print(image.shape)
image = np.array(image)
image = np.expand_dims(image, axis=0)
image = np.expand_dims(image, axis=3)
print(image.shape)
new_model = load_model("first_model_for_ocr.h5")

predictions = new_model.predict(image)
for i in predictions:
    for ii in i:
        print("Predicted: %s" % ii)

