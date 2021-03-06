import cv2
import numpy as np
import os
from PIL import Image as PImage
import io
import numpy as np
import cv2


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()
'''
tr25aining_path = "/home/huseyin/Downloads/datas/training"
classes = []
for counter, i in enumerate(os.listdir(tr25aining_path)):
    print(i)
img = cv2.imread("proje2.jpg")
img_man = ImageManager()
img_man.extract_roi_and_process(img)
img_man.find_blobs()
characters = img_man.classify_blobs()
print(characters)

my_array = np.load("/home/huseyin/PycharmProjects/Find Characters/network/train_data.npy")
test_labels = [i[1] for i in my_array]
print(test_labels)

image = PImage.open("/home/huseyin/PycharmProjects/Find Characters/0.png")
image1 = np.asarray(image)
r, g, b = cv2.split(image1)
image2 = cv2.merge([b, g, r])
image3 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
image4 = PImage.fromarray(image3, mode="L")
image4.save("output.png")
'''
'''
def load_images(path):
    # return array of images

    images_list = os.listdir(path)
    loadedImages = []
    for image in images_list:
        img = PImage.open(path + "/" + image)
        loadedImages.append(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img.save()


for file_name in os.listdir(training_path):
    load_images(training_path + "/" + file_name)
'''
'''
for item in classes:
    for file in os.path.realpath(training_path+classes[item]):
        image = cv2.imread("file")
  '''
'''
img = cv2.imread(os.path.realpath(training_path + "/" + classes[0] + "/" + file))
cv2.imshow("image", img)
cv2.waitKey(0)
'''

