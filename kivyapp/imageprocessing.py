import cv2
import math
from keras.models import load_model
import numpy as np


class ImageManager:

    def __init__(self, show_mode="off"):
        self.img = None
        self.blobs = None
        self.show_mode = show_mode

    @staticmethod
    def find_bounds(blob_points):  # finds rectangular frame of the blob
        top, left = blob_points[0]
        bottom, right = blob_points[0]
        for x, y in blob_points:
            if top > x:
                top = x
            elif bottom < x:
                bottom = x
            if left > y:
                left = y
            elif right < y:
                right = y
        return top, left, bottom, right

    @staticmethod
    def _blob_center_of_mass(blob_points):
        sum_x, sum_y = 0.0, 0.0
        for x, y in blob_points:
            sum_x += x
            sum_y += y
        return sum_x/len(blob_points), sum_y/len(blob_points)

    @staticmethod
    def _validate_point(x, y, img):
        return (x >= 0) and (y >= 0) and (x < img.shape[0]) and (y < img.shape[1]) and (img[x, y] != -1)

    def _blobs_filter(self):
        main_blobs = []
        aux_blobs = []
        blobs_average_points = sum([len(blob) for blob in self.blobs])/len(self.blobs)
        if blobs_average_points*0.3 > 20:  # we need at least 20 pts
            for blob in self.blobs:
                dots = len(blob)
                if dots < 10:
                    continue
                if dots >= blobs_average_points*0.35:
                    main_blobs.append(blob)
                elif dots >= blobs_average_points*0.2:
                    aux_blobs.append(blob)
            self.blobs = main_blobs
            for blob in aux_blobs:
                self._find_closest_and_merge(blob)
        else:
            self.blobs = []

    def _find_closest_and_merge(self, blob):
        cm_blob = np.array(self._blob_center_of_mass(blob))
        distance = np.array([])
        for current_blob in self.blobs:
            cm_current_blob = np.array(self._blob_center_of_mass(current_blob))
            diff = cm_current_blob-cm_blob
            out = np.sqrt(diff.dot(diff))
            distance = np.append(distance, [out])
        self.blobs[distance.argmin()] += blob

    def _split_into_blobs(self):
        if self.img is None:
            raise Exception("Image is not set")
        blobs = []
        img = np.array(self.img, np.int16)
        step = 0
        way = [(0, 0)]
        while True:
            # Try to find a blob
            blob_point = None

            while (step < len(way)) and (blob_point is None):
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        point = (way[step][0]+dx, way[step][1]+dy)
                        if self._validate_point(point[0], point[1], img):
                            if img[point] != 0:
                                blob_point = point
                                break
                            way.append(point)  # new non-blob point
                            img[point] = -1
                step += 1

            if blob_point is not None:  # found new blob
                blob_points = [blob_point]
                img[blob_point] = -1
                blob_step = 0
                while blob_step < len(blob_points):  # passing blob's points
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            point = (blob_points[blob_step][0]+dx, blob_points[blob_step][1]+dy)
                            if self._validate_point(point[0], point[1], img):
                                if img[point] != 0:
                                    blob_points.append(point)
                                else:
                                    way.append(point)
                                img[point] = -1
                    blob_step += 1
                blobs.append(blob_points)
            else:
                break

        self.blobs = blobs

    def extract_roi_and_process(self, img):
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.img = cv2.blur(self.img, (5, 5))
        self.img = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 5)
        self.img = cv2.erode(self.img, (15, 15), iterations=3)
        self.img = cv2.dilate(self.img, (25, 25), iterations=1)

    def set_img(self, img):
        self.img = img

    def find_blobs(self, filtering=True):
        self._split_into_blobs()

        if filtering:
            self._blobs_filter()
        print(str(len(self.blobs)) + " blobs are found.")

    def reproduce_blob(self, blob_points):
        top, left, bottom, right = self.find_bounds(blob_points)
        center_x, center_y = self._blob_center_of_mass(blob_points)
        if bottom-top > right - left:
            max_side = bottom-top
        else:
            max_side = right-left
        tmp_img = np.zeros((int(max_side*2), int(max_side*2), 1), np.uint8)
        dx = math.floor(max_side-center_x)
        dy = math.floor(max_side-center_y)
        for x, y in blob_points:
            if self._validate_point(x+int(dx), y+int(dy), tmp_img):
                tmp_img[x+int(dx), y+int(dy), 0] = 255
        tmp_img = cv2.resize(tmp_img, (28, 28), 1)
        tmp_img = cv2.bitwise_not(tmp_img)
        return tmp_img

    def classify_blobs(self):
        res = []
        if self.blobs is None:
            raise Exception("There are no blobs. Maybe you should run find_blobs first")
        for blob in self.blobs:
            img = np.array((28, 28, 1), np.uint8)
            img = self.reproduce_blob(blob)
            res.append(self.get_data_from_network(img))
        return res

    @staticmethod
    def get_real_character(number):
        for count, character in enumerate(character_array):
            if count == number:
                return character

    def get_data_from_network(self, image):
        new_model = load_model("second_model_for_ocr.h5")
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=3)
        predictions = new_model.predict(image)
        for prediction_first in predictions:
            for count, prediction_second in enumerate(prediction_first):
                if prediction_second > 0.5:
                    print(count, prediction_second)
                    return self.get_real_character(count)
            return "send a legit character"

character_array = ['p', 'J', 'W', 'T', 'i', '6', 'g', '1', 'z', 'A', 'Q', 'j', 'o', 'X', '8', 'Z', '3', 'l', 'P', 'U', 'd', 'I', 'R', 'e', '2', '7', 'v', 'n', 'C', 'y', 'm', 'r', 'O', '0', 'a', 'B', '4', 'w', 'N', 'F', 'D', 'G', '9', 'L', 'V', 'Y', 'E', 'k', 't', 'b', 'x', '5', 'c', 'H', 'S', 'q', 's', 'K', 'h', 'f', 'u', 'M']
FRAME_RESIZE_HEIGHT = 50







