from random import randint
from kivy.app import App
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager, Screen
import kivy
import cv2
import time
from imageprocessing import ImageManager
kivy.require('1.10.0')
__version__ = "0.1.1"


class CameraScreen(Screen):
    def on_enter(self, *args):
        camera = self.ids['camera']
        self.timestr = time.strftime("%Y%m%d_%H%M%S")
        self.camera = camera
        self.random = 0

    def capture(self):
        self.camera.play = False
        img_man = ImageManager()
        self.random = randint(0, 1000)
        self.camera.export_to_png("IMG_{}.png".format(self.random))
        time.sleep(5)
        self.predict(img_man)

    def predict(self, img_man):
        img = cv2.imread("IMG_{}.png".format(self.random))
        crop_img = img[10:10 + 480, 80:80 + 640]
        img_man.extract_roi_and_process(crop_img)
        img_man.find_blobs()
        characters = img_man.classify_blobs()
        popup = MsgPopup(str(characters))
        popup.open()
        self.camera.play = True


class MenuScreen(Screen):
    pass


class FindCharacters(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(MenuScreen(name='menu'))
        sm.add_widget(CameraScreen(name='camera'))
        sm.current = "menu"
        return sm


class MsgPopup(Popup):
    def __init__(self, msg):
        super(MsgPopup, self).__init__()
        self.ids.message_label.text = msg


if __name__ == '__main__':
    FindCharacters().run()
