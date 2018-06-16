from kivy.app import App
from kivy.core.window import Window
from kivy.core.camera import Camera
import kivy
from kivy.uix.widget import Widget
kivy.require('1.10.0')
__version__ = "1.0"
Window.size = (720, 1280)


class CameraApp(Camera):
    def __init__(self):
        super(CameraApp, self).__init__()

    def my_buttons(self):
        pass


class KivyApp(App):
    def build(self):
        return CameraApp()

if __name__ == "__main__":
    KivyApp().run()
