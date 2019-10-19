from pobalog.message_recogntion import MessageRecognition
from pobalog.recognition import Recognition


class RecognitionMessageWindow(Recognition):
    def __init__(self):
        self.message_recognition = MessageRecognition("cache")

    def process_frame(self, img, trigger):
        result = self.message_recognition.recognize(img[908:1065, 17:1342])
        return result

    @property
    def name(self):
        return 'message_window'
