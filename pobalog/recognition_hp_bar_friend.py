from pobalog.hp_bar_recognition import HPBarRecognition
from pobalog.message_recogntion import MessageRecognition
from pobalog.recognition import Recognition


class RecognitionHPBarFriend(Recognition):
    def __init__(self):
        self.hp_bar_recognition = HPBarRecognition([1028, 1034, 29, 359])

    def process_frame(self, img, trigger):
        result = self.hp_bar_recognition.evaluate(img)
        return result

    @property
    def name(self):
        return 'hp_bar_friend'
