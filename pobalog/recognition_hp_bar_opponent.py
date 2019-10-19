from pobalog.hp_bar_recognition import HPBarRecognition
from pobalog.message_recogntion import MessageRecognition
from pobalog.recognition import Recognition


class RecognitionHPBarOpponent(Recognition):
    def __init__(self):
        self.hp_bar_recognition = HPBarRecognition([94, 102, 1559, 1889])

    def process_frame(self, img, trigger):
        result = self.hp_bar_recognition.evaluate(img)
        return result

    @property
    def name(self):
        return 'hp_bar_opponent'
