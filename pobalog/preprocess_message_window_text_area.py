from pobalog.preprocess import Preprocess
from pobalog.text_area_detection import TextAreaDetection
from pobalog.whole_image_matching import WholeImageMatching


class PreprocessMessageWindowTextArea(Preprocess):
    """
    メッセージウィンドウ内の文字の面積を計算する
    """
    MESSAGE_WINDOW_THRES = 0.05

    def __init__(self):
        self.text_area_detection = TextAreaDetection([908, 1065, 17, 1342], 100)

    @property
    def name(self):
        return 'message_window_text_area'

    def process_frame(self, img) -> dict:
        area = self.text_area_detection.evaluate(img)['text_area']
        return {'text_area': area}
