from pobalog.preprocess import Preprocess
from pobalog.whole_image_matching import WholeImageMatching


class PreprocessMessageWindow(Preprocess):
    """
    メッセージウィンドウの有無を検出する
    """
    MESSAGE_WINDOW_THRES = 0.05

    def __init__(self):
        self.whole_image_matching = WholeImageMatching("template/message_window.png")

    @property
    def name(self):
        return 'message_window'

    def process_frame(self, img) -> dict:
        match_score = self.whole_image_matching.evaluate(img)['diff']
        exist = match_score <= PreprocessMessageWindow.MESSAGE_WINDOW_THRES
        return {'exist': exist}
