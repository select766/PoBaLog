from pobalog.preprocess import Preprocess
from pobalog.whole_image_matching import WholeImageMatching


class PreprocessHPAreaOpponent(Preprocess):
    """
    メッセージウィンドウの有無を検出する
    """
    HP_AREA_OPPONENT_THRES = 0.10

    def __init__(self):
        self.whole_image_matching = WholeImageMatching("template/hp_area_opponent.png")

    @property
    def name(self):
        return 'hp_area_opponent'

    def process_frame(self, img) -> dict:
        match_score = self.whole_image_matching.evaluate(img)['diff']
        exist = match_score <= PreprocessHPAreaOpponent.HP_AREA_OPPONENT_THRES
        return {'exist': exist}
