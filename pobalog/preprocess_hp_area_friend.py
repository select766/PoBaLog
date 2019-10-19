from pobalog.preprocess import Preprocess
from pobalog.whole_image_matching import WholeImageMatching


class PreprocessHPAreaFriend(Preprocess):
    """
    メッセージウィンドウの有無を検出する
    """
    HP_AREA_FRIEND_THRES = 0.10

    def __init__(self):
        self.whole_image_matching = WholeImageMatching("template/hp_area_friend.png")

    @property
    def name(self):
        return 'hp_area_friend'

    def process_frame(self, img) -> dict:
        match_score = self.whole_image_matching.evaluate(img)['diff']
        exist = match_score <= PreprocessHPAreaFriend.HP_AREA_FRIEND_THRES
        return {'exist': exist}
