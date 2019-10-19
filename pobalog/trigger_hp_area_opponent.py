from pobalog.trigger import Trigger


class TriggerHPAreaOpponent(Trigger):
    def __init__(self):
        pass

    @property
    def name(self):
        return 'hp_area_opponent'

    def process_preprocess(self, frame_idx: int, preprocess: dict):
        # HPエリアが存在してれば認識器を呼ぶ
        if preprocess['hp_area_opponent']['exist']:
            return [{'frame_idx': frame_idx, 'recognition': 'hp_bar_opponent', 'params': {}}]
        return []
