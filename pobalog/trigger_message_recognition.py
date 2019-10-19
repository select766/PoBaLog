from pobalog.trigger import Trigger


class TriggerMessageRecognition(Trigger):
    TEXT_AREA_MIN_THRES = 500
    TEXT_AREA_PEAK_THRES = 0.8

    def __init__(self):
        self.curr_peak = 0

    @property
    def name(self):
        return 'message_recognition'

    def process_preprocess(self, frame_idx: int, preprocess: dict):
        tarea = 0  # メッセージウィンドウがないときはメッセージの面積は0
        if preprocess['message_window']['exist']:
            tarea = preprocess['message_window_text_area']['text_area']
        if self.curr_peak > TriggerMessageRecognition.TEXT_AREA_MIN_THRES and \
                tarea < (self.curr_peak * TriggerMessageRecognition.TEXT_AREA_PEAK_THRES):
            # メッセージ面積が減少した
            # 直前のフレームを認識すべき
            self.curr_peak = tarea
            return [{'frame_idx': frame_idx - 1, 'recognition': 'message_window', 'params': {}}]
        else:
            self.curr_peak = max(tarea, self.curr_peak)
        return []
