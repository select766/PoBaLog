class Trigger:
    def process_preprocess(self, frame_idx: int, preprocess: dict):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError
