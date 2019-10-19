class Preprocess:
    def process_frame(self, img) -> dict:
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError
