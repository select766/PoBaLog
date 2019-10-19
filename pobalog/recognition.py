class Recognition:
    def process_frame(self, img, trigger):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError
