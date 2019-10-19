"""
文字の面積を検出する
"""

import numpy as np
import cv2


class TextAreaDetection:
    def __init__(self, rectangle, threshold):
        """

        :param rectangle: 対象領域の上、下、左、右の座標
        :param threshold: 明るさ閾値（0~255,これ以下が文字とみなされる)
        """
        self.rectangle = rectangle
        self.threshold = threshold

    def evaluate(self, img):
        img_crop = img[self.rectangle[0]:self.rectangle[1], self.rectangle[2]:self.rectangle[3]]
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        text_area = np.count_nonzero(gray <= self.threshold)
        return {'text_area': text_area}
