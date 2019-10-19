"""
HPバーのHP率を読み取る
"""

import numpy as np
import cv2


class HPBarRecognition:
    V_THRES = 127

    def __init__(self, rectangle):
        """

        :param rectangle: 対象領域の上、下、左、右の座標(上左はinclusive, 下右はexclusive)
        """
        self.rectangle = rectangle

    def evaluate(self, img):
        img_crop = img[self.rectangle[0]:self.rectangle[1], self.rectangle[2]:self.rectangle[3]]
        full_px = img_crop.shape[1]
        hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
        bar = np.mean(hsv, axis=0)  # y方向につぶす
        # エンコードの都合で端がぼやける場合があるので、色を見てより詳細な認識があったほうがよさそう
        thresed = bar[:, 2] > HPBarRecognition.V_THRES  # Vのみ
        remaining_px = int(np.count_nonzero(thresed))
        if remaining_px > 0:
            mean_h = float(np.mean(bar[:remaining_px, 0]))
            if mean_h > 100:
                # 角度360寄りの赤
                color = 'red'
            elif mean_h > 40:
                color = 'green'
            else:
                color = 'yellow'
        else:
            color = 'zero'
        return {'hp_ratio': remaining_px / full_px, 'full_px': full_px, 'remaining_px': remaining_px, 'color': color}
