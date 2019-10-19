"""
画像全体がマスクと一致しているか判定する
"""

import numpy as np
import cv2


class WholeImageMatching:
    MASK_COLOR = [255, 0, 255]  # テンプレート画像でマスク領域を表す色BGR

    def __init__(self, template_path: str):
        self.template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        self.mask = np.any(self.template != WholeImageMatching.MASK_COLOR, axis=2)  # (h, w)のboolean maskで有効領域True
        self.valid_template = self.template[self.mask]  # (有効ピクセル数, 3=BGR)

    def evaluate(self, img):
        img_masked = img[self.mask]
        diff_map = self.valid_template.astype(np.float32) - img_masked.astype(np.float32)
        # ピクセル値の差のmean squared rootでテンプレートとの差を計算。
        # 0(完全一致)~1で正規化したい。
        # ピクセル値の差の最大値は255。
        diff = float(np.sqrt(np.mean(np.square(diff_map))) / 255.0)
        return {'diff': diff}
