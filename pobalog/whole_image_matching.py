"""
画像全体がマスクと一致しているか判定する
"""

import numpy as np
import cv2


def _find_first_nonzero(vec, start, stop, step):
    for x in range(start, stop, step):
        if vec[x]:
            return x
    raise ValueError


def _find_nonzero_slice(vec):
    first = _find_first_nonzero(vec, 0, vec.size, 1)
    last = _find_first_nonzero(vec, vec.size - 1, -1, -1)
    return slice(first, last + 1)  # index=lastの要素を含むには+1をスライスに含める


class WholeImageMatching:
    MASK_COLOR = [255, 0, 255]  # テンプレート画像でマスク領域を表す色BGR

    def __init__(self, template_path: str):
        whole_template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        whole_mask = np.any(whole_template != WholeImageMatching.MASK_COLOR, axis=2)  # (h, w)のboolean maskで有効領域True
        # マスクのうち有効な矩形領域を抽出し、後の処理の領域を狭める
        self._match_area = (_find_nonzero_slice(np.any(whole_mask, axis=1)),
                            _find_nonzero_slice(np.any(whole_mask, axis=0)))
        self.mask = whole_mask[self._match_area]
        self.valid_template = whole_template[self._match_area][self.mask]  # (有効ピクセル数, 3=BGR)

    def evaluate(self, img):
        img_masked = img[self._match_area][self.mask]
        diff_map = self.valid_template.astype(np.float32) - img_masked.astype(np.float32)
        # ピクセル値の差のmean squared rootでテンプレートとの差を計算。
        # 0(完全一致)~1で正規化したい。
        # ピクセル値の差の最大値は255。
        diff = float(np.sqrt(np.mean(np.square(diff_map))) / 255.0)
        return {'diff': diff}
