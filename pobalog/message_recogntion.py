"""
テキストメッセージの文字認識
Google Vision APIのDOCUMENT_TEXT_DETECTIONを利用。
有料なので注意。1000回あたり1.5ドル。
デバッグ中に同じ画像を何度も認識しないよう、クエリ画像とレスポンスの組を保存しておく。
まったく同じ画像が再度来たら保存した結果を読む。
"""

import os
import pickle
import os
import hashlib
import cv2
import numpy as np

from google.cloud import vision

client = vision.ImageAnnotatorClient()


class MessageRecognition:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def recognize(self, img):
        img_hash = hashlib.sha1(img.tobytes()).hexdigest()
        cache_result_path = os.path.join(self.cache_dir, img_hash + ".bin")
        if os.path.exists(cache_result_path):
            with open(cache_result_path, "rb") as f:
                response = pickle.load(f)
        else:
            cache_image_path = os.path.join(self.cache_dir, img_hash + ".png")
            cv2.imwrite(cache_image_path, img)
            with open(cache_image_path, "rb") as f:
                image_file = f.read()
            vis_image = vision.types.Image(content=image_file)
            response = client.document_text_detection(image=vis_image)
            with open(cache_result_path, "wb") as f:
                pickle.dump(response, f, pickle.HIGHEST_PROTOCOL)
        return self.get_text(response)

    def get_text(self, response):
        # APIの結果から単純なテキスト部分を取得
        return response.full_text_annotation.text
