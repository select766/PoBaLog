"""
時系列処理の試行
メッセージウィンドウを認識すべきタイミングを抽出、スクリーンショット保存
"""

import argparse
import os
import time
import numpy as np
import cv2

from pobalog.text_area_detection import TextAreaDetection
from pobalog.whole_image_matching import WholeImageMatching

MESSAGE_WINDOW_THRES = 0.05
TEXT_AREA_MIN_THRES = 500
TEXT_AREA_PEAK_THRES = 0.8


def run(video, frame_start, frame_end, screenshot_dir):
    cap = cv2.VideoCapture(video)
    # 最初のフレームを読まずにシークするとフレーム番号と画像の対応が変になる？
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame 0")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    frame = None

    message_window_detection = WholeImageMatching("template/message_window.png")
    text_area_detection = TextAreaDetection([908, 1065, 17, 1342], 100)

    curr_peak = 0
    last_frame = None
    for frame_idx in range(frame_start, frame_end):
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_idx}")
            break
        mdet = message_window_detection.evaluate(frame)
        tdet = text_area_detection.evaluate(frame)
        print(frame_idx, mdet, tdet)
        tarea = 0
        if mdet['diff'] <= MESSAGE_WINDOW_THRES:
            # メッセージウィンドウあり
            tarea = tdet['text_area']
        if curr_peak > TEXT_AREA_MIN_THRES and tarea < (curr_peak * TEXT_AREA_PEAK_THRES):
            # メッセージ面積が減少した
            # 直前のフレームを認識すべき
            curr_peak = tarea
            ss_path = os.path.join(screenshot_dir, f"text_area_{frame_idx - 1}.png")
            cv2.imwrite(ss_path, last_frame)
            print("sshot")
        else:
            curr_peak = max(tarea, curr_peak)
        last_frame = frame
    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    parser.add_argument("--screenshot", default=".", help="directory to save screenshot")
    args = parser.parse_args()
    if not os.path.isdir(args.screenshot):
        print("Screenshot directory does not exist")
        return
    run(args.video, args.start, args.end, args.screenshot)


if __name__ == '__main__':
    main()
