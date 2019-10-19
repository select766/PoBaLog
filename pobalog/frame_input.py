"""
フレームをGUIで選択、静止画処理モジュールに入力して認識結果を表示する
"""

import argparse
import os
import time
import numpy as np
import cv2

from pobalog.hp_bar_recognition import HPBarRecognition
from pobalog.text_area_detection import TextAreaDetection
from pobalog.whole_image_matching import WholeImageMatching


def nothing(x):
    pass


def run(video, screenshot_dir, engines):
    cap = cv2.VideoCapture(video)
    print('width', cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print('height', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('fps', cap.get(cv2.CAP_PROP_FPS))
    # cap.get(cv2.CAP_PROP_FRAME_COUNT)-2をsetしてフレームを読もうとすると失敗するので、
    # cap.get(cv2.CAP_PROP_FRAME_COUNT)-2フレームある動画とみなす
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 2
    print('total frames', total_frames)

    cv2.namedWindow('video')
    cv2.namedWindow('seek')
    cv2.createTrackbar('frame', 'seek', 0, total_frames - 1, nothing)

    frame_idx = 0
    last_frame_idx = -1
    frame = None

    while True:
        # Capture frame-by-frame
        frame_idx = cv2.getTrackbarPos('frame', 'seek')
        if frame_idx != last_frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            last_frame_idx = frame_idx
            ret, frame = cap.read()
            for key, engine in engines.items():
                print(key, engine.evaluate(frame))

        # Display the resulting frame
        cv2.imshow('video', frame)
        key = cv2.waitKeyEx(1)  # waitKeyだと矢印が取れない
        if key != -1:
            print('keycode', key)
        if key == ord('q'):
            break
        if key == ord('s'):
            # スクリーンショットを保存
            ss_path = os.path.join(screenshot_dir, f"{time.strftime('%Y%m%d%H%M%S')}_{frame_idx}.png")
            cv2.imwrite(ss_path, frame)
        # 矢印キーでフレーム移動
        frame_offset = 0
        if key == 0x250000:  # arrow left
            frame_offset = -1
        if key == 0x260000:  # arrow up
            frame_offset = -60
        if key == 0x270000:  # arrow right
            frame_offset = 1
        if key == 0x280000:  # arrow down
            frame_offset = 60
        if key == 0x240000:  # home
            frame_offset = -total_frames
        if key == 0x230000:  # end
            frame_offset = total_frames
        if frame_offset != 0:
            cv2.setTrackbarPos('frame', 'seek', max(0, min(total_frames - 1, frame_idx + frame_offset)))

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--screenshot", default=".", help="directory to save screenshot")
    args = parser.parse_args()
    if not os.path.isdir(args.screenshot):
        print("Screenshot directory does not exist")
        return
    engines = {}
    engines['message_window'] = WholeImageMatching("template/message_window.png")
    engines['hp_area_friend'] = WholeImageMatching("template/hp_area_friend.png")
    engines['hp_area_opponent'] = WholeImageMatching("template/hp_area_opponent.png")
    engines['hp_bar_friend'] = HPBarRecognition([1028, 1034, 29, 359])
    engines['hp_bar_opponent'] = HPBarRecognition([94, 102, 1559, 1889])
    engines['text_area'] = TextAreaDetection([908, 1065, 17, 1342], 100)
    run(args.video, args.screenshot, engines)


if __name__ == '__main__':
    main()
