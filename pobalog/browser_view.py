"""
ブラウザ上で状態をレンダリングする
OpenCVのGUIで動画をシークし、表示されているフレームに対応する解析結果をブラウザ側に送信する。
"""
import pickle

"""
フレームをGUIで選択、静止画処理モジュールに入力して認識結果を表示する
"""

import argparse
import os
import time
import json
import numpy as np
import cv2
from websocket import create_connection


def nothing(x):
    pass


def run(video, analyzed_dir):
    ws = create_connection("ws://127.0.0.1:15100")
    with open(os.path.join(analyzed_dir, 'states.bin'), 'rb') as f:
        video_results = pickle.load(f)
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
    frame_states = video_results['frame_states']

    while True:
        # Capture frame-by-frame
        frame_idx = cv2.getTrackbarPos('frame', 'seek')
        if frame_idx != last_frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            last_frame_idx = frame_idx
            ret, frame = cap.read()
            # 直近のstateを探す
            latest_state = None
            for state_frame_idx in range(frame_idx, -1, -1):
                if state_frame_idx in frame_states:
                    latest_state = frame_states[state_frame_idx]
                    break
            else:
                latest_state = list(frame_states.values())[0]
            ws.send(json.dumps({
                'frame_idx': frame_idx,
                'state': latest_state
            }))
            ws.recv()  # broadcastされてくるメッセージを読み捨てる

        # Display the resulting frame
        cv2.imshow('video', frame)
        key = cv2.waitKeyEx(1)  # waitKeyだと矢印が取れない
        if key != -1:
            print('keycode', key)
        if key == ord('q'):
            break
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
    ws.close()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("analyzed")
    args = parser.parse_args()
    run(args.video, args.analyzed)


if __name__ == '__main__':
    main()
