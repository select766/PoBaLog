"""
キャプチャボードからのリアルタイム入力を認識
"""

"""
動画の時系列処理
"""

import argparse
import os
import pickle
import time
import json
import multiprocessing
import queue
from collections import defaultdict
import numpy as np
import cv2
from websocket import create_connection

from pobalog.preprocess_hp_area_friend import PreprocessHPAreaFriend
from pobalog.preprocess_hp_area_opponent import PreprocessHPAreaOpponent
from pobalog.preprocess_message_window import PreprocessMessageWindow
from pobalog.preprocess_message_window_text_area import PreprocessMessageWindowTextArea
from pobalog.recognition_hp_bar_friend import RecognitionHPBarFriend
from pobalog.recognition_hp_bar_opponent import RecognitionHPBarOpponent
from pobalog.recognition_message_window import RecognitionMessageWindow
from pobalog.trigger_hp_area_friend import TriggerHPAreaFriend
from pobalog.trigger_hp_area_opponent import TriggerHPAreaOpponent
from pobalog.trigger_message_recognition import TriggerMessageRecognition
from pobalog import semantic_analysis


def get_preproceses():
    pmw = PreprocessMessageWindow()
    pmwta = PreprocessMessageWindowTextArea()
    phaf = PreprocessHPAreaFriend()
    phao = PreprocessHPAreaOpponent()
    return {pmw.name: pmw, pmwta.name: pmwta, phaf.name: phaf, phao.name: phao}


def get_triggers():
    tmr = TriggerMessageRecognition()
    thaf = TriggerHPAreaFriend()
    thao = TriggerHPAreaOpponent()
    return {tmr.name: tmr, thaf.name: thaf, thao.name: thao}


def get_recognitions():
    rmw = RecognitionMessageWindow()
    rhbf = RecognitionHPBarFriend()
    rhbo = RecognitionHPBarOpponent()
    return {rmw.name: rmw, rhbf.name: rhbf, rhbo.name: rhbo}


def run(frame_queue, result_dir):
    ws = create_connection("ws://127.0.0.1:15100")
    frames = {}
    remove_old = 10
    preprocesses = get_preproceses()
    triggers = get_triggers()
    recognitions = get_recognitions()
    frame_preprocess_results = {}
    frame_trigger_results = {}
    frame_recognition_results = defaultdict(dict)

    state = semantic_analysis.get_initial_state()
    frame_idx = -1
    while True:
        frame_idx += 1
        frame = frame_queue.get()
        if frame is None:
            break
        frames[frame_idx] = frame
        old_frame_idx = frame_idx - remove_old
        if old_frame_idx in frame_recognition_results:
            if semantic_analysis.update_by_frame(state, frame_recognition_results[old_frame_idx]):
                ws.send(json.dumps({
                    'frame_idx': frame_idx,
                    'state': state
                }))
                ws.recv()  # broadcastされてくるメッセージを読み捨てる
                print(state)
        if old_frame_idx in frames:
            del frames[old_frame_idx]

        preps = {}
        for name, prepro in preprocesses.items():
            preps[name] = prepro.process_frame(frame)
        frame_preprocess_results[frame_idx] = preps

        trigs = []
        for name, trig in triggers.items():
            trigs.extend(trig.process_preprocess(frame_idx, preps))
        frame_trigger_results[frame_idx] = trigs

        for trig in trigs:
            recog_frame = frames[trig['frame_idx']]
            recog = recognitions[trig['recognition']].process_frame(recog_frame, trig['params'])
            frame_recognition_results[trig['frame_idx']][trig['recognition']] = recog
    ws.close()


def capture(device: int, frame_queue: multiprocessing.Queue):
    """
    画像をキャプチャするプロセス
    :param device:
    :return:
    """
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cv2.namedWindow('video')
    frame_idx = -1

    start_time = time.time()
    while True:
        frame_idx += 1
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('video', frame)
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            print("Queue Full")
        key = cv2.waitKeyEx(1)  # waitKeyだと矢印が取れない
        if key == ord('q'):
            break
        print(f"\rfps={(frame_idx + 1) / (time.time() - start_time)}", end="")
    frame_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=int)
    parser.add_argument("result", help="directory to save result")
    args = parser.parse_args()
    if not os.path.isdir(args.result):
        os.makedirs(args.result)
    frame_queue = multiprocessing.Queue(maxsize=300)
    p = multiprocessing.Process(target=run, args=(frame_queue, args.result))
    p.start()
    capture(args.device, frame_queue)


if __name__ == '__main__':
    main()
