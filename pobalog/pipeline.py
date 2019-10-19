"""
動画の時系列処理
"""

import argparse
import os
import pickle
import time
from collections import defaultdict
import numpy as np
import cv2

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


def run(video, frame_start, frame_end, result_dir):
    cap = cv2.VideoCapture(video)
    # 最初のフレームを読まずにシークするとフレーム番号と画像の対応が変になる？
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame 0")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    frames = {}
    remove_old = 10
    preprocesses = get_preproceses()
    triggers = get_triggers()
    recognitions = get_recognitions()
    frame_preprocess_results = {}
    frame_trigger_results = {}
    frame_recognition_results = defaultdict(dict)

    for frame_idx in range(frame_start, frame_end):
        print(frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_idx}")
            break
        frames[frame_idx] = frame
        if (frame_idx - remove_old) in frames:
            del frames[frame_idx - remove_old]

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

    cap.release()
    with open(os.path.join(result_dir, 'pipeline.bin'), 'wb') as f:
        pickle.dump({
            'frame_preprocess_results': frame_preprocess_results,
            'frame_trigger_results': frame_trigger_results,
            'frame_recognition_results': frame_recognition_results,
        }, f, pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    parser.add_argument("result", help="directory to save result")
    args = parser.parse_args()
    if not os.path.isdir(args.result):
        os.makedirs(args.result)
    run(args.video, args.start, args.end, args.result)


if __name__ == '__main__':
    main()
