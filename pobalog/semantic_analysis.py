"""
画像認識結果から文脈を解析する
"""

import argparse
import os
import pickle
from collections import defaultdict
import re
import copy

message_processors = {}


def message_processor(pattern):
    def _message_processor(func):
        def wrapper(state, match):
            return func(state, match)

        message_processors[pattern] = func
        return wrapper

    return _message_processor


@message_processor('(?:いっておいで!|ゆけっ!|がんばれ!|あとすこしだ!\nがんばれ!)(.+)!')
def mp_friend_kuridasi(state, match):
    # 自分のポケモンを繰り出した時のメッセージ
    state['friend']['name'] = match.group(1)


@message_processor('(.+)の(.+)は\n(.+)をくりだした!')
def mp_opponent_kuridasi(state, match):
    # 相手がポケモンを繰り出した時のメッセージ
    state['opponent']['name'] = match.group(3)


def update_message_window(state, recog_value):
    msg = recog_value['text'].replace(' ', '')  # 空白は認識されたりされなかったりするので削除（とりあえず日本語ひらがな想定）
    for pattern, proc in message_processors.items():
        m = re.match(pattern, msg)
        if m is not None:
            proc(state, m)
            break
    else:
        print(f"No processor matched '{msg}'")


def update_hp_bar_friend(state, recog_value):
    state['friend']['hp_ratio'] = recog_value['hp_ratio']


def update_hp_bar_opponent(state, recog_value):
    state['opponent']['hp_ratio'] = recog_value['hp_ratio']


def run(result_dir):
    with open(os.path.join(result_dir, 'pipeline.bin'), 'rb') as f:
        video_results = pickle.load(f)

    frame_recognition_results = video_results['frame_recognition_results']
    state = {
        'friend': {
            'name': None,
            'hp_ratio': 0.0,
        },
        'opponent': {
            'name': None,
            'hp_ratio': 0.0,
        }
    }
    # 前から順に認識結果を受け取り、状態を更新していく
    frame_states = {}

    for frame_idx in sorted(frame_recognition_results.keys()):
        updated = False
        for recog_key, recog_value in frame_recognition_results[frame_idx].items():
            updated = True
            if recog_key == 'message_window':
                update_message_window(state, recog_value)
            elif recog_key == 'hp_bar_friend':
                update_hp_bar_friend(state, recog_value)
            elif recog_key == 'hp_bar_opponent':
                update_hp_bar_opponent(state, recog_value)
            else:
                raise KeyError
        if updated:
            frame_states[frame_idx] = copy.deepcopy(state)
    video_results['frame_states'] = frame_states
    with open(os.path.join(result_dir, 'states.bin'), 'wb') as f:
        pickle.dump(video_results, f, pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", help="directory to save result")
    args = parser.parse_args()
    run(args.result)


if __name__ == '__main__':
    main()
