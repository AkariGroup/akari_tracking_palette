#!/usr/bin/env python3
import argparse
import math
from typing import Any

import cv2
import threading
import numpy
import numpy as np

from akari_client import AkariClient
from akari_client.color import Colors
from akari_client.position import Positions
from lib.palette import RoiPalette, OakdTrackingYoloWithPalette
import openai
from lib.akari_chatgpt_bot.lib.chat import chat_stream
from lib.akari_chatgpt_bot.lib.conf import OPENAI_APIKEY
from lib.akari_chatgpt_bot.lib.transcribe_google_speech import (
    MicrophoneStream,
    get_db_thresh,
    listen_print_loop,
)


# OAK-D LITEの視野角
fov = 56.7
POS_DIFF = 0.1

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
POWER_THRESH_DIFF = 20  # 周辺音量にこの値を足したものをpower_threshouldとする

openai.api_key = OPENAI_APIKEY
host: str = ""
port: str = ""
tracklets = None

def is_voice_command(text: str) -> bool:
    if "右向いて" in text or "右むいて" in text or "右見て" in text or "右みて" in text:
        joints.move_joint_positions(pan=-0.45, sync=False)
        return True
    elif "左向いて" in text or "左むいて" in text or "左見て" in text or "左みて" in text:
        joints.move_joint_positions(pan=0.45, sync=False)
        return True
    elif "戻って" in text or "もどって" in text:
        joints.move_joint_positions(pan=0.0, sync=False)
        return True
    return False

def tracklet_to_text(oakd_palette, roi_palette):
    text = ""
    labels = oakd_palette.get_labels()
    if(tracklets is None):
        return text
    for tracklet in tracklets:
        text += f"種類: {labels[tracklet.label]} "
        text += f"x: {tracklet.spatialCoordinates.x / 10:.0f}cm, z: {tracklet.spatialCoordinates.z / 10:.0f}cm. \n"
        for i in range(0, 3):
            if (
                tracklet.status.name == "TRACKED"
                and roi_palette.is_point_in_roi(
                    i,
                    (
                        tracklet.spatialCoordinates.x,
                        tracklet.spatialCoordinates.z,
                    ),
                )
            ):
                text += f"エリア{i}にいる。 "
    print(f"track_info:")
    print(f"    {text}")
    return text

def convert_to_pos_from_akari(pos: Any, pitch: float, yaw: float) -> Any:
    pitch = -1 * pitch
    yaw = -1 * yaw
    cur_pos = np.array([[pos.x], [pos.y], [pos.z]])
    arr_y = np.array(
        [
            [math.cos(yaw), 0, math.sin(yaw)],
            [0, 1, 0],
            [-math.sin(yaw), 0, math.cos(yaw)],
        ]
    )
    arr_p = np.array(
        [
            [1, 0, 0],
            [
                0,
                math.cos(pitch),
                -math.sin(pitch),
            ],
            [0, math.sin(pitch), math.cos(pitch)],
        ]
    )
    ans = arr_y @ arr_p @ cur_pos
    return ans


def talk(args,oakd_palette,roi_palette):
    timeout: float = args.timeout
    power_threshold: float = args.power_threshold
    if power_threshold == 0:
        power_threshold = get_db_thresh() + POWER_THRESH_DIFF
    print(f"power_threshold set to {power_threshold:.3f}db")
    if args.voicevox_local:
        from lib.akari_chatgpt_bot.lib.voicevox import TextToVoiceVox
        host = args.voicevox_host
        port = args.voicevox_port
        text_to_voice = TextToVoiceVox(host, port)
    else:
        from lib.akari_chatgpt_bot.lib.conf import VOICEVOX_APIKEY
        from lib.akari_chatgpt_bot.lib.voicevox import TextToVoiceVoxWeb

        text_to_voice = TextToVoiceVoxWeb(apikey=VOICEVOX_APIKEY)
    messages = [
        {
            "role": "system",
            "content": "「あかり」という名前の見守りロボットとしてロールプレイをします。質問と合わせて周囲にある物の名前と見守りロボットから見た位置情報が伝えられるので、それを元に質問に回答してください。xは左右で右方向が+, zは奥行き方向が+です。あなたは人がたくさんいると嬉しく、誰もいないと寂しく感じます。",
        }
    ]
    while True:
        # 音声認識
        text = ""
        responses = None
        with MicrophoneStream(RATE, CHUNK, timeout, power_threshold) as stream:
            print("Enterを入力してください")
            input()
            responses = stream.transcribe()
            if responses is not None:
                text = listen_print_loop(responses)

        if not is_voice_command(text):
            # chatGPT
            attention = "。120文字以内で回答してください。"
            object_info = tracklet_to_text(oakd_palette,roi_palette)
            messages.append({"role": "user", "content": text + attention + object_info})
            print(f"User   : {text}")
            print("ChatGPT: ")
            response = ""
            # 音声合成
            for sentence in chat_stream(messages):
                text_to_voice.put_text(sentence)
                response += sentence
                print(sentence, end="")
            messages.append({"role": "assistant", "content": response})
            print("")
        print("")



def main() -> None:
    global tracklets
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="Provide model name or model path for inference",
        default="yolov4_tiny_coco_416x416",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Provide config path for inference",
        default="lib/akari_yolo_inference/json/yolov4-tiny.json",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--fps",
        help="Camera frame fps. This should be smaller than nn inference fps",
        default=7,
        type=int,
    )
    parser.add_argument(
        "-d",
        "--display_camera",
        help="Display camera rgb and depth frame",
        action="store_true",
    )
    parser.add_argument(
        "--roi_path",
        help="Roi json file path",
        default=None,
        type=str,
    )
    parser.add_argument("--voicevox_local", action="store_true")
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=0.5,
        help="Microphone input power timeout",
    )
    parser.add_argument(
        "-p",
        "--power_threshold",
        type=float,
        default=0,
        help="Microphone input power threshold",
    )
    args = parser.parse_args()
    oakd_palette = OakdTrackingYoloWithPalette(
        config_path=args.config,
        model_path=args.model,
        fps=args.fps,
        fov=fov,
        cam_debug=args.display_camera,
        robot_coordinate=True,
        track_targets=[0],
    )
    roi_palette = RoiPalette(fov, roi_path=args.roi_path)
    talk_thread = threading.Thread(
        target=talk, args=(args,oakd_palette,roi_palette)
    )
    talk_thread.start()
    akari = AkariClient()
    joints = akari.joints
    joints.enable_all_servo()
    joints.set_joint_accelerations(pan=20, tilt=20)
    joints.set_joint_velocities(pan=3, tilt=3)
    m5 = akari.m5stack
    m5.set_display_text(
        text="人数カウンタ",
        size=4,
        pos_y=10,
        text_color=Colors.BLACK,
        refresh=True,
        sync=True,
    )
    m5.set_display_text(
        text="area0",
        size=3,
        pos_x=10,
        pos_y=70,
        text_color=Colors.BLUE,
        refresh=False,
        sync=True,
    )
    m5.set_display_text(
        text="area1",
        size=3,
        pos_x=Positions.CENTER,
        pos_y=70,
        text_color=Colors.GREEN,
        refresh=False,
        sync=True,
    )
    m5.set_display_text(
        text="area2",
        size=3,
        pos_x=220,
        pos_y=70,
        text_color=Colors.RED,
        refresh=False,
        sync=True,
    )
    while True:
        frame = None
        detections = []
        frame, detections, tracklets = oakd_palette.get_frame()
        if frame is not None:
            roi_palette.set_tracklets(tracklets)
            roi_palette.draw_frame()
            oakd_palette.display_frame("nn", frame, tracklets, roi_palette)
            count = [0, 0, 0]
            for tracklet in tracklets:
                for i in range(0, 3):
                    if (
                        tracklet.status.name == "TRACKED"
                        and roi_palette.is_point_in_roi(
                            i,
                            (
                                tracklet.spatialCoordinates.x,
                                tracklet.spatialCoordinates.z,
                            ),
                        )
                    ):
                        count[i] += 1
            m5.set_display_text(
                text=f" {count[0]}  {count[1]}  {count[2]} \n",
                size=10,
                pos_y=120,
                text_color=Colors.NAVY,
                refresh=False,
                sync=False,
            )
        key = cv2.waitKeyEx(10)
        pos = joints.get_joint_positions()
        changed = False
        if key == ord("q"):
            break
        elif key == ord("r"):
            roi_palette.set_mode("rectangle")
        elif key == ord("c"):
            roi_palette.set_mode("circle")
        elif key == ord("d"):
            roi_palette.reset()
        elif key == ord("s"):
            roi_palette.save_roi()
        elif key == ord("0"):
            roi_palette.set_roi_id(0)
        elif key == ord("1"):
            roi_palette.set_roi_id(1)
        elif key == ord("2"):
            roi_palette.set_roi_id(2)
        if key == ord("l"):
            pos["pan"] += POS_DIFF
            changed = True
        if key == ord("j"):
            pos["pan"] -= POS_DIFF
            changed = True
        if key == ord("m"):
            pos["tilt"] -= POS_DIFF
            changed = True
        if key == ord("k"):
            pos["tilt"] += POS_DIFF
            changed = True
        if changed:
            joints.move_joint_positions(pan=pos["pan"], tilt=pos["tilt"], sync=False)
    talk_thread.join()

if __name__ == "__main__":
    main()
