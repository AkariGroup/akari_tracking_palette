#!/usr/bin/env python3
import argparse
import math
from typing import Any

import cv2
import numpy as np

from akari_client import AkariClient
from akari_client.color import Colors
from akari_client.position import Positions
from lib.palette import RoiPalette, trackDataList, OakdTrackingYoloWithPalette

# OAK-D LITEの視野角
fov = 56.7
INPUT_POS_DIFF = 0.05


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


def main() -> None:
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
        default=8,
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
    args = parser.parse_args()
    # personのみをtracking対象に指定。他のものをtracking対象にしたい時はここを変更する。
    # target_listの引数指定をしない場合、すべての認識対象がtracking対象になる。
    target_list = [0]

    akari = AkariClient()
    joints = akari.joints
    joints.enable_all_servo()
    joints.set_joint_accelerations(pan=10, tilt=10)
    joints.set_joint_velocities(pan=3, tilt=3)
    limit = joints.get_joint_limits()
    roi_palette = RoiPalette(fov, roi_path=args.roi_path)
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
    end = False
    while not end:
        oakd_palette = OakdTrackingYoloWithPalette(
            config_path=args.config,
            model_path=args.model,
            fps=args.fps,
            fov=fov,
            cam_debug=args.display_camera,
            robot_coordinate=True,
            track_targets=target_list,
        )

        labels = oakd_palette.get_labels()
        track_data_list = trackDataList(labels,roi_palette)
        while True:
            frame = None
            detections = []
            try:
                frame, detections, tracklets = oakd_palette.get_frame()
            except BaseException:
                print("===================")
                print("get_frame() error! Reboot OAK-D.")
                print("If reboot occur frequently, Bandwidth may be too much.")
                print("Set lower FPS.")
                print("==================")
                break
            if frame is not None:
                track_data_list.update_track_data_list(tracklets)
                track_data_list.debug_track_data_list()
                roi_palette.set_tracklets(tracklets)
                roi_palette.draw_frame()
                oakd_palette.display_frame("nn", frame, tracklets, roi_palette)
                count = [0, 0, 0]
                for tracklet in tracklets:
                    # trackletがトラッキング中かつ指定エリアに存在するかを判定し、存在したらカウントアップする
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
            joint_command = False
            # キーボードコントロール
            if key == ord("q"):
                end = True
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
            elif key == ord("w"):
                roi_palette.set_mode("rectangle")
            if key == ord("j"):
                joint_command = True
                pos["pan"] = pos["pan"] + INPUT_POS_DIFF
            if key == ord("l"):
                joint_command = True
                pos["pan"] = pos["pan"] - INPUT_POS_DIFF
            if key == ord("i"):
                joint_command = True
                pos["tilt"] = pos["tilt"] + INPUT_POS_DIFF
            if key == ord("m"):
                joint_command = True
                pos["tilt"] = pos["tilt"] - INPUT_POS_DIFF
            if key == ord("k"):
                joint_command = True
                pos["pan"] = 0
                pos["tilt"] = 0
            # リミット範囲内に収める
            if pos["pan"] < limit["pan"][0]:
                pos["pan"] = limit["pan"][0]
            elif pos["pan"] > limit["pan"][1]:
                pos["pan"] = limit["pan"][1]
            if pos["tilt"] < limit["tilt"][0]:
                pos["tilt"] = limit["tilt"][0]
            elif pos["tilt"] > limit["tilt"][1]:
                pos["tilt"] = limit["tilt"][1]
            if joint_command:
                joints.move_joint_positions(
                    pan=pos["pan"], tilt=pos["tilt"], sync=False
                )
        oakd_palette.close()


if __name__ == "__main__":
    main()
