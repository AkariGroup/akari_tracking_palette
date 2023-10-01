#!/usr/bin/env python3
import argparse
import math
from typing import Any

import cv2
import numpy
import numpy as np

from akari_client import AkariClient
from akari_client.color import Colors
from akari_client.position import Positions
from lib.palette import RoiPalette, OakdTrackingYoloWithPalette

# OAK-D LITEの視野角
fov = 56.7


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
        default=10,
        type=int,
    )
    parser.add_argument(
        "-d",
        "--display_camera",
        help="Display camera rgb and depth frame",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--robot_coordinate",
        help="Convert object pos from camera coordinate to robot coordinate",
        action="store_true",
    )
    args = parser.parse_args()

    oakd_palette = OakdTrackingYoloWithPalette(
        args.config, args.model, args.fps, fov, args.display_camera
    )

    akari = AkariClient()
    joints = akari.joints
    trackings = None
    roi_palette = RoiPalette(fov)
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
        text="area0  area1  area2",
        size=3,
        pos_y=70,
        text_color=Colors.BLACK,
        refresh=False,
        sync=True,
    )
    while True:
        frame = None
        detections = []
        frame, detections, tracklets = oakd_palette.get_frame()
        if args.robot_coordinate:
            head_pos = joints.get_joint_positions()
            pitch = head_pos["tilt"]
            yaw = head_pos["pan"]
        if frame is not None:
            if args.robot_coordinate:
                for detection in detections:
                    converted_pos = convert_to_pos_from_akari(
                        detection.spatialCoordinates, pitch, yaw
                    )
                    detection.spatialCoordinates.x = converted_pos[0][0]
                    detection.spatialCoordinates.y = converted_pos[1][0]
                    detection.spatialCoordinates.z = converted_pos[2][0]
                for tracklet in tracklets:
                    converted_pos = convert_to_pos_from_akari(
                        tracklet.spatialCoordinates, pitch, yaw
                    )
                    tracklet.spatialCoordinates.x = converted_pos[0][0]
                    tracklet.spatialCoordinates.y = converted_pos[1][0]
                    tracklet.spatialCoordinates.z = converted_pos[2][0]
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
                text_color=Colors.BLUE,
                refresh=False,
                sync=False,
            )
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        elif key == ord("r"):
            roi_palette.set_mode("rectangle")
        elif key == ord("c"):
            roi_palette.set_mode("circle")
        elif key == ord("d"):  # 'd'キーを押すと図形をクリア
            roi_palette.reset()
        elif key == ord("0"):
            roi_palette.set_roi_id(0)
        elif key == ord("1"):
            roi_palette.set_roi_id(1)
        elif key == ord("2"):
            roi_palette.set_roi_id(2)


if __name__ == "__main__":
    main()
