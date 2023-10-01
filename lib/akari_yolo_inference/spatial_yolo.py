#!/usr/bin/env python3
import argparse
import math
from typing import Any

import cv2
import numpy
import numpy as np
from oakd_yolo.oakd_spatial_yolo import OakdSpatialYolo

fov = 56.7


def convert_to_pos_from_akari(pos: Any, pitch: float, yaw: float) -> Any:
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
        default="json/yolov4-tiny.json",
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

    oakd_spatial_yolo = OakdSpatialYolo(
        args.config, args.model, args.fps, fov, args.display_camera
    )

    if args.robot_coordinate:
        from akari_client import AkariClient

        akari = AkariClient()
        joints = akari.joints

    while True:
        frame = None
        detections = []
        frame, detections = oakd_spatial_yolo.get_frame()
        if args.robot_coordinate:
            head_pos = joints.get_joint_positions()
            pitch = head_pos["tilt"]
            yaw = head_pos["pan"]
        for detection in detections:
            if args.robot_coordinate:
                converted_pos = convert_to_pos_from_akari(
                    detection.spatialCoordinates, -1 * pitch, -1 * yaw
                )
                detection.spatialCoordinates.x = converted_pos[0][0]
                detection.spatialCoordinates.y = converted_pos[1][0]
                detection.spatialCoordinates.z = converted_pos[2][0]
        oakd_spatial_yolo.display_frame("nn", frame, detections)
        if cv2.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    main()
