#!/usr/bin/env python3
import argparse
import math
from typing import Any

import cv2
import numpy
import numpy as np
from oakd_yolo.oakd_tracking_yolo import OakdTrackingYolo

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

    oakd_tracking_yolo = OakdTrackingYolo(
        args.config, args.model, args.fps, fov, args.display_camera
    )

    if args.robot_coordinate:
        from akari_client import AkariClient

        akari = AkariClient()
        joints = akari.joints
    trackings = None

    while True:
        frame = None
        detections = []
        frame, detections, tracklets = oakd_tracking_yolo.get_frame()
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
            oakd_tracking_yolo.display_frame("nn", frame, tracklets)
        if cv2.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    main()
