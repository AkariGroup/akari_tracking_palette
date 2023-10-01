#!/usr/bin/env python3
import argparse

import cv2
import numpy as np
from oakd_yolo.oakd_yolo import OakdYolo


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
    args = parser.parse_args()
    oakd_yolo = OakdYolo(args.config, args.model, args.fps)

    while True:
        frame = None
        detections = []
        frame, detections = oakd_yolo.get_frame()
        oakd_yolo.display_frame("nn", frame, detections)
        if cv2.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    main()
