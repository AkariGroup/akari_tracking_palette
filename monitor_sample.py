#!/usr/bin/env python3
import argparse

import cv2
from lib.palette import OakdTrackingYoloWithPalette, RoiPalette

# OAK-D LITEの視野角
fov = 56.7


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
    roi_palette = RoiPalette(fov, roi_path=args.roi_path)
    end = False
    while not end:
        oakd_palette = OakdTrackingYoloWithPalette(
            config_path=args.config,
            model_path=args.model,
            fps=args.fps,
            fov=fov,
            cam_debug=args.display_camera,
            robot_coordinate=True,
        )
        while True:
            frame = None
            detections = []
            # frame,trackletsの取得
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
                # 俯瞰図上にtrackletsをセット
                roi_palette.set_tracklets(tracklets)
                # 俯瞰図を描画
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
                print(f"area0: {count[0]} area1: {count[1]} area2: {count[2]}")
            key = cv2.waitKeyEx(10)
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
        oakd_palette.close()


if __name__ == "__main__":
    main()
