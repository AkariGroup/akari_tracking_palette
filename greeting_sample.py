#!/usr/bin/env python3
import argparse
import time
from typing import List

import cv2
from akari_client import AkariClient
from lib.palette import OakdTrackingYoloWithPalette, RoiPalette

# OAK-D LITEの視野角
fov = 56.7

LOCK_TIME = 3.0
AREA_COUNT_THRESHOULD = 5


# 人のIDごとにエリア内で検出された回数、エリアフラグを記録
class TrackPerson(object):
    def __init__(self) -> None:
        self.id: int = 0
        self.area0_count: int = 0
        self.area1_count: int = 0
        self.area0_flag: bool = False
        self.area1_flag: bool = False


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
    roi_palette = RoiPalette(fov, roi_path=args.roi_path)
    akari = AkariClient()
    m5 = akari.m5stack
    prev_time = time.time()
    end = False
    is_reset = True
    person_list: List[TrackPerson] = []
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
        while True:
            # 音声出力&画面変更後、一定時間経過するまでロックする
            if time.time() - prev_time >= LOCK_TIME and is_reset is False:
                is_reset = True
                m5.set_display_image("/jpg/logo320.jpg", sync=False)
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
                for tracklet in tracklets:
                    if (
                        tracklet.status.name == "NEW"
                        or tracklet.status.name == "TRACKED"
                    ):
                        is_tracking = False
                        for person in person_list:
                            # 既にトラッキングしている人だった場合、area0にいたらカウントアップ。もしくはarea0で挨拶済みの場合は、area1にいたらカウントアップ
                            if tracklet.id == person.id:
                                is_tracking = True
                                # area0に人がいたら、カウントアップ
                                if roi_palette.is_point_in_roi(
                                    0,
                                    (
                                        tracklet.spatialCoordinates.x,
                                        tracklet.spatialCoordinates.z,
                                    ),
                                ):
                                    person.area0_count += 1
                                # area0_flagが立っている人がarea1にいたら、カウントアップ
                                elif (
                                    roi_palette.is_point_in_roi(
                                        1,
                                        (
                                            tracklet.spatialCoordinates.x,
                                            tracklet.spatialCoordinates.z,
                                        ),
                                    )
                                    and person.area0_flag
                                ):
                                    person.area1_count += 1
                        # 新しい人だった場合はperson_listにid追加
                        if not is_tracking:
                            new_person = TrackPerson()
                            new_person.id = tracklet.id
                            person_list.append(new_person)

                # area_countから挨拶するかどうか決める。
                for person in person_list:
                    # area0に一定時間いてまだ挨拶していない人の場合
                    if (
                        person.area0_count > AREA_COUNT_THRESHOULD
                        and is_reset
                        and not person.area0_flag
                    ):
                        is_reset = False
                        person.area0_flag = True
                        prev_time = time.time()
                        m5.set_display_text(text="こんにちは", sync=False)
                    # area0で挨拶していて、area1に一定時間いた場合
                    if (
                        person.area1_count > AREA_COUNT_THRESHOULD
                        and is_reset
                        and person.area0_flag
                        and not person.area1_flag
                    ):
                        is_reset = False
                        person.area1_flag = True
                        prev_time = time.time()
                        m5.set_display_text(text="さようなら", sync=False)
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
