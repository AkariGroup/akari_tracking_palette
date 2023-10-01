import json
import math
import os
import time
from datetime import datetime
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
from akari_client import AkariClient
from geojson import Feature, Point, Polygon
from turfpy.measurement import boolean_point_in_polygon

from .akari_yolo_inference.oakd_yolo.oakd_tracking_yolo import OakdTrackingYolo

DISPLAY_WINDOW_SIZE_RATE = 3.0


class RectRoi(object):
    def __init__(self, p1: Tuple[float, float], p2: Tuple[float, float]):
        self.p1: Tuple[float, float] = p1
        self.p2: Tuple[float, float] = p2

    def is_pos_in_rect(self, pos: Tuple[float, float]) -> bool:
        """
        posが矩形エリア内に存在するか確認する。
        """
        point = Feature(geometry=Point(pos))
        polygon = Polygon(
            [
                [
                    (self.p1[0], self.p1[1]),
                    (self.p1[0], self.p2[1]),
                    (self.p2[0], self.p2[1]),
                    (self.p2[0], self.p1[1]),
                ]
            ]
        )
        return boolean_point_in_polygon(point, polygon)


class CircleRoi(object):
    def __init__(self, p1: Tuple[float, float], radius):
        self.p1: Tuple[float, float] = p1
        self.radius: float = radius

    def is_pos_in_circle(self, pos: Tuple[float, float]) -> bool:
        """
        posが円形エリア内に存在するか確認する。
        """
        distance = math.sqrt((pos[0] - self.p1[0]) ** 2 + (pos[1] - self.p1[1]) ** 2)
        if distance <= self.radius:
            return True
        else:
            return False


class RoiList(object):
    def __init__(self):
        self.rect: List[RectRoi] = []
        self.circle: List[CircleRoi] = []

    def add_rect(self, p1, p2):
        """
        矩形エリアをroiリストに追加する。
        """
        self.rect.append(RectRoi(p1, p2))

    def add_circle(self, p1, radius):
        """
        円形エリアをroiリストに追加する。
        """
        self.circle.append(CircleRoi(p1, radius))

    def reset(self):
        """
        roiを初期化する。
        """
        self.rect.clear()
        self.circle.clear()


class RoiPalette(object):
    def __init__(
        self,
        fov: float = 180,
        roi_path: Optional[str] = None,
        show_labels: bool = False,
    ) -> None:
        self.MAX_ROI_ID = 3
        self.ROI_COLOR = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        self.roi: List[RoiList] = [RoiList() for _ in range(self.MAX_ROI_ID)]
        self.mode = "rectangle"
        self.drawing = False
        self.ix = -1
        self.iy = -1
        self.cur_ix = -1
        self.cur_iy = -1
        self.cur_roi_id = 0
        self.fov = fov
        self.show_labels = False
        self.FRAME_HEIGHT = 480
        self.FRAME_WIDTH = 640
        self.Z_RANGE = 10000
        self.X_RANGE = self.Z_RANGE * self.FRAME_WIDTH / self.FRAME_HEIGHT
        self.LATTICE_INTERVAL = 2000
        self.window_name = "palette"
        self.tracklets: Any = None
        akari = AkariClient()
        self.joints = akari.joints
        self.bird_eye_frame = self.create_bird_frame()
        if roi_path is not None:
            self.load_roi(roi_path)

    def load_roi(self, path: str) -> None:
        """
        saveしたroiをpathから取得し、初期状態として使用する。
        """
        if not os.path.isfile(path):
            print(f"ROI setting file path {path} is not available.")
            return
        with open(path, "r") as roi_json:
            roi_data = json.load(roi_json)
            for i in range(0, self.MAX_ROI_ID):
                id = str(i)
                if id in roi_data:
                    if "rectangle" in roi_data[id]:
                        for rect in roi_data[id]["rectangle"]:
                            try:
                                self.roi[i].add_rect(rect["p1"], rect["p2"])
                            except BaseException:
                                print(f"ROI file id:{id} rect type wrong")
                    if "circle" in roi_data[id]:
                        for circle in roi_data[id]["circle"]:
                            try:
                                self.roi[i].add_circle(circle["p1"], circle["radius"])
                            except BaseException:
                                print(f"ROI file id:{id} circle type wrong")

    def save_roi(self) -> None:
        """
        roiをjsonファイルにsaveする。
        """
        current_time = datetime.now()
        file_name = current_time.strftime("%Y%m%d_%H%M%S.json")
        file_path = "roi_json/" + file_name
        roi_dict = {}
        for i in range(0, self.MAX_ROI_ID):
            roi_dict[str(i)] = {}
            if self.roi[i].rect:
                roi_dict[str(i)]["rectangle"] = []
                for rect in self.roi[i].rect:
                    cur_roi = {}
                    cur_roi["p1"] = rect.p1
                    cur_roi["p2"] = rect.p2
                    roi_dict[str(i)]["rectangle"].append(cur_roi)
            if self.roi[i].circle:
                roi_dict[str(i)]["circle"] = []
                for circle in self.roi[i].circle:
                    cur_roi = {}
                    cur_roi["p1"] = circle.p1
                    cur_roi["radius"] = circle.radius
                    roi_dict[str(i)]["circle"].append(cur_roi)
        with open(file_path, "w") as json_file:
            json.dump(roi_dict, json_file, indent=4)

    def set_mode(self, mode: str) -> bool:
        """
        矩形描画と円の描画を切り替える。
        """
        if mode == "rectangle" or mode == "circle":
            self.mode = mode
            return True
        return False

    def set_roi_id(self, id: int) -> bool:
        """
        現在のroiのIDを切り替える。
        """
        if 0 <= id < self.MAX_ROI_ID:
            self.cur_roi_id = id
            return True
        return False

    def set_tracklets(self, tracklets) -> None:
        """
        trackletをセットする。
        """
        self.tracklets = tracklets

    def reset(self) -> None:
        """
        roiを初期化する。
        """
        for roi in self.roi:
            roi.reset()

    def point_to_pos(self, point: Tuple[int, int]) -> Tuple[float, float]:
        """
        bird_frame上の座標をロボットから見た3次元位置に変換する。
        """
        pos_x = (
            -1 * (point[0] - self.FRAME_WIDTH / 2) / self.FRAME_WIDTH * (self.X_RANGE)
        )
        pos_z = (-point[1] + self.FRAME_HEIGHT) / self.FRAME_HEIGHT * (self.Z_RANGE)
        return (pos_x, pos_z)

    def point_diff_to_pos_diff(self, diff: float) -> float:
        """
        bird_frame上の座標の差分をロボットから見た3次元位置の差分に変換する。
        """
        return diff / self.FRAME_HEIGHT * self.Z_RANGE

    def pos_diff_to_point_diff(self, diff: float) -> float:
        """
        ロボットから見た3次元位置の差分をbird_frame上の座標の差分に変換する。
        """
        return diff / self.Z_RANGE * self.FRAME_HEIGHT

    def pos_to_point(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        """
        ロボットから見た3次元位置をbird_frame上の座標に変換する。
        """
        point_x = int(
            -1 * pos[0] / self.X_RANGE * self.FRAME_WIDTH + self.FRAME_WIDTH / 2
        )
        point_y = self.FRAME_HEIGHT - int(pos[1] / self.Z_RANGE * self.FRAME_HEIGHT)
        return (point_x, point_y)

    def create_bird_frame(self) -> np.ndarray:
        """
        bird_frameを作成する。
        """
        # ウィンドウを作成
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.draw_shape)
        frame = np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH, 3), np.uint8)
        center = int(frame.shape[1] / 2)
        # 2m単位の格子線
        lattice_x = center
        while lattice_x < frame.shape[1]:
            cv2.line(
                frame,
                (lattice_x, 0),
                (lattice_x, frame.shape[1]),
                (255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_8,
                shift=0,
            )
            lattice_x += int(self.FRAME_WIDTH / self.X_RANGE * self.LATTICE_INTERVAL)
        lattice_x = center
        while lattice_x > 0:
            cv2.line(
                frame,
                (lattice_x, 0),
                (lattice_x, frame.shape[1]),
                (255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_8,
                shift=0,
            )
            lattice_x -= int(self.FRAME_WIDTH / self.X_RANGE * self.LATTICE_INTERVAL)
        lattice_y = frame.shape[0]
        while lattice_y > 0:
            cv2.line(
                frame,
                (0, lattice_y),
                (self.FRAME_WIDTH, lattice_y),
                (255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_8,
                shift=0,
            )
            lattice_y -= int(self.FRAME_HEIGHT / self.Z_RANGE * self.LATTICE_INTERVAL)
        return frame

    def add_tracklet_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        bird_frame上にtrackletの情報を追加する。
        """
        if self.tracklets is None:
            return frame
        for tracklet in self.tracklets:
            if tracklet.status.name == "TRACKED":
                pos = self.pos_to_point(
                    (tracklet.spatialCoordinates.x, tracklet.spatialCoordinates.z)
                )
                if self.show_labels:
                    cv2.putText(
                        frame,
                        self.labels[tracklet.label],
                        (pos[0] - 30, pos[1] + 5),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.5,
                        (0, 241, 255),
                    )
                cv2.circle(
                    frame,
                    pos,
                    2,
                    (0, 241, 255),
                    thickness=5,
                    lineType=8,
                    shift=0,
                )
        for roi_id in range(0, self.MAX_ROI_ID):
            for rect in self.roi[roi_id].rect:
                cv2.rectangle(
                    frame,
                    self.pos_to_point(rect.p1),
                    self.pos_to_point(rect.p2),
                    self.ROI_COLOR[roi_id],
                    thickness=2,
                )
            for circle in self.roi[roi_id].circle:
                cv2.circle(
                    frame,
                    self.pos_to_point(circle.p1),
                    int(self.pos_diff_to_point_diff(circle.radius)),
                    self.ROI_COLOR[roi_id],
                    thickness=2,
                )
        return frame

    def draw_fov(self, frame: np.ndarray) -> np.ndarray:
        """
        AKARIのヘッドの向きに応じてFOVを描画する。
        """
        center = int(frame.shape[1] / 2)
        alpha = self.fov / 2
        yaw = self.joints.get_joint_positions()["pan"]
        # 正方向fovの境界描画
        ang_p = math.radians(alpha) + yaw
        if ang_p >= 1.57:
            pass
        if 0.464 < ang_p < 1.57:
            fov_cnt = np.array(
                [
                    (center, frame.shape[0]),
                    (frame.shape[1], frame.shape[0]),
                    (
                        frame.shape[1],
                        int(frame.shape[0] - (frame.shape[1] / (2 * math.tan(ang_p)))),
                    ),
                    (center, frame.shape[0]),
                ]
            )
            cv2.fillPoly(frame, [fov_cnt], color=(70, 70, 70))
        else:
            fov_cnt = np.array(
                [
                    (center, frame.shape[0]),
                    (frame.shape[1], frame.shape[0]),
                    (frame.shape[1], 0),
                    (center + int(frame.shape[0] * math.tan(ang_p)), 0),
                    (center, frame.shape[0]),
                ]
            )
            cv2.fillPoly(frame, [fov_cnt], color=(70, 70, 70))
        # 負方向fovの境界描画
        ang_n = -math.radians(alpha) + yaw
        if ang_n <= -1.57:
            pass
        if -0.464 > ang_n > -1.57:
            fov_cnt = np.array(
                [
                    (center, frame.shape[0]),
                    (0, frame.shape[0]),
                    (
                        0,
                        int(frame.shape[0] + (frame.shape[1] / (2 * math.tan(ang_n)))),
                    ),
                    (center, frame.shape[0]),
                ]
            )
            cv2.fillPoly(frame, [fov_cnt], color=(70, 70, 70))
        else:
            fov_cnt = np.array(
                [
                    (center, frame.shape[0]),
                    (0, frame.shape[0]),
                    (0, 0),
                    (center + int(frame.shape[0] * math.tan(ang_n)), 0),
                    (center, frame.shape[0]),
                ]
            )
            cv2.fillPoly(frame, [fov_cnt], color=(70, 70, 70))
        # ロボット向きの描画
        robot_tri = np.array(
            [
                (center, frame.shape[0]),
                (
                    int(center + 20 * math.sin(ang_p)),
                    int(frame.shape[0] - 20 * math.cos(ang_p)),
                ),
                (
                    int(center + 20 * math.sin(ang_n)),
                    int(frame.shape[0] - 20 * math.cos(ang_n)),
                ),
            ]
        )
        cv2.fillPoly(frame, [robot_tri], color=(0, 241, 255))
        return frame

    def draw_frame(self) -> None:
        """
        bird_frameを描画する。
        """
        frame = self.bird_eye_frame.copy()
        frame = self.draw_fov(frame)
        frame = self.add_tracklet_to_frame(frame)
        if self.drawing:
            if self.mode == "rectangle":
                cv2.rectangle(
                    frame,
                    (self.ix, self.iy),
                    (self.cur_ix, self.cur_iy),
                    self.ROI_COLOR[self.cur_roi_id],
                    thickness=2,
                )
            elif self.mode == "circle":
                cv2.circle(
                    frame,
                    (self.ix, self.iy),
                    int(
                        math.sqrt(
                            (self.ix - self.cur_ix) ** 2 + (self.iy - self.cur_iy) ** 2
                        )
                    ),
                    self.ROI_COLOR[self.cur_roi_id],
                    thickness=2,
                )
        pos = self.joints.get_joint_positions()
        cv2.putText(
            frame,
            f"pan: {math.degrees(pos['pan']):.1f}, tilt: {math.degrees(pos['tilt']):.1f}",
            (0, frame.shape[0] - 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"mode: {self.mode}",
            (0, frame.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "roi id: ",
            (0, frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"{self.cur_roi_id}",
            (60, frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.ROI_COLOR[self.cur_roi_id],
            2,
        )
        # マウスの現在座標を表示
        pos_i = self.point_to_pos((self.cur_ix, self.cur_iy))
        cv2.putText(
            frame,
            f"x: {pos_i[0]/1000:.2f}m, z: {pos_i[1]/1000:.2f}m",
            (0, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        cv2.imshow("palette", frame)

    def draw_shape(self, event, x, y, flags, param) -> None:
        """
        bird_frame上に矩形、円形を描画する。
        """
        self.cur_ix = x
        self.cur_iy = y
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = self.cur_ix, self.cur_iy
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.mode == "rectangle":
                self.roi[self.cur_roi_id].add_rect(
                    self.point_to_pos((self.ix, self.iy)), self.point_to_pos((x, y))
                )
            elif self.mode == "circle":
                self.roi[self.cur_roi_id].add_circle(
                    self.point_to_pos((self.ix, self.iy)),
                    self.point_diff_to_pos_diff(
                        math.sqrt(
                            (self.ix - self.cur_ix) ** 2 + (self.iy - self.cur_iy) ** 2
                        )
                    ),
                )
        self.draw_frame()

    def is_point_in_roi(self, id: int, pos: Tuple[float, float]) -> bool:
        """
        bird_frame上の座標が指定されたroiの内部にあるかを判定して返す。
        """
        if not 0 <= id < self.MAX_ROI_ID:
            return False
        for rect in self.roi[id].rect:
            if rect.is_pos_in_rect(pos):
                return True
        for circle in self.roi[id].circle:
            if circle.is_pos_in_circle(pos):
                return True
        return False


class OakdTrackingYoloWithPalette(OakdTrackingYolo):
    def display_frame(
        self,
        name: str,
        frame: np.ndarray,
        tracklets: List[Any],
        roi_palette: RoiPalette,
    ) -> None:
        """
        roi_paletteの情報を元に、認識結果の枠の色を変えて描画する。
        """
        if frame is not None:
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1] * DISPLAY_WINDOW_SIZE_RATE),
                    int(frame.shape[0] * DISPLAY_WINDOW_SIZE_RATE),
                ),
            )
            if tracklets is not None:
                for tracklet in tracklets:
                    if tracklet.status.name == "TRACKED":
                        roi = tracklet.roi.denormalize(frame.shape[1], frame.shape[0])
                        x1 = int(roi.topLeft().x)
                        y1 = int(roi.topLeft().y)
                        x2 = int(roi.bottomRight().x)
                        y2 = int(roi.bottomRight().y)
                        try:
                            label = self.labels[tracklet.label]
                        except BaseException:
                            label = tracklet.label
                        self.text.put_text(frame, str(label), (x1 + 10, y1 + 20))
                        self.text.put_text(
                            frame,
                            f"ID: {[tracklet.id]}",
                            (x1 + 10, y1 + 45),
                        )
                        self.text.put_text(
                            frame, tracklet.status.name, (x1 + 10, y1 + 70)
                        )
                        color = (168, 87, 167)
                        for i in range(0, 3):
                            if roi_palette.is_point_in_roi(
                                i,
                                (
                                    tracklet.spatialCoordinates.x,
                                    tracklet.spatialCoordinates.z,
                                ),
                            ):
                                color = roi_palette.ROI_COLOR[i]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=3)
                        if tracklet.spatialCoordinates.z != 0:
                            self.text.put_text(
                                frame,
                                "X: {:.2f} m".format(
                                    tracklet.spatialCoordinates.x / 1000
                                ),
                                (x1 + 10, y1 + 95),
                            )
                            self.text.put_text(
                                frame,
                                "Y: {:.2f} m".format(
                                    tracklet.spatialCoordinates.y / 1000
                                ),
                                (x1 + 10, y1 + 120),
                            )
                            self.text.put_text(
                                frame,
                                "Z: {:.2f} m".format(
                                    tracklet.spatialCoordinates.z / 1000
                                ),
                                (x1 + 10, y1 + 145),
                            )
            cv2.putText(
                frame,
                "NN fps: {:.2f}".format(
                    self.counter / (time.monotonic() - self.startTime)
                ),
                (2, frame.shape[0] - 4),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.3,
                (255, 255, 255),
            )
            # Show the frame
            cv2.imshow(name, frame)
