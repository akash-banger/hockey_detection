from ultralytics import YOLO
import supervision as sv
import pickle
import os
from utils.bbox_utils import *
import pandas as pd
import cv2
import numpy as np


class Tracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(sekf, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if object == "ball":
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]["position"] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(
            ball_positions, columns=["x1", "y1", "x2", "y2"]
        )

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [
            {1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()
        ]

        return ball_positions

    def detect_frames(self, frames):
        detections = []
        for i in range(0, len(frames), 24):
            batch_frames = frames[i : i + 24]
            batch_detections = self.model.predict(batch_frames, verbose=False)
            detections.extend(batch_detections)
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {"players": [], "referees": [], "puck": []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision
            )

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["puck"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if "player" in cls_names_inv and cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if "referee" in cls_names_inv and cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if "puck" in cls_names_inv and cls_id == cls_names_inv["puck"]:
                    tracks["puck"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, jersey_number=None, name=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=3,
            lineType=cv2.LINE_4,
        )

        # Calculate text dimensions and positions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        
        # Determine text content and measurements
        if jersey_number and name:
            text = f"#{jersey_number} {name}"
        elif jersey_number:
            text = f"#{jersey_number}"
        elif name:
            text = name
        else:
            text = "Player"  # Default text when no name or number is available
        
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate background rectangle dimensions with padding
        padding = 5
        rect_width = text_width + (padding * 2)
        rect_height = text_height + (padding * 2)
        
        # Position the text box above the ellipse
        x_text = x_center - (rect_width // 2)
        y_text = y2 - 25  # Offset above the ellipse
        
        # Draw background rectangle
        cv2.rectangle(
            frame,
            (int(x_text), int(y_text - rect_height)),
            (int(x_text + rect_width), int(y_text)),
            color,
            cv2.FILLED,
        )
        
        # Add a black border around the rectangle
        cv2.rectangle(
            frame,
            (int(x_text), int(y_text - rect_height)),
            (int(x_text + rect_width), int(y_text)),
            (0, 0, 0),
            1,
        )
        
        # Draw text
        cv2.putText(
            frame,
            text,
            (int(x_text + padding), int(y_text - padding)),
            font,
            font_scale,
            (0, 0, 0),  # Black text
            thickness,
        )

        return frame
    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array(
            [
                [x, y],
                [x - 10, y - 20],
                [x + 10, y - 20],
            ]
        )
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            puck_dict = tracks["puck"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for _, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(
                    frame,
                    player["bbox"],
                    color,
                    player.get("jersey_number", None),
                    player.get("name", None),
                )

                if player.get("has_ball", False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(
                    frame,
                    referee["bbox"],
                    (0, 255, 255),
                    jersey_number="",
                    name="Referee",
                )

            # Draw ball
            for _, puck in puck_dict.items():
                frame = self.draw_traingle(frame, puck["bbox"], (0, 255, 0))

            output_video_frames.append(frame)

        return output_video_frames
