import pandas as pd
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import ast
from PIL import Image

df = pd.read_csv("2024-11-22-12-59-55.csv")
keypoints_names = [e for e in mp.solutions.pose.PoseLandmark]


def process_row(row):
    landmarks = []
    for keypoint_name in keypoints_names:
        pos = row[keypoint_name.name]
        pos = ast.literal_eval(pos)
        x, y, z, vis = pos

        landmarks.append(
            landmark_pb2.NormalizedLandmark(
                x=x,
                y=y,
                z=z,
                visibility=vis,
            )
        )

    return landmark_pb2.NormalizedLandmarkList(landmark=landmarks)


all_frames = []
for _, row in df.iterrows():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        f"=== {row['action']} ===",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    landmarks = process_row(row)
    mp.solutions.drawing_utils.draw_landmarks(
        frame,
        landmarks,
        mp.solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
    )
    all_frames.append(Image.fromarray(frame))

all_frames[0].save(
    "animation.gif", save_all=True, append_images=all_frames[1:], duration=1, loop=1000
)
