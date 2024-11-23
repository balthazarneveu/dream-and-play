import safetensors.torch
import mediapipe as mp
import time
import torch

from dreamnplay.training.dataset import CLASSES_NAMES
from dreamnplay.training.model import MLPBaseline, WINDOW_SIZE

keypoints_names = [e for e in mp.solutions.pose.PoseLandmark]
labels2actions = {v: k for k, v in CLASSES_NAMES.items()}


def process_keypoints_list(keypoints_list):
    positions = []
    for keypoints in keypoints_list:
        for name in keypoints_names:
            x, y, z, v = keypoints[name.name]
            positions.extend([x, y, z, v])

    return torch.tensor(positions)


class DetectorBaseline():
    def __init__(self):
        self.model = MLPBaseline().eval()
        self.model.load_state_dict(
            torch.load("mlpbaseline_model.pth")
        )
        self.start_time = time.time()
        self.all_kps = []
        self.current_action = None

    def infer_action(self, keypoint, capture_time=None):
        keypoint["time"] = capture_time - self.start_time
        self.all_kps.append(keypoint)

        if len(self.all_kps) > WINDOW_SIZE:
            self.all_kps = self.all_kps[-WINDOW_SIZE:]
        if len(self.all_kps) != WINDOW_SIZE:
            self.current_action = None
            return

        # Convert to tensor
        positions = process_keypoints_list(self.all_kps).unsqueeze(0)

        with torch.no_grad():
            res = self.model(positions)
            res = torch.softmax(res, dim=-1)
            index = res[0].argmax().item()
            action = labels2actions[index]
        if action == "JUMP":
            self.current_action = "JUMP"
        elif action == "CROUCH":
            self.current_action = "CROUCH"
        else:
            print("IDLE")
            self.current_action = None
