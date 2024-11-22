from dreamnplay.model import TransformerForPoseMoveDetection
import safetensors.torch
import mediapipe as mp
import time
import torch

keypoints_names = [e for e in mp.solutions.pose.PoseLandmark]

actions2labels = {
    "ACTIVATE": 1,
    "CROUCH": 2,
    "NOTHING": 0,  # nothing
    "JUMP": 3,
    "SHAKE": 4,
}
labels2actions = {v: k for k, v in actions2labels.items()}


def convert_input(all_kps):
    def convert_kp(kp):
        coords = []
        for name in keypoints_names:
            x, y, _, v = kp[name.name]
            coords += [x, y, v]

        return coords, kp["time"]

    all_pos = []
    all_ts = []
    for kp in all_kps:
        pos, t = convert_kp(all_kps[0])
        all_pos.append(pos)
        all_ts.append(t)

    positions = torch.asarray(all_pos).float().clone()
    ts = torch.asarray(all_ts).float().clone()

    return positions, ts

class Detector():
    def __init__(self):
        self.model = TransformerForPoseMoveDetection(
            embed_size=64, num_layers=1, heads=2).eval()
        self.model.load_state_dict(
            safetensors.torch.load_file(
                "checkpoint-27500-500epochs/model.safetensors"
            )
        )
        self.start_time = time.time()
        self.all_kps = []
        self.current_action = None

    def infer_action(self, keypoint, capture_time=None):
        keypoint["time"] = capture_time - self.start_time
        self.all_kps.append(keypoint)
        context_length = 39*2
        if len(self.all_kps) > context_length:
            self.all_kps = self.all_kps[-context_length:]

        # Convert to tensor
        pos, ts = convert_input(self.all_kps)

        pos = pos.unsqueeze(0)
        ts = ts.unsqueeze(0)

        with torch.no_grad():
            res = self.model(None, pos, ts, None)
            res = res.logits
            index = res[0].argmax().item()
            action = labels2actions[index]
        if action == "JUMP":
            self.current_action = "JUMP"
        elif action == "CROUCH":
            self.current_action = "CROUCH"
        else:
            self.current_action = None
