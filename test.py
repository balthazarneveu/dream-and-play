import dreamnplay.controller.webcame_controller
from dreamnplay.model import TransformerForPoseMoveDetection
import mediapipe as mp
import time
import torch
import safetensors.torch

webcame_controller = dreamnplay.controller.webcame_controller.Controller(
    webcam_show=True, allow_hand_control=False, allow_body_control=True
)

keypoints_names = [e for e in mp.solutions.pose.PoseLandmark]

model = TransformerForPoseMoveDetection(embed_size=64, num_layers=1, heads=2).eval()
model.load_state_dict(
    safetensors.torch.load_file(
        "/home/victor/Documents/hackaton/dream-and-play/ckpts/checkpoint-22000/model.safetensors"
    )
)

actions2labels = {
    "ACTIVATE": 1,
    "CROUCH": 2,
    "NOTHING": 0,  # nothing
    "JUMP": 3,
    "SHAKE": 4,
}
labels2actions = {v: k for k, v in actions2labels.items()}


def get_keypoint(
    webcame_controller: dreamnplay.controller.webcame_controller.Controller,
    is_showing_action=False,
):
    webcame_controller.webcam_show_info = is_showing_action
    webcame_controller.process_webcam("====")
    results_pose = webcame_controller.results_pose

    if results_pose.pose_landmarks is None:
        return None

    positions = {
        keypoint_name.name: results_pose.pose_landmarks.landmark[keypoint_name]
        for keypoint_name in keypoints_names
    }

    converted_positions = {
        keypoint_name: (position.x, position.y, position.z, position.visibility)
        for keypoint_name, position in positions.items()
    }

    return converted_positions


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


all_kps = []

t = time.time()
# Fill once
for window in range(15):
    keypoints = get_keypoint(webcame_controller)
    if keypoints is not None:
        e = time.time()
        keypoints["time"] = e - t
        all_kps.append(keypoints)

i = 0
while True:
    i += 1
    all_kps = all_kps[1:]
    while (keypoint := get_keypoint(webcame_controller)) is None:
        pass

    keypoint["time"] = e - t
    all_kps.append(keypoint)

    # Convert to tensor
    pos, ts = convert_input(all_kps)

    pos = pos.unsqueeze(0)
    ts = ts.unsqueeze(0)

    with torch.no_grad():
        res = model(None, pos, ts, None)
        res = res[0].argmax().item()

    # display label
    print(i, labels2actions[res])
