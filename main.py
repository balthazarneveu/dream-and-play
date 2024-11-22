import dreamnplay.controller.webcame_controller
import mediapipe as mp
import pandas as pd
import time
import random
import datetime


def get_keypoint(
    webcame_controller: dreamnplay.controller.webcame_controller.Controller,
    is_showing_action=False,
):
    webcame_controller.webcam_show_info = is_showing_action
    webcame_controller.process_webcam(action)
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


def process_action(
    webcame_controller, action, time_seconds=5, display_info=False, start_time=0
):
    t_end = time.time() + time_seconds
    action_keypoint = []
    while (cur_time := time.time()) < t_end:
        keypoint = get_keypoint(webcame_controller, display_info)
        if keypoint is not None:
            keypoint = {
                "relative_time": cur_time - start_time,
                **keypoint,
                "action": action if display_info is False else "INFO_DISPLAYED",
            }

            action_keypoint.append(keypoint)

    return action_keypoint


if __name__ == "__main__":
    webcame_controller = dreamnplay.controller.webcame_controller.Controller(
        webcam_show=True, allow_hand_control=False, allow_body_control=True
    )

    keypoints_names = [e for e in mp.solutions.pose.PoseLandmark]

    actions = ["JUMP", "CROUCH", "ACTIVATE", "IDLE", "LEFT_STEP", "RIGHT_STEP", "SHAKE"]

    all_dataframes = []
    try:
        start_time = time.time()
        while True:
            # Select action
            action = random.choice(actions)

            keypoints = process_action(
                webcame_controller,
                action,
                0.3,
                display_info=True,
                start_time=start_time,
            )
            if keypoints is not None:
                df = pd.DataFrame(keypoints)
                all_dataframes.append(df)

            # Record
            total_time_second = 1 if action == "JUMP" else 3
            keypoints = process_action(
                webcame_controller,
                action,
                min(0.8 + random.random() * 2, total_time_second),
                start_time=start_time,
            )

            # Save action
            if keypoints is not None:
                df = pd.DataFrame(keypoints)
                all_dataframes.append(df)

    except KeyboardInterrupt:
        print("stopping recording")
        pass

    date = datetime.datetime.utcnow()
    date = date.strftime("%Y-%m-%d-%H-%M-%S")
    pd.concat(all_dataframes).to_csv(f"{date}.csv", index=False)
