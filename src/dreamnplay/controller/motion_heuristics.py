import mediapipe as mp
keypoints_names = [e for e in mp.solutions.pose.PoseLandmark]
NOSE = mp.solutions.pose.PoseLandmark.NOSE.name


class HeuristicsDetector():
    def __init__(self):
        self.all_kps = []
        self.current_action = None

    def infer_action(self, keypoint, capture_time=None):
        if keypoint is None:
            self.current_action = None
            return
        self.all_kps.append(keypoint)
        context_length = 39*2
        if len(self.all_kps) > context_length:
            self.all_kps = self.all_kps[-context_length:]
        window = 5

        action = None
        if len(self.all_kps) > window+1:
            prev_kp = self.all_kps[-window]
            prev_nose = prev_kp[NOSE]
            curr_nose = keypoint[NOSE]
            if (curr_nose[1] - prev_nose[1]) < -0.05 :
                action = "JUMP"

            if (curr_nose[1] - prev_nose[1]) > 0.05:
                action = "CROUCH"

        if action == "JUMP":
            self.current_action = "JUMP"
        elif action == "CROUCH":
            self.current_action = "CROUCH"
        else:
            self.current_action = None
