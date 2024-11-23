import sys
import cv2
import mediapipe as mp
import time
from dreamnplay.controller.motion_detection import Detector
from dreamnplay.controller.motion_detection_baseline import DetectorBaseline
from dreamnplay.controller.motion_heuristics import HeuristicsDetector
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
keypoints_names = [e for e in mp.solutions.pose.PoseLandmark]


class Controller:
    def __init__(self, webcam_show: bool = False, allow_hand_control: bool = True, allow_body_control: bool = True,
                 heuristics=False):
        self.allow_body_control = allow_body_control
        self.allow_hand_control = allow_hand_control
        assert allow_hand_control or allow_body_control, "At least one control mode must be enabled."
        self.frame_count = 0
        if not heuristics:
            # self.motion_detector = Detector()
            self.motion_detector = DetectorBaseline()
        else:
            self.motion_detector = HeuristicsDetector()
        # Initialize MediaPipe hands and pose
        if allow_hand_control:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
        if allow_body_control:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.7
            )

        # OpenCV webcam setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        # Control variables
        self.hand_position = None
        self.hand_position_y = None
        self.hand_size = None
        self.nose_position = None
        self.hand_control = False
        self.body_control = False
        self.webcam_show = webcam_show
        self.current_position = None
        self.current_action = None

    def process_webcam(self):
        """Process webcam input to detect hands or body."""

        if self.frame_count % 3 == 0:
            # Reset control flags
            self.hand_control = False
            self.body_control = False
            self.current_position = None
            _, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Hand detection
            if self.allow_hand_control:
                results_hands = self.hands.process(rgb_frame)
                if results_hands.multi_hand_landmarks:
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        index_finger_tip = hand_landmarks.landmark[8]
                        wrist = hand_landmarks.landmark[0]
                        hand_position = index_finger_tip.x
                        self.hand_size = ((index_finger_tip.x - wrist.x) ** 2 +
                                          (index_finger_tip.y - wrist.y) ** 2) ** 0.5
                        self.hand_control = self.hand_size > 0.2
                        if self.hand_control:
                            self.current_position = hand_position
                        if self.webcam_show:
                            mp_draw.draw_landmarks(frame, hand_landmarks,
                                                   mp_hands.HAND_CONNECTIONS)

            if not self.hand_control and self.allow_body_control:
                # Body detection
                results_pose = self.pose.process(rgb_frame)
                if results_pose.pose_landmarks:
                    
                    nose = results_pose.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
                    # Body control only if hand control is inactive
                    self.body_control = not self.hand_control
                    self.current_position = nose.x
                    current_time = time.time()

                    positions = {
                        keypoint_name.name: results_pose.pose_landmarks.landmark[keypoint_name]
                        for keypoint_name in keypoints_names
                    }

                    converted_positions = {
                        keypoint_name: (position.x, position.y,
                                        position.z, position.visibility)
                        for keypoint_name, position in positions.items()
                    }
                    # print(results_pose.pose_landmarks)
                    self.motion_detector.infer_action(
                        converted_positions, capture_time=current_time)
                    self.current_action = self.motion_detector.current_action
                    if self.webcam_show:
                        mp_draw.draw_landmarks(frame, results_pose.pose_landmarks,
                                               mp_pose.POSE_CONNECTIONS)
            if self.webcam_show:
                cv2.imshow("Webcam Feed", frame)
                cv2.waitKey(1)
        self.frame_count += 1
        return



    def release_resources(self):
        """Release webcam and cleanup resources."""
        self.cap.release()
        cv2.destroyAllWindows()
