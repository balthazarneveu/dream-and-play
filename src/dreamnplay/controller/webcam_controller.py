import sys
import pygame
import cv2
import mediapipe as mp
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


class Controller:
    def __init__(self, webcam_show: bool = False):
        self.frame_count = 0
        # Initialize MediaPipe hands and pose
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
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

    def process_webcam(self):
        """Process webcam input to detect hands or body."""

        if self.frame_count % 3 == 0:
            # Reset control flags
            self.hand_control = False
            self.body_control = False
            _, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Hand detection
            results_hands = self.hands.process(rgb_frame)
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    index_finger_tip = hand_landmarks.landmark[8]
                    wrist = hand_landmarks.landmark[0]
                    self.hand_position = index_finger_tip.x
                    self.hand_size = ((index_finger_tip.x - wrist.x) ** 2 +
                                      (index_finger_tip.y - wrist.y) ** 2) ** 0.5
                    self.hand_control = self.hand_size > 0.2

                    if self.webcam_show:
                        mp_draw.draw_landmarks(frame, hand_landmarks,
                                               mp_hands.HAND_CONNECTIONS)

            if not self.hand_control:
                # Body detection
                results_pose = self.pose.process(rgb_frame)
                if results_pose.pose_landmarks:
                    nose = results_pose.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
                    self.nose_position = nose.x
                    # Body control only if hand control is inactive
                    self.body_control = not self.hand_control
                    mp_draw.draw_landmarks(frame, results_pose.pose_landmarks,
                                           mp_pose.POSE_CONNECTIONS)
            if self.webcam_show:
                cv2.imshow("Webcam Feed", frame)
                cv2.waitKey(1)
        self.frame_count += 1
        return

    def get_control(self, player_lane):
        """
        Determine the control method (hand, body, or keyboard) and return the updated lane.
        """

        if self.hand_control and self.hand_position is not None:
            # Hand gesture control
            if self.hand_position < 0.3:
                player_lane = 0  # Left lane
            elif self.hand_position > 0.7:
                player_lane = 2  # Right lane
            else:
                player_lane = 1  # Middle lane

        elif self.body_control and self.nose_position is not None:
            # Body position control
            if self.nose_position < 0.3:
                player_lane = 0  # Left lane
            elif self.nose_position > 0.7:
                player_lane = 2  # Right lane
            else:
                player_lane = 1  # Middle lane
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    print("QUIT!")
                    pygame.quit()
                    sys.exit()
                if not self.hand_control and not self.body_control:
                    if event.key == pygame.K_LEFT:
                        player_lane -= 1
                    elif event.key == pygame.K_RIGHT:
                        player_lane += 1
        # Ensure the lane remains within bounds
        return max(0, min(player_lane, 2))

    def release_resources(self):
        """Release webcam and cleanup resources."""
        self.cap.release()
        cv2.destroyAllWindows()
