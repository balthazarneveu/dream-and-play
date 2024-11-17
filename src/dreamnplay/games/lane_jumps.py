import numpy as np
import pygame
import random
import sys
import cv2
import mediapipe as mp
from pygame_emojis import load_emoji

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3-Lane Jumping Game with Webcam Control")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

# Lane properties
LANE_WIDTH = WIDTH // 3
HOLE_HEIGHT = 50
SPEED = 5

# Player properties
PLAYER_WIDTH = 30
PLAYER_HEIGHT = 30
PLAYER_START_X = LANE_WIDTH // 2 - PLAYER_WIDTH // 2
PLAYER_START_Y = HEIGHT - 80

# Setup
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

# Emoji icons
size = (40, 40)
hand_emoji = load_emoji('🖐️', size)
keyboard_emoji = load_emoji('⌨️', size)
body_emoji = load_emoji('🕺', size)

# Game variables
player_x = PLAYER_START_X
player_lane = 1
holes = []
score = 0
running = True
mp_draw = mp.solutions.drawing_utils

# OpenCV setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# MediaPipe hands and pose
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7)

def create_hole():
    lane = random.randint(0, 2)
    x = lane * LANE_WIDTH
    y = -HOLE_HEIGHT
    return pygame.Rect(x, y, LANE_WIDTH, HOLE_HEIGHT)

def draw_holes():
    for hole in holes:
        pygame.draw.rect(screen, WHITE, hole)

def move_holes():
    global holes, score
    for hole in holes:
        hole.y += SPEED
    holes = [hole for hole in holes if hole.y < HEIGHT]
    if len(holes) == 0 or holes[-1].y > 150:
        holes.append(create_hole())
    score += 1

def check_collision():
    player_rect = pygame.Rect(player_x, PLAYER_START_Y,
                              PLAYER_WIDTH, PLAYER_HEIGHT)
    for hole in holes:
        if hole.y + HOLE_HEIGHT > PLAYER_START_Y and hole.y < PLAYER_START_Y + PLAYER_HEIGHT:
            if hole.x <= player_x <= hole.x + LANE_WIDTH:
                return True
    return False

frame_count = 0
hand_position = None
hand_position_y = None
hand_size = None
nose_position = None
body_control = False

# Main game loop
while running:
    screen.fill(BLACK)

    # Process webcam frames at intervals
    frame_count += 1
    if frame_count % 3 == 0:
        hand_position = None
        hand_position_y = None
        hand_size = None
        nose_position = None
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Hand detection
        results_hands = hands.process(rgb_frame)
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[8]
                wrist = hand_landmarks.landmark[0]
                hand_position = index_finger_tip.x
                hand_position_y = index_finger_tip.y
                hand_size = ((index_finger_tip.x - wrist.x) ** 2 +
                             (index_finger_tip.y - wrist.y) ** 2) ** 0.5

        # Pose detection
        results_pose = pose.process(rgb_frame)
        if results_pose.pose_landmarks:
            nose = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            nose_position = nose.x  # Normalized X position (0 to 1)

    # Hand control
    hand_control = False
    if False:
        pass
    if (hand_position is not None) and (hand_size is not None and hand_size > 0.2):
        hand_control = True
        if hand_position < 0.3:
            player_lane = 0  # Left lane
        elif hand_position > 0.7:
            player_lane = 2  # Right lane
        else:
            player_lane = 1  # Middle lane

    # Body control as fallback
    elif nose_position is not None:
        body_control = True
        if nose_position < 0.3:
            player_lane = 0  # Left lane
        elif nose_position > 0.7:
            player_lane = 2  # Right lane
        else:
            player_lane = 1  # Middle lane

    # Keyboard input fallback
    else:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    player_lane -= 1
                elif event.key == pygame.K_RIGHT:
                    player_lane += 1

    player_lane = max(0, min(player_lane, 2))
    player_x = player_lane * LANE_WIDTH + PLAYER_START_X

    move_holes()
    draw_holes()

    # Display control method emoji
    bottom_live_position = HEIGHT - 40
    disp_emoji_location = (WIDTH - 50, 20)
    if hand_control:
        screen.blit(hand_emoji, disp_emoji_location)
        disp_emoji_location = (int(WIDTH*hand_position), bottom_live_position)
        screen.blit(hand_emoji, disp_emoji_location)
    elif body_control:
        screen.blit(body_emoji, disp_emoji_location)
        disp_emoji_location = (int(WIDTH*nose_position), bottom_live_position)
        screen.blit(body_emoji, disp_emoji_location)
    else:
        screen.blit(keyboard_emoji, disp_emoji_location)

    pygame.draw.rect(screen, BLUE, (player_x, PLAYER_START_Y,
                     PLAYER_WIDTH, PLAYER_HEIGHT))

    if check_collision():
        running = False

    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

    pygame.display.flip()
    clock.tick(30)  # Ensure consistent frame rate

# Cleanup
cap.release()
cv2.destroyAllWindows()

screen.fill(BLACK)
game_over_text = font.render("Game Over!", True, RED)
score_text = font.render(f"Final Score: {score}", True, WHITE)
screen.blit(game_over_text, (WIDTH // 2 -
            game_over_text.get_width() // 2, HEIGHT // 3))
screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, HEIGHT // 2))
pygame.display.flip()
pygame.time.wait(3000)

pygame.quit()
sys.exit()
