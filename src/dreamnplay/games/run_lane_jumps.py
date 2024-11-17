import pygame
import random
import sys
from pygame_emojis import load_emoji
from dreamnplay.controller.webcam_controller import Controller

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3-Lane Jumping Game")

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

# Load emojis
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

# Initialize Controller
controller = Controller()


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


# Main game loop
while running:
    screen.fill(BLACK)

    # Process input
    controller.process_webcam()
    player_lane = controller.get_control(player_lane)

    # Update player position
    player_x = player_lane * LANE_WIDTH + PLAYER_START_X

    move_holes()
    draw_holes()

    # Display control method emoji
    bottom_live_position = HEIGHT - 40
    disp_emoji_location = (WIDTH - 50, 20)
    if controller.hand_control:
        screen.blit(hand_emoji, disp_emoji_location)
        disp_emoji_location = (int(WIDTH*controller.current_position), bottom_live_position)
        screen.blit(hand_emoji, disp_emoji_location)
    elif controller.body_control:
        screen.blit(body_emoji, disp_emoji_location)
        disp_emoji_location = (int(WIDTH*controller.current_position), bottom_live_position)
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
controller.release_resources()

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
