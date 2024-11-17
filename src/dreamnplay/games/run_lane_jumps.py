import pygame
import random
import sys

from dreamnplay.controller.webcam_controller import Controller
from dreamnplay.view.overlay_controller import display_control_mode

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


def process_motion(controller, player_lane):
    """
    Determine the control method (hand, body, or keyboard) and return the updated lane.
    """

    # self.hand_control and self.hand_position is not None:
    if controller.current_position is not None:
        # Hand gesture control
        if controller.current_position < 0.3:
            player_lane = 0  # Left lane
        elif controller.current_position > 0.7:
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
            if controller.current_position is None:
                if event.key == pygame.K_LEFT:
                    player_lane -= 1
                elif event.key == pygame.K_RIGHT:
                    player_lane += 1
    # Ensure the lane remains within bounds
    return max(0, min(player_lane, 2))


# Main game loop
while running:
    screen.fill(BLACK)
    # Process input
    controller.process_webcam()

    # Main game loop logic
    # --------------------------------
    player_lane = process_motion(controller, player_lane)

    # Update player position
    player_x = player_lane * LANE_WIDTH + PLAYER_START_X

    move_holes()
    draw_holes()

    pygame.draw.rect(screen, BLUE, (player_x, PLAYER_START_Y,
                                    PLAYER_WIDTH, PLAYER_HEIGHT))

    if check_collision():
        running = False

    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

    display_control_mode(controller, screen, WIDTH, HEIGHT)
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
