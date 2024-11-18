import pygame
import sys
from dreamnplay.engine.runnable_game import RunnableGame
from dreamnplay.engine.game import Game


class VerticalPongGame(Game):
    def __init__(self, width=400, height=800, controller=None):
        super().__init__(controller=controller, width=width, height=height)

        # Paddle dimensions
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 20
        self.BALL_SIZE = 20

        self.PADDLE_SPEED = 10
        self.BALL_SPEED_X = 8
        self.BALL_SPEED_Y = 8

        # Player paddle position
        self.player_paddle_x = (self.WIDTH - self.PADDLE_WIDTH) // 2
        self.bot_paddle_x = (self.WIDTH - self.PADDLE_WIDTH) // 2
        self.ball_x = self.WIDTH // 2 - self.BALL_SIZE // 2
        self.ball_y = self.HEIGHT // 2 - self.BALL_SIZE // 2

        self.ball_dir_x = 1  # 1 means right, -1 means left
        self.ball_dir_y = 1  # 1 means down, -1 means up

        self.player_score = 0
        self.bot_score = 0

        pygame.display.set_caption("Vertical Pong Game")
        self.font = pygame.font.Font(None, 36)
        self.clock = pygame.time.Clock()

    def draw_objects(self):
        # Background
        self.screen.fill((0, 0, 0))

        # Draw paddles
        pygame.draw.rect(
            self.screen, (255, 255, 255),
            (self.player_paddle_x, self.HEIGHT - 10 -
             self.PADDLE_HEIGHT, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        )
        pygame.draw.rect(
            self.screen, (255, 255, 255),
            (self.bot_paddle_x, 10, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        )

        # Draw ball
        pygame.draw.rect(
            self.screen, (255, 255, 255),
            (self.ball_x, self.ball_y, self.BALL_SIZE, self.BALL_SIZE)
        )

        # Draw score
        score_text = self.font.render(
            f"Player: {self.player_score}  Bot: {self.bot_score}", True, (
                255, 255, 255)
        )
        self.screen.blit(
            score_text,
            (self.WIDTH // 2 - score_text.get_width() // 2, self.HEIGHT // 2 - 50)
        )

    def move_paddles(self):
        keys = pygame.key.get_pressed()

        # Player paddle movement
        if keys[pygame.K_LEFT]:
            self.player_paddle_x -= self.PADDLE_SPEED
        if keys[pygame.K_RIGHT]:
            self.player_paddle_x += self.PADDLE_SPEED

        # Keep player paddle within bounds
        self.player_paddle_x = max(
            0, min(self.WIDTH - self.PADDLE_WIDTH, self.player_paddle_x))

        # Bot paddle movement (simple AI)
        if self.ball_x < self.bot_paddle_x + self.PADDLE_WIDTH // 2:
            self.bot_paddle_x -= self.PADDLE_SPEED
        elif self.ball_x > self.bot_paddle_x + self.PADDLE_WIDTH // 2:
            self.bot_paddle_x += self.PADDLE_SPEED

        # Keep bot paddle within bounds
        self.bot_paddle_x = max(
            0, min(self.WIDTH - self.PADDLE_WIDTH, self.bot_paddle_x))

    def move_ball(self):
        self.ball_x += self.ball_dir_x * self.BALL_SPEED_X
        self.ball_y += self.ball_dir_y * self.BALL_SPEED_Y

        # Ball collision with side walls
        if self.ball_x <= 0 or self.ball_x + self.BALL_SIZE >= self.WIDTH:
            self.ball_dir_x *= -1

        # Ball collision with paddles
        if (
            self.ball_y + self.BALL_SIZE >= self.HEIGHT - 10 - self.PADDLE_HEIGHT and
            self.player_paddle_x <= self.ball_x <= self.player_paddle_x + self.PADDLE_WIDTH
        ):
            self.ball_dir_y *= -1

        if (
            self.ball_y <= 10 + self.PADDLE_HEIGHT and
            self.bot_paddle_x <= self.ball_x <= self.bot_paddle_x + self.PADDLE_WIDTH
        ):
            self.ball_dir_y *= -1

        # Ball out of bounds
        if self.ball_y <= 0:
            self.player_score += 1
            self.reset_ball()
        if self.ball_y + self.BALL_SIZE >= self.HEIGHT:
            self.bot_score += 1
            self.reset_ball()

    def reset_ball(self):
        self.ball_x = self.WIDTH // 2 - self.BALL_SIZE // 2
        self.ball_y = self.HEIGHT // 2 - self.BALL_SIZE // 2
        self.ball_dir_x *= -1
        self.ball_dir_y *= -1

    def update(self):
        self.move_paddles()
        self.move_ball()
        self.draw_objects()

    def process_motion(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if self.controller and self.controller.current_position is not None:
            # Map controller positions to the player's paddle
            self.player_paddle_x = int(
                self.controller.current_position * (self.WIDTH - self.PADDLE_WIDTH))


if __name__ == "__main__":
    RunnableGame(VerticalPongGame).run()
