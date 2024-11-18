import pygame
import sys
import random
from dreamnplay.engine.runnable_game import RunnableGame
from dreamnplay.engine.game import Game


class BrickBreakerGame(Game):
    def __init__(self, width=800, height=600, controller=None):
        super().__init__(controller=controller, width=width, height=height)

        # Paddle properties
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 20
        self.PADDLE_SPEED = 10

        # Ball properties
        self.BALL_SIZE = 20
        self.BALL_SPEED_X = 6
        self.BALL_SPEED_Y = 6

        # Brick properties
        self.BRICK_WIDTH = 75
        self.BRICK_HEIGHT = 30
        self.BRICK_ROWS = 5
        self.BRICK_COLUMNS = self.WIDTH // self.BRICK_WIDTH

        # Paddle position
        self.paddle_x = (self.WIDTH - self.PADDLE_WIDTH) // 2
        self.paddle_y = self.HEIGHT - 40

        # Ball position
        self.ball_x = self.WIDTH // 2 - self.BALL_SIZE // 2
        self.ball_y = self.HEIGHT // 2
        self.ball_dir_x = random.choice([-1, 1])
        self.ball_dir_y = -1

        # Bricks
        self.bricks = self.create_bricks()

        # Score
        self.score = 0

        # Game state
        self.lost_game = False

        pygame.display.set_caption("Brick Breaker")
        self.font = pygame.font.Font(None, 36)

    def create_bricks(self):
        bricks = []
        for row in range(self.BRICK_ROWS):
            for col in range(self.BRICK_COLUMNS):
                brick_x = col * self.BRICK_WIDTH
                brick_y = row * self.BRICK_HEIGHT
                bricks.append(pygame.Rect(brick_x, brick_y, self.BRICK_WIDTH, self.BRICK_HEIGHT))
        return bricks

    def draw_objects(self):
        # Background
        self.screen.fill((0, 0, 0))

        # Draw paddle
        pygame.draw.rect(
            self.screen, (255, 255, 255),
            (self.paddle_x, self.paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        )

        # Draw ball
        pygame.draw.circle(
            self.screen, (255, 255, 255),
            (self.ball_x, self.ball_y), self.BALL_SIZE // 2
        )

        # Draw bricks
        for brick in self.bricks:
            pygame.draw.rect(self.screen, (random.randint(100, 255), random.randint(100, 255), 255), brick)

        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

    def move_paddle(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.paddle_x -= self.PADDLE_SPEED
        if keys[pygame.K_RIGHT]:
            self.paddle_x += self.PADDLE_SPEED

        # Keep paddle within bounds
        self.paddle_x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle_x))

    def move_ball(self):
        self.ball_x += self.ball_dir_x * self.BALL_SPEED_X
        self.ball_y += self.ball_dir_y * self.BALL_SPEED_Y

        # Ball collision with walls
        if self.ball_x <= 0 or self.ball_x + self.BALL_SIZE >= self.WIDTH:
            self.ball_dir_x *= -1
        if self.ball_y <= 0:
            self.ball_dir_y *= -1

        # Ball collision with paddle
        if (
            self.paddle_y <= self.ball_y + self.BALL_SIZE // 2 <= self.paddle_y + self.PADDLE_HEIGHT and
            self.paddle_x <= self.ball_x <= self.paddle_x + self.PADDLE_WIDTH
        ):
            self.ball_dir_y *= -1

        # Ball out of bounds (lose condition)
        if self.ball_y > self.HEIGHT:
            self.lost_game = True

    def check_brick_collision(self):
        ball_rect = pygame.Rect(
            self.ball_x - self.BALL_SIZE // 2, self.ball_y - self.BALL_SIZE // 2, self.BALL_SIZE, self.BALL_SIZE
        )
        for brick in self.bricks:
            if ball_rect.colliderect(brick):
                self.bricks.remove(brick)
                self.ball_dir_y *= -1
                self.score += 10
                break

    def reset_game(self):
        self.ball_x = self.WIDTH // 2 - self.BALL_SIZE // 2
        self.ball_y = self.HEIGHT // 2
        self.ball_dir_x = random.choice([-1, 1])
        self.ball_dir_y = -1
        self.paddle_x = (self.WIDTH - self.PADDLE_WIDTH) // 2

    def update(self):
        self.move_paddle()
        self.move_ball()
        self.check_brick_collision()
        self.draw_objects()

        # Win condition
        if not self.bricks:
            self.display_win_screen()
        if self.lost_game:
            self.game_over()

    def process_motion(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if self.controller and self.controller.current_position is not None:
            # Map controller position to paddle
            self.paddle_x = int(self.controller.current_position * (self.WIDTH - self.PADDLE_WIDTH))

    def display_win_screen(self):
        self.screen.fill((0, 0, 0))
        win_text = self.font.render("You Win!", True, (0, 255, 0))
        final_score_text = self.font.render(f"Final Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(
            win_text, (self.WIDTH // 2 - win_text.get_width() // 2, self.HEIGHT // 3)
        )
        self.screen.blit(
            final_score_text, (self.WIDTH // 2 - final_score_text.get_width() // 2, self.HEIGHT // 2)
        )
        pygame.display.flip()
        pygame.time.wait(3000)
        self.running = False

    def game_over(self):
        self.screen.fill((0, 0, 0))
        game_over_text = self.font.render("Game Over!", True, (255, 0, 0))
        final_score_text = self.font.render(f"Final Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(
            game_over_text, (self.WIDTH // 2 - game_over_text.get_width() // 2, self.HEIGHT // 3)
        )
        self.screen.blit(
            final_score_text, (self.WIDTH // 2 - final_score_text.get_width() // 2, self.HEIGHT // 2)
        )
        pygame.display.flip()
        pygame.time.wait(3000)
        self.running = False
        


if __name__ == "__main__":
    RunnableGame(BrickBreakerGame).run()
