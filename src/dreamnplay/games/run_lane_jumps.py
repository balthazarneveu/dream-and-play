import pygame
import random
import sys
from dreamnplay.engine.runnable_game import RunnableGame
from dreamnplay.engine.game import Game


class ThreeLaneGame(Game):
    def __init__(self, width=400, height=600, controller=None):
        super().__init__(controller=controller, width=width, height=height)
        self.LANE_WIDTH = self.WIDTH // 3
        self.HOLE_HEIGHT = 50
        self.SPEED = 5
        self.PLAYER_WIDTH = 30
        self.PLAYER_HEIGHT = 30
        self.PLAYER_START_Y = self.HEIGHT - 80
        self.PLAYER_START_X = self.LANE_WIDTH // 2 - self.PLAYER_WIDTH // 2

        
        pygame.display.set_caption("3-Lane Jumping Game")
        self.font = pygame.font.Font(None, 36)
        self.clock = pygame.time.Clock()

        self.player_lane = 1
        self.player_x = self.PLAYER_START_X
        self.holes = []
        self.score = 0
        self.running = True

    def create_hole(self):
        lane = random.randint(0, 2)
        x = lane * self.LANE_WIDTH
        y = -self.HOLE_HEIGHT
        return pygame.Rect(x, y, self.LANE_WIDTH, self.HOLE_HEIGHT)

    def draw_holes(self):
        for hole in self.holes:
            pygame.draw.rect(self.screen, (255, 255, 255), hole)

    def move_holes(self):
        for hole in self.holes:
            hole.y += self.SPEED
        self.holes = [hole for hole in self.holes if hole.y < self.HEIGHT]
        if len(self.holes) == 0 or self.holes[-1].y > 150:
            self.holes.append(self.create_hole())
        self.score += 1

    def check_collision(self):
        player_rect = pygame.Rect(
            self.player_x, self.PLAYER_START_Y, self.PLAYER_WIDTH, self.PLAYER_HEIGHT
        )
        for hole in self.holes:
            if hole.y + self.HOLE_HEIGHT > self.PLAYER_START_Y and hole.y < self.PLAYER_START_Y + self.PLAYER_HEIGHT:
                if hole.x <= self.player_x <= hole.x + self.LANE_WIDTH:
                    return True
        return False

    def process_motion(self):
        if self.controller.current_position is not None:
            if self.controller.current_position < 0.3:
                self.player_lane = 0
            elif self.controller.current_position > 0.7:
                self.player_lane = 2
            else:
                self.player_lane = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    pygame.quit()
                    sys.exit()
                if self.controller.current_position is None:
                    if event.key == pygame.K_LEFT:
                        self.player_lane -= 1
                    elif event.key == pygame.K_RIGHT:
                        self.player_lane += 1
        self.player_lane = max(0, min(self.player_lane, 2))

    def update(self):
        self.screen.fill((0, 0, 0))
        self.player_x = self.player_lane * self.LANE_WIDTH + self.PLAYER_START_X

        self.move_holes()
        self.draw_holes()

        pygame.draw.rect(
            self.screen,
            (0, 0, 255),
            (self.player_x, self.PLAYER_START_Y,
             self.PLAYER_WIDTH, self.PLAYER_HEIGHT),
        )

        if self.check_collision():
            self.running = False

        score_text = self.font.render(
            f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

    def game_over(self):
        self.running = False

        self.screen.fill((0, 0, 0))
        game_over_text = self.font.render("Game Over!", True, (255, 0, 0))
        final_score_text = self.font.render(
            f"Final Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(
            game_over_text, (self.WIDTH // 2 -
                             game_over_text.get_width() // 2, self.HEIGHT // 3)
        )
        self.screen.blit(
            final_score_text,
            (self.WIDTH // 2 - final_score_text.get_width() // 2, self.HEIGHT // 2),
        )
        pygame.display.flip()
        pygame.time.wait(3000)


if __name__ == "__main__":
    RunnableGame(ThreeLaneGame).run()
