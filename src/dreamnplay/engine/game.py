import pygame


class Game:
    def __init__(self, *args, width: int = None, height: int = None,  controller=None, **kwargs):
        self.WIDTH = width
        self.HEIGHT = height
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.controller = controller
        self.running = True

    def process_motion(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def game_over(self):
        raise NotImplementedError
