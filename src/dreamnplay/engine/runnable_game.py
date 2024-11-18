import pygame
import sys
from dreamnplay.controller.webcam_controller import Controller
from dreamnplay.view.overlay_controller import display_control_mode


class RunnableGame:
    def __init__(self, game_class):
        self.controller = Controller(webcam_show=False)
        pygame.init()
        self.game = game_class(controller=self.controller)

    def run(self):
        while self.game.running:
            self.controller.process_webcam()
            self.game.process_motion()
            self.game.update()
            display_control_mode(self.game.controller, self.game.screen,
                                 self.game.WIDTH, self.game.HEIGHT)
            pygame.display.flip()
            self.game.clock.tick(30)
        self.controller.release_resources()
        self.game.game_over()
        pygame.quit()
        sys.exit()
