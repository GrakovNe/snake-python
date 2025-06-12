import pygame

from engine.game_engine import GameEngine


class GameUI:
    CELL = 32
    PANEL = 220
    MARGIN = 24
    FPS = 12
    CLR_A = (52, 52, 64)
    CLR_B = (62, 62, 76)
    CLR_HEAD = (0, 210, 160)
    CLR_BODY = (0, 170, 130)
    CLR_FOOD = (240, 70, 70)
    CLR_BG = (22, 22, 32)
    CLR_TXT = (240, 240, 240)

    def __init__(self, engine: GameEngine):
        pygame.init()
        self.engine = engine
        s = engine.state.size
        self.w = s * self.CELL + self.PANEL + self.MARGIN * 2
        self.h = s * self.CELL + self.MARGIN * 2
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake ✦")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 36, bold=True)
        self.board_x = self.MARGIN
        self.board_y = self.MARGIN
        self.panel_x = self.board_x + s * self.CELL + self.MARGIN

    def _draw_board(self):
        s = self.engine.state.size
        pygame.draw.rect(
            self.screen,
            (40, 40, 54),
            pygame.Rect(self.board_x - 4, self.board_y - 4, s * self.CELL + 8, s * self.CELL + 8),
            border_radius=6,
        )
        for y in range(s):
            for x in range(s):
                c = self.CLR_A if (x + y) % 2 == 0 else self.CLR_B
                pygame.draw.rect(
                    self.screen,
                    c,
                    pygame.Rect(self.board_x + x * self.CELL, self.board_y + y * self.CELL, self.CELL, self.CELL),
                )

    def _draw_snake(self):
        for i, (x, y) in enumerate(self.engine.state.snake):
            c = self.CLR_HEAD if i == 0 else self.CLR_BODY
            pygame.draw.rect(
                self.screen,
                c,
                pygame.Rect(
                    self.board_x + x * self.CELL + 2,
                    self.board_y + y * self.CELL + 2,
                    self.CELL - 4,
                    self.CELL - 4,
                ),
                border_radius=4,
            )

    def _draw_food(self):
        fx, fy = self.engine.state.food
        pygame.draw.rect(
            self.screen,
            self.CLR_FOOD,
            pygame.Rect(
                self.board_x + fx * self.CELL + 2,
                self.board_y + fy * self.CELL + 2,
                self.CELL - 4,
                self.CELL - 4,
            ),
            border_radius=4,
        )

    def _draw_panel(self):
        pygame.draw.rect(
            self.screen,
            (45, 45, 60),
            pygame.Rect(self.panel_x, 0, self.PANEL, self.h),
        )
        txt = self.font.render(f"Score: {self.engine.state.score}", True, self.CLR_TXT)
        self.screen.blit(txt, (self.panel_x + 20, 40))
        if self.engine.state.done:
            over = self.font.render("GAME OVER", True, (255, 80, 80))
            r = over.get_rect(center=(self.panel_x + self.PANEL // 2, 120))
            self.screen.blit(over, r)

    def run(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.engine.tick()
            self.screen.fill(self.CLR_BG)
            self._draw_board()
            self._draw_snake()
            self._draw_food()
            self._draw_panel()
            pygame.display.flip()
            self.clock.tick(self.FPS)