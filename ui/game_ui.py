import sys
import pygame
from engine.game_engine import GameEngine


class GameUI:
    WIN_W = 980
    WIN_H = 640
    PANEL = 220
    MARGIN = 24
    BORDER = 8
    FPS = 60

    CLR_BOARD_BORDER = (120, 120, 120)
    CLR_BG          = (255, 255, 255)
    CLR_HEAD        = (220,   0,   0)
    CLR_BODY        = (  0,   0,   0)
    CLR_FOOD        = (215,   0, 215)
    CLR_TXT         = (  0,   0,   0)
    CLR_GRID        = (230, 230, 230)
    CLR_CHART_LINE  = (220,  40,  40)

    def __init__(self, engine: GameEngine):
        pygame.init()
        self.engine = engine
        s = engine.state.size

        board_max_w = self.WIN_W - self.PANEL - self.MARGIN * 3 - self.BORDER * 2
        board_max_h = self.WIN_H - self.MARGIN * 2 - self.BORDER * 2
        self.cell = min(board_max_w // s, board_max_h // s)
        if self.cell < 4:
            raise ValueError("Board too large for fixed window; increase WIN_W/H or decrease size.")

        self.board_w = self.cell * s
        self.board_h = self.cell * s

        outer_board_w = self.board_w + self.BORDER * 2
        outer_board_h = self.board_h + self.BORDER * 2

        self.board_x = self.MARGIN + self.BORDER
        self.board_y = (self.WIN_H - outer_board_h) // 2 + self.BORDER
        self.panel_x = self.MARGIN + outer_board_w + self.MARGIN

        self.screen = pygame.display.set_mode((self.WIN_W, self.WIN_H))
        pygame.display.set_caption("org.grakovne.Snake")
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.SysFont("monospace", 64, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 14)

        self.steps = 0
        self.size_history = [len(self.engine.state.snake)]

    def _draw_board(self):
        pygame.draw.rect(
            self.screen,
            self.CLR_BG,
            pygame.Rect(self.board_x, self.board_y, self.board_w, self.board_h),
        )
        pygame.draw.rect(
            self.screen,
            self.CLR_BOARD_BORDER,
            pygame.Rect(
                self.board_x - self.BORDER,
                self.board_y - self.BORDER,
                self.board_w + self.BORDER * 2,
                self.board_h + self.BORDER * 2,
            ),
            width=self.BORDER,
        )

    def _draw_snake(self):
        for i, (x, y) in enumerate(self.engine.state.snake):
            c = self.CLR_HEAD if i == 0 else self.CLR_BODY
            pygame.draw.rect(
                self.screen,
                c,
                pygame.Rect(
                    self.board_x + x * self.cell,
                    self.board_y + y * self.cell,
                    self.cell,
                    self.cell,
                ),
            )

    def _draw_food(self):
        fx, fy = self.engine.state.food
        pygame.draw.rect(
            self.screen,
            self.CLR_FOOD,
            pygame.Rect(
                self.board_x + fx * self.cell,
                self.board_y + fy * self.cell,
                self.cell,
                self.cell,
            ),
        )

    def _draw_panel(self):
        pygame.draw.rect(
            self.screen,
            self.CLR_BG,
            pygame.Rect(self.panel_x, 0, self.PANEL, self.WIN_H),
        )

        score_txt = f"{self.engine.state.score:08d}"
        txt = self.font_big.render(score_txt, True, self.CLR_TXT)
        self.screen.blit(txt, (self.panel_x + 10, 40))

        footer = self.font_small.render(
            "https://github.com/GrakovNe/snake-python", True, self.CLR_TXT
        )
        self.screen.blit(
            footer,
            footer.get_rect(bottomleft=(self.panel_x, self.WIN_H - 30)),
        )

    def run(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.engine.tick()

            self.steps += 1
            self.size_history.append(len(self.engine.state.snake))

            self.screen.fill(self.CLR_BG)
            self._draw_board()
            self._draw_snake()
            self._draw_food()
            self._draw_panel()
            pygame.display.flip()
            self.clock.tick(self.FPS)
