from __future__ import annotations

import random
import sys
from enum import Enum
from typing import List, Tuple

import pygame

CELL = 32
GRID = 20
PANEL = 220
MARGIN = 24
FPS = 12
C_A = (52, 52, 64)
C_B = (62, 62, 76)
C_HEAD = (0, 210, 160)
C_BODY = (0, 170, 130)
C_FOOD = (240, 70, 70)
C_BG = (22, 22, 32)
C_TXT = (240, 240, 240)


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    @property
    def dx(self):
        return self.value[0]

    @property
    def dy(self):
        return self.value[1]

    def opposite(self, other: "Direction") -> bool:
        return self.dx == -other.dx and self.dy == -other.dy


class GameState:
    def __init__(self, size: int = GRID):
        self.size = size
        self.reset()

    def reset(self):
        m = self.size // 2
        self.snake: List[Tuple[int, int]] = [(m, m)]
        self.direction = random.choice(list(Direction))
        self.score = 0
        self.done = False
        self._place_food()

    def _place_food(self):
        while True:
            p = (random.randrange(self.size), random.randrange(self.size))
            if p not in self.snake:
                self.food = p
                return

    def step(self, direction: Direction):
        if self.done:
            return
        if direction.opposite(self.direction):
            direction = self.direction
        self.direction = direction
        hx, hy = self.snake[0]
        nx, ny = hx + direction.dx, hy + direction.dy
        if (nx, ny) in self.snake or not (0 <= nx < self.size and 0 <= ny < self.size):
            self.done = True
            return
        self.snake.insert(0, (nx, ny))
        if (nx, ny) == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

    def grid(self):
        g = [[" " for _ in range(self.size)] for _ in range(self.size)]
        for x, y in self.snake:
            g[y][x] = "S"
        fx, fy = self.food
        g[fy][fx] = "F"
        return g


class Agent:
    def get_move(self, grid, snake, food):
        raise NotImplementedError


class GreedyAgent(Agent):
    def get_move(self, grid, snake, food):
        hx, hy = snake[0]
        fx, fy = food
        pref = []
        if fx < hx:
            pref.append(Direction.LEFT)
        if fx > hx:
            pref.append(Direction.RIGHT)
        if fy < hy:
            pref.append(Direction.UP)
        if fy > hy:
            pref.append(Direction.DOWN)
        for d in Direction:
            if d not in pref:
                pref.append(d)
        for d in pref:
            nx, ny = hx + d.dx, hy + d.dy
            if not (0 <= nx < GRID and 0 <= ny < GRID):
                continue
            if (nx, ny) in snake:
                continue
            return d
        return random.choice(list(Direction))


class GameUI:
    def __init__(self):
        pygame.init()
        self.w = GRID * CELL + PANEL + MARGIN * 2
        self.h = GRID * CELL + MARGIN * 2
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake ✦")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 36, bold=True)
        self.game = GameState()
        self.agent: Agent = GreedyAgent()
        self.board_x = MARGIN
        self.board_y = MARGIN
        self.panel_x = self.board_x + GRID * CELL + MARGIN

    def draw_board(self):
        r = pygame.Rect(self.board_x - 4, self.board_y - 4, GRID * CELL + 8, GRID * CELL + 8)
        pygame.draw.rect(self.screen, (40, 40, 54), r, border_radius=6)
        for y in range(GRID):
            for x in range(GRID):
                c = C_A if (x + y) % 2 == 0 else C_B
                pygame.draw.rect(
                    self.screen,
                    c,
                    pygame.Rect(self.board_x + x * CELL, self.board_y + y * CELL, CELL, CELL),
                )

    def draw_snake(self):
        for i, (x, y) in enumerate(self.game.snake):
            c = C_HEAD if i == 0 else C_BODY
            pygame.draw.rect(
                self.screen,
                c,
                pygame.Rect(self.board_x + x * CELL + 2, self.board_y + y * CELL + 2, CELL - 4, CELL - 4),
                border_radius=4,
            )

    def draw_food(self):
        fx, fy = self.game.food
        pygame.draw.rect(
            self.screen,
            C_FOOD,
            pygame.Rect(self.board_x + fx * CELL + 2, self.board_y + fy * CELL + 2, CELL - 4, CELL - 4),
            border_radius=4,
        )

    def draw_panel(self):
        pygame.draw.rect(
            self.screen,
            (45, 45, 60),
            pygame.Rect(self.panel_x, 0, PANEL, self.h),
        )
        txt = self.font.render(f"Score: {self.game.score}", True, C_TXT)
        self.screen.blit(txt, (self.panel_x + 20, 40))
        if self.game.done:
            over = self.font.render("GAME OVER", True, (255, 80, 80))
            r = over.get_rect(center=(self.panel_x + PANEL // 2, 120))
            self.screen.blit(over, r)

    def run(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            if not self.game.done:
                d = self.agent.get_move(self.game.grid(), self.game.snake[:], self.game.food)
                self.game.step(d)
            self.screen.fill(C_BG)
            self.draw_board()
            self.draw_snake()
            self.draw_food()
            self.draw_panel()
            pygame.display.flip()
            self.clock.tick(FPS)


def main():
    GameUI().run()


if __name__ == "__main__":
    main()
