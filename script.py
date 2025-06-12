"""Snake Game Environment with GUI and Agent interface.

Requirements satisfied:
 - GUI draws an NxN field and score on the right.
 - On every tick the engine calls an Agent object's `get_move` method with the current game state.
 - The Agent returns a direction, after which the engine updates the game logic according to classic Snake rules.
 - A very simple agent (`GreedyAgent`) is included for testing.

Run:
    python snake_env.py

Dependencies:
    pip install pygame
"""
from __future__ import annotations

import random
import sys
from enum import Enum
from typing import List, Tuple

import pygame

# ------------- Config -------------
CELL_SIZE = 24               # pixel size of one square cell
GRID_SIZE = 20               # N — dimensions of the board (NxN)
UI_MARGIN = 150              # width reserved on the right for the score panel (px)
FPS = 10                     # frames per second (game speed)

# ------------- Direction Enum -------------
class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    @property
    def dx(self) -> int:
        return self.value[0]

    @property
    def dy(self) -> int:
        return self.value[1]

    @staticmethod
    def opposite(dir1: 'Direction', dir2: 'Direction') -> bool:
        return dir1.dx == -dir2.dx and dir1.dy == -dir2.dy

# ------------- Agent Interface -------------
class Agent:
    """Базовый класс агента. Наследуйте и переопределяйте `get_move`."""

    def get_move(self, grid: List[List[str]], snake: List[Tuple[int, int]], food: Tuple[int, int]) -> Direction:
        """Возвращает одно из Direction в ответ на текущее состояние игры."""
        raise NotImplementedError

# ------------- Simple test agent -------------
class GreedyAgent(Agent):
    """Простейший агент: двигается к еде по прямой, избегая немедленных столкновений."""

    def get_move(self, grid, snake, food):
        head_x, head_y = snake[0]
        fx, fy = food
        # Список предпочтений по направлению
        prefs = []
        if fx < head_x:
            prefs.append(Direction.LEFT)
        if fx > head_x:
            prefs.append(Direction.RIGHT)
        if fy < head_y:
            prefs.append(Direction.UP)
        if fy > head_y:
            prefs.append(Direction.DOWN)
        # Добавляем остальные направления для случайности
        for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            if d not in prefs:
                prefs.append(d)
        # Выбираем первое безопасное направление
        for d in prefs:
            nx, ny = head_x + d.dx, head_y + d.dy
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                continue
            if (nx, ny) in snake:
                continue
            return d
        # Если нет безопасного хода — сдаёмся (но игра всё равно закончится)
        return random.choice(list(Direction))

# ------------- Game Logic -------------
class SnakeGame:
    EMPTY = 0
    SNAKE = 1
    FOOD = 2

    def __init__(self, size: int = GRID_SIZE):
        self.size = size
        self.reset()

    def reset(self):
        mid = self.size // 2
        self.snake: List[Tuple[int, int]] = [(mid, mid)]  # голова первой
        self.direction = random.choice(list(Direction))
        self.score = 0
        self.done = False
        self._place_food()

    def _place_food(self):
        while True:
            pos = (random.randrange(self.size), random.randrange(self.size))
            if pos not in self.snake:
                self.food = pos
                return

    def step(self, direction: Direction):
        if self.done:
            return
        # не разрешаем мгновенный разворот на 180°
        if Direction.opposite(direction, self.direction):
            direction = self.direction
        self.direction = direction

        head_x, head_y = self.snake[0]
        new_head = (head_x + direction.dx, head_y + direction.dy)

        # Проверяем столкновения
        if (
            not (0 <= new_head[0] < self.size and 0 <= new_head[1] < self.size)
            or new_head in self.snake
        ):
            self.done = True
            return

        # Перемещаем змейку
        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()  # убираем хвост

    # ------- State for Agent -------
    def get_grid(self) -> List[List[str]]:
        grid = [[" " for _ in range(self.size)] for _ in range(self.size)]
        for x, y in self.snake:
            grid[y][x] = "S"
        fx, fy = self.food
        grid[fy][fx] = "F"
        return grid

# ------------- GUI & Main loop -------------

def draw(game: SnakeGame, screen: pygame.Surface):
    screen.fill((30, 30, 30))  # фон
    # поле
    for y in range(game.size):
        for x in range(game.size):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (40, 40, 40), rect, 1)  # сетка
    # змейка
    for i, (x, y) in enumerate(game.snake):
        color = (0, 180, 0) if i == 0 else (0, 120, 0)
        rect = pygame.Rect(x * CELL_SIZE + 1, y * CELL_SIZE + 1, CELL_SIZE - 2, CELL_SIZE - 2)
        pygame.draw.rect(screen, color, rect)
    # еда
    fx, fy = game.food
    rect = pygame.Rect(fx * CELL_SIZE + 1, fy * CELL_SIZE + 1, CELL_SIZE - 2, CELL_SIZE - 2)
    pygame.draw.rect(screen, (200, 0, 0), rect)
    # счёт
    font = pygame.font.SysFont(None, 36)
    txt = font.render(f"Score: {game.score}", True, (250, 250, 250))
    screen.blit(txt, (game.size * CELL_SIZE + 10, 10))


def run():
    pygame.init()
    screen = pygame.display.set_mode((GRID_SIZE * CELL_SIZE + UI_MARGIN, GRID_SIZE * CELL_SIZE))
    pygame.display.set_caption("Snake Environment")
    clock = pygame.time.Clock()

    game = SnakeGame(GRID_SIZE)
    agent: Agent = GreedyAgent()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if not game.done:
            # Получаем направление от агента
            direction = agent.get_move(game.get_grid(), game.snake.copy(), game.food)
            game.step(direction)

        draw(game, screen)
        if game.done:
            # Отображаем сообщение об окончании
            font = pygame.font.SysFont(None, 48)
            msg = font.render("Game Over", True, (255, 50, 50))
            rect = msg.get_rect(center=(GRID_SIZE * CELL_SIZE // 2, GRID_SIZE * CELL_SIZE // 2))
            screen.blit(msg, rect)
        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    run()
