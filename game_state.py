
from typing import List, Tuple
import random
from direction import Direction

class GameState:
    def __init__(self, size: int = 20):
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

