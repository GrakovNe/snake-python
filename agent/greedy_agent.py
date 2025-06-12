import random

from agent.agent import Agent
from typing import List, Tuple

from direction import Direction


class GreedyAgent(Agent):
    def get_move(self, grid, snake, food):
        size = len(grid)
        hx, hy = snake[0]
        fx, fy = food
        order: List[Direction] = []
        if fx < hx:
            order.append(Direction.LEFT)
        if fx > hx:
            order.append(Direction.RIGHT)
        if fy < hy:
            order.append(Direction.UP)
        if fy > hy:
            order.append(Direction.DOWN)
        for d in Direction:
            if d not in order:
                order.append(d)
        for d in order:
            nx, ny = hx + d.dx, hy + d.dy
            if not (0 <= nx < size and 0 <= ny < size):
                continue
            if (nx, ny) in snake:
                continue
            return d
        return random.choice(list(Direction))
