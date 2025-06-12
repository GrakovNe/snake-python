from agent.agent import Agent
from common.game_state import GameState
from typing import Tuple, List


class GameEngine:
    def __init__(self, state: GameState, agent: Agent):
        self.state = state
        self.agent = agent

    # ————————————————————————————————————————————————
    #  Игровой цикл для визуального раннера
    # ————————————————————————————————————————————————
    def tick(self) -> None:
        if self.state.done:
            return
        direction = self.agent.get_move(
            self.state.grid(), self.state.snake[:], self.state.food
        )
        self.state.step(direction)

    # ————————————————————————————————————————————————
    #  Интерфейс, необходимый DQN-раннеру
    # ————————————————————————————————————————————————
    def reset(self) -> Tuple[List[List[int]], List[Tuple[int, int]], Tuple[int, int]]:
        """Начинает новый эпизод и возвращает стартовое состояние."""
        self.state.reset()
        return self.state.grid(), self.state.snake[:], self.state.food

    def step(
            self, direction
    ) -> Tuple[List[List[int]], List[Tuple[int, int]], Tuple[int, int], float, bool]:
        """
        Выполняет один ход.
        Возвращает (grid, snake, food, reward, done).
        """
        prev_len = len(self.state.snake)
        prev_head = self.state.snake[0]
        food_pos = self.state.food

        dist_before = abs(prev_head[0] - food_pos[0]) + abs(prev_head[1] - food_pos[1])

        self.state.step(direction)

        new_head = self.state.snake[0]
        dist_after = abs(new_head[0] - food_pos[0]) + abs(new_head[1] - food_pos[1])

        if self.state.done:
            reward = -5.0
        elif len(self.state.snake) > prev_len:
            reward = 5.0
        else:
            reward = 0

            if dist_after < dist_before:
                reward += 0.1
            elif dist_after > dist_before:
                reward -= 0.1

        return (
            self.state.grid(),
            self.state.snake[:],
            self.state.food,
            reward,
            self.state.done,
        )
