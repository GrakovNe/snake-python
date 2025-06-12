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
    ) -> Tuple[List[List[int]], List[Tuple[int, int]], Tuple[int, int], bool]:
        """Выполняет один ход. Возвращает (grid, snake, food, done)."""
        self.state.step(direction)
        return (
            self.state.grid(),
            self.state.snake[:],
            self.state.food,
            self.state.done,
        )
