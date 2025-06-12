from agent.agent import Agent
from common.game_state import GameState


class GameEngine:
    def __init__(self, state: GameState, agent: Agent):
        self.state = state
        self.agent = agent

    def tick(self) -> None:
        if self.state.done:
            return
        direction = self.agent.get_move(
            self.state.grid(), self.state.snake[:], self.state.food
        )
        self.state.step(direction)
