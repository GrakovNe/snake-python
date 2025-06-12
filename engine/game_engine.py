from agent.agent import Agent
from common.game_state import GameState


class GameEngine:
    def __init__(self, state: GameState, agent: Agent):
        self.state = state
        self.agent = agent

    def tick(self):
        if self.state.done:
            return
        d = self.agent.get_move(self.state.grid(), self.state.snake[:], self.state.food)
        self.state.step(d)