from __future__ import annotations

from agent.greedy_agent import GreedyAgent
from engine.game_engine import GameEngine
from game_state import GameState
from ui.game_ui import GameUI


def main():
    state = GameState(30)
    agent = GreedyAgent()
    engine = GameEngine(state, agent)
    GameUI(engine).run()


if __name__ == "__main__":
    main()
