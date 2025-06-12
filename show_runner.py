from agent.dqn_agent import DQNAgent
from engine.game_engine import GameEngine
from common.game_state import GameState
from ui.game_ui import GameUI


def main() -> None:
    state = GameState(30)
    engine = GameEngine(state, DQNAgent(30))
    ui = GameUI(engine)

    while True:
        engine.tick()
        ui.draw()


if __name__ == "__main__":
    main()
