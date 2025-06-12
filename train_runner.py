# train_runner.py
#!/usr/bin/env python3
import random, numpy as np, torch
from collections import deque
from pathlib import Path

from agent.dqn_agent import DQNAgent
from engine.game_engine import GameEngine
from common.game_state import GameState

# ───── параметры ─────
BOARD   = 12
EPISODES, MAX_STEPS = 6_000, 500
STEP_PEN, FOOD_REW, DEATH_PEN = -0.01, 1.0, -1.0
STUCK_LIMIT = 100

device = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))
print("Device:", device)

agent = DQNAgent(size=BOARD, device=device)
env   = GameEngine(GameState(BOARD), agent)

scores = deque(maxlen=100)

for ep in range(1, EPISODES+1):
    grid, snake, food = env.reset()
    state, tot, nofood = agent._encode_state(grid, snake, food), 0.0, 0

    for t in range(MAX_STEPS):
        move = agent.get_move(grid, snake, food)
        prev_food = food

        grid, snake, food, done = env.step(move)
        next_state = agent._encode_state(grid, snake, food)

        if done:
            rew = DEATH_PEN
        elif snake[0] == prev_food:
            rew, nofood = FOOD_REW, 0
        else:
            rew, nofood = STEP_PEN, nofood + 1

        agent.remember(state, agent.ACTIONS.index(move), rew, next_state, done)
        agent.steps += 1
        agent.train_step()

        state, tot = next_state, tot + rew
        if done or nofood > STUCK_LIMIT: break

    scores.append(tot)
    print(f"Ep {ep:4} | steps {t+1:3} | len {len(snake):2} | "
          f"score {tot:6.2f} | avg {np.mean(scores):6.2f} | ε={agent._eps():.3f}")

    # curriculum: расширяем поле
    if ep == 800:
        BOARD = 24
        env = GameEngine(GameState(BOARD), agent)
        agent.size = BOARD
        print("→ поле увеличено до 24×24\n")

    if ep % 300 == 0:
        Path("weights").mkdir(exist_ok=True)
        agent.save(f"weights/snake_{BOARD}_{ep:05}.pt")
