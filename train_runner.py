#!/usr/bin/env python3
# train_runner.py  ─ «быстрое» обучение за ~500 эпизодов
import os, random, numpy as np
from collections import deque
from pathlib import Path
import torch

from agent.dqn_agent import DQNAgent
from engine.game_engine import GameEngine
from common.game_state import GameState

# ───── гиперпараметры ─────
BOARD          = 20
EPISODES       = 500
MAX_STEPS_EP   = 1000
SAVE_EVERY     = 100
# ──────────────────────────

device = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)
print("Device:", device)

agent = DQNAgent(size=BOARD, device=device)
env   = GameEngine(GameState(size=BOARD), agent)

scores = deque(maxlen=100)

for ep in range(1, EPISODES+1):
    grid, snake, food = env.reset()
    state  = agent._encode_state(grid, snake, food)
    total  = 0.0
    steps  = 0

    for _ in range(MAX_STEPS_EP):
        move = agent.get_move(grid, snake, food)
        grid, snake, food, done = env.step(move)
        next_state = agent._encode_state(grid, snake, food)

        # награда (как в dqn_agent)
        head = snake[0]
        reward = (
            10.0  if head == food            else
           -10.0  if done                    else
            -0.1
        )

        agent.remember(state, agent.ACTIONS.index(move), reward, next_state, done)
        agent.train_step()

        state  = next_state
        total += reward
        steps += 1
        if done: break

    scores.append(total)
    avg = float(np.mean(scores))
    print(f"Ep {ep:4} | steps {steps:4} | len {len(snake)} | score {total:7.2f} | avg {avg:7.2f}")

    if ep % SAVE_EVERY == 0:
        Path("weights").mkdir(exist_ok=True)
        agent.save(f"weights/snake_tiny_{ep:05}.pt")

print("✔ Обучение завершено")
