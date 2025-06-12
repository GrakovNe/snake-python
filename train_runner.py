#!/usr/bin/env python3
# train_runner.py  —  «быстрообучаемая» змейка (500 эпизодов)

import random, os
from collections import deque
from pathlib import Path

import numpy as np
import torch

from agent.dqn_agent import DQNAgent
from engine.game_engine import GameEngine
from common.game_state import GameState
from common.direction import Direction


# ─────────────── гиперпараметры обучения ───────────────
BOARD        = 22
EPISODES     = 500
MAX_STEPS    = 800
STUCK_LIMIT  = 150
SAVE_EVERY   = 100

FOOD_REWARD   = 200.0
STEP_PENALTY  = -0.05
DEATH_PENALTY = -10.0

EPS_START, EPS_END = 1.0, 0.05
EPS_DECAY_STEPS    = 5_000
# ────────────────────────────────────────────────────────

WEIGHTS = Path("weights"); WEIGHTS.mkdir(exist_ok=True)

device = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)
print("Device:", device)

agent = DQNAgent(size=BOARD, device=device,
                 eps_start=EPS_START, eps_final=EPS_END,
                 eps_decay_steps=EPS_DECAY_STEPS)
env   = GameEngine(GameState(size=BOARD), agent)

scores = deque(maxlen=50)

for ep in range(1, EPISODES + 1):
    grid, snake, food = env.reset()
    state        = agent._encode_state(grid, snake, food)
    total_reward = 0.0
    steps_nofood = 0

    for step in range(1, MAX_STEPS + 1):
        move = agent.get_move(grid, snake, food)
        prev_food = tuple(food)

        grid, snake, food, done = env.step(move)
        next_state = agent._encode_state(grid, snake, food)

        # ─── награда ───
        if done:
            reward = DEATH_PENALTY
        elif snake[0] == list(prev_food):
            reward = FOOD_REWARD
            steps_nofood = 0
        else:
            reward = STEP_PENALTY
            steps_nofood += 1

        # запоминаем переход и тренируемся
        agent.remember(state, agent.ACTIONS.index(move),
                       reward, next_state, done)
        agent.train_step()

        state         = next_state
        total_reward += reward

        if done or steps_nofood > STUCK_LIMIT:
            break

    scores.append(total_reward)
    avg = float(np.mean(scores))

    print(f"Ep {ep:3} | steps {step:3} | len {len(snake):2} | "
          f"score {total_reward:7.2f} | avg {avg:7.2f} | ε={agent._eps():.3f}")

    if ep % SAVE_EVERY == 0:
        agent.save(WEIGHTS / f"snake_{BOARD}_{ep:05}.pt")
а