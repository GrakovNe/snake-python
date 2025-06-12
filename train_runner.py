#!/usr/bin/env python3
# train_runner.py
# ------------------------------------------------------------
# RL-обучение «Змейки» (без графиков) — полный скрипт
# ------------------------------------------------------------
import os
import random
from collections import deque
from pathlib import Path

import torch
import numpy as np

from agent.dqn_agent import DQNAgent
from engine.game_engine import GameEngine
from common.game_state import GameState
from common.direction import Direction

# ────────────── гиперпараметры ────────────── #
BOARD_SIZE       = 32
EPISODES         = 10_000
MAX_STEPS_EP     = 1_000
SAVE_EVERY       = 200
STUCK_LIMIT      = 200          # прерывать эпизод, если долго нет еды
TRAIN_EVERY      = 1

FOOD_REWARD      = 500.0
STEP_PENALTY     = -0.10
DEATH_PENALTY    = -20.0

EPS_START        = 1.0
EPS_END          = 0.05
EPS_DECAY_STEPS  = 15_000       # шагов до минимального ε
# ------------------------------------------- #

WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)

device = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)
print("Device:", device)

agent = DQNAgent(size=BOARD_SIZE, device=device)
env   = GameEngine(GameState(size=BOARD_SIZE), agent)

scores       = deque(maxlen=100)
epsilon      = EPS_START
global_step  = 0

for ep in range(1, EPISODES + 1):
    grid, snake, food = env.reset()
    state         = agent._encode_state(grid, snake, food)
    total_reward  = 0.0
    steps         = 0
    steps_no_food = 0
    head_pos      = tuple(snake[0])
    dist_before   = abs(head_pos[0] - food[0]) + abs(head_pos[1] - food[1])

    for _ in range(MAX_STEPS_EP):
        # ε-жадное действие
        if random.random() < epsilon:
            move = random.choice(agent.ACTIONS)
        else:
            move = agent.get_move(grid, snake, food)

        # запоминаем координаты еды ДО хода
        prev_food_pos = tuple(food)

        grid, snake, food, done = env.step(move)
        next_state = agent._encode_state(grid, snake, food)

        # вычисление награды
        head_pos    = tuple(snake[0])
        ate_food    = head_pos == prev_food_pos
        dist_after  = abs(head_pos[0] - food[0]) + abs(head_pos[1] - food[1])

        if done:
            reward = DEATH_PENALTY
        elif ate_food:
            reward = FOOD_REWARD
            steps_no_food = 0
        else:
            delta  = dist_before - dist_after
            reward = STEP_PENALTY + 0.5 * delta
            steps_no_food += 1

        dist_before = dist_after

        agent.remember(state, agent.ACTIONS.index(move), reward, next_state, done)
        if agent._steps_done % TRAIN_EVERY == 0:
            agent.train_step()

        state         = next_state
        total_reward += reward
        steps        += 1
        global_step  += 1

        # линейное затухание ε
        epsilon = max(
            EPS_END,
            EPS_START - global_step / EPS_DECAY_STEPS * (EPS_START - EPS_END)
        )

        if done or steps_no_food > STUCK_LIMIT:
            break

    scores.append(total_reward)
    avg_score = float(np.mean(scores))

    print(f"Episode {ep:5}: steps={steps:4} len={len(snake):3} "
          f"score={total_reward:7.2f}  avg={avg_score:7.2f}  ε={epsilon:.3f}")

    if ep % SAVE_EVERY == 0:
        agent.save(WEIGHTS_DIR / f"snake_{ep:05}.pt")
