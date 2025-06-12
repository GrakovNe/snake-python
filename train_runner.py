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

# ————————————————————————————————————————————————
# Настройки обучения
# ————————————————————————————————————————————————
BOARD_SIZE     = 32
EPISODES       = 10_000
MAX_STEPS_EP   = 1000
NUM_ENVS       = min(8, os.cpu_count() or 1)
SAVE_EVERY     = 200
STUCK_LIMIT    = 200    # прерывать эпизод, если змея не ест слишком долго
TRAIN_EVERY    = 1

from collections import deque
from pathlib import Path

import torch
import numpy as np

from agent.dqn_agent import DQNAgent
from engine.game_engine import GameEngine
from common.game_state import GameState
from common.direction import Direction

# ————————————————————————————————————————————————
# Настройки обучения
# ————————————————————————————————————————————————
BOARD_SIZE     = 32
EPISODES       = 10_000
MAX_STEPS_EP   = 1000
SAVE_EVERY     = 200
STUCK_LIMIT    = 200
TRAIN_EVERY    = 1

WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)

# ————————————————————————————————————————————————
# Устройство
# ————————————————————————————————————————————————
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print("Device:", device)

# ————————————————————————————————————————————————
# Инициализация
# ————————————————————————————————————————————————
agent = DQNAgent(size=BOARD_SIZE, device=device)
env = GameEngine(GameState(size=BOARD_SIZE), agent)

# ————————————————————————————————————————————————
# Цикл обучения
# ————————————————————————————————————————————————
scores = deque(maxlen=100)

for ep in range(1, EPISODES + 1):
    grid, snake, food = env.reset()
    state = agent._encode_state(grid, snake, food)
    total_reward = 0.0
    steps = 0
    since_last_food = 0
    prev_len = len(snake)
    prev_head = snake[0]
    food_pos = food
    dist_before = abs(prev_head[0] - food_pos[0]) + abs(prev_head[1] - food_pos[1])

    for _ in range(MAX_STEPS_EP):
        move = agent.get_move(grid, snake, food)
        grid, snake, food, done = env.step(move)
        next_state = agent._encode_state(grid, snake, food)

        # ——— Расчёт награды ———
        dist_after = abs(snake[0][0] - food[0]) + abs(snake[0][1] - food[1])

        if done:
            reward = -10.0
        elif len(snake) > prev_len:
            reward = 20.0
        else:
            reward = -0.01
            if dist_after < dist_before:
                reward += 0.2
            elif dist_after > dist_before:
                reward -= 0.2

        dist_before = dist_after

        agent.remember(state, agent.ACTIONS.index(move), reward, next_state, done)

        if agent._steps_done % TRAIN_EVERY == 0:
            agent.train_step()

        state = next_state
        total_reward += reward
        steps += 1

        # Проверка застоя
        if len(snake) > prev_len:
            since_last_food = 0
            prev_len = len(snake)
        else:
            since_last_food += 1

        if done or since_last_food > STUCK_LIMIT:
            break

    scores.append(total_reward)
    avg_score = np.mean(scores)

    print(f"Episode {ep:5}: steps={steps:4} len={len(snake):3} score={total_reward:6.2f}  avg={avg_score:6.2f}")

    if ep % SAVE_EVERY == 0:
        agent.save(WEIGHTS_DIR / f"snake_{ep:05}.pt")


WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)

# ————————————————————————————————————————————————
# Устройство
# ————————————————————————————————————————————————
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print("Device:", device)

# ————————————————————————————————————————————————
# Инициализация
# ————————————————————————————————————————————————
agent = DQNAgent(size=BOARD_SIZE, device=device)
env = GameEngine(GameState(size=BOARD_SIZE), agent)

# ————————————————————————————————————————————————
# Цикл обучения
# ————————————————————————————————————————————————
scores = deque(maxlen=100)

for ep in range(1, EPISODES + 1):
    grid, snake, food = env.reset()
    state = agent._encode_state(grid, snake, food)
    total_reward = 0.0
    steps = 0
    since_last_food = 0
    prev_len = len(snake)

    for _ in range(MAX_STEPS_EP):
        move = agent.get_move(grid, snake, food)
        grid, snake, food, reward, done = env.step(move)
        next_state = agent._encode_state(grid, snake, food)

        agent.remember(state, agent.ACTIONS.index(move), reward, next_state, done)

        if agent._steps_done % TRAIN_EVERY == 0:
            agent.train_step()

        state = next_state
        total_reward += reward
        steps += 1

        # Проверка застоя
        if len(snake) > prev_len:
            since_last_food = 0
            prev_len = len(snake)
        else:
            since_last_food += 1

        if done or since_last_food > STUCK_LIMIT:
            break

    scores.append(total_reward)
    avg_score = np.mean(scores)

    print(f"Episode {ep:5}: steps={steps:4} len={len(snake):3} score={total_reward:6.2f}  avg={avg_score:6.2f}")

    if ep % SAVE_EVERY == 0:
        agent.save(WEIGHTS_DIR / f"snake_{ep:05}.pt")
