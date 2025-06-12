import os
import multiprocessing
from pathlib import Path

import torch

from agent.dqn_agent import DQNAgent  # проверьте: файл dqn_agent.py
from engine.game_engine import GameEngine
from common.game_state import GameState
from common.direction import Direction

# -------------------- параметры обучения --------------------
BOARD_SIZE = 30
EPISODES   = 10_000
MAX_STEPS  = 1_000        # тайм‑аут одного эпизода
SAVE_EVERY = 200          # периодичность сохранений

WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)

# -------------------- выбор устройства ----------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Silicon / Metal
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Device:", device)

# -------------------- CPU‑параллелизм -----------------------
num_cores = multiprocessing.cpu_count()
torch.set_num_threads(num_cores)
torch.set_num_interop_threads(num_cores)
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)
print(f"CPU threads set to {num_cores}")

# -------------------- инициализация -------------------------
state_proto = GameState(BOARD_SIZE)
agent       = DQNAgent(size=BOARD_SIZE, device=device)
engine      = GameEngine(state_proto, agent)

# -------------------- цикл эпизодов -------------------------
for episode in range(1, EPISODES + 1):
    grid, snake, food = engine.reset()
    done  = False
    steps = 0
    score = 0.0

    while not done and steps < MAX_STEPS:
        state_t = agent._encode_state(grid, snake, food)
        move    = agent.get_move(grid, snake, food)
        action  = DQNAgent.ACTIONS.index(move)

        grid, snake, food, reward, done = engine.step(move)
        next_state_t = agent._encode_state(grid, snake, food)

        agent.remember(state_t, action, reward, next_state_t, done)
        agent.train_step()

        score += reward
        steps += 1

    print(f"Episode {episode:5d}: steps={steps:4d} len={len(snake):3d}  score={score:6.2f}")

    if episode % SAVE_EVERY == 0:
        w_path = WEIGHTS_DIR / f"snake_{episode:05d}.pt"
        agent.save(w_path)
        print("Saved", w_path)

print("Training finished.")
