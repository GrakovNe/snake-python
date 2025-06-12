# agent/dqn_agent.py
from __future__ import annotations
import random, math
from collections import deque
from pathlib import Path
from typing import Deque, Tuple, Optional, List

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from common.direction import Direction
from agent.agent import Agent


class _Net(nn.Module):
    """Простейшая CNN-"алекснет": вход 4×N×N."""
    def __init__(self, board: int, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * board * board, 256), nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x): return self.net(x)


class DQNAgent(Agent):
    ACTIONS: Tuple[Direction, ...] = (
        Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT
    )

    def __init__(
        self,
        size: int,
        *,
        device: Optional[torch.device] = None,
        lr: float = 2e-4,
        gamma: float = .99,
        batch_size: int = 128,
        memory_capacity: int = 30_000,
        eps_start: float = 1.0,
        eps_final: float = 0.05,
        eps_decay_steps: int = 25_000,
    ):
        self.size = size
        self.device = device or torch.device("cpu")
        self.policy = _Net(size).to(self.device)
        self.target = _Net(size).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory: Deque = deque(maxlen=memory_capacity)
        self.batch, self.gamma = batch_size, gamma
        self.sync_every = 1_000
        self.steps = 0

        self.eps_start, self.eps_final, self.eps_decay = eps_start, eps_final, eps_decay_steps

    # ε-жадная стратегия
    def _eps(self):
        t = min(1.0, self.steps / self.eps_decay)
        return self.eps_final + (self.eps_start - self.eps_final) * (1.0 - t)

    def get_move(self, grid, snake, food):
        state = self._encode_state(grid, snake, food)
        if random.random() < self._eps():
            idx = random.randrange(len(self.ACTIONS))
        else:
            with torch.no_grad():
                idx = int(self.policy(state).argmax(1).item())
        return self.ACTIONS[idx]

    # буфер
    def remember(self, *transition): self.memory.append(transition)

    # шаг обучения
    def train_step(self):
        if len(self.memory) < self.batch:  # ждём, пока буфер наполнится
            return
        batch = random.sample(self.memory, self.batch)
        s, a, r, s2, d = zip(*batch)
        s, s2 = torch.cat(s), torch.cat(s2)
        a = torch.tensor(a, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        d = torch.tensor(d, dtype=torch.bool, device=self.device).unsqueeze(1)

        q = self.policy(s).gather(1, a)
        with torch.no_grad():
            q_next = self.target(s2).max(1, True)[0]
            target = r + (~d) * self.gamma * q_next
        loss = F.smooth_l1_loss(q, target)

        self.opt.zero_grad(); loss.backward(); self.opt.step()

        if self.steps % self.sync_every == 0:
            self.target.load_state_dict(self.policy.state_dict())

    # agent/dqn_agent.py  ▸  _encode_state
    def _encode_state(self, grid, snake, food):
        s = torch.zeros((1, 4, self.size, self.size), device=self.device)

        # 0-й канал — голова
        hx, hy = snake[0]
        s[0, 0, hy, hx] = 1.0

        # 1-й канал — тело
        for x, y in snake[1:]:
            s[0, 1, y, x] = 1.0

        # 2-й канал — еда
        fx, fy = food
        s[0, 2, fy, fx] = 1.0

        # 3-й канал — стены/барьеры
        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                if cell:  # всё, что не «пусто» → стена
                    s[0, 3, y, x] = 1.0

        return s
