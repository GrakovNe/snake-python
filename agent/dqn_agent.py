#!/usr/bin/env python3
# agent/dqn_agent.py
from __future__ import annotations

import math, random
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agent.agent import Agent                # ваш базовый класс
from common.direction import Direction       # перечисление направлений


# ─────────────────── сверх-маленькая сеть (≈ 15 k параметров) ──────────────────
class _Net(nn.Module):
    """
    Fully-conv сеть, независимая от размера поля.
    Вход: 4×N×N   →   Выход: Q-значения для 4 действий.
    """
    def __init__(self, n_actions: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(),      #  592 параметра
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),     # 4 640
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),     # 9 248
        )
        self.head = nn.Sequential(
            nn.Conv2d(32, n_actions, 1),                    #   132
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# ─────────────────────────────── агент DQN ─────────────────────────────────────
class DQNAgent(Agent):
    ACTIONS: Tuple[Direction, ...] = (
        Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT
    )

    def __init__(
        self,
        size: int,
        *,
        device: Optional[torch.device] = None,
        lr: float = 1e-3,
        gamma: float = .99,
        batch_size: int = 64,
        memory_capacity: int = 10_000,
        eps_start: float = 1.0,
        eps_final: float = 0.05,
        eps_decay_steps: int = 5_000,
    ):
        self.size   = size
        self.device = device or torch.device("cpu")

        self.policy = _Net().to(self.device)
        self.target = _Net().to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.opt        = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory: Deque[
            Tuple[torch.Tensor, int, float, torch.Tensor, bool]
        ] = deque(maxlen=memory_capacity)

        self.gamma       = gamma
        self.batch       = batch_size
        self.sync_every  = 250
        self.steps       = 0

        self.eps_start, self.eps_final, self.eps_decay = \
            eps_start, eps_final, eps_decay_steps

    #  ε-жадная политика ---------------------------------------------------------
    def _eps(self) -> float:
        t = min(1.0, self.steps / self.eps_decay)
        return self.eps_final + (self.eps_start - self.eps_final) * (1.0 - t)

    def get_move(self, grid, snake, food):
        state = self._encode_state(grid, snake, food)
        self.steps += 1

        if random.random() < self._eps():
            idx = random.randrange(len(self.ACTIONS))
        else:
            with torch.no_grad():
                idx = int(self.policy(state).argmax(1).item())
        return self.ACTIONS[idx]

    #  переход в буфер -----------------------------------------------------------
    def remember(self, *transition):
        # transition = (state, action_idx, reward, next_state, done)
        self.memory.append(transition)

    #  один шаг обучения ---------------------------------------------------------
    def train_step(self):
        if len(self.memory) < self.batch:
            return

        batch = random.sample(self.memory, self.batch)
        s, a, r, s2, d = zip(*batch)

        s  = torch.cat(s)
        s2 = torch.cat(s2)
        a  = torch.tensor(a, device=self.device).unsqueeze(1)
        r  = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        d  = torch.tensor(d, dtype=torch.bool,   device=self.device).unsqueeze(1)

        q      = self.policy(s).gather(1, a)
        with torch.no_grad():
            qmax = self.target(s2).max(1, keepdim=True)[0]
            tgt  = r + (~d) * self.gamma * qmax

        loss = F.smooth_l1_loss(q, tgt)
        self.opt.zero_grad(); loss.backward(); self.opt.step()

        if self.steps % self.sync_every == 0:
            self.target.load_state_dict(self.policy.state_dict())

    #  кодирование сетки в 4-канальный тензор -----------------------------------
    def _encode_state(self, grid, snake: List[Tuple[int, int]], food):
        s = torch.zeros((1, 4, self.size, self.size), device=self.device)

        hx, hy = snake[0]                # голова
        s[0, 0, hy, hx] = 1.0

        for x, y in snake[1:]:           # тело
            s[0, 1, y, x] = 1.0

        fx, fy = food                    # еда
        s[0, 2, fy, fx] = 1.0

        for y, row in enumerate(grid):   # стены / барьеры
            for x, cell in enumerate(row):
                if cell:
                    s[0, 3, y, x] = 1.0
        return s

    #  сохранение / загрузка -----------------------------------------------------
    def save(self, path: str | Path):
        torch.save(self.policy.state_dict(), Path(path))

    def load(self, path: str | Path):
        sd = torch.load(Path(path), map_location=self.device)
        self.policy.load_state_dict(sd)
        self.target.load_state_dict(sd)
