# agent/dqn_agent.py
# ------------------------------------------------------------
# «Крошечный» DQN-агент: ~2.2 k параметров и вход-вектор из 12 признаков
# Совместим c вашим GameEngine: get_move(grid, snake, food) → Direction
# ------------------------------------------------------------
from __future__ import annotations
import math, random
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agent.agent import Agent
from common.direction import Direction

# ―――――――――――――――― небольшая полносвязная сеть ――――――――――――――――
class _TinyNet(nn.Module):
    """Вход 12 → FC-128 → FC-4   (≈ 2 180 обучаемых параметров)"""
    def __init__(self, in_dim: int = 12, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent(Agent):
    ACTIONS: Tuple[Direction, ...] = (
        Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT
    )

    # ────────────────────────── init ─────────────────────────
    def __init__(
        self,
        size: int,
        *,
        device: Optional[torch.device] = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 512,
        memory_capacity: int = 10_000,
        eps_start: float = 1.0,
        eps_final: float = 0.05,
        eps_decay_steps: int = 10_000,
    ) -> None:
        self.size          = size
        self.device        = device or torch.device("cpu")
        self.policy_net    = _TinyNet().to(self.device)
        self.target_net    = _TinyNet().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer     = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory: Deque = deque(maxlen=memory_capacity)

        self.gamma         = gamma
        self.batch_size    = batch_size
        self.sync_every    = 500            # шагов до копирования весов
        self.steps_done    = 0

        self.eps_start, self.eps_final, self.eps_decay = (
            eps_start, eps_final, eps_decay_steps)

    # ────────────────────────── действие ─────────────────────
    def _epsilon(self) -> float:
        t = min(1.0, self.steps_done / self.eps_decay)
        return self.eps_final + (self.eps_start - self.eps_final) * (1 - t)

    def get_move(self, grid, snake: List[Tuple[int,int]], food: Tuple[int,int]):
        state = self._encode_state(grid, snake, food)
        if random.random() < self._epsilon():
            idx = random.randrange(len(self.ACTIONS))
        else:
            with torch.no_grad():
                idx = int(self.policy_net(state).argmax(1).item())
        self.steps_done += 1
        return self.ACTIONS[idx]

    # ─────────────────────── реплей-буфер ────────────────────
    def remember(self, *tr): self.memory.append(tr)

    # ─────────────────────── train_step ──────────────────────
    def train_step(self) -> None:
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s2, d = zip(*batch)
        s  = torch.cat(s)
        s2 = torch.cat(s2)
        a  = torch.tensor(a, device=self.device).unsqueeze(1)
        r  = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        d  = torch.tensor(d, dtype=torch.bool,   device=self.device).unsqueeze(1)

        q = self.policy_net(s).gather(1, a)
        with torch.no_grad():
            q_next = self.target_net(s2).max(1, keepdim=True)[0]
            target = r + (~d)*self.gamma*q_next

        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

        if self.steps_done % self.sync_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # ───────────────────── кодировка состояния ───────────────
    def _encode_state(
        self, grid, snake: List[Tuple[int,int]], food: Tuple[int,int]
    ) -> torch.Tensor:
        """12-мерный вектор признаков → Tensor(1,12)"""
        n = self.size
        head_x, head_y = snake[0]

        # опасность (столкновение) в 4-х направлениях
        def danger(dx:int, dy:int) -> float:
            x, y = head_x+dx, head_y+dy
            return float(
                x < 0 or x >= n or y < 0 or y >= n or (x, y) in snake
            )

        up    = danger(0,-1); down  = danger(0,1)
        left  = danger(-1,0); right = danger(1,0)

        # текущий вектор движения
        if len(snake) > 1:
            neck_x, neck_y = snake[1]
            mov_x, mov_y   = head_x - neck_x, head_y - neck_y
        else:                                 # змейка длиной 1
            mov_x = mov_y = 0
        moving_up    = float(mov_y == -1)
        moving_down  = float(mov_y ==  1)
        moving_left  = float(mov_x == -1)
        moving_right = float(mov_x ==  1)

        # относительное положение еды
        food_up    = float(food[1] < head_y)
        food_down  = float(food[1] > head_y)
        food_left  = float(food[0] < head_x)
        food_right = float(food[0] > head_x)

        state = torch.tensor([
            up, down, left, right,
            moving_up, moving_down, moving_left, moving_right,
            food_up, food_down, food_left, food_right
        ], dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,12)

        return state

    # ───────────────────── сохранение / загрузка ─────────────
    def save(self, path: str | Path): torch.save(self.policy_net.state_dict(), Path(path))
    def load(self, path: str | Path):
        sd = torch.load(Path(path), map_location=self.device)
        self.policy_net.load_state_dict(sd); self.target_net.load_state_dict(sd)
