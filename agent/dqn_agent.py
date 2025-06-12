from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agent.agent import Agent
from common.direction import Direction


class _ConvNet(nn.Module):
    """A very small CNN that maps an RGB‑like board tensor to Q‑values."""

    def __init__(self, board_size: int, n_actions: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * board_size * board_size, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQNAgent(Agent):
    """Deep‑Q‑Learning agent for the classic Snake game.

    The agent is *fully self‑contained*: it can be used in inference‑only
    mode by simply instantiating it and calling :py:meth:`get_move`, or it
    can be trained online/offline through the provided helper methods
    :py:meth:`remember` and :py:meth:`train_step`.

    **Usage** (inference‑only):
    ```python
    agent = DQNAgent(size=12, model_path="weights.pt")
    move = agent.get_move(grid, snake, food)
    ```

    **Usage** (training loop – pseudocode):
    ```python
    agent = DQNAgent(size=12)
    for episode in range(N_EPISODES):
        state = engine.reset()
        done = False
        while not done:
            move = agent.get_move(*state)
            next_state, reward, done = engine.step(move)
            agent.remember(state, move, reward, next_state, done)
            agent.train_step()
            state = next_state
    agent.save("weights.pt")
    ```
    """

    # Mapping of index → Direction expected by the environment
    ACTIONS: Tuple[Direction, ...] = (
        Direction.UP,
        Direction.DOWN,
        Direction.LEFT,
        Direction.RIGHT,
    )

    def __init__(
        self,
        size: int,
        *,
        device: Optional[torch.device] = None,
        model_path: Optional[str | Path] = None,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.05,
        epsilon_decay: int = 10_000,
        gamma: float = 0.99,
        lr: float = 1e-3,
        memory_capacity: int = 50_000,
        batch_size: int = 128,
    ) -> None:
        self.size = size
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.policy_net = _ConvNet(size).to(self.device)
        self.target_net = _ConvNet(size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory: Deque[Tuple[torch.Tensor, int, float, torch.Tensor, bool]] = deque(maxlen=memory_capacity)

        # ε‑greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self._steps_done = 0

        # RL parameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.sync_every = 1_000  # steps after which to sync target_net

        if model_path:
            self.load(model_path)

    # ---------------------------------------------------------------------
    # Public API required by the game engine
    # ---------------------------------------------------------------------

    def get_move(self, grid, snake: List[Tuple[int, int]], food: Tuple[int, int]):
        """Return the next move direction according to ε‑greedy policy."""
        state = self._encode_state(grid, snake, food)

        # ε‑greedy action selection
        eps_threshold = self._current_epsilon()
        self._steps_done += 1
        if random.random() < eps_threshold:
            action_idx = random.randrange(len(self.ACTIONS))
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
                action_idx = int(q_values.argmax(dim=1).item())

        return self.ACTIONS[action_idx]

    # ------------------------------------------------------------------
    # Training utilities (optional)
    # ------------------------------------------------------------------

    def remember(
        self,
        state: torch.Tensor,
        action_idx: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.memory.append((state, action_idx, reward, next_state, done))

    def train_step(self) -> None:
        if len(self.memory) < self.batch_size:
            return  # not enough samples yet

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.cat(states).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        actions = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device).unsqueeze(1)

        # Q(s, a)
        q_values = self.policy_net(states).gather(1, actions)

        # max_a' Q_target(s', a')
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            targets = rewards + (~dones) * self.gamma * max_next_q_values

        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update target network
        if self._steps_done % self.sync_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str | Path) -> None:
        torch.save(self.policy_net.state_dict(), Path(path))

    def load(self, path: str | Path) -> None:
        state_dict = torch.load(Path(path), map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_epsilon(self) -> float:
        # Linear decay
        return max(
            self.epsilon_final,
            self.epsilon - (self.epsilon - self.epsilon_final) * self._steps_done / self.epsilon_decay,
        )

    def _encode_state(
        self, grid, snake: List[Tuple[int, int]], food: Tuple[int, int]
    ) -> torch.Tensor:
        """Convert the environment state into a 3‑channel (H, W) tensor:

        Channel 0 – snake head (1.0 at head position, 0 elsewhere)
        Channel 1 – snake body (1.0 on body, 0 elsewhere)
        Channel 2 – food (1.0 at food position, 0 elsewhere)
        """
        size = self.size
        head = snake[0]
        state = torch.zeros((1, 3, size, size), dtype=torch.float32, device=self.device)

        # Head
        state[0, 0, head[1], head[0]] = 1.0
        # Body
        for x, y in snake[1:]:
            state[0, 1, y, x] = 1.0
        # Food
        fx, fy = food
        state[0, 2, fy, fx] = 1.0

        return state
