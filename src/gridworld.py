"""Text-based gridworld for Stage-D curiosity visualization."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from .model import build_model
from .tokenizer import SentencePieceTokenizer
from .utils.logging import CSVLogger

Coord = Tuple[int, int]


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass
class GridworldConfig:
    size: int = 5
    max_steps: int = 24
    episodes: int = 8
    policy: str = "greedy"
    log_path: str = "logs/gridworld.csv"
    transcript_dir: str | None = "logs/gridworld_transcripts"


class TextGridworld:
    """Small deterministic gridworld with textual observations."""

    ACTIONS: Sequence[str] = ("NORTH", "SOUTH", "WEST", "EAST", "LOOK")

    def __init__(self, size: int = 5, max_steps: int = 24, seed: int | None = None) -> None:
        self.size = size
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.goal: Coord = (size - 1, size - 1)
        self.agent: Coord = (0, 0)
        self.steps = 0
        self.obstacles: set[Coord] = set()
        self.treasures: set[Coord] = set()
        self.collected: set[Coord] = set()

    def reset(self) -> str:
        self.agent = (0, 0)
        self.steps = 0
        self.collected = set()
        # Sample a couple of static obstacles/treasures away from spawn and goal
        all_cells = [(r, c) for r in range(self.size) for c in range(self.size) if (r, c) not in {(0, 0), self.goal}]
        self.obstacles = set(self.rng.sample(all_cells, k=min(2, len(all_cells))))
        remaining = [cell for cell in all_cells if cell not in self.obstacles]
        self.treasures = set(self.rng.sample(remaining, k=min(2, len(remaining))))
        return self._describe()

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, float]]:
        action = action.upper()
        reward = -0.02  # time penalty
        done = False

        if action == "LOOK":
            reward -= 0.01
        else:
            next_pos = self._move(action)
            if next_pos == self.agent:
                reward -= 0.05  # bumping walls/invalid moves
            elif next_pos in self.obstacles:
                reward -= 0.1
            else:
                self.agent = next_pos

        if self.agent in self.treasures and self.agent not in self.collected:
            self.collected.add(self.agent)
            reward += 0.2

        if self.agent == self.goal:
            reward += 1.0
            done = True

        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        obs = self._describe()
        info = {"distance": float(manhattan(self.agent, self.goal)), "treasures": len(self.collected)}
        return obs, reward, done, info

    def _move(self, action: str) -> Coord:
        r, c = self.agent
        if action == "NORTH" and r > 0:
            return (r - 1, c)
        if action == "SOUTH" and r < self.size - 1:
            return (r + 1, c)
        if action == "WEST" and c > 0:
            return (r, c - 1)
        if action == "EAST" and c < self.size - 1:
            return (r, c + 1)
        return self.agent

    def _describe(self) -> str:
        r, c = self.agent
        dist = manhattan(self.agent, self.goal)
        nearby = []
        for action, (dr, dc) in {
            "NORTH": (-1, 0),
            "SOUTH": (1, 0),
            "WEST": (0, -1),
            "EAST": (0, 1),
        }.items():
            pos = (r + dr, c + dc)
            if pos in self.obstacles:
                nearby.append(f"hazard to the {action.lower()}")
            elif 0 <= pos[0] < self.size and 0 <= pos[1] < self.size:
                nearby.append(f"open {action.lower()}")
        if not nearby:
            nearby.append("quiet all around")
        treasure_hint = "glint nearby" if any(manhattan(self.agent, t) <= 1 for t in self.treasures if t not in self.collected) else "no glints"
        return (
            f"You stand at row {r}, column {c} on a {self.size}x{self.size} map. "
            f"Goal feels {dist} steps away. Nearby: {', '.join(nearby)}. There is {treasure_hint}."
        )

    def sample_action(self) -> str:
        return self.rng.choice(tuple(self.ACTIONS))

    def greedy_action(self) -> str:
        r, c = self.agent
        goal_r, goal_c = self.goal
        if abs(goal_r - r) >= abs(goal_c - c):
            return "SOUTH" if goal_r > r else "NORTH"
        return "EAST" if goal_c > c else "WEST"

    def policy_action(self, policy: str) -> str:
        if policy == "random":
            return self.sample_action()
        if policy == "look":
            return "LOOK"
        return self.greedy_action()


def run_gridworld_stage(cfg: Dict, args) -> None:
    model_cfg = cfg["model"]
    grid_cfg = GridworldConfig(**cfg.get("gridworld", {}))

    env = TextGridworld(size=grid_cfg.size, max_steps=grid_cfg.max_steps, seed=cfg.get("seed"))
    tokenizer = SentencePieceTokenizer()
    model = build_model(model_cfg)
    if getattr(args, "load", None):
        model.load_state_dict(torch.load(args.load, map_location="cpu"), strict=False)
    model.eval()

    log_path = Path(grid_cfg.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(log_path, ["episode", "step", "reward", "intrinsic", "pred_error", "distance", "action"])
    transcript_dir = Path(grid_cfg.transcript_dir) if grid_cfg.transcript_dir else None
    if transcript_dir:
        transcript_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(grid_cfg.episodes):
        obs = env.reset()
        transcript: List[str] = [f"Episode {ep}", obs]
        for step in range(grid_cfg.max_steps):
            action = env.policy_action(grid_cfg.policy)
            transcript.append(f"Action: {action}")
            obs, reward, done, info = env.step(action)
            transcript.append(obs)
            intrinsic, pred_error = _curiosity_from_transcript(
                model, tokenizer, transcript, model_cfg.get("max_seq", 256)
            )
            logger.log(
                {
                    "episode": ep,
                    "step": step,
                    "reward": round(reward, 4),
                    "intrinsic": round(intrinsic, 6),
                    "pred_error": round(pred_error, 6),
                    "distance": info["distance"],
                    "action": action,
                }
            )
            if done:
                break
        if transcript_dir:
            (transcript_dir / f"episode_{ep:03d}.txt").write_text("\n".join(transcript), encoding="utf8")


def _curiosity_from_transcript(
    model,
    tokenizer: SentencePieceTokenizer,
    transcript: List[str],
    max_seq: int,
) -> tuple[float, float]:
    text = "\n".join(transcript)
    tokens = tokenizer.encode(text, add_bos=True, add_eos=False)
    if len(tokens) > max_seq:
        tokens = tokens[-max_seq:]
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        pred_out, _ = model(input_ids, use_memory=False)
    pred_error = float(pred_out.error_trace[:, -1].mean().item())
    bonus = model.pred_head.curiosity_bonus(pred_out.error_trace)
    return bonus, pred_error
