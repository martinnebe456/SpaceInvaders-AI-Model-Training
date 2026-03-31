"""Gymnasium wrapper around the core Space Invaders game logic."""

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from .game import FPS, STATE_VECTOR_SIZE, SpaceInvadersGame


class SpaceInvadersEnv(gym.Env):
    """Expose SpaceInvadersGame through the standard Gymnasium API."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        render_mode=None,
        max_steps=5000,
        seed=None,
        single_level_mode=True,
    ):
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"Unsupported render mode: {render_mode}")

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.single_level_mode = single_level_mode
        self.steps_in_episode = 0
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(STATE_VECTOR_SIZE,),
            dtype=np.float32,
        )
        self.game = SpaceInvadersGame(
            human_mode=render_mode == "human",
            seed=seed,
            single_level_mode=single_level_mode,
        )

    def _get_obs(self):
        """Return the current observation in Gym-friendly float32 format."""
        return np.asarray(self.game.get_state_vector(), dtype=np.float32)

    def _get_info(self):
        """Extend the game's info dict with episode-level wrapper metadata."""
        info = self.game.get_info().copy()
        info["steps_in_episode"] = self.steps_in_episode
        return info

    def reset(self, *, seed=None, options=None):
        """Reset both the wrapper counters and the underlying game instance."""
        super().reset(seed=seed)
        self.steps_in_episode = 0

        # Pass a deterministic integer seed into the game so Gymnasium-level
        # seeding and direct game seeding stay aligned.
        game_seed = None if seed is None else int(seed)
        observation = self.game.reset(seed=game_seed)
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return np.asarray(observation, dtype=np.float32), info

    def step(self, action):
        """Advance the game by one action and translate the result to Gymnasium."""
        observation, reward, terminated, info = self.game.step(int(action))
        self.steps_in_episode += 1

        # The game itself reports terminal win/lose states. The wrapper adds
        # a separate truncation signal when the episode hits the step budget.
        truncated = self.steps_in_episode >= self.max_steps and not terminated
        info = info.copy()
        info["steps_in_episode"] = self.steps_in_episode
        if truncated:
            info["truncated_by_max_steps"] = True

        if self.render_mode == "human":
            self.render()

        return (
            np.asarray(observation, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self):
        """Render either to a window or return an RGB frame array."""
        self.game._draw()

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.game.clock.tick(self.metadata["render_fps"])
            return None

        if self.render_mode == "rgb_array":
            frame = pygame.surfarray.array3d(self.game.screen)
            # Pygame returns width-first arrays; Gym-style image consumers
            # usually expect height-first arrays.
            return np.transpose(frame, (1, 0, 2))

        return None

    def close(self):
        """Release pygame resources owned by the underlying game."""
        self.game.close()
