from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from patient_dataset import PatientState, PatientTrajectory, ActionSpec
from rewards import SimpleReward


class GlucoseEnvironment:
    """
    Replay-based simulator for glucose management POMDP.

    Steps through historical patient trajectories, generating observations
    based on actions and computing rewards.
    """

    def __init__(
        self,
        trajectory: PatientTrajectory,
        reward_fn: SimpleReward,
        resample_freq: str = "15min",
    ):
        """
        Initialize environment with a patient trajectory.

        Args:
            trajectory: PatientTrajectory to simulate
            reward_fn: Reward function for computing rewards
            resample_freq: Time step frequency (default: 15min)
        """
        self.trajectory = trajectory
        self.reward_fn = reward_fn
        self.resample_freq = resample_freq

        # Episode state
        self.current_idx = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.done = False

    def reset(self) -> PatientState:
        """
        Reset environment to start of trajectory.

        Returns:
            Initial patient state
        """
        self.current_idx = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.done = False
        return self.trajectory.states[0]

    def step(self, action: ActionSpec) -> Tuple[PatientState, float, bool, Dict[str, Any]]:
        """
        Execute action and transition to next state.

        Args:
            action: Action to take

        Returns:
            next_state: Next patient state
            reward: Immediate reward
            done: Whether episode is complete
            info: Additional metadata dict

        Raises:
            RuntimeError: If episode already finished
        """
        if self.done:
            raise RuntimeError("Episode already finished. Call reset().")

        state = self.trajectory.states[self.current_idx]

        # Get next state (deterministic in replay mode)
        self.current_idx += 1
        if self.current_idx >= len(self.trajectory.states):
            self.done = True
            next_state = state  # Terminal state
        else:
            next_state = self.trajectory.states[self.current_idx]

        # Generate observation based on action
        observation = self._generate_observation(next_state, action)

        # Compute reward
        reward = self.reward_fn(state, action, next_state)

        # Update episode stats
        self.episode_reward += reward
        self.episode_length += 1

        info = {
            "observation": observation,
            "true_glucose": next_state.metadata.get("measurement_value", np.nan),
            "episode_reward": self.episode_reward,
            "episode_length": self.episode_length,
        }

        return next_state, reward, self.done, info

    def _generate_observation(
        self,
        state: PatientState,
        action: ActionSpec,
    ) -> Optional[float]:
        """
        Generate observation based on action type.

        Args:
            state: Current state
            action: Action taken

        Returns:
            Glucose measurement if measurement action, else None
        """
        if action.name == "wait":
            return None

        # For measurement actions, return measurement if available
        measurement = state.metadata.get("measurement_value", None)
        if measurement is not None and not np.isnan(measurement):
            return float(measurement)

        return None

    def get_episode_stats(self) -> Dict[str, float]:
        """
        Get episode statistics.

        Returns:
            Dictionary with total_reward, length, and avg_reward
        """
        return {
            "total_reward": self.episode_reward,
            "length": self.episode_length,
            "avg_reward": self.episode_reward / max(1, self.episode_length),
        }


__all__ = ["GlucoseEnvironment"]
