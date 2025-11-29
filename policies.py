from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from patient_dataset import PatientState, ActionSpec


class Policy(ABC):
    """Abstract base class for decision policies."""

    @abstractmethod
    def select_action(
        self,
        state: PatientState,
        action_space: Tuple[ActionSpec, ...],
    ) -> ActionSpec:
        """
        Select action given current state.

        Args:
            state: Current patient state
            action_space: Available actions

        Returns:
            Selected action
        """
        pass

    def reset(self) -> None:
        """Reset policy state (for stateful policies)."""
        pass


class GreedyPolicy(Policy):
    """
    Greedy policy: always select wait action (lowest cost).

    This is a worst-case baseline that validates measuring has value.
    """

    def __init__(self, action_space: Tuple[ActionSpec, ...]):
        """
        Initialize greedy policy.

        Args:
            action_space: Available actions
        """
        self.action_lookup = {a.name: a for a in action_space}

    def select_action(
        self,
        state: PatientState,
        action_space: Tuple[ActionSpec, ...],
    ) -> ActionSpec:
        """Always return wait action."""
        return self.action_lookup["wait"]


class ThresholdPolicy(Policy):
    """
    Threshold-based policy: measure when time since last measurement exceeds threshold.

    Mimics simple clinical protocols (e.g., "check glucose every 4 hours").
    """

    def __init__(
        self,
        time_threshold_minutes: float = 240.0,
        measurement_action: str = "finger_prick",
        action_space: Tuple[ActionSpec, ...] = None,
    ):
        """
        Initialize threshold policy.

        Args:
            time_threshold_minutes: Time threshold for measurements (default: 240 = 4 hours)
            measurement_action: Which measurement action to use (default: finger_prick)
            action_space: Available actions (optional, for initialization)
        """
        self.time_threshold = time_threshold_minutes
        self.measurement_action = measurement_action
        self.action_lookup = None
        if action_space is not None:
            self.action_lookup = {a.name: a for a in action_space}

    def select_action(
        self,
        state: PatientState,
        action_space: Tuple[ActionSpec, ...],
    ) -> ActionSpec:
        """Select action based on time threshold."""
        if self.action_lookup is None:
            self.action_lookup = {a.name: a for a in action_space}

        time_since_last = state.features[7].item()  # time_since_last_measurement_min

        if np.isnan(time_since_last) or time_since_last > self.time_threshold:
            return self.action_lookup[self.measurement_action]
        else:
            return self.action_lookup["wait"]


class UncertaintyPolicy(Policy):
    """
    Measure when uncertainty (glucose std or time since last measurement) is high.

    Captures basic VOI intuition: gather information when uncertain about patient state.
    """

    def __init__(
        self,
        std_threshold: float = 30.0,
        time_threshold_minutes: float = 180.0,
        measurement_action: str = "finger_prick",
        action_space: Tuple[ActionSpec, ...] = None,
    ):
        """
        Initialize uncertainty policy.

        Args:
            std_threshold: Glucose std threshold (mg/dL) for measurement (default: 30)
            time_threshold_minutes: Time threshold for measurements (default: 180 = 3 hours)
            measurement_action: Which measurement action to use (default: finger_prick)
            action_space: Available actions (optional, for initialization)
        """
        self.std_threshold = std_threshold
        self.time_threshold = time_threshold_minutes
        self.measurement_action = measurement_action
        self.action_lookup = None
        if action_space is not None:
            self.action_lookup = {a.name: a for a in action_space}

    def select_action(
        self,
        state: PatientState,
        action_space: Tuple[ActionSpec, ...],
    ) -> ActionSpec:
        """Select action based on uncertainty (variability or staleness)."""
        if self.action_lookup is None:
            self.action_lookup = {a.name: a for a in action_space}

        glucose_std = state.features[2].item()  # glucose_std
        time_since_last = state.features[7].item()  # time_since_last_measurement_min

        # Measure if high variability or long time since measurement
        high_variability = not np.isnan(glucose_std) and glucose_std > self.std_threshold
        stale_measurement = np.isnan(time_since_last) or time_since_last > self.time_threshold

        if high_variability or stale_measurement:
            return self.action_lookup[self.measurement_action]
        else:
            return self.action_lookup["wait"]


class MyopicVOIPolicy(Policy):
    """
    Myopic value-of-information policy with one-step lookahead.

    Uses heuristic to estimate expected value of each action, accounting for
    information gain vs. action cost.
    """

    def __init__(
        self,
        info_gain_weight: float = 1.5,
        action_space: Tuple[ActionSpec, ...] = None,
    ):
        """
        Initialize myopic VOI policy.

        Args:
            info_gain_weight: Weight for information gain heuristic (default: 1.5)
            action_space: Available actions (optional, for initialization)
        """
        self.info_gain_weight = info_gain_weight
        self.action_lookup = None
        if action_space is not None:
            self.action_lookup = {a.name: a for a in action_space}

    def select_action(
        self,
        state: PatientState,
        action_space: Tuple[ActionSpec, ...],
    ) -> ActionSpec:
        """
        Select action with highest expected value.

        Uses heuristic: value = -cost + info_gain_bonus
        """
        if self.action_lookup is None:
            self.action_lookup = {a.name: a for a in action_space}

        best_action = None
        best_value = -np.inf

        for action in action_space:
            value = self._estimate_action_value(state, action)

            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def _estimate_action_value(self, state: PatientState, action: ActionSpec) -> float:
        """
        Heuristic estimate of action value.

        For measurement actions: -cost + info_gain_bonus
        For wait: -cost (small)
        """
        # Base: negative action cost
        value = -action.cost

        # Information gain heuristic (only for measurement actions)
        if action.name != "wait":
            time_since_last = state.features[7].item()  # time_since_last_measurement_min
            glucose_std = state.features[2].item()  # glucose_std

            info_bonus = 0.0

            # Higher value if we haven't measured in a while
            if not np.isnan(time_since_last) and time_since_last > 120:
                # Normalize by 2 hours, scale by weight
                info_bonus += self.info_gain_weight * (time_since_last / 120)

            # Higher value if glucose is variable (uncertain)
            if not np.isnan(glucose_std) and glucose_std > 30:
                # Normalize by threshold, scale by weight
                info_bonus += self.info_gain_weight * 0.5 * (glucose_std / 30)

            value += info_bonus

        return value


__all__ = [
    "Policy",
    "GreedyPolicy",
    "ThresholdPolicy",
    "UncertaintyPolicy",
    "MyopicVOIPolicy",
]
