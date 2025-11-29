from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from patient_dataset import PatientState, ActionSpec


@dataclass
class RewardConfig:
    """Configuration for reward parameters (for future extensibility)."""

    # Action cost weight (multiplier for action.cost)
    action_cost_weight: float = 1.0  # Use raw action costs directly

    # Glycemic penalties
    hypo_penalty: float = -10.0
    hyper_penalty: float = -5.0
    severe_hypo_penalty: float = -20.0
    severe_hyper_penalty: float = -10.0

    # Thresholds
    hypo_threshold: float = 70.0
    hyper_threshold: float = 180.0
    severe_hypo_threshold: float = 54.0
    severe_hyper_threshold: float = 250.0

    # Time-in-range
    tir_hourly_rate: float = 1.0

    # Uncertainty penalty (VOI component)
    use_uncertainty_penalty: bool = True
    uncertainty_time_threshold: float = 120.0  # 2 hours in minutes
    uncertainty_time_coeff: float = -2.0  # Stronger penalty for staleness
    uncertainty_variability_coeff: float = -1.5  # Stronger penalty for variability
    uncertainty_variability_threshold: float = 30.0


class SimpleReward:
    """
    Dense reward function for glucose management with uncertainty penalty.

    Balances:
    1. Action costs (financial/time burden on healthcare system)
    2. Glycemic safety (patient outcomes)
    3. Quality of control (time-in-range)
    4. Uncertainty penalty (value of information)
    """

    def __init__(self, config: RewardConfig = RewardConfig()):
        self.config = config

    def __call__(
        self,
        state: PatientState,
        action: ActionSpec | None,
        next_state: PatientState,
    ) -> float:
        """
        Compute reward for transition.

        Args:
            state: Current patient state
            action: Action taken (or None)
            next_state: Next patient state after action

        Returns:
            Scalar reward value
        """
        reward = 0.0

        # Component 1: Action cost (weighted to be very small)
        if action is not None:
            reward -= action.cost * self.config.action_cost_weight

        # Component 2: Glycemic penalty (when measurement available)
        glucose = next_state.metadata.get("measurement_value", np.nan)
        if not np.isnan(glucose):
            if glucose < self.config.hypo_threshold:
                reward += self.config.hypo_penalty  # -10.0
            elif glucose > self.config.hyper_threshold:
                reward += self.config.hyper_penalty  # -5.0

        # Component 3: Time-in-range reward (dense positive signal)
        time_delta = (next_state.timestamp - state.timestamp).total_seconds() / 3600
        tir = next_state.features[6].item()  # time_in_range_frac feature
        if not np.isnan(tir):
            reward += tir * time_delta * self.config.tir_hourly_rate

        # Component 4: Uncertainty penalty (VOI - encourages information gathering)
        if self.config.use_uncertainty_penalty:
            reward += self._uncertainty_penalty(next_state)

        return reward

    def _uncertainty_penalty(self, state: PatientState) -> float:
        """
        Penalize high uncertainty about patient state.

        Encourages measuring when:
        - Long time since last measurement (>2 hours)
        - High glucose variability (std > 30 mg/dL)

        Args:
            state: Patient state to evaluate

        Returns:
            Negative penalty value (or 0 if no penalty)
        """
        penalty = 0.0

        time_since_last = state.features[7].item()  # time_since_last_measurement_min
        glucose_std = state.features[2].item()  # glucose_std

        # Time penalty: nonlinear after threshold (2 hours)
        if not np.isnan(time_since_last) and time_since_last > self.config.uncertainty_time_threshold:
            excess_hours = (time_since_last - self.config.uncertainty_time_threshold) / 60
            penalty += self.config.uncertainty_time_coeff * (excess_hours ** 1.5)

        # Variability penalty: higher when glucose is unstable
        if not np.isnan(glucose_std) and glucose_std > self.config.uncertainty_variability_threshold:
            excess_variability = glucose_std / self.config.uncertainty_variability_threshold - 1
            penalty += self.config.uncertainty_variability_coeff * excess_variability

        return penalty


__all__ = ["RewardConfig", "SimpleReward"]
