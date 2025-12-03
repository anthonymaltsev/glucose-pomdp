from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

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
    uncertainty_variability_coeff: float = -1.0  # Stronger penalty for variability
    uncertainty_variability_threshold: float = 50.0

    # Treatment/intervention monitoring penalty
    treatment_no_measure_penalty: float = -20.0
    treatment_window_minutes: float = 120.0


class SimpleReward:
    """
    Dense reward function for glucose management with uncertainty penalty.

    Balances:
    1. Action costs (financial/time burden on healthcare system)
    2. Glycemic safety (patient outcomes)
    3. Quality of control (time-in-range)
    4. Uncertainty penalty (value of information)
    5. Treatment monitoring penalty (missing measurements around interventions)
    """

    def __init__(self, config: RewardConfig = RewardConfig()):
        self.config = config
        # Track which intervention events have already incurred a penalty
        self._last_penalized_event_time: pd.Timestamp | None = None
        # Track last measurement time under the evaluated policy (not historical)
        self._last_measurement_time_policy: pd.Timestamp | None = None

    def reset(self) -> None:
        """Reset any episode-specific bookkeeping."""
        self._last_penalized_event_time = None
        self._last_measurement_time_policy = None

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

        # Update policy-dependent bookkeeping: record when this policy measures.
        if action is not None and action.name != "wait":
            # Treat the measurement as occurring at the current decision time.
            self._last_measurement_time_policy = state.timestamp

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

        # Component 5: Treatment/intervention penalty
        reward += self._treatment_penalty(state, action, next_state)

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

    def _treatment_penalty(
        self,
        state: PatientState,
        action: ActionSpec | None,
        next_state: PatientState,
    ) -> float:
        """
        Penalize not measuring BEFORE insulin/treatment interventions (prospective check).

        Logic:
        - When an intervention occurs (detected via recent_bolus in metadata)
        - Check if there was a measurement within `treatment_window_minutes` BEFORE the intervention
        - If no measurement before intervention, apply penalty
        - Otherwise, no penalty (0 reward)

        This encourages measuring before interventions to ensure safe treatment decisions.
        """
        metadata = next_state.metadata
        recent_bolus = bool(metadata.get("recent_bolus", False))
        if not recent_bolus:
            return 0.0

        minutes_since = metadata.get("recent_bolus_minutes", np.nan)
        if np.isnan(minutes_since):
            return 0.0
        try:
            intervention_time = next_state.timestamp - pd.Timedelta(minutes=float(minutes_since))
        except Exception:
            return 0.0

        if self._last_penalized_event_time is not None:
            if intervention_time == self._last_penalized_event_time:
                return 0.0

        # Make the penalty depend on the *policy's* measurement history, not the
        # historical clinician measurements baked into the features.
        if self._last_measurement_time_policy is None:
            time_since_last_policy = np.inf
        else:
            time_since_last_policy = (
                (intervention_time - self._last_measurement_time_policy).total_seconds() / 60.0
            )

        # If the policy has not measured within the configured window before
        # the intervention, apply the penalty once for this event.
        if time_since_last_policy > self.config.treatment_window_minutes:
            self._last_penalized_event_time = intervention_time
            return self.config.treatment_no_measure_penalty

        return 0.0


__all__ = ["RewardConfig", "SimpleReward"]
