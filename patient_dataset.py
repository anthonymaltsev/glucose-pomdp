from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ActionSpec:
    """Metadata describing a diagnostic action in the VOI framing."""

    id: int
    name: str
    description: str
    cost: float
    expected_information_gain: float


@dataclass
class PatientState:
    """State snapshot summarizing a patient's partially observed glucose status."""

    timestamp: pd.Timestamp
    features: Dict[str, float]
    metadata: Dict[str, float | int | str | bool] = field(default_factory=dict)


@dataclass
class PatientTrajectory:
    """Chronological state rollout for a single ICU stay."""

    subject_id: int
    hadm_id: int
    icustay_id: int
    los_icu_days: float
    first_icu_stay: bool
    states: List[PatientState]


class PatientStateDataset(Iterable[PatientTrajectory]):
    """Builds VOI-ready patient state trajectories from glucose/insulin ICU events."""

    ACTION_SPACE: Tuple[ActionSpec, ...] = (
        ActionSpec(
            id=0,
            name="wait",
            description="No-op; allow latent glucose dynamics to evolve without measurement.",
            cost=0.25,
            expected_information_gain=0.0,
        ),
        ActionSpec(
            id=1,
            name="finger_prick",
            description="Minimal point-of-care glucose measurement (fingerstick).",
            cost=1.0,
            expected_information_gain=0.6,
        ),
        ActionSpec(
            id=2,
            name="lab_test",
            description="Extensive laboratory glucose panel (venous draw).",
            cost=3.0,
            expected_information_gain=1.0,
        ),
    )

    def __init__(
        self,
        csv_path: Path | str,
        *,
        resample_freq: str = "15min",
        history_window: str | None = None,
        intervention_horizon: str = "1h",
        time_in_range: Tuple[int, int] = (70, 180),
    ) -> None:
        self.csv_path = Path(csv_path)
        self.resample_freq = pd.Timedelta(resample_freq)
        self.history_window = (
            None if history_window in (None, "inf", "infinite") else pd.Timedelta(history_window)
        )
        self.intervention_horizon = pd.Timedelta(intervention_horizon)
        self.time_in_range_bounds = time_in_range

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.csv_path}")

        date_columns = ["TIMER", "STARTTIME", "GLCTIMER", "ENDTIME"]
        header = pd.read_csv(self.csv_path, nrows=0)
        parse_dates = [col for col in date_columns if col in header.columns]
        self._events = pd.read_csv(self.csv_path, parse_dates=parse_dates)
        self._normalize_columns()

    def __iter__(self) -> Iterator[PatientTrajectory]:
        for _, stay_df in self._events.groupby(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"], sort=False):
            yield self._build_trajectory(stay_df.copy())

    def get_action_space(self) -> Tuple[ActionSpec, ...]:
        return self.ACTION_SPACE

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _normalize_columns(self) -> None:
        """Ensure consistent dtypes and derived columns."""
        self._events = self._events.sort_values(["ICUSTAY_ID", "TIMER"]).reset_index(drop=True)
        self._events["GLC"] = pd.to_numeric(self._events["GLC"], errors="coerce")
        self._events["INPUT"] = pd.to_numeric(self._events["INPUT"], errors="coerce")
        self._events["INPUT_HRS"] = pd.to_numeric(self._events["INPUT_HRS"], errors="coerce")
        self._events["LOS_ICU_days"] = pd.to_numeric(self._events["LOS_ICU_days"], errors="coerce")
        self._events["first_ICU_stay"] = self._events["first_ICU_stay"].astype(bool)
        def _to_naive_datetime(series: pd.Series) -> pd.Series:
            dt_series = pd.to_datetime(series, errors="coerce", utc=True)
            return dt_series.dt.tz_localize(None)

        datetime_columns = ["TIMER", "STARTTIME", "GLCTIMER", "ENDTIME"]
        for col in datetime_columns:
            if col in self._events.columns:
                self._events[col] = _to_naive_datetime(self._events[col])

    def _build_trajectory(self, stay_df: pd.DataFrame) -> PatientTrajectory:
        stay_df = stay_df.sort_values("TIMER").reset_index(drop=True)
        subject_id = int(stay_df["SUBJECT_ID"].iloc[0])
        hadm_id = int(stay_df["HADM_ID"].iloc[0])
        icustay_id = int(stay_df["ICUSTAY_ID"].iloc[0])
        los_icu_days = float(stay_df["LOS_ICU_days"].iloc[0])
        first_stay = bool(stay_df["first_ICU_stay"].iloc[0])

        timeline = self._build_timeline(stay_df["TIMER"])
        glucose_events = self._extract_glucose_series(stay_df)
        insulin_events = self._extract_insulin_events(stay_df)

        states: List[PatientState] = []
        last_measurement_time: Optional[pd.Timestamp] = None
        last_measurement_value = np.nan
        glucose_idx = 0
        glucose_len = len(glucose_events)
        glucose_times = glucose_events.index

        for ts in timeline:
            if self.history_window is None:
                window_series = glucose_events.loc[: ts]
            else:
                window_start = ts - self.history_window
                window_series = glucose_events.loc[window_start:ts]

            while glucose_idx < glucose_len and glucose_times[glucose_idx] <= ts:
                last_measurement_time = glucose_times[glucose_idx]
                last_measurement_value = float(glucose_events.iloc[glucose_idx])
                glucose_idx += 1

            time_since_last = (
                (ts - last_measurement_time).total_seconds() / 60 if last_measurement_time is not None else np.nan
            )

            features = self._compute_features(window_series, last_measurement_value, time_since_last)
            metadata = self._compute_metadata(insulin_events, ts)
            states.append(PatientState(timestamp=ts, features=features, metadata=metadata))

        return PatientTrajectory(
            subject_id=subject_id,
            hadm_id=hadm_id,
            icustay_id=icustay_id,
            los_icu_days=los_icu_days,
            first_icu_stay=first_stay,
            states=states,
        )

    def _build_timeline(self, timestamps: pd.Series) -> pd.DatetimeIndex:
        start = timestamps.min().floor(self.resample_freq)
        end = timestamps.max().ceil(self.resample_freq)
        if start == end:
            end = start + self.resample_freq
        return pd.date_range(start=start, end=end, freq=self.resample_freq)

    def _extract_glucose_series(self, stay_df: pd.DataFrame) -> pd.Series:
        glc_df = stay_df.dropna(subset=["GLC"]).copy()
        if glc_df.empty:
            return pd.Series(dtype=float)
        timestamps = pd.to_datetime(glc_df["GLCTIMER"].fillna(glc_df["TIMER"]), errors="coerce", utc=True)
        timestamps = timestamps.dt.tz_localize(None)
        series = pd.Series(glc_df["GLC"].values, index=timestamps)
        series = series[~series.index.duplicated(keep="last")]
        return series.sort_index()

    def _extract_insulin_events(self, stay_df: pd.DataFrame) -> pd.DataFrame:
        insulin_mask = stay_df["INPUT"].notna() | stay_df["EVENT"].notna()
        insulin_df = stay_df.loc[insulin_mask, ["TIMER", "INPUT", "INSULINTYPE", "EVENT"]].copy()
        if insulin_df.empty:
            return pd.DataFrame(columns=["INPUT", "INSULINTYPE", "EVENT"])
        insulin_df["TIMER"] = pd.to_datetime(insulin_df["TIMER"], errors="coerce", utc=True).dt.tz_localize(None)
        insulin_df["INPUT"] = insulin_df["INPUT"].fillna(0.0)
        insulin_df["dose_class"] = insulin_df["INSULINTYPE"].fillna("Unknown")
        return insulin_df.set_index("TIMER")

    def _compute_features(
        self,
        window_series: pd.Series,
        last_measurement_value: float,
        time_since_last: float,
    ) -> Dict[str, float]:
        if window_series.empty:
            return {
                "glucose_last": last_measurement_value,
                "glucose_mean": np.nan,
                "glucose_std": np.nan,
                "glucose_min": np.nan,
                "glucose_max": np.nan,
                "glucose_trend_mgdl_per_hr": np.nan,
                "time_in_range_frac": np.nan,
                "time_since_last_measurement_min": time_since_last,
            }

        time_delta_hours = (window_series.index[-1] - window_series.index[0]).total_seconds() / 3600
        glucose_trend = np.nan
        if time_delta_hours > 0:
            glucose_trend = (window_series.iloc[-1] - window_series.iloc[0]) / time_delta_hours

        lower, upper = self.time_in_range_bounds
        time_in_range_frac = float((window_series.between(lower, upper)).mean())

        return {
            "glucose_last": float(window_series.iloc[-1]),
            "glucose_mean": float(window_series.mean()),
            "glucose_std": float(window_series.std(ddof=0)),
            "glucose_min": float(window_series.min()),
            "glucose_max": float(window_series.max()),
            "glucose_trend_mgdl_per_hr": float(glucose_trend),
            "time_in_range_frac": time_in_range_frac,
            "time_since_last_measurement_min": time_since_last,
        }

    def _compute_metadata(
        self, insulin_events: pd.DataFrame, ts: pd.Timestamp
    ) -> Dict[str, float | int | str | bool]:
        if insulin_events.empty:
            return {
                "recent_bolus": False,
                "recent_bolus_minutes": np.nan,
                "recent_dose_total_units": 0.0,
                "recent_bolus_type": "None",
                "recent_event_kind": "None",
            }

        window_start = ts - self.intervention_horizon
        recent_events = insulin_events.loc[window_start:ts]
        if recent_events.empty:
            return {
                "recent_bolus": False,
                "recent_bolus_minutes": np.nan,
                "recent_dose_total_units": 0.0,
                "recent_bolus_type": "None",
                "recent_event_kind": "None",
            }

        last_event_time = recent_events.index.max()
        minutes_since_event = (ts - last_event_time).total_seconds() / 60
        dose_total = float(recent_events["INPUT"].sum())
        last_event = recent_events.iloc[-1]
        return {
            "recent_bolus": True,
            "recent_bolus_minutes": minutes_since_event,
            "recent_dose_total_units": dose_total,
            "recent_bolus_type": str(last_event.get("INSULINTYPE", "Unknown")),
            "recent_event_kind": str(last_event.get("EVENT", "Unknown")),
        }


__all__ = [
    "ActionSpec",
    "PatientState",
    "PatientTrajectory",
    "PatientStateDataset",
]

