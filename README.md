# Glucose Management POMDP

A research implementation of a Partially Observable Markov Decision Process (POMDP) for sequential decision-making in glucose monitoring for hospitalized ICU patients. This project frames glucose management as a value-of-information (VOI) problem, where the goal is to determine when to perform diagnostic tests to maximize expected clinical utility under uncertainty.

## Overview

This project uses real patient data from the MIMIC-IV glucose management dataset to study when clinicians should perform glucose measurements (finger prick vs. lab test) versus waiting to observe glucose dynamics. The framework supports:

- **State representation**: 8D feature vector capturing glucose statistics, trends, time-in-range, and time since last measurement
- **Action space**: 3 actions (wait, finger prick, lab test) with different costs and information gains
- **Reward function**: 4-component dense reward balancing glycemic safety, action costs, time-in-range, and uncertainty
- **Policy evaluation**: Compare different measurement strategies on real ICU patient trajectories

## Project Structure

```
glucose-pomdp/
├── patient_dataset.py      # Dataset processing from MIMIC-IV glucose data
├── rewards.py              # 4-component reward function with VOI
├── simulator.py            # Replay-based environment for policy evaluation
├── policies.py             # Baseline policies (Greedy, Threshold, Uncertainty, Myopic VOI)
├── gm.py                   # Demo script showing complete infrastructure
├── requirements.txt        # Python dependencies
└── data/                   # MIMIC-IV glucose management dataset (not in repo)
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download MIMIC-IV glucose management dataset from PhysioNet
# Unzip into data/ directory
# https://physionet.org/content/glucose-management-mimic/1.0.1/
```

## Quick Start

```python
from pathlib import Path
from patient_dataset import PatientStateDataset
from rewards import SimpleReward
from simulator import GlucoseEnvironment
from policies import ThresholdPolicy

# Setup
data_dir = Path("data/physionet.org/files/glucose-management-mimic/1.0.1/Datasets")
reward_fn = SimpleReward()

# Load dataset with reward function
dataset = PatientStateDataset(data_dir / "glucose_insulin_ICU.csv", reward_fn=reward_fn)

# Get a trajectory
trajectory = next(dataset.iter_trajectories())

# Create simulator
env = GlucoseEnvironment(trajectory, reward_fn)

# Run a policy
policy = ThresholdPolicy(time_threshold_minutes=240)  # Measure every 4 hours
state = env.reset()

while not env.done:
    action = policy.select_action(state, dataset.get_action_space())
    next_state, reward, done, info = env.step(action)
    state = next_state

print(f"Total reward: {env.episode_reward:.2f}")
```

Or simply run the demo:

```bash
python3 gm.py
```

## State Representation

Each patient state includes an 8-dimensional feature vector:

1. **glucose_last**: Most recent glucose measurement (mg/dL)
2. **glucose_mean**: Mean glucose over history window
3. **glucose_std**: Standard deviation of glucose
4. **glucose_min**: Minimum glucose value
5. **glucose_max**: Maximum glucose value
6. **glucose_trend_mgdl_per_hr**: Rate of change (mg/dL per hour)
7. **time_in_range_frac**: Fraction of time in target range (70-180 mg/dL)
8. **time_since_last_measurement_min**: Time since last observation (minutes)

Plus metadata:
- Recent insulin interventions (bolus timing, dose, type)
- Measurement source (when available): FINGERSTICK, BLOOD, LAB
- Actual glucose value (when measured)

## Action Space

Three actions with different costs and information gains:

| Action | Cost | Info Gain | Description |
|--------|------|-----------|-------------|
| **wait** | 0.25 | 0.0 | No-op; allow glucose dynamics to evolve |
| **finger_prick** | 1.0 | 0.6 | Point-of-care fingerstick measurement |
| **lab_test** | 3.0 | 1.0 | Laboratory venous draw (more accurate) |

## Reward Function

The reward function uses 4 components to balance competing objectives:

### 1. Action Costs
- Reflects financial/time burden on healthcare system
- Encourages efficient resource use
- Uses raw action costs directly (weight = 1.0)
- `reward -= action.cost * 1.0`

### 2. Glycemic Event Penalties (sparse)
- **Hypoglycemia (<70 mg/dL)**: -10.0
- **Hyperglycemia (>180 mg/dL)**: -5.0
- Hypoglycemia penalized ~2x more (clinically more dangerous)

### 3. Time-in-Range Reward (dense)
- Continuous positive signal for maintaining glucose in 70-180 mg/dL
- `reward += tir_frac * time_delta_hours * 1.0`
- For 15-min steps: +0.25 if fully in range

### 4. Uncertainty Penalty (VOI component)
- Penalizes long gaps without measurements (>2 hours)
- Penalizes high glucose variability (std > 30 mg/dL)
- Superlinear penalty after threshold
- Encourages information gathering when uncertain

**Configurable via RewardConfig:**
```python
from rewards import SimpleReward, RewardConfig

config = RewardConfig(
    action_cost_weight=1.0,  # Use raw action costs directly
    hypo_penalty=-10.0,
    hyper_penalty=-5.0,
    tir_hourly_rate=1.0,
    uncertainty_time_coeff=-2.0,
    uncertainty_variability_coeff=-1.5,
)
reward_fn = SimpleReward(config)
```

**Interpretable Ratios:**
With `action_cost_weight=1.0`, the reward components have natural interpretable relationships:
- 1 hypoglycemic event (penalty -10.0) = cost of ~10 finger pricks (1.0 each)
- 1 hour in target range (reward +1.0) = 1 finger prick cost
- 1 lab test (cost 3.0) = 3 finger pricks

## Baseline Policies

### GreedyPolicy
Always wait (never measure). Worst-case baseline that validates measuring has value.

### ThresholdPolicy
Measure when time since last measurement exceeds threshold. Mimics simple clinical protocols.

```python
policy = ThresholdPolicy(time_threshold_minutes=240)  # Every 4 hours
```

### UncertaintyPolicy
Measure when glucose variability is high OR time since last measurement exceeds threshold. Captures basic VOI intuition.

```python
policy = UncertaintyPolicy(
    std_threshold=30.0,           # mg/dL
    time_threshold_minutes=180.0, # 3 hours
)
```

### MyopicVOIPolicy
One-step lookahead with information gain heuristic. More sophisticated VOI reasoning.

```python
policy = MyopicVOIPolicy(info_gain_weight=1.5)
```

## Implementing Custom Policies

Extend the `Policy` base class:

```python
from policies import Policy
from patient_dataset import PatientState, ActionSpec

class MyCustomPolicy(Policy):
    def select_action(self, state: PatientState, action_space) -> ActionSpec:
        glucose_last = state.features[0].item()
        time_since_last = state.features[7].item()

        action_lookup = {a.name: a for a in action_space}

        # Your custom logic here
        if glucose_last > 200 and time_since_last > 60:
            return action_lookup["lab_test"]
        elif time_since_last > 180:
            return action_lookup["finger_prick"]
        else:
            return action_lookup["wait"]
```

## Dataset

Uses **Curated Data for Describing Blood Glucose Management in the ICU** from PhysioNet (derived from MIMIC-IV):

- **Patients**: 9,517 unique patients
- **ICU stays**: 11,724 ICU admissions
- **Events**: 603,764 glucose/insulin events
- **Source**: https://physionet.org/content/glucose-management-mimic/

The dataset includes:
- Glucose measurements with timestamps and sources (FINGERSTICK, BLOOD, LAB)
- Insulin interventions (dose, type, timing)
- ICU stay metadata (length of stay, first stay indicator)

## Research Goals

This project supports research on:

1. **Value of Information (VOI) in glucose monitoring**: When is the expected utility of measuring glucose greater than the cost?
2. **Implicit utility inference**: Can we infer utility from clinician behavior (tests followed by interventions)?
3. **POMDP planning**: Testing online planning algorithms (POMCP, DESPOT) for glucose management
4. **Policy comparison**: How do different measurement strategies compare on real patient data?

## Example Output

```
============================================================
Glucose POMDP - Phase 1 Demo
============================================================

3. Getting sample trajectory...
   ✓ Trajectory loaded:
     - Subject ID: 11861
     - ICU Stay ID: 200010
     - Length of stay: 1.0 days
     - Number of states: 83

5. Running policies on trajectory...
------------------------------------------------------------

Greedy (always wait):
  Total reward:      -370.14
  Episode length:         83 steps
  Measurements:            0

Threshold (4hr):
  Total reward:      -370.33
  Episode length:         83 steps
  Measurements:           25

Uncertainty:
  Total reward:      -370.60
  Episode length:         83 steps
  Measurements:           61
```

## Future Extensions

- **Evaluation framework**: Systematic policy comparison across all 11,724 ICU stays
- **Belief state tracking**: Particle filters or Kalman filters for true POMDP reasoning
- **Learned policies**: Q-learning, DQN, or other RL algorithms
- **Generative forward model**: Build simulator for planning algorithms beyond replay
- **Clinical validation**: Compare policy recommendations to actual clinician decisions

## References

- MIMIC-IV dataset: Johnson, A., et al. (2020)
- Glucose management dataset: https://physionet.org/content/glucose-management-mimic/
- Time-in-Range guidelines: American Diabetes Association

## License

Research use only. Data usage subject to PhysioNet credentialing and data use agreements.
