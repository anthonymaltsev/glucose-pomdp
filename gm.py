from __future__ import annotations

from pathlib import Path

from patient_dataset import PatientStateDataset
from rewards import SimpleReward
from simulator import GlucoseEnvironment
from policies import GreedyPolicy, ThresholdPolicy, UncertaintyPolicy, MyopicVOIPolicy


def run_policy_on_trajectory(
    policy,
    trajectory,
    reward_fn,
    dataset,
    verbose: bool = False,
):
    """
    Run a single policy on a trajectory and return statistics.

    Args:
        policy: Policy to evaluate
        trajectory: PatientTrajectory to run on
        reward_fn: Reward function
        dataset: Dataset (for action space)
        verbose: Print step-by-step info

    Returns:
        Dictionary with episode statistics
    """
    env = GlucoseEnvironment(trajectory, reward_fn)
    state = env.reset()

    total_reward = 0.0
    num_measurements = 0
    num_waits = 0
    action_counts = {}

    while not env.done:
        action = policy.select_action(state, dataset.get_action_space())

        # Track actions
        action_counts[action.name] = action_counts.get(action.name, 0) + 1
        if action.name == "wait":
            num_waits += 1
        else:
            num_measurements += 1

        next_state, reward, done, info = env.step(action)
        total_reward += reward

        if verbose:
            print(f"  Step {env.episode_length}: action={action.name}, reward={reward:.2f}, obs={info['observation']}")

        state = next_state

    stats = env.get_episode_stats()
    stats["num_measurements"] = num_measurements
    stats["num_waits"] = num_waits
    stats["action_counts"] = action_counts

    return stats


def main() -> None:
    # Setup
    dataset_dir = Path(__file__).parent / "data" / "physionet.org" / "files" / "glucose-management-mimic" / "1.0.1" / "Datasets"
    csv_path = dataset_dir / "glucose_insulin_ICU.csv"

    print("=" * 60)
    print("Glucose POMDP - Phase 1 Demo")
    print("=" * 60)

    # Create reward function
    print("\n1. Creating reward function...")
    reward_fn = SimpleReward()
    print("   ✓ SimpleReward initialized")

    # Load dataset with reward function
    print("\n2. Loading dataset with reward function...")
    dataset = PatientStateDataset(csv_path, reward_fn=reward_fn)
    print(f"   ✓ Dataset loaded from {csv_path}")

    # Get one trajectory for testing
    print("\n3. Getting sample trajectory...")
    trajectory_iter = dataset.iter_trajectories()
    trajectory = next(trajectory_iter)
    print(f"   ✓ Trajectory loaded:")
    print(f"     - Subject ID: {trajectory.subject_id}")
    print(f"     - ICU Stay ID: {trajectory.icustay_id}")
    print(f"     - Length of stay: {trajectory.los_icu_days:.1f} days")
    print(f"     - Number of states: {len(trajectory.states)}")

    # Create policies
    print("\n4. Creating policies...")
    policies = [
        ("Greedy (always wait)", GreedyPolicy(dataset.get_action_space())),
        ("Threshold (4hr)", ThresholdPolicy(time_threshold_minutes=240)),
        ("Threshold (2hr)", ThresholdPolicy(time_threshold_minutes=120)),
        ("Uncertainty", UncertaintyPolicy(std_threshold=30, time_threshold_minutes=180)),
        ("Myopic VOI", MyopicVOIPolicy(info_gain_weight=1.5)),
    ]
    print(f"   ✓ Created {len(policies)} policies")

    # Test each policy
    print("\n5. Running policies on trajectory...")
    print("-" * 60)

    results = []
    for name, policy in policies:
        stats = run_policy_on_trajectory(policy, trajectory, reward_fn, dataset, verbose=False)
        results.append((name, stats))

        print(f"\n{name}:")
        print(f"  Total reward:     {stats['total_reward']:>8.2f}")
        print(f"  Episode length:   {stats['length']:>8} steps")
        print(f"  Avg reward/step:  {stats['avg_reward']:>8.3f}")
        print(f"  Measurements:     {stats['num_measurements']:>8}")
        print(f"  Wait actions:     {stats['num_waits']:>8}")
        print(f"  Action breakdown: {stats['action_counts']}")

    # Summary comparison
    print("\n" + "=" * 60)
    print("Summary Comparison")
    print("=" * 60)
    print(f"{'Policy':<20} {'Total Reward':>15} {'Measurements':>15}")
    print("-" * 60)
    for name, stats in results:
        print(f"{name:<20} {stats['total_reward']:>15.2f} {stats['num_measurements']:>15}")

    print("\n✓ Phase 1 implementation complete!")
    print("\nYou can now:")
    print("  - Implement custom policies by extending the Policy class")
    print("  - Experiment with different reward parameters")
    print("  - Run policies on different trajectories")
    print("  - Build evaluation framework to compare across all trajectories")


if __name__ == "__main__":
    main()
