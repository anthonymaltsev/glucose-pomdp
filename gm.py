from __future__ import annotations

from pathlib import Path

from patient_dataset import PatientStateDataset, PatientTrajectory
from rewards import SimpleReward
from simulator import GlucoseEnvironment
from policies import (
    GreedyPolicy,
    HistoricalPolicy,
    MyopicVOIPolicy,
    ThresholdPolicy,
    UncertaintyPolicy,
)
from qlearn import QLearner
from bc import BCPolicy

def run_policy_on_trajectory(
    policy,
    trajectory: PatientTrajectory,
    reward_fn,
    dataset: PatientStateDataset,
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
    # Reset any internal policy state between episodes
    if hasattr(policy, "reset"):
        policy.reset()
    if hasattr(reward_fn, "reset"):
        reward_fn.reset()
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
    
    # Get reward component breakdown
    stats["reward_breakdown"] = reward_fn.get_component_breakdown()

    return stats


def main() -> None:
    # Setup
    # dataset_dir = Path(__file__).parent / "data" / "physionet.org" / "files" / "glucose-management-mimic" / "1.0.1" / "Datasets"
    dataset_dir = Path(__file__).parent / "data" / "Datasets"
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
    dataset = PatientStateDataset(csv_path, reward_fn=reward_fn, split="test")
    print(f"   ✓ Dataset loaded from {csv_path} (test split)")

    # Create policy factories so we can build a fresh policy per trajectory.
    print("\n3. Creating policies...")
    qlearner = QLearner.load("checkpoints/qlearner_ep2500.pth")
    bc_policy = BCPolicy.load("checkpoints/bc_policy_ep8.pth")
    qlearner.training = False  # Set to eval mode
    # bc_policy.training = False  # keep in train mode for now so that it performs like a stochastic policy
    policy_factories = [
        ("Greedy (always wait)", lambda ds, traj: GreedyPolicy(ds.get_action_space())),
        # ("Threshold (4hr)", lambda ds, traj: ThresholdPolicy(time_threshold_minutes=240)),
        ("Threshold (2hr)", lambda ds, traj: ThresholdPolicy(time_threshold_minutes=120)),
        (
            "Uncertainty",
            lambda ds, traj: UncertaintyPolicy(std_threshold=30, time_threshold_minutes=180),
        ),
        ("Myopic VOI", lambda ds, traj: MyopicVOIPolicy(info_gain_weight=1.5)),
        ("Historical", lambda ds, traj: HistoricalPolicy(traj)),
        ("QLearner", lambda ds, traj: qlearner),
        ("BCPolicy", lambda ds, traj: bc_policy)
    ]
    print(f"   ✓ Created {len(policy_factories)} policy definitions")

    # Initialize results storage for all trajectories
    print("\n4. Running policies on all trajectories...")
    print("-" * 60)
    
    # Store aggregated results for each policy
    policy_results = {
        name: {
            "total_rewards": [],
            "total_measurements": [],
            "total_waits": [],
            "episode_lengths": [],
            "avg_rewards": [],
            "reward_components": {
                "action_cost": [],
                "glycemic_penalty": [],
                "time_in_range": [],
                "uncertainty_penalty": [],
                "treatment_penalty": [],
            },
        }
        for name, _ in policy_factories
    }
    
    trajectory_iter = dataset.iter_trajectories()
    trajectory_count = 0
    
    # Process all trajectories
    for trajectory in trajectory_iter:
        trajectory_count += 1
        
        # Verify trajectory structure: should have len(states) - 1 actions
        if len(trajectory.actions) != len(trajectory.states) - 1:
            print(f"WARNING: Trajectory {trajectory_count} has {len(trajectory.states)} states but {len(trajectory.actions)} actions (expected {len(trajectory.states) - 1})")
        
        # Run each policy on this trajectory
        for name, factory in policy_factories:
            policy = factory(dataset, trajectory)
            stats = run_policy_on_trajectory(policy, trajectory, reward_fn, dataset, verbose=False)
            
            # Accumulate statistics
            policy_results[name]['total_rewards'].append(stats['total_reward'])
            policy_results[name]['total_measurements'].append(stats['num_measurements'])
            policy_results[name]['total_waits'].append(stats['num_waits'])
            policy_results[name]['episode_lengths'].append(stats['length'])
            policy_results[name]['avg_rewards'].append(stats['avg_reward'])
            
            # Accumulate reward component breakdown
            breakdown = stats.get('reward_breakdown', {})
            for component in policy_results[name]['reward_components']:
                policy_results[name]['reward_components'][component].append(
                    breakdown.get(component, 0.0)
                )
        
        # Print progress every 100 iterations
        if trajectory_count % 100 == 0:
            print(f"  Processed {trajectory_count} trajectories...")
        if trajectory_count == 100:
            print("Stopping at 100 trajectories...")
            break
    
    print(f"   ✓ Completed processing {trajectory_count} trajectories")

    # Summary comparison across all trajectories
    print("\n" + "=" * 60)
    print("Summary Comparison (Across All Trajectories)")
    print("=" * 60)
    print(f"{'Policy':<25} {'Avg Reward':>15} {'Total Reward':>15} {'Avg Measurements':>18} {'Total Measurements':>20}")
    print("-" * 100)
    
    for name, _ in policy_factories:
        results = policy_results[name]
        avg_reward = sum(results['total_rewards']) / len(results['total_rewards']) if results['total_rewards'] else 0
        total_reward = sum(results['total_rewards'])
        avg_measurements = sum(results['total_measurements']) / len(results['total_measurements']) if results['total_measurements'] else 0
        total_measurements = sum(results['total_measurements'])
        
        print(f"{name:<25} {avg_reward:>15.2f} {total_reward:>15.2f} {avg_measurements:>18.2f} {total_measurements:>20}")
    
    # Reward component breakdown
    print("\n" + "=" * 60)
    print("Reward Component Breakdown (Average per Trajectory)")
    print("=" * 60)
    print(f"{'Policy':<25} {'Action Cost':>15} {'Uncertainty':>15} {'Treatment':>15} {'Time-in-Range':>15} {'Glycemic':>15}")
    print("-" * 100)
    
    for name, _ in policy_factories:
        results = policy_results[name]
        components = results['reward_components']
        
        avg_action_cost = sum(components['action_cost']) / len(components['action_cost']) if components['action_cost'] else 0.0
        avg_uncertainty = sum(components['uncertainty_penalty']) / len(components['uncertainty_penalty']) if components['uncertainty_penalty'] else 0.0
        avg_treatment = sum(components['treatment_penalty']) / len(components['treatment_penalty']) if components['treatment_penalty'] else 0.0
        avg_tir = sum(components['time_in_range']) / len(components['time_in_range']) if components['time_in_range'] else 0.0
        avg_glycemic = sum(components['glycemic_penalty']) / len(components['glycemic_penalty']) if components['glycemic_penalty'] else 0.0
        
        print(f"{name:<25} {avg_action_cost:>15.2f} {avg_uncertainty:>15.2f} {avg_treatment:>15.2f} {avg_tir:>15.2f} {avg_glycemic:>15.2f}")


if __name__ == "__main__":
    main()
