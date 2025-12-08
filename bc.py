from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from patient_dataset import ActionSpec, PatientState, PatientStateDataset
from policies import Policy
from rewards import SimpleReward


class BCNetwork(nn.Module):
    """Lightweight MLP for behavior cloning."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...] = (64, 32)):
        super().__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class BCPolicy(Policy):
    """Behavior cloning policy that learns to mimic historical actions."""
    
    def __init__(
        self,
        action_space: Tuple[ActionSpec, ...],
        learning_rate: float = 1e-3,
        hidden_dims: Tuple[int, ...] = (64, 32),
        device: str = "cpu",
    ):
        self.action_space = action_space
        self.action_lookup = {a.id: a for a in action_space}
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        
        state_dim = 8
        action_dim = len(action_space)
        self.network = BCNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.training = True

    def _state_to_tensor(self, state: PatientState) -> torch.Tensor:
        features = state.features.clone().detach()
        features = torch.nan_to_num(features, nan=0.0)
        return features.flatten().unsqueeze(0).to(self.device)

    def select_action(self, state: PatientState, action_space: Tuple[ActionSpec, ...]) -> ActionSpec:
        with torch.no_grad():
            state_tensor = self._state_to_tensor(state)
            logits = self.network(state_tensor)
            probs = torch.softmax(logits, dim=1)
            
            if self.training:
                # Sample from distribution during training
                action_id = torch.multinomial(probs, 1).item()
            else:
                # Greedy selection during evaluation
                action_id = probs.argmax().item()
            
            return self.action_lookup[action_id]

    def update(self, states: torch.Tensor, actions: torch.Tensor):
        """Update policy using supervised learning on state-action pairs."""
        states = states.to(self.device)
        actions = actions.to(self.device)

        logits = self.network(states)
        loss = nn.CrossEntropyLoss()(logits, actions)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def reset(self) -> None:
        pass

    def save(self, path: Path):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'action_space': self.action_space,
            'learning_rate': self.learning_rate,
        }, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> BCPolicy:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        policy = cls(
            action_space=checkpoint['action_space'],
            learning_rate=checkpoint['learning_rate'],
            device=device,
        )
        policy.network.load_state_dict(checkpoint['network_state_dict'])
        policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return policy


class BehaviorCloningTrainer:
    """Trainer for behavior cloning on historical trajectories."""
    
    def __init__(
        self,
        dataset_path: Path,
        checkpoint_dir: Path = Path("checkpoints"),
        learning_rate: float = 1e-3,
        hidden_dims: Tuple[int, ...] = (64, 32),
        batch_size: int = 32,
        checkpoint_freq: int = 10,
        device: str = "cpu",
    ):
        self.dataset_path = dataset_path
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.checkpoint_freq = checkpoint_freq
        
        self.reward_fn = SimpleReward()
        self.dataset = PatientStateDataset(dataset_path, reward_fn=self.reward_fn, split="train")
        self.action_space = self.dataset.get_action_space()
        
        self.policy = BCPolicy(
            action_space=self.action_space,
            learning_rate=learning_rate,
            hidden_dims=hidden_dims,
            device=device,
        )
        
        self.step_count = 0

    def _state_to_tensor(self, state: PatientState) -> torch.Tensor:
        features = state.features.clone().detach()
        features = torch.nan_to_num(features, nan=0.0)
        return features.flatten()

    def _collect_training_data(self, max_trajectories: int = None, sample_fraction: float = None):
        """Collect state-action pairs from historical trajectories.
        
        Uses lazy trajectory building with caching for faster iteration.
        
        Args:
            max_trajectories: Maximum number of trajectories to process (None = all)
            sample_fraction: Randomly sample this fraction of trajectories (None = use max_trajectories)
        """
        states = []
        actions = []
        
        # Get trajectory identifiers first (fast - just grouping keys, no trajectory building)
        print("Getting trajectory keys...")
        trajectory_keys = []
        for key, _ in self.dataset._events.groupby(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"], sort=False):
            trajectory_keys.append(key)
        
        print(f"Found {len(trajectory_keys)} total trajectories")
        
        # Sample if requested
        if sample_fraction is not None and sample_fraction < 1.0:
            num_samples = max(1, int(len(trajectory_keys) * sample_fraction))
            selected_keys = np.random.choice(len(trajectory_keys), num_samples, replace=False)
            trajectory_keys = [trajectory_keys[i] for i in selected_keys]
            print(f"Sampled {len(trajectory_keys)} trajectories ({sample_fraction*100:.1f}%)")
        elif max_trajectories is not None:
            trajectory_keys = trajectory_keys[:max_trajectories]
            print(f"Using first {len(trajectory_keys)} trajectories")
        
        # Lazy trajectory cache - only build when accessed
        trajectory_cache = {}
        
        def get_trajectory(key):
            if key not in trajectory_cache:
                stay_df = self.dataset._events[
                    (self.dataset._events["SUBJECT_ID"] == key[0]) &
                    (self.dataset._events["HADM_ID"] == key[1]) &
                    (self.dataset._events["ICUSTAY_ID"] == key[2])
                ]
                trajectory_cache[key] = self.dataset._build_trajectory(stay_df.copy())
            return trajectory_cache[key]
        
        # Collect data from selected trajectories
        print("Collecting state-action pairs...")
        for idx, key in enumerate(trajectory_keys):
            trajectory = get_trajectory(key)
            
            # Use historical actions as ground truth
            for i, state in enumerate(trajectory.states):
                if i < len(trajectory.actions):
                    # Only include states where we have a historical action
                    historical_action = trajectory.actions[i]
                    states.append(state)
                    actions.append(historical_action.id)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(trajectory_keys)} trajectories, collected {len(states)} pairs so far...")
        
        return states, actions

    def train(self, num_epochs: int = 100, max_trajectories: int = None, sample_fraction: float = None):
        """Train behavior cloning policy on historical data.
        
        Args:
            num_epochs: Number of training epochs
            max_trajectories: Maximum number of trajectories to use (None = all)
            sample_fraction: Randomly sample this fraction of trajectories for faster iteration
                            (e.g., 0.1 = 10% of trajectories). If None, uses max_trajectories.
        """
        print("Collecting training data from historical trajectories...")
        states, actions = self._collect_training_data(
            max_trajectories=max_trajectories, 
            sample_fraction=sample_fraction
        )
        
        if len(states) == 0:
            raise ValueError("No training data collected. Check dataset and trajectories.")
        
        print(f"Collected {len(states)} state-action pairs from historical data")
        
        state_tensors = torch.stack([self._state_to_tensor(s) for s in states])
        action_tensors = torch.tensor(actions, dtype=torch.long)
        
        dataset = torch.utils.data.TensorDataset(state_tensors, action_tensors)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        print("Starting training...")
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch_states, batch_actions in dataloader:
                loss = self.policy.update(batch_states, batch_actions)
                epoch_losses.append(loss)
                self.step_count += 1
            
            avg_loss = np.mean(epoch_losses)
            
            if epoch % self.checkpoint_freq == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(epoch)
                print(f"Epoch {epoch}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
        
        self.policy.training = False
        self.save_checkpoint(num_epochs - 1, final=True)
        print("Training complete.")

    def save_checkpoint(self, epoch: int, final: bool = False):
        if final:
            path = self.checkpoint_dir / "bc_policy_final.pth"
        else:
            path = self.checkpoint_dir / f"bc_policy_ep{epoch}.pth"
        
        self.policy.save(path)
        
        config = {
            'epoch': epoch,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
        }
        
        config_path = path.with_suffix('.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


def main():
    dataset_dir = Path(__file__).parent / "data" / "Datasets"
    csv_path = dataset_dir / "glucose_insulin_ICU.csv"
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    trainer = BehaviorCloningTrainer(
        dataset_path=csv_path,
        checkpoint_dir=checkpoint_dir,
        learning_rate=1e-3,
        hidden_dims=(64, 32),
        batch_size=32,
        checkpoint_freq=1,
        device=device,
    )
    
    print(f"Starting behavior cloning training on {device}...")
    # For quick iteration, use sample_fraction (e.g., 0.1 for 10% of trajectories)
    # For full training, use max_trajectories=None and sample_fraction=None
    trainer.train(num_epochs=10000, max_trajectories=None)
    print(f"Training complete. Checkpoints saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()

