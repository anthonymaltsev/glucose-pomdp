from __future__ import annotations

import json
import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from patient_dataset import ActionSpec, PatientState, PatientStateDataset
from policies import Policy
from rewards import SimpleReward
from simulator import GlucoseEnvironment


@dataclass
class Transition:
    state: PatientState
    action: ActionSpec
    reward: float
    next_state: PatientState
    done: bool


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...] = (128, 128)):
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


class QLearner(Policy):
    def __init__(
        self,
        action_space: Tuple[ActionSpec, ...],
        learning_rate: float = 1e-3,
        discount: float = 0.99,
        epsilon: float = 0.1,
        hidden_dims: Tuple[int, ...] = (128, 128),
        device: str = "cpu",
    ):
        self.action_space = action_space
        self.action_lookup = {a.id: a for a in action_space}
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.device = torch.device(device)
        
        state_dim = 8
        action_dim = len(action_space)
        self.q_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.training = True

    def _state_to_tensor(self, state: PatientState) -> torch.Tensor:
        features = state.features.clone().detach()
        features = torch.nan_to_num(features, nan=0.0)
        return features.flatten().unsqueeze(0).to(self.device)

    def select_action(self, state: PatientState, action_space: Tuple[ActionSpec, ...]) -> ActionSpec:
        if self.training and np.random.random() < self.epsilon:
            return action_space[np.random.randint(len(action_space))]
        
        with torch.no_grad():
            state_tensor = self._state_to_tensor(state)
            q_values = self.q_network(state_tensor)
            best_action_id = q_values.argmax().item()
            return self.action_lookup[best_action_id]

    def update(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, 
               next_states: torch.Tensor, dones: torch.Tensor):
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q = self.q_network(next_states).max(1)[0]
            target_q = rewards + self.discount * next_q * (1 - dones.float())
        
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def reset(self) -> None:
        pass

    def save(self, path: Path):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'action_space': self.action_space,
            'learning_rate': self.learning_rate,
            'discount': self.discount,
            'epsilon': self.epsilon,
        }, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> QLearner:
        checkpoint = torch.load(path, map_location=device)
        
        learner = cls(
            action_space=checkpoint['action_space'],
            learning_rate=checkpoint['learning_rate'],
            discount=checkpoint['discount'],
            epsilon=checkpoint['epsilon'],
            device=device,
        )
        learner.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        learner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return learner


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


class QLearningTrainer:
    def __init__(
        self,
        dataset_path: Path,
        checkpoint_dir: Path = Path("checkpoints"),
        learning_rate: float = 1e-3,
        discount: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        hidden_dims: Tuple[int, ...] = (128, 128),
        replay_capacity: int = 10000,
        batch_size: int = 32,
        update_freq: int = 4,
        checkpoint_freq: int = 10,
        device: str = "cpu",
    ):
        self.dataset_path = dataset_path
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.checkpoint_freq = checkpoint_freq
        
        self.reward_fn = SimpleReward()
        self.dataset = PatientStateDataset(dataset_path, reward_fn=self.reward_fn)
        self.action_space = self.dataset.get_action_space()
        
        self.learner = QLearner(
            action_space=self.action_space,
            learning_rate=learning_rate,
            discount=discount,
            epsilon=epsilon_start,
            hidden_dims=hidden_dims,
            device=device,
        )
        
        self.replay_buffer = ReplayBuffer(replay_capacity)
        self.step_count = 0

    def _state_to_tensor(self, state: PatientState) -> torch.Tensor:
        features = state.features.clone().detach()
        features = torch.nan_to_num(features, nan=0.0)
        return features.flatten()

    def train(self, num_episodes: int = 100):
        trajectories = list(self.dataset.iter_trajectories())
        episode = 0
        
        while episode < num_episodes:
            trajectory = np.random.choice(trajectories)
            env = GlucoseEnvironment(trajectory, self.reward_fn)
            
            if hasattr(self.learner, 'reset'):
                self.learner.reset()
            if hasattr(self.reward_fn, 'reset'):
                self.reward_fn.reset()
            
            state = env.reset()
            total_reward = 0.0
            
            while not env.done:
                action = self.learner.select_action(state, self.action_space)
                next_state, reward, done, info = env.step(action)
                
                transition = Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )
                self.replay_buffer.push(transition)
                total_reward += reward
                
                if len(self.replay_buffer) >= self.batch_size and self.step_count % self.update_freq == 0:
                    batch = self.replay_buffer.sample(self.batch_size)
                    
                    states = torch.stack([self._state_to_tensor(t.state) for t in batch])
                    actions = torch.tensor([t.action.id for t in batch], dtype=torch.long)
                    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
                    next_states = torch.stack([self._state_to_tensor(t.next_state) for t in batch])
                    dones = torch.tensor([t.done for t in batch], dtype=torch.bool)
                    
                    loss = self.learner.update(states, actions, rewards, next_states, dones)
                
                state = next_state
                self.step_count += 1
            
            episode += 1
            self.learner.epsilon = max(
                self.epsilon_end,
                self.epsilon_start * (self.epsilon_decay ** episode)
            )
            
            if episode % self.checkpoint_freq == 0:
                self.save_checkpoint(episode)
                print(f"Episode {episode}/{num_episodes}, Epsilon: {self.learner.epsilon:.3f}, Reward: {total_reward:.2f}")
        
        self.learner.training = False
        self.save_checkpoint(episode, final=True)

    def save_checkpoint(self, episode: int, final: bool = False):
        if final:
            path = self.checkpoint_dir / "qlearner_final.pth"
        else:
            path = self.checkpoint_dir / f"qlearner_ep{episode}.pth"
        
        self.learner.save(path)
        
        config = {
            'episode': episode,
            'learning_rate': self.learning_rate,
            'discount': self.discount,
            'epsilon': self.learner.epsilon,
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
    
    trainer = QLearningTrainer(
        dataset_path=csv_path,
        checkpoint_dir=checkpoint_dir,
        learning_rate=1e-3,
        discount=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        hidden_dims=(128, 128),
        replay_capacity=10000,
        batch_size=32,
        update_freq=4,
        checkpoint_freq=10,
        device=device,
    )
    
    print(f"Starting DQN training on {device}...")
    trainer.train(num_episodes=100)
    print(f"Training complete. Checkpoints saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()
