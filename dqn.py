import torch
import torch.nn as nn
import random
import torch.optim as optim

from scoring import Scoring
from tnn_generator import TransformerGenerator


class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def reinforced_training(self, generator: TransformerGenerator, score_calc: Scoring, target_pdb: str, epochs: int = 1000, gamma: float = 0.99):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        buffer = ReplayBuffer(10000)

        for epoch in range(epochs):
            state = generator.get_initial_state()
            done = False
            total_reward = 0

            while not done:
                q_values = self(torch.tensor(state).float())
                action = torch.argmax(q_values).item()

                next_smiles, next_state = generator.step(state, action)
                score_calc.set_args({'smiles': next_smiles, 'target': target_pdb})
                reward = score_calc.compute()
                done = generator.is_terminal(next_smiles)

                buffer.push((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                if len(buffer.buffer) > 64:
                    states, actions, rewards, next_states, dones = buffer.sample(64)
                    states = torch.tensor(states).float()
                    actions = torch.tensor(actions)
                    rewards = torch.tensor(rewards)
                    next_states = torch.tensor(next_states).float()
                    dones = torch.tensor(dones).float()

                    q_values = self(states)
                    next_q_values = self(next_states).detach()
                    target_q = rewards + gamma * (1 - dones) * next_q_values.max(1)[0]

                    loss = nn.functional.mse_loss(q_values.gather(1, actions.unsqueeze(1)).squeeze(), target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            print(f"Epoch {epoch}, Total reward: {total_reward}")


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = []
        self.capacity = capacity

    def push(self, transition: tuple):
        self.buffer.append(transition)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size: int):
        return zip(*random.sample(self.buffer, batch_size))
