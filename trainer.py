import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from typing import List, Optional, Dict
import numpy as np
from tqdm import tqdm
import os
import random
import csv
from collections import deque, namedtuple

from tnn import TNN
from tokenizer import SMILESTokenizer
from dataset import SMILESDataset

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

def custom_reward(smiles, qed, sa, docking):
    return -0.01 * sa + 0.4 * qed + (abs(docking) - 3.3) / (2 * (9.8 - 3.3))

def compute_mc_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def populate_replay_buffer_from_csv(buffer, csv_path, tokenizer, max_length):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            smiles, qed, sa, docking = row[0], float(row[1]), float(row[2]), float(row[3])
            state = tokenizer.encode(smiles, max_length).float()
            reward = custom_reward(smiles, qed, sa, docking)
            buffer.push(state, 0, reward, None, False)

class RLAgent:
    def __init__(self, state_dim, action_dim, buffer_capacity=10000, batch_size=64, gamma=0.99, lr=1e-3):
        self.dqn = DQN(state_dim, action_dim)
        self.target_dqn = DQN(state_dim, action_dim)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.steps = 0

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.stack(batch.state)
        action_batch = torch.tensor(batch.action)
        reward_batch = torch.tensor(batch.reward)
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None]) if any(non_final_mask) else None
        done_batch = torch.tensor(batch.done, dtype=torch.float32)

        q_values = self.dqn(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_q_values = torch.zeros(self.batch_size)
        if non_final_next_states is not None:
            next_q_values[non_final_mask] = self.target_dqn(non_final_next_states).max(1)[0].detach()
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        loss = self.loss_fn(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.steps % 100 == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.steps += 1

    def act(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.dqn.net[-1].out_features - 1)
        with torch.no_grad():
            q_values = self.dqn(state.unsqueeze(0))
            return q_values.argmax().item()

class SMILESTrainer:
    def __init__(
        self,
        model: TNN,
        tokenizer: SMILESTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token2idx[tokenizer.PAD_token])
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        self.current_epoch = 0
        self.best_loss = float('inf')

    def save_checkpoint(self, save_path: str, is_best: bool = False) -> None:
        """Save model checkpoint including optimizer state and training progress."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'tokenizer_state': {
                'token2idx': self.tokenizer.token2idx,
                'idx2token': self.tokenizer.idx2token,
                'vocab_size': self.tokenizer.vocab_size
            }
        }
        
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = os.path.join(os.path.dirname(save_path), 'best_model.pt')
            torch.save(checkpoint, best_path)
            
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint including optimizer state and training progress."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        tokenizer_state = checkpoint['tokenizer_state']
        self.tokenizer.token2idx = tokenizer_state['token2idx']
        self.tokenizer.idx2token = tokenizer_state['idx2token']
        self.tokenizer.vocab_size = tokenizer_state['vocab_size']

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(tqdm(dataloader)):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(input_ids)
            
            loss = self.criterion(
                outputs.view(-1, outputs.size(-1)),
                target_ids.view(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    @torch.no_grad()
    def generate(
        self,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: Optional[int] = 50,
        num_samples: int = 1
    ) -> List[str]:
        """Generate SMILES strings using the model."""
        self.model.eval()
        generated = []
        
        for _ in range(num_samples):
            current_ids = torch.tensor([[self.tokenizer.token2idx[self.tokenizer.START_token]]], device=self.device)
            for _ in range(max_length):
                outputs = self.model(current_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                if top_k is not None:
                    values, _ = torch.topk(next_token_logits, top_k)
                    min_topk = values[..., -1, None]
                    indices_to_remove = next_token_logits < min_topk
                    next_token_logits[indices_to_remove] = float('-inf')
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                if next_token.item() == self.tokenizer.token2idx[self.tokenizer.END_token]:
                    break
                current_ids = torch.cat([current_ids, next_token], dim=1)
            generated_smiles = self.tokenizer.decode(current_ids[0], skip_special_tokens=True)
            generated.append(generated_smiles)
        return generated

    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        save_dir: str = "checkpoints",
        save_frequency: int = 10,
        eval_dataloader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None
    ):
        os.makedirs(save_dir, exist_ok=True)
        
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming training from checkpoint: {resume_from}")
            self.load_checkpoint(resume_from)
            
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            train_loss = self.train_epoch(train_dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")
            
            if eval_dataloader is not None:
                eval_loss = self.evaluate(eval_dataloader)
                print(f"Eval Loss: {eval_loss:.4f}")
                is_best = eval_loss < self.best_loss
                if is_best:
                    self.best_loss = eval_loss
            else:
                is_best = train_loss < self.best_loss
                if is_best:
                    self.best_loss = train_loss
            
            if (epoch + 1) % save_frequency == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
                self.save_checkpoint(checkpoint_path, is_best=is_best)
                print(f"Saved checkpoint at epoch {epoch+1}")
            

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            outputs = self.model(input_ids)
            loss = self.criterion(
                outputs.view(-1, outputs.size(-1)),
                target_ids.view(-1)
            )
            total_loss += loss.item()
            
        return total_loss / len(dataloader) 