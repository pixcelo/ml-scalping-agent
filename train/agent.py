import os
import torch
import torch.optim as optim
import numpy as np
from qnetwork import QNetwork
import torch.optim.lr_scheduler as lr_scheduler

class Agent:
    def __init__(self, input_dim, action_dim=5, learning_rate=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.999, step_size=100, gamma_lr=0.9):
        self.qnetwork = QNetwork(input_dim, action_dim)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.loss_fn = torch.nn.MSELoss()
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma_lr)
        self.action_dim = action_dim

    def step_scheduler(self):
        self.scheduler.step()

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(self.action_dim))
        else:
            with torch.no_grad():
                q_values = self.qnetwork(torch.FloatTensor([state]))
                return torch.argmax(q_values).item()

    def train(self, state, action, reward, next_state, done):
        q_values = self.qnetwork(torch.FloatTensor([state]))
        
        with torch.no_grad():
            next_q_values = self.qnetwork(torch.FloatTensor([next_state]))
            target = reward + (1 - done) * self.gamma * torch.max(next_q_values)

        loss = self.loss_fn(q_values[0, action], target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filename='checkpoint.pth'):
        save_path = os.path.join('..', 'model', filename)
        torch.save(self.qnetwork.state_dict(), save_path)

    def load_model(self, filename='checkpoint.pth', eval_mode=True):
        load_path = os.path.join('..', 'model', filename)
        self.qnetwork.load_state_dict(torch.load(load_path))
        if eval_mode:
            self.qnetwork.eval()
            self.epsilon = 0 # Disable random actions
        else:
            self.qnetwork.train()