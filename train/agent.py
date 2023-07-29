import torch
import torch.optim as optim
import numpy as np
from qnetwork import QNetwork

class Agent:
    def __init__(self, input_dim, action_dim=5, learning_rate=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.qnetwork = QNetwork(input_dim, action_dim)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.loss_fn = torch.nn.MSELoss()
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1, 2, 3, 4])
        else:
            with torch.no_grad():
                q_values = self.qnetwork(torch.FloatTensor([state]))
                return torch.argmax(q_values).item()

    def train(self, state, action, reward, next_state, done):
        q_values = self.qnetwork(torch.FloatTensor([state]))
        
        with torch.no_grad():
            next_q_values = self.qnetwork(torch.FloatTensor([next_state]))
            target = reward + (1 - done) * self.gamma * torch.max(next_q_values)

        loss = self.loss_fn(q_values[action], target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
