import numpy as np
from reward_scheme import RewardScheme

class TradingEnv:
    def __init__(self, data, initial_balance=1000, transaction_fee=0.001):
        self.data = data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0  # positive: long, negative: short
        self.current_step = 0
        self.transaction_fee = transaction_fee

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        return self.data[self.current_step]

    def step(self, action):
        prev_balance = self.get_balance()

        # 0: Do nothing, 1: Buy, 2: Sell, 3: Short, 4: Short-cover
        if action == 1 and self.balance > 0:
            self.position += (self.balance / self.data[self.current_step]) * (1 - self.transaction_fee)
            self.balance = 0
        elif action == 2 and self.position > 0:
            self.balance += self.position * self.data[self.current_step] * (1 - self.transaction_fee)
            self.position = 0
        elif action == 3 and self.balance > 0:  # For simplicity, shorting using balance
            self.position -= (self.balance / self.data[self.current_step]) * (1 - self.transaction_fee)
            self.balance = 0
        elif action == 4 and self.position < 0:
            self.balance += abs(self.position) * self.data[self.current_step] * (1 + self.transaction_fee)
            self.position = 0

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
        else:
            done = False
 
        reward_calculator = RewardScheme()
        total_reward = reward_calculator.get_reward(prev_balance, self.get_balance(), action)

        return self.data[self.current_step], total_reward, done

    def get_balance(self):
        return self.balance + self.position * self.data[self.current_step]
