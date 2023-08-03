import numpy as np

class RewardScheme:
    def __init__(self):
        # Counter for consecutive wins and losses
        self.win_streak = 0
        self.loss_streak = 0

    def reset(self):
        """Reset the win and loss streak counters."""
        self.win_streak = 0
        self.loss_streak = 0

    def get_reward(self, prev_balance, current_balance, action_taken):
        """Calculate the reward based on balance difference and action taken."""
        # Compute the basic reward
        immediate_reward = current_balance - prev_balance
        
        # Clip the reward to scale it. Assuming we want to clip between -1 and 1. Adjust as needed.
        immediate_reward = np.clip(immediate_reward, -1, 1)
        
        # Calculate the win bonus and consecutive win bonus
        if immediate_reward > 0:
            self.win_streak += 1
            self.loss_streak = 0
            win_bonus = 0.1  # Bonus for a single win
            streak_bonus = 0.05 * self.win_streak  # Bonus for consecutive wins
        else:
            self.loss_streak += 1
            self.win_streak = 0
            win_bonus = 0
            streak_bonus = 0

        # Optional: Compute the penalty for losses
        # loss_penalty = -0.05 * self.loss_streak if immediate_reward < 0 else 0
        
        # Compute the total reward
        total_reward = immediate_reward + win_bonus + streak_bonus # + loss_penalty

        return total_reward
