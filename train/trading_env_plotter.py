from matplotlib import pyplot as plt

class TradingEnvPlotter:
    def __init__(self, env):
        self.env = env
        self.balances = []
        self.actions = []
        self.prices = []

    def record_step(self, action):
        """ 現在の状態を記録する """
        balance = self.env.get_balance()
        price = self.env.data[self.env.current_step]

        self.balances.append(balance)
        self.actions.append(action)
        self.prices.append(price)

    def plot_balance(self):
        """ バランスの履歴をプロットする """
        plt.figure(figsize=(10, 5))
        plt.plot(self.balances, label='Balance')
        plt.legend()
        plt.title('Balance over time')
        plt.xlabel('Step')
        plt.ylabel('Balance')
        plt.grid(True)
        plt.show()

    def plot_actions(self):
        """ エージェントの行動の履歴をプロットする """
        plt.figure(figsize=(10, 5))
        plt.plot(self.actions, label='Actions', color='gray', alpha=0.7)
        plt.legend()
        plt.title('Agent actions over time')
        plt.xlabel('Step')
        plt.ylabel('Action')
        plt.grid(True)
        plt.show()

    def plot_prices_and_trades(self):
        """ 価格と取引の履歴をプロットする """
        plt.figure(figsize=(10, 5))
        plt.plot(self.prices, label='Price', color='blue', alpha=0.6)

        # Buy actions
        buy_steps = [step for step, action in enumerate(self.actions) if action == 1]
        plt.scatter(buy_steps, [self.prices[i] for i in buy_steps], color='green', label='Buy', zorder=5)

        # Sell actions
        sell_steps = [step for step, action in enumerate(self.actions) if action == 2]
        plt.scatter(sell_steps, [self.prices[i] for i in sell_steps], color='red', label='Sell', zorder=5)

        # Short actions
        short_steps = [step for step, action in enumerate(self.actions) if action == 3]
        plt.scatter(short_steps, [self.prices[i] for i in short_steps], color='purple', label='Short', zorder=5)

        # Cover actions
        cover_steps = [step for step, action in enumerate(self.actions) if action == 4]
        plt.scatter(cover_steps, [self.prices[i] for i in cover_steps], color='orange', label='Short Cover', zorder=5)

        plt.legend()
        plt.title('Price and trades over time')
        plt.xlabel('Step')
        plt.ylabel('Price')
        plt.grid(True)
        plt.show()
