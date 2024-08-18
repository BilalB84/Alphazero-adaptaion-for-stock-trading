import numpy as np
import gym
from gym import spaces
import pandas as pd

class StockTradingEnvWithIndicators(gym.Env):
    """A custom environment for stock trading using historical data and trading indicators."""
    
    def __init__(self, df):
        super(StockTradingEnvWithIndicators, self).__init__()
        
        self.df = df
        self.current_step = 0

        # Initial conditions
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.holdings = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trades = []

        # Define action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

        # Define observation space: Adding custom indicators (e.g., 10 features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        
        # Ensure the indicators are calculated in the DataFrame
        self._add_indicators()

    def _add_indicators(self):
        """Adds custom trading indicators to the DataFrame."""
        # Simple Moving Average (SMA)
        self.df['SMA'] = self.df['Close'].rolling(window=14).mean()

        # Exponential Moving Average (EMA)
        self.df['EMA'] = self.df['Close'].ewm(span=14, adjust=False).mean()

        # Relative Strength Index (RSI)
        delta = self.df['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        self.df['RSI'] = 100 - (100 / (1 + rs))

        # Moving Average Convergence Divergence (MACD)
        short_ema = self.df['Close'].ewm(span=12, adjust=False).mean()
        long_ema = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = short_ema - long_ema
        self.df['Signal Line'] = self.df['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        self.df['20_SMA'] = self.df['Close'].rolling(window=20).mean()
        self.df['Upper Band'] = self.df['20_SMA'] + (self.df['Close'].rolling(window=20).std() * 2)
        self.df['Lower Band'] = self.df['20_SMA'] - (self.df['Close'].rolling(window=20).std() * 2)

        # Fill NaN values
        self.df.fillna(0, inplace=True)

    def _get_observation(self):
        obs = np.array([
            self.df.loc[self.current_step, 'Open'],
            self.df.loc[self.current_step, 'High'],
            self.df.loc[self.current_step, 'Low'],
            self.df.loc[self.current_step, 'Close'],
            self.df.loc[self.current_step, 'SMA'],
            self.df.loc[self.current_step, 'EMA'],
            self.df.loc[self.current_step, 'RSI'],
            self.df.loc[self.current_step, 'MACD'],
            self.df.loc[self.current_step, 'Upper Band'],
            self.df.loc[self.current_step, 'Lower Band'],
            self.balance,
            self.holdings,
            self.net_worth
        ])
        return obs

    def _take_action(self, action):
        current_price = self.df.loc[self.current_step, 'Close']
        
        if action == 1:  # Buy
            if self.balance > current_price:
                self.holdings += self.balance // current_price
                self.balance -= self.holdings * current_price
                self.trades.append({
                    'step': self.current_step,
                    'amount': self.holdings,
                    'total': self.holdings * current_price,
                    'type': 'buy'
                })

        elif action == 2:  # Sell
            if self.holdings > 0:
                self.balance += self.holdings * current_price
                self.trades.append({
                    'step': self.current_step,
                    'amount': self.holdings,
                    'total': self.holdings * current_price,
                    'type': 'sell'
                })
                self.holdings = 0

        # Update net worth
        self.net_worth = self.balance + self.holdings * current_price
        
        # Track the maximum net worth achieved
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

    def reset(self):
        # Reset environment state
        self.balance = self.initial_balance
        self.holdings = 0
        self.net_worth = self.initial_balance
        self.trades = []
        
        self.current_step = 0  # Start at the first step
        return self._get_observation()

    def step(self, action):
        self._take_action(action)
        self.current_step += 1

        if self.current_step >= len(self.df):
            self.current_step = 0
            done = True  # End of episode
        else:
            done = False

        reward = self.net_worth - self.initial_balance
        obs = self._get_observation()

        return obs, reward, done, {}

    def render(self, mode='human'):
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Holdings: {self.holdings}')
        print(f'Net Worth: {self.net_worth}')
        print(f'Max Net Worth: {self.max_net_worth}')
        print('Trades:', self.trades)
