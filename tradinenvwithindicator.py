import numpy as np
import gym
from gym import spaces
import pandas as pd

class StockTradingEnvWithIndicators(gym.Env):
    
    def __init__(self, df):
        super(StockTradingEnvWithIndicators, self).__init__()
        
        self.df = df
        self.current_step = 0

        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.holdings = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trades = []

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        
        self._add_indicators()

    def _add_indicators(self):

        self.df['SMA'] = self.df['Close'].rolling(window=14).mean()

        self.df['EMA'] = self.df['Close'].ewm(span=14, adjust=False).mean()

        delta = self.df['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        self.df['RSI'] = 100 - (100 / (1 + rs))

        short_ema = self.df['Close'].ewm(span=12, adjust=False).mean()
        long_ema = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = short_ema - long_ema
        self.df['Signal Line'] = self.df['MACD'].ewm(span=9, adjust=False).mean()

        self.df['20_SMA'] = self.df['Close'].rolling(window=20).mean()
        self.df['Upper Band'] = self.df['20_SMA'] + (self.df['Close'].rolling(window=20).std() * 2)
        self.df['Lower Band'] = self.df['20_SMA'] - (self.df['Close'].rolling(window=20).std() * 2)

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
        
        if action == 1:  
            if self.balance > current_price:
                self.holdings += self.balance // current_price
                self.balance -= self.holdings * current_price
                self.trades.append({
                    'step': self.current_step,
                    'amount': self.holdings,
                    'total': self.holdings * current_price,
                    'type': 'buy'
                })

        elif action == 2:  
            if self.holdings > 0:
                self.balance += self.holdings * current_price
                self.trades.append({
                    'step': self.current_step,
                    'amount': self.holdings,
                    'total': self.holdings * current_price,
                    'type': 'sell'
                })
                self.holdings = 0

        self.net_worth = self.balance + self.holdings * current_price
        
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

    def reset(self):
        self.balance = self.initial_balance
        self.holdings = 0
        self.net_worth = self.initial_balance
        self.trades = []
        
        self.current_step = 0  
        return self._get_observation()

    def step(self, action):
        self._take_action(action)
        self.current_step += 1

        if self.current_step >= len(self.df):
            self.current_step = 0
            done = True 
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
