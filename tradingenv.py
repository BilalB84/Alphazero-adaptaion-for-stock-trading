import numpy as np
import gym
from gym import spaces

class StockTradingEnv(gym.Env):
    
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        
        self.df = df
        self.current_step = 0 
        self.net_worth = 0  
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.holdings = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trades = []

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        
    def _get_observation(self):
        if self.current_step < 0 or self.current_step >= len(self.df):
            raise ValueError(f"Invalid step index: {self.current_step}. It should be within the range [0, {len(self.df) - 1}].")
        
        obs = np.array([
            self.df.loc[self.current_step, 'Open'],
            self.df.loc[self.current_step, 'High'],
            self.df.loc[self.current_step, 'Low'],
            self.df.loc[self.current_step, 'Close'],
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
        
        if self.df.empty:
            raise ValueError("DataFrame is empty. Please check your data loading process.")
        
        self.current_step = 0  
        print(f"Environment reset. Starting at step index: {self.current_step}")

        return self._get_observation()

    def step(self, action):
        self._take_action(action)
        self.current_step += 1

        if self.current_step >= len(self.df):
            print(f"End of data reached at step: {self.current_step - 1}, resetting to 0.")
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
