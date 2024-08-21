import torch
import numpy as np
from MCTS import MCTS, MCTSNode 
from tradingenv import StockTradingEnv
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def preprocess_stock_data(stock_file, year, train_months):
    data = pd.read_csv(stock_file)
    data['Date'] = pd.to_datetime(data['Date'])
    train_data = data[(data['Date'].dt.year == year) & (data['Date'].dt.month.isin(train_months))].copy()
    train_data.reset_index(drop=True, inplace=True)

    scaler = StandardScaler()
    train_data.loc[:, ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = scaler.fit_transform(
        train_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    )
    
    return train_data

def calculate_profit(reward, cost):
    return reward - cost

def evaluate(model, stock_file, year, test_months, mcts, num_episodes, load_path="model.pth"):
    test_data = preprocess_stock_data(stock_file, year, test_months)
    
    test_env = StockTradingEnv(test_data)
    
    state_dict = torch.load(load_path)
    model.load_state_dict(state_dict)
    model.eval()  
    
    total_rewards = []
    total_profits = []
    
    for episode in range(num_episodes):
        state = test_env.reset()  
        done = False
        cumulative_profit = 0
        cumulative_reward = 0
        
        while not done:
            root = MCTSNode(state) 
            mcts.run(root) 
            
            action_probs = np.array([child.visit_count for child in root.children.values()])
            action_probs = action_probs / np.sum(action_probs)
            
            action = np.random.choice(len(action_probs), p=action_probs)
            
            next_state, reward, done, _ = test_env.step(action)
            
            reward = np.clip(reward, -1, 1)  
            cost = 0.1 
            profit = calculate_profit(reward, cost)  
            
            cumulative_profit += profit
            cumulative_reward += reward
            
            state = next_state 
        
        total_rewards.append(cumulative_reward)
        total_profits.append(cumulative_profit)
        
        print(f"Episode {episode + 1}/{num_episodes}, Profit: {cumulative_profit}, Reward: {cumulative_reward}")
    
    avg_profit = np.mean(total_profits)
    avg_reward = np.mean(total_rewards)
    
    print(f"Average Profit: {avg_profit}")
    print(f"Average Reward: {avg_reward}")
    
    plt.figure(figsize=(20, 6))
    plt.plot(range(1, num_episodes + 1), total_rewards, label="Cumulative Reward")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward Over Episodes")
    plt.legend()
    plt.show()

    plt.figure(figsize=(20, 6))
    plt.plot(range(1, num_episodes + 1), total_profits, label="Cumulative Profit", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Profit")
    plt.title("Cumulative Profit Over Episodes")
    plt.legend()
    plt.show()