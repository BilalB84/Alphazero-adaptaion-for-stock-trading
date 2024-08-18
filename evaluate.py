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
    # Example profit calculation: profit is reward minus some cost
    return reward - cost

def evaluate(model, stock_file, year, test_months, mcts, num_episodes, load_path="model.pth"):
    # Preprocess the test data using the preprocess_stock_data function
    test_data = preprocess_stock_data(stock_file, year, test_months)
    
    # Initialize the test environment with the filtered and scaled test data
    test_env = StockTradingEnv(test_data)
    
    # Load the model's state_dict (weights) from the file
    state_dict = torch.load(load_path)
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode
    
    total_rewards = []
    total_profits = []
    
    for episode in range(num_episodes):
        state = test_env.reset()  # Reset the environment at the beginning of each episode
        done = False
        cumulative_profit = 0
        cumulative_reward = 0
        
        while not done:
            root = MCTSNode(state)  # Initialize the MCTS node with the current state
            mcts.run(root)  # Run MCTS to determine the best action
            
            # Calculate the action probabilities from the MCTS results
            action_probs = np.array([child.visit_count for child in root.children.values()])
            action_probs = action_probs / np.sum(action_probs)
            
            # Select an action based on the probabilities
            action = np.random.choice(len(action_probs), p=action_probs)
            
            # Step the environment with the chosen action
            next_state, reward, done, _ = test_env.step(action)
            
            reward = np.clip(reward, -1, 1)  # Clip the reward to a reasonable range
            
            cost = 0.1  # Assume a fixed cost for each transaction
            profit = calculate_profit(reward, cost)  # Implement this based on your context
            
            cumulative_profit += profit
            cumulative_reward += reward
            
            state = next_state  # Update the state for the next iteration
        
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

    # Plot cumulative profits
    plt.figure(figsize=(20, 6))
    plt.plot(range(1, num_episodes + 1), total_profits, label="Cumulative Profit", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Profit")
    plt.title("Cumulative Profit Over Episodes")
    plt.legend()
    plt.show()