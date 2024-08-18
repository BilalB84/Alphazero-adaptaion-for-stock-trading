import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import gc
from MCTS_indicators import MCTSIndicators, MCTSNode  # Adjusted to use the MCTS with indicators
from tradinenvwithindicator import StockTradingEnvWithIndicators  # Adjusted to use the environment with indicators
from MCTS import MCTS, MCTSNode
from alphazeroagent import AlphaZeroNetwork


def discount_rewards(rewards, gamma=0.99):
    discounted = []
    cumulative = 0.0
    for reward in reversed(rewards):
        cumulative = reward + gamma * cumulative
        discounted.insert(0, cumulative)
    return discounted

def calculate_profit(reward, cost):
    return reward - cost

def train_indicator(model, env, mcts, num_iterations, batch_size, save_path="model.pth"):
    weight_decay = 1e-3  
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=weight_decay)
    
    total_profits = []  
    total_rewards = []  
    total_losses = []  
    
    for iteration in range(num_iterations):
        states, mcts_probs, rewards = [], [], []
        state = env.reset()
        done = False
        cumulative_profit = 0  
        cumulative_reward = 0  
        iteration_loss = 0  
        
        while not done:
            root = MCTSNode(state)
            mcts.run(root)
            
            action_probs = np.array([child.visit_count for child in root.children.values()])
            action_probs = action_probs / np.sum(action_probs)
            action = np.random.choice(len(action_probs), p=action_probs)
            
            states.append(state)
            mcts_probs.append(action_probs)
            state, reward, done, _ = env.step(action)
            
            reward = np.clip(reward, -1, 1)
            rewards.append(reward)
            
            cost = 0.1
            profit = calculate_profit(reward, cost)
            
            cumulative_profit += profit
            cumulative_reward += reward
        
        discounted_rewards = discount_rewards(rewards)
        
        states_tensor = torch.tensor(states, dtype=torch.float32)
        mcts_probs_tensor = torch.tensor(mcts_probs, dtype=torch.float32)
        rewards_tensor = torch.tensor(discounted_rewards, dtype=torch.float32)
        
        if len(states_tensor.shape) == 1:
            states_tensor = states_tensor.unsqueeze(0)

        for batch in range(0, len(states), batch_size):
            batch_states = states_tensor[batch:batch + batch_size]
            batch_mcts_probs = mcts_probs_tensor[batch:batch + batch_size]
            batch_rewards = rewards_tensor[batch:batch + batch_size]
            
            optimizer.zero_grad()
            policy, value = model(batch_states)
            
            value_loss = torch.mean((batch_rewards - value.squeeze())**2)
            policy_loss = -torch.mean(batch_mcts_probs * torch.log(policy + 1e-8))
            loss = value_loss + policy_loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.7)  # Try different values like 0.3 or 0.7
            optimizer.step()
            
            iteration_loss += loss.item()
        
        total_losses.append(iteration_loss)
        total_profits.append(cumulative_profit)
        total_rewards.append(cumulative_reward)

        print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {iteration_loss:.4f}, Profit: {cumulative_profit:.4f}, Reward: {cumulative_reward:.4f}")

        # Save model at certain checkpoints
        if (iteration + 1) % 1000 == 0:
            torch.save(model.state_dict(), f"{save_path}_checkpoint_{iteration + 1}.pth")
        
        # Free memory manually
        del states_tensor, mcts_probs_tensor, rewards_tensor
        gc.collect()
    
    avg_profit = np.mean(total_profits)
    avg_reward = np.mean(total_rewards)
    
    print(f"Average Profit: {avg_profit:.4f}")
    print(f"Average Reward: {avg_reward:.4f}")
    
    torch.save(model.state_dict(), save_path)
    
    return total_losses, total_profits, total_rewards

def plot_training_progress(total_losses, total_profits, total_rewards, num_iterations):
    # Plot for Training Loss
    plt.figure(figsize=(20, 6))
    plt.plot(range(1, num_iterations + 1), total_losses, label="Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Iterations")
    plt.legend()
    plt.show()

    # Plot for Cumulative Profit
    plt.figure(figsize=(20, 6))
    plt.plot(range(1, num_iterations + 1), total_profits, label="Cumulative Profit", color="orange")
    plt.xlabel("Iteration")
    plt.ylabel("Cumulative Profit")
    plt.title("Cumulative Profit Over Iterations")
    plt.legend()
    plt.show()

    # Plot for Cumulative Reward
    plt.figure(figsize=(20, 6))
    plt.plot(range(1, num_iterations + 1), total_rewards, label="Cumulative Reward", color="green")
    plt.xlabel("Iteration")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward Over Iterations")
    plt.legend()
    plt.show()

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

def train_stock_custom_Indicator(stock_file, year, train_months, input_size, hidden_size, action_size, num_iterations=100, batch_size=32, save_path="model.pth"):
    train_data = preprocess_stock_data(stock_file, year, train_months)
    
    train_env = StockTradingEnvWithIndicators(train_data)  # Use environment with custom indicators
    agent = AlphaZeroNetwork(input_size, hidden_size, action_size)
    mcts = MCTSIndicators(agent, c_puct=1.0, num_simulations=100)

    total_losses, total_profits, total_rewards = train_indicator(agent, train_env, mcts, num_iterations=num_iterations, batch_size=batch_size, save_path=save_path)
    
    plot_training_progress(total_losses, total_profits, total_rewards, num_iterations)

    return agent, train_env, mcts
