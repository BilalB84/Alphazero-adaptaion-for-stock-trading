import pandas as pd
from stable_baselines3 import DQN, PPO, A2C
from tradingenv import StockTradingEnv
import gym
from Train import preprocess_stock_data

def test_model(stock_file, year, train_months, model_path, algo="DQN"):
    df = preprocess_stock_data(stock_file, year, train_months)
    env = StockTradingEnv(df)

    # Load the appropriate model based on the algorithm
    if algo == "DQN":
        model = DQN.load(model_path, env=env)
    elif algo == "PPO":
        model = PPO.load(model_path, env=env)
    elif algo == "A2C":
        model = A2C.load(model_path, env=env)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    obs = env.reset()
    results = []
    for _ in range(len(df)):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        results.append({
            'step': env.current_step,
            'balance': env.balance,
            'holdings': env.holdings,
            'net_worth': env.net_worth,
            'action': action,
            'reward': rewards
        })
        if done:
            break

    # Convert results to a DataFrame for easy inspection
    results_df = pd.DataFrame(results)
    return results_df