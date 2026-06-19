import pandas as pd
import numpy as np
from stable_baselines3 import SAC, PPO
from src.environment.battery_env import BatteryDispatchEnv

class Backtester:
    """
    Evaluates the trained agent against the test Parquet dataset.
    Compares the agent's performance with heuristics and Perfect Foresight.
    """
    def __init__(self, model_path: str, data_path: str, algo: str = "sac"):
        self.config = {
            'data_path': data_path,
            'capacity_mwh': 100.0,
            'power_mw': 50.0,
            'reward_config_path': 'src/config/reward_weights.yaml',
            'episode_length': 288 * 30 # ~1 month
        }
        self.env = BatteryDispatchEnv(self.config)
        
        if algo == "sac":
            self.model = SAC.load(model_path)
        else:
            self.model = PPO.load(model_path)
            
    def run_backtest(self):
        obs, info = self.env.reset()
        done = False
        
        results = []
        
        while not done:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            results.append({
                'step': info['step'],
                'soc': info['soc'],
                'soh': info['soh'],
                'actual_arb_power': info['actual_arb_power'],
                'rev_arb': info['rev_arb'],
                'rev_fcas': info['rev_fcas'],
                'cost_deg': info['cost_deg'],
                'profit': info['rev_arb'] + info['rev_fcas'] - info['cost_deg']
            })
            
            done = terminated or truncated
            
        df_results = pd.DataFrame(results)
        df_results.to_csv("reports/backtest_results.csv", index=False)
        print(f"Total Profit: ${df_results['profit'].sum():.2f}")
        return df_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/processed/test.parquet")
    parser.add_argument("--algo", type=str, default="sac")
    args = parser.parse_args()
    
    bt = Backtester(args.model, args.data, args.algo)
    bt.run_backtest()
