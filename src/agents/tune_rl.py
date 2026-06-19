import optuna
import argparse
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from src.environment.battery_env import BatteryDispatchEnv
from src.agents.rl_agent import BatteryFeatureExtractor

def make_env(config):
    def _init():
        env = BatteryDispatchEnv(config)
        env = Monitor(env)
        return env
    return _init

def optimize_agent(trial, algo="sac"):
    config = {
        'data_path': 'data/processed/train.parquet',
        'capacity_mwh': 100.0,
        'power_mw': 50.0,
        'reward_config_path': 'src/config/reward_weights.yaml',
        'episode_length': 1000
    }
    
    n_envs = 4
    env = SubprocVecEnv([make_env(config) for _ in range(n_envs)], start_method='spawn')
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    
    policy_kwargs = dict(
        features_extractor_class=BatteryFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )
    
    if algo == "sac":
        tau = trial.suggest_float("tau", 0.001, 0.05)
        model = SAC("MlpPolicy", env, learning_rate=learning_rate, batch_size=batch_size, 
                    gamma=gamma, tau=tau, policy_kwargs=policy_kwargs, verbose=0)
    else:
        ent_coef = trial.suggest_float("ent_coef", 0.0, 0.01)
        model = PPO("MlpPolicy", env, learning_rate=learning_rate, batch_size=batch_size, 
                    gamma=gamma, ent_coef=ent_coef, policy_kwargs=policy_kwargs, verbose=0)
                    
    eval_env = SubprocVecEnv([make_env(config) for _ in range(1)], start_method='spawn')
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    eval_callback = EvalCallback(eval_env, best_model_save_path=f'./models/optuna/{algo}/',
                                 log_path=f'./logs/{algo}/', eval_freq=2000,
                                 deterministic=True, render=False)
                                 
    try:
        model.learn(total_timesteps=10000, callback=eval_callback)
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('-inf')
        
    return eval_callback.best_mean_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="sac")
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optimize_agent(trial, args.algo), n_trials=args.trials)
    
    print("Best hyperparameters: ", study.best_params)
