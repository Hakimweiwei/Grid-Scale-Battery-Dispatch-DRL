import os
import argparse
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from src.environment.battery_env import BatteryDispatchEnv
from src.agents.rl_agent import BatteryFeatureExtractor
import wandb
from wandb.integration.sb3 import WandbCallback

def make_env(config, rank):
    def _init():
        env = BatteryDispatchEnv(config)
        env = Monitor(env) # For SB3 statistics
        return env
    return _init

import yaml

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train(algo="sac", n_envs=4, total_timesteps=None):
    state_config = load_yaml('src/config/state_space_config.yaml')
    hyperparams = load_yaml('src/config/hyperparameters.yaml')
    
    # Merge for env
    config = state_config['battery']
    config.update(state_config['env'])
    config.update(state_config['paths'])
    
    if total_timesteps is None:
        total_timesteps = hyperparams['common']['total_timesteps']
        
    hp = hyperparams.get(algo.lower(), {})
    common_hp = hyperparams.get('common', {})
    
    os.makedirs(f"models/{algo}", exist_ok=True)
    os.makedirs(f"models/scalers", exist_ok=True)
    
    env = DummyVecEnv([make_env(config, i) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    wandb.init(project="grid_battery_drl", sync_tensorboard=True, monitor_gym=True)
    
    policy_kwargs = dict(
        features_extractor_class=BatteryFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=common_hp.get('features_dim', 256)),
    )
    
    if algo.lower() == "sac":
        model = SAC("MlpPolicy", env, device="auto", policy_kwargs=policy_kwargs, 
                    tensorboard_log="runs/", 
                    batch_size=hp.get('batch_size', 512), 
                    buffer_size=hp.get('buffer_size', 500000), 
                    gradient_steps=hp.get('gradient_steps', 2),
                    learning_rate=hp.get('learning_rate', 0.0003))
    else:
        model = PPO("MlpPolicy", env, device="auto", policy_kwargs=policy_kwargs,
                    tensorboard_log="runs/", 
                    batch_size=hp.get('batch_size', 512),
                    n_steps=hp.get('n_steps', 2048),
                    learning_rate=hp.get('learning_rate', 0.0003))
        
    checkpoint_callback = CheckpointCallback(save_freq=max(1000, total_timesteps // 10), save_path=f'models/{algo}/', name_prefix='rl_model')
    wandb_callback = WandbCallback(gradient_save_freq=1000, model_save_path=f"models/{algo}/", verbose=2)
    
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, wandb_callback])
    
    model.save(f"models/{algo}/best_model")
    env.save(f"models/scalers/vecnormalize_{algo}.pkl")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="sac", choices=["sac", "ppo"])
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--envs", type=int, default=4)
    args = parser.parse_args()
    
    train(algo=args.algo, total_timesteps=args.steps, n_envs=args.envs)
