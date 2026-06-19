import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class BatteryFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor for Battery Dispatch State Space.
    Uses LayerNorm and ELU for stable training with volatile market data.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, features_dim),
            nn.LayerNorm(features_dim),
            nn.ELU()
        )
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)
