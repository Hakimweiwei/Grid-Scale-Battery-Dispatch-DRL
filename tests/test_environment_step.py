import pytest
import numpy as np
from src.environment.battery_env import BatteryDispatchEnv

class MockDataFeeder:
    def __init__(self, *args, **kwargs):
        self.length = 100
    
    def get_step_data(self, index):
        return {
            'RRP': 50.0,
            'hour_of_day': 12,
            'RAISE6SECRRP': 1.0,
            'LOWER6SECRRP': 0.5,
            'RAISE60SECRRP': 1.0,
            'LOWER60SECRRP': 0.5,
            'RAISE5MINRRP': 1.0,
            'LOWER5MINRRP': 0.5
        }
        
    def get_length(self):
        return self.length

def test_environment_initialization(monkeypatch):
    monkeypatch.setattr("src.environment.battery_env.AEMODataFeeder", MockDataFeeder)
    config = {'data_path': 'dummy.parquet', 'capacity_mwh': 100.0, 'power_mw': 50.0}
    env = BatteryDispatchEnv(config)
    obs, info = env.reset()
    assert obs.shape == (32,)
    assert info['soc'] == 0.5

def test_environment_step_random(monkeypatch):
    monkeypatch.setattr("src.environment.battery_env.AEMODataFeeder", MockDataFeeder)
    config = {'data_path': 'dummy.parquet', 'capacity_mwh': 100.0, 'power_mw': 50.0, 'episode_length': 10}
    env = BatteryDispatchEnv(config)
    env.reset()
    action = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (32,)
    assert not terminated
    assert isinstance(reward, float)
    assert 'rev_arb' in info

def test_action_masking(monkeypatch):
    monkeypatch.setattr("src.environment.battery_env.AEMODataFeeder", MockDataFeeder)
    config = {'data_path': 'dummy.parquet', 'capacity_mwh': 100.0, 'power_mw': 50.0}
    env = BatteryDispatchEnv(config)
    env.reset()
    action = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert info['rev_fcas'] == 0.0
    assert info['actual_arb_power'] > 0.0 
