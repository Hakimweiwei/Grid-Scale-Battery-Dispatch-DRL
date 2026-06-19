import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from src.environment.battery_model import BatteryModel
from src.environment.data_feeder import AEMODataFeeder
from src.environment.market_simulator import MarketSimulator

class BatteryDispatchEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        self.data_feeder = AEMODataFeeder(config.get('data_path', 'data/processed/train.parquet'))
        self.market_sim = MarketSimulator(config.get('reward_config_path', 'src/config/reward_weights.yaml'))
        
        self.battery = BatteryModel(
            capacity_mwh=config.get('capacity_mwh', 100.0),
            max_power_mw=config.get('power_mw', 50.0),
            efficiency=config.get('efficiency', 0.90),
            initial_soc=config.get('initial_soc', 0.5)
        )
        
        self.episode_length = config.get('episode_length', 105120)
        self.current_step_idx = 0
        self.start_idx = 0
        self.steps_taken = 0
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32
        )
        
        # SB3 requires flat box for continuous action
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        max_start = max(0, self.data_feeder.get_length() - self.episode_length - 1)
        if max_start > 0:
            self.start_idx = self.np_random.integers(0, max_start)
        else:
            self.start_idx = 0
            
        self.current_step_idx = self.start_idx
        self.steps_taken = 0
        
        self.battery.reset(initial_soc=self.config.get('initial_soc', 0.5))
        return self._get_obs(), self._get_info()
        
    def step(self, action):
        market_data = self.data_feeder.get_step_data(self.current_step_idx)
        
        arb_power, fcas_powers, soc_violation = self._apply_constraints(action)
        
        prev_soh = self.battery.soh
        actual_arb_power = self.battery.step(arb_power, duration_hours=5/60)
        delta_soh = prev_soh - self.battery.soh
        
        rev_arb, rev_fcas = self.market_sim.calculate_revenue(actual_arb_power, fcas_powers, market_data, 5/60)
        cost_deg = self.market_sim.calculate_degradation_cost(delta_soh)
        penalty = self.market_sim.calculate_penalties(soc_violation=soc_violation, ramp_violation=0.0)
        
        hour = market_data.get('hour_of_day', 0)
        if hour == 0 and 'SETTLEMENTDATE_UTC' in market_data:
            hour = pd.to_datetime(market_data['SETTLEMENTDATE_UTC']).hour
            
        bonus = self.market_sim.calculate_shaping_bonus(hour, self.battery.soc)
        
        raw_reward = (rev_arb + rev_fcas) - cost_deg - penalty + bonus
        normalized_reward = raw_reward / self.market_sim.config['reward_scaling']['scale_factor']
        
        terminated = self.battery.soh < self.market_sim.config['degradation']['end_of_life_soh']
        
        self.steps_taken += 1
        self.current_step_idx += 1
        
        truncated = self.steps_taken >= self.episode_length or self.current_step_idx >= self.data_feeder.get_length()
        
        info = self._get_info()
        info.update({
            "rev_arb": rev_arb,
            "rev_fcas": rev_fcas,
            "cost_deg": cost_deg,
            "penalty": penalty,
            "raw_reward": raw_reward,
            "actual_arb_power": actual_arb_power
        })
        
        return self._get_obs(), normalized_reward, terminated, truncated, info

    def _apply_constraints(self, action):
        max_power = self.battery.max_power_capacity
        raw_arb = float(action[0])
        arb_power = raw_arb * max_power
        
        energy_req = arb_power * (5/60)
        soc_violation = 0.0
        
        if arb_power < 0: # Charge
            stored = -energy_req * self.battery.charging_eff
            proposed_soc = self.battery.soc + (stored / self.battery.max_energy_capacity)
            if proposed_soc > self.battery.max_soc:
                soc_violation = proposed_soc - self.battery.max_soc
        elif arb_power > 0: # Discharge
            extracted = energy_req / self.battery.discharging_eff
            proposed_soc = self.battery.soc - (extracted / self.battery.max_energy_capacity)
            if proposed_soc < self.battery.min_soc:
                soc_violation = self.battery.min_soc - proposed_soc
                
        available_for_fcas = max_power - abs(arb_power)
        fcas_powers = []
        remaining_capacity = available_for_fcas
        
        for i in range(6):
            requested = float(action[i + 1]) * remaining_capacity
            actual = min(requested, remaining_capacity)
            fcas_powers.append(actual)
            remaining_capacity -= actual
            
        return arb_power, fcas_powers, soc_violation

    def _get_obs(self):
        market_data = self.data_feeder.get_step_data(self.current_step_idx)
        obs = np.zeros(32, dtype=np.float32)
        
        features_to_extract = [
            'RRP', 'cumulative_price_ratio', 'net_demand', 'NETINTERCHANGE',
            'RAISE6SECRRP', 'LOWER6SECRRP', 'RAISE60SECRRP', 'LOWER60SECRRP', 'RAISE5MINRRP', 'LOWER5MINRRP',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'is_weekend',
            'price_lag_1', 'price_lag_12', 'price_lag_288',
            'price_rolling_mean_1h', 'price_rolling_std_1h', 'demand_forecast_error'
        ]
        
        for i, key in enumerate(features_to_extract):
            if i < 28:
                obs[i] = float(market_data.get(key, 0.0))
                
        obs[28] = self.battery.soc
        obs[29] = self.battery.soh
        obs[30] = self.battery.max_power_capacity
        obs[31] = self.battery.cycles_count
        
        return obs
        
    def _get_info(self):
        return {
            "soc": self.battery.soc,
            "soh": self.battery.soh,
            "step": self.current_step_idx
        }
