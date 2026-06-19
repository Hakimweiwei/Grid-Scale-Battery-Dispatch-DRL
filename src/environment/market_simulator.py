import yaml
from pathlib import Path

class MarketSimulator:
    """
    Simulates the market clearing process for Energy Arbitrage and FCAS.
    """
    def __init__(self, config_path: str = "src/config/reward_weights.yaml"):
        # Make it robust if file doesn't exist
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception:
            self.config = {
                'reward_scaling': {'scale_factor': 10000.0},
                'financials': {'include_fcas': True, 'include_arbitrage': True},
                'degradation': {'battery_capex_aud': 100000000.0, 'end_of_life_soh': 0.80},
                'penalties': {'soc_violation_weight': 10000.0, 'ramp_rate_violation_weight': 500.0},
                'shaping': {'duck_curve_bonus': 10.0}
            }
            
    def calculate_revenue(self, arb_power_mw: float, fcas_powers_mw: list, market_data: dict, duration_hours: float = 5/60):
        rrp = market_data.get('RRP', 0.0)
        rev_arb = arb_power_mw * rrp * duration_hours if self.config['financials'].get('include_arbitrage', True) else 0.0
            
        rev_fcas = 0.0
        fcas_keys = [
            'RAISE6SECRRP', 'LOWER6SECRRP', 
            'RAISE60SECRRP', 'LOWER60SECRRP', 
            'RAISE5MINRRP', 'LOWER5MINRRP'
        ]
        
        if self.config['financials'].get('include_fcas', True) and len(fcas_powers_mw) == 6:
            for i, key in enumerate(fcas_keys):
                fcas_price = market_data.get(key, 0.0)
                rev_fcas += fcas_powers_mw[i] * fcas_price * duration_hours
                
        return rev_arb, rev_fcas
        
    def calculate_degradation_cost(self, delta_soh: float):
        capex = self.config['degradation']['battery_capex_aud']
        end_of_life = self.config['degradation']['end_of_life_soh']
        usable_soh_range = 1.0 - end_of_life
        cost_per_soh_unit = capex / usable_soh_range
        return delta_soh * cost_per_soh_unit
        
    def calculate_penalties(self, soc_violation: float, ramp_violation: float):
        penalty_soc = soc_violation * self.config['penalties']['soc_violation_weight']
        penalty_ramp = ramp_violation * self.config['penalties']['ramp_rate_violation_weight']
        return penalty_soc + penalty_ramp
        
    def calculate_shaping_bonus(self, hour: int, soc: float):
        bonus = 0.0
        duck_bonus = self.config['shaping']['duck_curve_bonus']
        if hour == 15 and soc > 0.80:
            bonus += duck_bonus
        if hour == 10 and soc < 0.20:
            bonus += duck_bonus
        return bonus
