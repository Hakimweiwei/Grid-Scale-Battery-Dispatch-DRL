import numpy as np
from loguru import logger

class BatteryModel:
    """
    Simulates the physical dynamics of a grid-scale Lithium-Ion battery.
    Includes State of Charge (SOC) tracking and State of Health (SOH) degradation.
    """
    
    def __init__(self, 
                 capacity_mwh: float = 100.0, 
                 max_power_mw: float = 50.0,
                 efficiency: float = 0.90,
                 initial_soc: float = 0.5,
                 min_soc: float = 0.1,
                 max_soc: float = 1.0):
        self.nominal_capacity_mwh = capacity_mwh
        self.nominal_power_mw = max_power_mw
        
        # Current degraded capacity
        self.max_energy_capacity = capacity_mwh
        self.max_power_capacity = max_power_mw
        
        self.efficiency = efficiency  # Round-trip efficiency
        self.charging_eff = np.sqrt(efficiency)
        self.discharging_eff = np.sqrt(efficiency)
        
        self.min_soc = min_soc
        self.max_soc = max_soc
        
        self.soc = initial_soc
        self.soh = 1.0  # 100% Health
        self.cycles_count = 0.0
        
        # State tracking
        self.current_power_mw = 0.0
        
    def step(self, action_power_mw: float, duration_hours: float = 5/60) -> float:
        """
        Applies a power action (MW) for a given duration.
        Negative power = Charging, Positive power = Discharging.
        Returns the actual power delivered/consumed, accounting for constraints.
        """
        # Constraint: Power cannot exceed max rating
        action_power_mw = np.clip(action_power_mw, -self.max_power_capacity, self.max_power_capacity)
        
        # Constraint: SOC limits
        energy_req = action_power_mw * duration_hours
        
        if action_power_mw < 0: # Charging
            energy_to_store = -energy_req * self.charging_eff
            max_can_store = (self.max_soc - self.soc) * self.max_energy_capacity
            
            if energy_to_store > max_can_store:
                energy_to_store = max_can_store
                actual_energy_req = -(energy_to_store / self.charging_eff)
                action_power_mw = actual_energy_req / duration_hours
                
            self.soc += energy_to_store / self.max_energy_capacity
            
        elif action_power_mw > 0: # Discharging
            energy_to_discharge = energy_req / self.discharging_eff
            max_can_discharge = (self.soc - self.min_soc) * self.max_energy_capacity
            
            if energy_to_discharge > max_can_discharge:
                energy_to_discharge = max_can_discharge
                actual_energy_req = energy_to_discharge * self.discharging_eff
                action_power_mw = actual_energy_req / duration_hours
                
            self.soc -= energy_to_discharge / self.max_energy_capacity
            
        else: # Idle
            pass
            
        # Update Cycles (simplified: full equivalent cycles)
        if action_power_mw != 0:
            # Absolute energy throughput in MWh
            throughput = abs(action_power_mw * duration_hours)
            # Cycle = half throughput / capacity (1 full cycle = 1 full charge + 1 full discharge)
            cycle_increment = throughput / (2 * self.nominal_capacity_mwh)
            self.cycles_count += cycle_increment
            
            # Apply Degradation
            self._apply_degradation(cycle_increment, duration_hours)
            
        self.current_power_mw = action_power_mw
        return action_power_mw
        
    def _apply_degradation(self, cycle_increment: float, duration_hours: float):
        """
        Applies cycle and calendar aging to SOH.
        """
        # Cycle aging: ~0.003% per cycle (so 300 cycles = 0.9% degradation)
        cycle_aging = cycle_increment * 0.00003
        
        # Calendar aging: ~1% per year
        years_passed = duration_hours / 8760
        calendar_aging = years_passed * 0.01
        
        self.soh -= (cycle_aging + calendar_aging)
        self.soh = max(0.0, self.soh)
        
        # Degrade capacity linearly with SOH
        self.max_energy_capacity = self.nominal_capacity_mwh * self.soh
        
    def get_state(self) -> dict:
        return {
            "soc": self.soc,
            "soh": self.soh,
            "max_capacity": self.max_energy_capacity,
            "cycles": self.cycles_count
        }

    def reset(self, initial_soc: float = 0.5):
        self.soc = initial_soc
        self.soh = 1.0
        self.max_energy_capacity = self.nominal_capacity_mwh
        self.cycles_count = 0.0
        self.current_power_mw = 0.0
