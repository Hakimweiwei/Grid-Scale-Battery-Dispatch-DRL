import pytest
from src.environment.battery_model import BatteryModel

def test_battery_initialization():
    battery = BatteryModel(capacity_mwh=100.0, max_power_mw=50.0, initial_soc=0.5)
    assert battery.soc == 0.5
    assert battery.soh == 1.0
    assert battery.max_energy_capacity == 100.0

def test_charging_limits():
    battery = BatteryModel(capacity_mwh=100.0, max_power_mw=50.0, initial_soc=0.9, max_soc=1.0, efficiency=1.0)
    actual_power = battery.step(action_power_mw=-50.0, duration_hours=1.0)
    assert actual_power == -10.0 # Clipped to 10 MW
    assert battery.soc == 1.0

def test_discharging_limits():
    battery = BatteryModel(capacity_mwh=100.0, max_power_mw=50.0, initial_soc=0.2, min_soc=0.1, efficiency=1.0)
    actual_power = battery.step(action_power_mw=50.0, duration_hours=1.0)
    assert actual_power == 10.0 # Clipped to 10 MW
    assert battery.soc == 0.1

def test_efficiency_losses():
    # Efficiency is 0.81 (0.9 in, 0.9 out)
    battery = BatteryModel(capacity_mwh=100.0, max_power_mw=50.0, initial_soc=0.5, efficiency=0.81)
    # Charge with 10 MWh from grid
    battery.step(action_power_mw=-10.0, duration_hours=1.0)
    assert battery.soc == 0.59
    # Discharge to get 10 MWh to grid
    battery.step(action_power_mw=10.0, duration_hours=1.0)
    assert battery.soc == pytest.approx(0.59 - 0.111111, rel=1e-3)

def test_degradation_monotonically_decreasing():
    battery = BatteryModel(capacity_mwh=100.0)
    initial_soh = battery.soh
    battery.step(action_power_mw=50.0, duration_hours=1.0) # Discharge
    assert battery.soh < initial_soh
    battery.step(action_power_mw=-50.0, duration_hours=1.0) # Charge
    assert battery.soh < initial_soh
