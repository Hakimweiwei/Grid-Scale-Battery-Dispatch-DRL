import pyomo.environ as pyo
import pandas as pd
import numpy as np
from loguru import logger

class OracleMILP:
    """
    Perfect foresight optimization using Mixed Integer Linear Programming (MILP).
    Uses Pyomo and HiGHS solver to find the absolute maximum profit possible.
    """
    def __init__(self, data_path: str, battery_config: dict):
        self.data_path = data_path
        self.config = battery_config
        self.df = pd.read_parquet(self.data_path, engine="pyarrow").head(288 * 7) # 1 week
        
    def solve(self):
        logger.info(f"Solving Oracle MILP for {len(self.df)} intervals (1 week)")
        
        solver = pyo.SolverFactory('appsi_highs')
        if not solver.available():
            logger.warning("HiGHS not available. Trying glpk...")
            solver = pyo.SolverFactory('glpk')
            if not solver.available():
                logger.warning("No solver available! Install highs or glpk.")
                return 0.0
            
        model = pyo.ConcreteModel()
        T = len(self.df)
        model.T = pyo.RangeSet(0, T - 1)
        
        prices = self.df['RRP'].values
        dt = 5/60 # hours
        
        max_p = self.config.get('power_mw', 50.0)
        max_e = self.config.get('capacity_mwh', 100.0)
        eff = self.config.get('efficiency', 0.90)
        ch_eff = np.sqrt(eff)
        dis_eff = np.sqrt(eff)
        
        model.P_ch = pyo.Var(model.T, bounds=(0, max_p))
        model.P_dis = pyo.Var(model.T, bounds=(0, max_p))
        model.SOC = pyo.Var(model.T, bounds=(0.1, 1.0))
        
        def obj_rule(m):
            return sum((m.P_dis[t] - m.P_ch[t]) * prices[t] * dt for t in m.T)
        model.Profit = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
        
        def soc_rule(m, t):
            if t == 0:
                return m.SOC[t] == 0.5 + (m.P_ch[t] * ch_eff * dt / max_e) - (m.P_dis[t] * dt / (dis_eff * max_e))
            else:
                return m.SOC[t] == m.SOC[t-1] + (m.P_ch[t] * ch_eff * dt / max_e) - (m.P_dis[t] * dt / (dis_eff * max_e))
        model.soc_con = pyo.Constraint(model.T, rule=soc_rule)
        
        results = solver.solve(model, tee=False)
        profit = pyo.value(model.Profit)
        
        logger.info(f"Oracle Optimal Profit (1 week): {profit:.2f} AUD")
        return profit

if __name__ == '__main__':
    oracle = OracleMILP('data/processed/test.parquet', {'power_mw': 50.0, 'capacity_mwh': 100.0, 'efficiency': 0.90})
    oracle.solve()