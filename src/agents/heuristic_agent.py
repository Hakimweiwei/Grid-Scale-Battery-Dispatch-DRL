import numpy as np

class HeuristicAgent:
    """
    A simple rule-based agent for Battery Dispatch.
    Rule: 
    - Charge (action = -1.0) if RRP < 0 (negative price)
    - Discharge (action = 1.0) if RRP > 150 (price spike)
    - Else Idle (action = 0.0)
    """
    def __init__(self, charge_threshold: float = 0.0, discharge_threshold: float = 150.0):
        self.charge_threshold = charge_threshold
        self.discharge_threshold = discharge_threshold
        
    def predict(self, observation: np.ndarray, deterministic: bool = True):
        rrp = observation[0] 
        action = np.zeros(7, dtype=np.float32)
                  
        if rrp < self.charge_threshold:
            action[0] = -1.0
        elif rrp > self.discharge_threshold:
            action[0] = 1.0
        else:
            action[0] = 0.0
            
        action[1] = 0.5
        action[2] = 0.5
        
        return action, None
