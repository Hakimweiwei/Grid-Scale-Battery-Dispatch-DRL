import pandas as pd
from pathlib import Path

class AEMODataFeeder:
    """
    Feeds historical AEMO market data step-by-step into the Gymnasium environment.
    """
    def __init__(self, parquet_path: str):
        self.parquet_path = Path(parquet_path)
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.parquet_path}")
            
        self.df = pd.read_parquet(self.parquet_path, engine="pyarrow")
        if 'SETTLEMENTDATE_UTC' in self.df.columns:
            self.df = self.df.sort_values('SETTLEMENTDATE_UTC').reset_index(drop=True)
            
        self.length = len(self.df)
        
    def get_step_data(self, index: int) -> dict:
        if index >= self.length or index < 0:
            index = self.length - 1
            
        return self.df.iloc[index].to_dict()
    
    def get_length(self) -> int:
        return self.length
