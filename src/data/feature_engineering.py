import pandas as pd
import numpy as np
from pathlib import Path

def engineer_features(input_path: Path, output_path: Path):
    """
    Creates the 32-dim features required for the RL Observation Space.
    """
    print(f"Engineering features for {input_path}...")
    df = pd.read_parquet(input_path, engine="pyarrow")
    
    # 1. Temporal (Cyclical) Encoding
    df['hour_of_day'] = df['SETTLEMENTDATE_UTC'].dt.hour + df['SETTLEMENTDATE_UTC'].dt.minute / 60.0
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24.0)
    
    df['month_of_year'] = df['SETTLEMENTDATE_UTC'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month_of_year'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month_of_year'] / 12.0)
    
    df['is_weekend'] = df['SETTLEMENTDATE_UTC'].dt.weekday >= 5
    df['is_weekend'] = df['is_weekend'].astype(float)
    
    # 2. Market Features
    df['net_demand'] = df['TOTALDEMAND'] - df['TOTALINTERMITTENTGENERATION']
    df['cumulative_price_ratio'] = df['RRP'].rolling(window=288, min_periods=1).sum() / 1359900.0
    
    # 3. Lags
    df['price_lag_1'] = df['RRP'].shift(1).fillna(df['RRP'])
    df['price_lag_12'] = df['RRP'].shift(12).fillna(df['RRP'])
    df['price_lag_288'] = df['RRP'].shift(288).fillna(df['RRP'])
    
    # 4. Volatility & Trend
    df['price_rolling_mean_1h'] = df['RRP'].rolling(window=12, min_periods=1).mean()
    df['price_rolling_std_1h'] = df['RRP'].rolling(window=12, min_periods=1).std().fillna(0)
    
    # 5. Forecast Error (Simulated for synthetic)
    df['demand_forecast_error'] = np.random.normal(0, 50, len(df))
    
    # Add ambient temperature (mock)
    df['ambient_temp'] = 25.0 + 10.0 * np.sin(np.pi * (df['hour_of_day'] - 8) / 12)
    
    df.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"Features engineered and saved to {output_path}")

def main():
    input_file = Path("data/processed/cleaned_AEMO.parquet")
    output_file = Path("data/processed/featured_AEMO.parquet")
    
    if not input_file.exists():
        raise FileNotFoundError(f"{input_file} not found.")
        
    engineer_features(input_file, output_file)

if __name__ == "__main__":
    main()
