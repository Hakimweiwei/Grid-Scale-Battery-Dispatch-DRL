import pandas as pd
import numpy as np

def run_stress_test(base_data_path: str, output_path: str):
    """
    Creates a stress test dataset simulating a System Black event or extreme price spikes.
    """
    df = pd.read_parquet(base_data_path, engine="pyarrow")
    
    # Introduce extreme price spikes for a 24-hour period
    spike_start = len(df) // 2
    spike_end = spike_start + 288
    
    df.loc[spike_start:spike_end, 'RRP'] = np.random.uniform(10000, 16000, spike_end - spike_start + 1)
    df.loc[spike_start:spike_end, 'RAISE6SECRRP'] = np.random.uniform(500, 1000, spike_end - spike_start + 1)
    
    # Save stress test data
    df.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"Stress test data saved to {output_path}")

if __name__ == "__main__":
    run_stress_test("data/processed/test.parquet", "data/processed/stress_test.parquet")
