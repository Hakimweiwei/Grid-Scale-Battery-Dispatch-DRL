import pandas as pd
from pathlib import Path

def clean_data(input_path: Path, output_path: Path):
    """
    Cleans data, converts timezone to UTC, and handles missing values.
    """
    print(f"Cleaning {input_path}...")
    df = pd.read_parquet(input_path, engine="pyarrow")
    
    # Convert SETTLEMENTDATE to datetime if it's not already
    df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
    
    # Real AEMO data is in Brisbane timezone (no DST)
    # We localize and convert to UTC
    try:
        df['SETTLEMENTDATE'] = df['SETTLEMENTDATE'].dt.tz_localize('Australia/Brisbane').dt.tz_convert('UTC')
    except TypeError:
        # Already localized
        df['SETTLEMENTDATE'] = df['SETTLEMENTDATE'].dt.tz_convert('UTC')
        
    df.rename(columns={'SETTLEMENTDATE': 'SETTLEMENTDATE_UTC'}, inplace=True)
    
    # Handle missing values (forward fill then backward fill)
    df = df.ffill().bfill()
    
    df.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"Cleaned data saved to {output_path}")

def main():
    input_file = Path("data/processed/parsed_AEMO.parquet")
    output_file = Path("data/processed/cleaned_AEMO.parquet")
    
    if not input_file.exists():
        raise FileNotFoundError(f"{input_file} not found.")
        
    clean_data(input_file, output_file)

if __name__ == "__main__":
    main()
