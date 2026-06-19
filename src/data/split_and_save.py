import pandas as pd
from pathlib import Path

def split_and_save(input_path: Path):
    """
    Splits the data temporally into train (80%) and test (20%) and saves as Parquet.
    """
    print(f"Splitting data from {input_path}...")
    df = pd.read_parquet(input_path, engine="pyarrow")
    
    df = df.sort_values('SETTLEMENTDATE_UTC').reset_index(drop=True)
    
    train_size = int(len(df) * 0.8)
    
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    train_df.to_parquet("data/processed/train.parquet", engine="pyarrow", index=False)
    test_df.to_parquet("data/processed/test.parquet", engine="pyarrow", index=False)
    
    print(f"Saved {len(train_df)} rows to train.parquet")
    print(f"Saved {len(test_df)} rows to test.parquet")

def main():
    input_file = Path("data/processed/featured_AEMO.parquet")
    
    if not input_file.exists():
        raise FileNotFoundError(f"{input_file} not found.")
        
    split_and_save(input_file)

if __name__ == "__main__":
    main()
