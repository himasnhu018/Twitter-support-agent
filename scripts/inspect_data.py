from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/raw")


def find_data_files():
    """Find CSV files inside data/raw."""
    files = list(DATA_DIR.glob("*.csv"))

    if not files:
        print("❌ No CSV files found in data/raw/")
        print(f"Expected location: {DATA_DIR.resolve()}")
        return []

    return files


def inspect_file(file_path: Path):
    print("\n" + "=" * 70)
    print(f"FILE: {file_path.name}")
    print("=" * 70)

    # Read only a small sample first
    df = pd.read_csv(file_path, nrows=1000)

    print(f"\nColumns ({len(df.columns)}):")
    for column in df.columns:
        print(f"  - {column}")

    print(f"\nSample shape: {df.shape}")

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    print("\nMissing values:")
    missing = df.isnull().sum()

    for column, count in missing.items():
        if count > 0:
            print(f"  {column}: {count}")


def main():
    print("=" * 70)
    print("HIVER SUPPORT AGENT - DATA INSPECTION")
    print("=" * 70)

    files = find_data_files()

    if not files:
        return

    print(f"\nFound {len(files)} CSV file(s):")

    for file in files:
        print(f"  - {file}")

    for file in files:
        inspect_file(file)


if __name__ == "__main__":
    main()