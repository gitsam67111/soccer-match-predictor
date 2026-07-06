from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
csv_path = BASE_DIR / "data" / "raw" / "E0.csv"

df = pd.read_csv(csv_path)

print(df.head())