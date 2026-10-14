import pandas as pd

data = pd.load_parquet('statcast_2025.parquet')

print(data.columns())