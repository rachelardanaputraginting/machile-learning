import pandas as pd

df = pd.read_csv('../dataset/data.csv')

new_df = df.dropna()

print(new_df.to_string())