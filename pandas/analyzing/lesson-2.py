# Get a quick overview by printing the first 10 rows of the DataFrame:
import pandas as pd

df = pd.read_csv("../dataset/diabetes.csv")

print(df.head())