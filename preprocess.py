import sklearn
import pandas as pd

data_matrix = pd.read_csv("flowering_time.csv")

# Transform Target Matrix
target_matrix = pd.read_csv("data.csv")
target_matrix = (target_matrix>40).astype(int)

print("Import Successful")

# 0. Sample 600 rows randomly
sampled_rows = data_matrix.sample(n = 600)


# 1. Normal Naive Decision Tree