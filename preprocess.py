import sklearn
import pandas as pd
import logging
import numpy as np
import sklearn.model_selection

data_matrix = pd.read_csv("flowering_time.csv")
print("Import Successful")

# Transform Target Matrix
target_matrix = pd.read_csv("data.csv")
target_matrix = (target_matrix>40).astype(int)
print("Binary Target Variable created")

# 0. Sample 600 rows randomly

# First, generate indexes of selected rows
indexes = np.random.randint(low = 0, high = len(target_matrix)-1, size = 600).tolist()
print("Random Indexes generated")

sampled_predictors = data_matrix.iloc[indexes,:]
sampled_targets = target_matrix.iloc[indexes,:]
print("Random Rows Selected based on generated indexes")

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
    sampled_predictors,
    sampled_targets,
    train_size = 2/3,
    random_state = 12345)

print("Train Test Split created")

# In order to import dataset use "from preprocess import X_train, X_test, Y_train, Y_test"