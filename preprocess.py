import sklearn
import pandas as pd
import logging
import numpy as np
import sklearn.model_selection
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

data_matrix = pd.read_csv("data.csv")
print("Import Successful")

# Transform Target Matrix
target_matrix = pd.read_csv("flowering_time.csv")
target_matrix = (target_matrix>40).astype(int)
print("Binary Target Variable created")

# 0. Sample 600 rows randomly
np.random.seed(420)

# First, generate indexes of selected rows
indexes = np.random.randint(low = 0, high = len(target_matrix)-1, size = 600).tolist()
print("Random Indexes generated")

sampled_predictors = data_matrix.iloc[indexes,:]
sampled_targets = target_matrix.iloc[indexes,:]
print("Random Rows Selected based on generated indexes")


# Identify categorical and continuous columns
categorical_columns = [categorical for categorical in list(sampled_predictors.columns) if categorical[0] == "g"]
continuous_columns = [categorical for categorical in list(sampled_predictors.columns) if categorical[0] != "g"]  # Replace with your actual continuous column names

# Define transformers for categorical and continuous columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('cont', StandardScaler(), continuous_columns)
    ])

sampled_predictors = preprocessor.fit_transform(sampled_predictors)

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
    sampled_predictors,
    sampled_targets,
    train_size = 2/3,
    random_state = 12345)


print("Train Test Split created")

# In order to import dataset use "from preprocess import X_train, X_test, Y_train, Y_test"