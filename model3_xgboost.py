import xgboost
from preprocess import X_train, X_test, Y_train, Y_test
from sklearn.metrics import accuracy_score

XGTree = xgboost.XGBClassifier(
    tree_method = "hist",
    early_stopping_rounds = 3,
    random_state = 420,
    
)

XGTree.fit(X_train, Y_train)

predictions = XGTree.predict(X_test)

accuracy = accuracy_score(Y_test, predictions)
print(f'Accuracy: {accuracy}')