from sklearn.ensemble import RandomForestClassifier
from preprocess import X_train, X_test, Y_train, Y_test
from sklearn.metrics import accuracy_score

from sklearn import random

random.seed(42)
randomForest = RandomForestClassifier(
    n_estimators=100, # To Play around with
    criterion="entropy"
)

randomForest.fit(X_train, Y_train.to_numpy().ravel())

predictions = randomForest.predict(X_test)

accuracy = accuracy_score(Y_test, predictions)
print(f'Accuracy: {accuracy}')