from sklearn.tree import DecisionTreeClassifier
from preprocess import X_train, X_test, Y_train, Y_test
from sklearn.metrics import accuracy_score

basicTreeClassifier = DecisionTreeClassifier(
    criterion = "entropy",
    splitter = "best",
    #max_depth = 10,
    min_samples_split = 2,
    random_state=42,
    max_depth=10
)

basicTreeClassifier.fit(X=X_train, y=Y_train)

y_pred = basicTreeClassifier.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)
print(f'Accuracy: {accuracy}')