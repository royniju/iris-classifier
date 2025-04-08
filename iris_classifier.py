import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled, y_train)

decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)

random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, y_train)

# Accuracy
print("Model Accuracies:")
print(f"Logistic Regression: {accuracy_score(y_test, logistic_model.predict(X_test_scaled)):.2f}")
print(f"Decision Tree: {accuracy_score(y_test, decision_tree_model.predict(X_test)):.2f}")
print(f"Random Forest: {accuracy_score(y_test, random_forest_model.predict(X_test)):.2f}")

# Sample prediction
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
sample_scaled = scaler.transform(sample)

print("\nSample Prediction:")
print(f"Logistic Regression: {target_names[logistic_model.predict(sample_scaled)[0]]}")
print(f"Decision Tree: {target_names[decision_tree_model.predict(sample)[0]]}")
print(f"Random Forest: {target_names[random_forest_model.predict(sample)[0]]}")
