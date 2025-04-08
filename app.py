import streamlit as st
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

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled, y_train)
logistic_accuracy = accuracy_score(y_test, logistic_model.predict(X_test_scaled))

decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)
decision_tree_accuracy = accuracy_score(y_test, decision_tree_model.predict(X_test))

random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, y_train)
random_forest_accuracy = accuracy_score(y_test, random_forest_model.predict(X_test))

# Streamlit UI
st.title("🌸 Iris Classifier with Streamlit")

model_choice = st.selectbox("Choose a model:", 
                            ["Logistic Regression", "Decision Tree", "Random Forest"])

st.header("Input Features")
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
input_scaled = scaler.transform(input_data)

st.subheader("Prediction")
if st.button("Predict"):
    if model_choice == "Logistic Regression":
        pred = logistic_model.predict(input_scaled)[0]
        st.success(f"Prediction: {target_names[pred]}")
    elif model_choice == "Decision Tree":
        pred = decision_tree_model.predict(input_data)[0]
        st.success(f"Prediction: {target_names[pred]}")
    else:
        pred = random_forest_model.predict(input_data)[0]
        st.success(f"Prediction: {target_names[pred]}")

st.subheader("Model Accuracy")
if model_choice == "Logistic Regression":
    st.info(f"Accuracy: {logistic_accuracy:.2f}")
elif model_choice == "Decision Tree":
    st.info(f"Accuracy: {decision_tree_accuracy:.2f}")
else:
    st.info(f"Accuracy: {random_forest_accuracy:.2f}")
