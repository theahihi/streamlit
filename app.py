import streamlit as st
import pickle
from sklearn.datasets import load_iris

iris = load_iris()

# Load the trained model
clf = pickle.load(open('iris_model.pkl', 'rb'))

# Sidebar for user input
st.sidebar.title('Iris Classifier')
sepal_length = st.sidebar.number_input('Sepal Length', min_value=4.0, max_value=8.0, value=5.0)
sepal_width = st.sidebar.number_input('Sepal Width', min_value=2.0, max_value=4.5, value=3.0)
petal_length = st.sidebar.number_input('Petal Length', min_value=1.0, max_value=7.0, value=4.0)
petal_width = st.sidebar.number_input('Petal Width', min_value=0.1, max_value=2.5, value=1.0)

# Make predictions
prediction = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])

# Display prediction
st.write('## Prediction:')
st.write(iris.target_names[prediction[0]])
