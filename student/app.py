import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
try:
    model_path = 'student_grade_model3.pkl'  # Updated to use student_grade_model3.pkl
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model file 'student_grade_model3.pkl' not found. Please ensure it’s in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Title of the app
st.title("Student Final Grade Prediction")

# Create input fields for features (using exact feature names from model training)
st.header("Enter Student Information")

G1 = st.number_input("Previous Grade 1 (G1)", min_value=0.0, max_value=20.0, value=10.0)
G2 = st.number_input("Previous Grade 2 (G2)", min_value=0.0, max_value=20.0, value=10.0)
failures = st.number_input("Number of Past Failures", min_value=0, max_value=10, value=0)
absences = st.number_input("Number of Absences", min_value=0, max_value=100, value=5)

# Create a button to make prediction
if st.button("Predict Final Grade"):
    try:
        # Prepare input data as a DataFrame with exact feature names and order from model training
        input_data = pd.DataFrame({
            'G1': [G1],
            'G2': [G2],
            'failures': [failures],
            'absences': [absences]  # Matches model training
        })

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Display the result
        st.success(f"Predicted Final Grade (G3): {prediction:.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# Add information about the model
st.markdown("""
This app predicts a student's final grade (G3) based on their previous grades, failures, and absences using a Multiple Linear Regression model.
""")