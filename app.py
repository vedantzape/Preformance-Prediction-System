import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# App title
st.title("Student Performance Prediction System")

st.write("Enter student details to predict whether the student will PASS or FAIL.")

# Input fields
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100)
internal_marks = st.number_input("Internal Marks", min_value=0, max_value=100)
study_hours = st.number_input("Study Hours per Day", min_value=0, max_value=24)
assignments = st.number_input("Assignments Completed", min_value=0, max_value=10)

# Predict button
if st.button("Predict Result"):

    # Create input array
    input_data = np.array([[attendance, internal_marks, study_hours, assignments]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    # Output result
    if prediction[0] == 1:
        st.success("Prediction: Student will PASS")
    else:
        st.error("Prediction: Student will FAIL")

# Footer
st.write("Built using Streamlit and Machine Learning")
