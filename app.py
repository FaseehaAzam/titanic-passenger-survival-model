
import streamlit as st
import pandas as pd
import joblib

# Load Titanic model
saved_data = joblib.load("titanic_survival_model (1).pkl")
model = saved_data["model"]
encoders = saved_data["encoders"]
features = saved_data["feature_columns"]

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢"
)

st.title("🚢 Titanic Passenger Survival Prediction")
st.write("Enter passenger details to predict survival.")

pclass = st.selectbox(
    "Passenger Class",
    ["First", "Second", "Third"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

sibling = st.selectbox(
    "Number of Siblings",
    ["Zero", "One", "Two", "Three"]
)

embarked = st.selectbox(
    "Port of Embarkation",
    ["Cherbourg", "Queenstown", "Southampton"]
)

if st.button("Predict Survival"):
    input_values = {
        "PClass": pclass,
        "Gender": gender,
        "Sibling": sibling,
        "Embarked": embarked
    }

    input_data = pd.DataFrame([input_values], columns=features)

    for column in features:
        input_data[column] = encoders[column].transform(
            input_data[column]
        )

    prediction = model.predict(input_data)[0]
    result = encoders["Survived"].inverse_transform(
        [prediction]
    )[0]

    if result == "Yes":
        st.success("Prediction: The passenger is predicted to survive! 🎉")
    else:
        st.error("Prediction: The passenger is predicted not to survive.")

st.caption("This is a machine-learning educational project.")
