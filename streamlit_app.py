import streamlit as st
import pandas as pd
import joblib

# =========================================
# Load Trained Model
# =========================================
model = joblib.load("models/final_catboost_model.pkl")

# =========================================
# Streamlit Page Configuration
# =========================================
st.set_page_config(
    page_title="Healthcare Cost Prediction",
    page_icon="🏥",
    layout="centered"
)

# =========================================
# Title and Description
# =========================================
st.title("🏥 Healthcare Cost Prediction System")

st.markdown("""
Predict medical insurance charges using Machine Learning.

Enter patient details below to estimate the healthcare insurance cost.
""")

# =========================================
# User Inputs
# =========================================

st.subheader("Enter Patient Details")

age = st.slider(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0,
    step=0.1
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    [
        "southwest",
        "southeast",
        "northwest",
        "northeast"
    ]
)

# =========================================
# Feature Engineering Functions
# =========================================

def bmi_category(bmi):

    if bmi < 18.5:
        return "Underweight"

    elif bmi < 25:
        return "Normal"

    elif bmi < 30:
        return "Overweight"

    else:
        return "Obese"


def age_group(age):

    if age < 30:
        return "Young"

    elif age <= 50:
        return "Adult"

    else:
        return "Senior"

# =========================================
# Internal Feature Engineering
# =========================================

smoker_numeric = 1 if smoker == "yes" else 0

risk_score = (
    bmi * 0.3 +
    age * 0.2 +
    smoker_numeric * 50
)

bmi_smoker_interaction = bmi * smoker_numeric

# =========================================
# Data For Model Prediction
# =========================================

input_df = pd.DataFrame({

    "age": [age],

    "sex": [sex],

    "bmi": [bmi],

    "children": [children],

    "smoker": [smoker],

    "region": [region],

    "bmi_category": [bmi_category(bmi)],

    "age_group": [age_group(age)],

    "smoker_numeric": [smoker_numeric],

    "risk_score": [risk_score],

    "bmi_smoker_interaction": [
        bmi_smoker_interaction
    ]

})

# =========================================
# Clean Display Data For Users
# =========================================

display_df = pd.DataFrame({

    "Age": [age],

    "Sex": [sex],

    "BMI": [bmi],

    "Children": [children],

    "Smoker": [smoker],

    "Region": [region]

})

# =========================================
# Display User Inputs
# =========================================

st.subheader("Patient Information")

st.dataframe(
    display_df,
    use_container_width=True
)

# =========================================
# Prediction Section
# =========================================

if st.button("Predict Healthcare Cost"):

    prediction = model.predict(input_df)

    predicted_cost = prediction[0]

    st.subheader("Prediction Result")

    st.success(
        f"Estimated Healthcare Cost: ${predicted_cost:,.2f}"
    )

    # Risk Interpretation
    if predicted_cost < 10000:

        st.info(
            "Risk Level: Low Healthcare Risk"
        )

    elif predicted_cost < 30000:

        st.warning(
            "Risk Level: Moderate Healthcare Risk"
        )

    else:

        st.error(
            "Risk Level: High Healthcare Risk"
        )

# =========================================
# Footer
# =========================================

# st.markdown("---")

# st.markdown(
#     "Developed using Machine Learning, CatBoost, and Streamlit"
# )
