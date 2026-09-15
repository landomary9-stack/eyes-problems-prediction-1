from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "eye_problem_logistic_model.joblib"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model file not found. Please train the model first so 'eye_problem_logistic_model.joblib' is created in this folder.")
        st.stop()

    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as exc:  # pragma: no cover - UI-level error handling
        st.error(f"Failed to load the model: {exc}")
        st.stop()


def predict_eye_problem(model, age, sex, year, landmark, visual_acuity_re, visual_acuity_le, disability_type, screen_time):
    input_data = pd.DataFrame(
        [{
            "Age": age,
            "Sex": sex,
            "year": year,
            "Landmark / Village / Estate": landmark,
            "Visual Acuity (RE)": visual_acuity_re,
            "Visual Acuity (LE)": visual_acuity_le,
            "Disability Type": disability_type,
            "Screen time": screen_time,
        }]
    )

    probability = model.predict_proba(input_data)[0, 1]
    prediction = int(probability >= 0.50)

    return {
        "Prediction": "Eye Problem" if prediction == 1 else "No Eye Problem",
        "Probability": round(float(probability), 4),
    }


def main():
    st.set_page_config(page_title="Eye Problem Risk Predictor", layout="centered")
    st.title("Eye Problem Risk Prediction")
    st.caption("Machine learning app for eye problem assessment based on patient details.")

    model = load_model()

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=35)
            sex = st.selectbox("Sex", ["F", "M", "Other"])
            year = st.number_input("Year", min_value=2000, max_value=2100, value=2026)
            landmark = st.text_input("Landmark / Village / Estate", value="Miathene")

        with col2:
            visual_acuity_re = st.text_input("Visual Acuity (RE)", value="6/6")
            visual_acuity_le = st.text_input("Visual Acuity (LE)", value="6/6")
            disability_type = st.number_input("Disability Type", min_value=0, max_value=100, value=7)
            screen_time = st.number_input("Screen time (hours/day)", min_value=0.0, max_value=24.0, value=6.0, step=0.5)

        submitted = st.form_submit_button("Predict")

    if submitted:
        result = predict_eye_problem(
            model=model,
            age=age,
            sex=sex,
            year=year,
            landmark=landmark,
            visual_acuity_re=visual_acuity_re,
            visual_acuity_le=visual_acuity_le,
            disability_type=disability_type,
            screen_time=screen_time,
        )

        st.subheader("Prediction Result")
        st.metric(label="Classification", value=result["Prediction"])
        st.metric(label="Risk Probability", value=f"{result['Probability'] * 100:.2f}%")

        if result["Prediction"] == "Eye Problem":
            st.warning("This patient is predicted to have a potential eye problem risk.")
        else:
            st.success("This patient is predicted to have no eye problem risk.")


if __name__ == "__main__":
    main()
