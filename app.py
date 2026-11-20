import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------------------------------------------------------
# Electric Vehicle High-Efficiency Prediction - Streamlit App
# ---------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "ev_efficiency_model.pkl"
COLUMNS_PATH = APP_DIR / "model_columns.pkl"

st.set_page_config(
    page_title="EV High-Efficiency Prediction",
    page_icon="🔋",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Load the trained preprocessing pipeline with the selected model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "ev_efficiency_model.pkl was not found. "
            "Run the training script first and keep the "
            "generated model file in the same folder as app.py."
        )

    if not COLUMNS_PATH.exists():
        raise FileNotFoundError(
            "model_columns.pkl was not found. Keep it in the same folder as app.py."
        )

    return joblib.load(MODEL_PATH), joblib.load(COLUMNS_PATH)


st.title("EV High-Efficiency Prediction")
st.write(
    "Enter the electric vehicle specifications below. The trained machine learning "
    "model will predict whether the vehicle qualifies as a High-Efficiency EV "
    "(Efficiency Score > 5.0 km/kWh)."
)

try:
    model, feature_columns = load_model()
except Exception as error:
    st.error(str(error))
    st.stop()

st.info(
    "The prediction is an estimate based on the vehicle specifications entered. "
    "Use it as supporting information when evaluating EV performance."
)

with st.form("ev_prediction_form"):
    st.subheader("Vehicle Specifications")
    col1, col2 = st.columns(2)

    with col1:
        manufacturer = st.text_input("Manufacturer", value="BYD", placeholder="e.g., Tesla, BYD, BMW")
        model_name = st.text_input("Model", value="Model S", placeholder="e.g., Model 3, EV6, ID.4")
        vehicle_type = st.selectbox("Vehicle Type", ["Sedan", "SUV", "Hatchback", "Coupe", "Truck"])
        drive_type = st.selectbox("Drive Type", ["AWD", "FWD", "RWD"])
        fuel_type = st.selectbox("Fuel Type", ["Electric", "Hybrid"])
        fast_charging = st.selectbox("Fast Charging Support", ["True", "False"])
        country = st.text_input("Country", value="China", placeholder="e.g., China, Norway, UK, Japan")
        release_year = st.number_input("Release Year", min_value=2010, max_value=2026, value=2024, step=1)
        seats = st.number_input("Number of Seats", min_value=2, max_value=9, value=5, step=1)

    with col2:
        battery_kwh = st.number_input("Battery Capacity (kWh)", min_value=10.0, value=75.0, step=5.0)
        range_km = st.number_input("Estimated Range (km)", min_value=50.0, value=420.0, step=10.0)
        charging_time_hr = st.number_input("Charging Time (hours)", min_value=0.1, value=1.0, step=0.1)
        price_usd = st.number_input("Price (USD)", min_value=5000.0, value=32000.0, step=1000.0)
        acceleration = st.number_input("0-100 km/h Acceleration (sec)", min_value=1.5, value=5.5, step=0.1)
        top_speed = st.number_input("Top Speed (km/h)", min_value=100.0, value=200.0, step=5.0)
        warranty_years = st.number_input("Warranty (Years)", min_value=1, max_value=15, value=5, step=1)
        cargo_space = st.number_input("Cargo Space (Liters)", min_value=50.0, value=420.0, step=10.0)
        safety_rating = st.number_input("Safety Rating (1.0 - 5.0)", min_value=1.0, max_value=5.0, value=4.5, step=0.1)

    submitted = st.form_submit_button("Predict Efficiency Class", use_container_width=True)

if submitted:
    vehicle_age = 2026 - release_year
    price_per_kwh = price_usd / battery_kwh if battery_kwh > 0 else np.nan

    input_values = {
        "manufacturer": manufacturer.strip() or np.nan,
        "model": model_name.strip() or np.nan,
        "type": vehicle_type.strip() or np.nan,
        "drive_type": drive_type.strip() or np.nan,
        "fuel_type": fuel_type.strip() or np.nan,
        "battery_kwh": battery_kwh,
        "range_km": range_km,
        "charging_time_hr": charging_time_hr,
        "fast_charging": fast_charging,
        "release_year": release_year,
        "country": country.strip() or np.nan,
        "seats": seats,
        "price_usd": price_usd,
        "acceleration_0_100_kmph": acceleration,
        "top_speed_kmph": top_speed,
        "warranty_years": warranty_years,
        "cargo_space_liters": cargo_space,
        "safety_rating": safety_rating,
        "vehicle_age": vehicle_age,
        "price_per_kwh": price_per_kwh,
    }

    input_data = pd.DataFrame([input_values]).reindex(columns=feature_columns)

    # Match the training script's conversion of categorical values to strings.
    for column in input_data.select_dtypes(include=["object"]).columns:
        input_data[column] = input_data[column].apply(
            lambda value: str(value) if pd.notna(value) else np.nan
        )

    try:
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.classes_)
        high_eff_index = classes.index(1)
        high_eff_probability = float(probabilities[high_eff_index])

        st.subheader("Prediction Result")
        if prediction == 1:
            st.success("Predicted Result: High-Efficiency EV")
        else:
            st.warning("Predicted Result: Standard-Efficiency EV")

        st.metric("High-Efficiency Probability", f"{high_eff_probability:.2%}")
        st.progress(float(np.clip(high_eff_probability, 0.0, 1.0)))
        st.caption(
            "Class 1 means High-Efficiency EV (efficiency > 5.0). Class 0 means Standard-Efficiency EV. "
            "The probability is the model's estimated likelihood of high efficiency."
        )
    except Exception as error:
        st.error(
            "The prediction could not be completed. Check the entered values "
            f"and make sure the model file was generated by the updated script. Details: {error}"
        )

st.divider()
st.caption("EV High-Efficiency Prediction | Decision Tree & Naive Bayes Pipeline")