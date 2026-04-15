import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "Deep Sleep Predictor",
    page_icon  = "😴",
    layout     = "centered"
)

# ── Feature config ────────────────────────────────────────────
# 🔧 Customize each feature:
#    "column_name": {
#        "label":       Display name shown on the app
#        "description": Tooltip / helper text
#        "min":         Minimum allowed input value (None = no limit)
#        "max":         Maximum allowed input value (None = no limit)
#        "default":     Default value shown on load
#        "step":        How much each click increases/decreases
#        "type":        "int" or "float"
#    }

FEATURE_CONFIG = {
    "Sleep efficiency": {
        "label":       "Sleep Efficiency",
        "description": "A measure of the proportion of time in bed spent asleep (0–10 scale).",
        "min":         0.0,
        "max":         10.0,
        "default":     5.0,
        "step":        0.1,
        "type":        "float"
    },
    "Alcohol consumption": {
        "label":       "Alcohol Consumption",
        "description": "The amount of alcohol consumed in the 24 hours prior to bedtime (in oz). Range: 0–5.",
        "min":         0.0,
        "max":         5.0,
        "default":     0.0,
        "step":        0.1,
        "type":        "float"
    },
    "Awakenings": {
        "label":       "Awakenings",
        "description": "The number of times the test subject wakes up during the night. Range: 0–5.",
        "min":         0,
        "max":         5,
        "default":     0,
        "step":        1,
        "type":        "int"
    },
    # ── Add or remove features below to match your dataset ────
    # "Age": {
    #     "label":       "Age",
    #     "description": "Age of the subject in years.",
    #     "min":         0,
    #     "max":         None,
    #     "default":     30,
    #     "step":        1,
    #     "type":        "int"
    # },
    # "Exercise frequency": {
    #     "label":       "Exercise Frequency",
    #     "description": "Number of times per week the subject exercises.",
    #     "min":         0,
    #     "max":         None,
    #     "default":     3,
    #     "step":        1,
    #     "type":        "int"
    # },
}


# ── Load model artifacts ──────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model        = joblib.load("svr_model.pkl")
    scaler_X     = joblib.load("scaler_X.pkl")
    scaler_y     = joblib.load("scaler_y.pkl")
    with open("target_info.json")  as f: target_info  = json.load(f)
    with open("feature_names.json") as f: feature_names = json.load(f)
    return model, scaler_X, scaler_y, target_info, feature_names

model, scaler_X, scaler_y, target_info, feature_names = load_artifacts()


# ── Header ────────────────────────────────────────────────────
st.title("😴 Deep Sleep Percentage Predictor")
st.markdown(
    "Fill in your sleep details below and click **Predict** "
    "to estimate your **Deep Sleep Percentage**."
)
st.divider()


# ── Input form ────────────────────────────────────────────────
st.subheader("Sleep Factors")
st.caption("Hover over the **?** icon next to each field for more info.")

input_values = {}
col1, col2 = st.columns(2)

for i, feature_key in enumerate(feature_names):
    # Use config if defined, otherwise fall back to a generic unlimited input
    if feature_key in FEATURE_CONFIG:
        cfg = FEATURE_CONFIG[feature_key]
    else:
        cfg = {
            "label":       feature_key,
            "description": "",
            "min":         None,
            "max":         None,
            "default":     0.0,
            "step":        0.1,
            "type":        "float"
        }

    col = col1 if i % 2 == 0 else col2
    with col:
        if cfg["type"] == "int":
            val = st.number_input(
                label     = cfg["label"],
                min_value = int(cfg["min"]) if cfg["min"] is not None else None,
                max_value = int(cfg["max"]) if cfg["max"] is not None else None,
                value     = int(cfg["default"]),
                step      = int(cfg["step"]),
                help      = cfg["description"]
            )
        else:
            val = st.number_input(
                label     = cfg["label"],
                min_value = float(cfg["min"]) if cfg["min"] is not None else None,
                max_value = float(cfg["max"]) if cfg["max"] is not None else None,
                value     = float(cfg["default"]),
                step      = float(cfg["step"]),
                format    = "%.2f",
                help      = cfg["description"]
            )
        input_values[feature_key] = val

st.divider()


# ── Predict ───────────────────────────────────────────────────
if st.button("Predict Deep Sleep %", type="primary", use_container_width=True):

    # Build input in the same feature order as training
    input_df    = pd.DataFrame([[input_values[f] for f in feature_names]],
                               columns=feature_names)
    input_sc    = scaler_X.transform(input_df)
    pred_scaled = model.predict(input_sc).reshape(-1, 1)
    prediction  = scaler_y.inverse_transform(pred_scaled)[0][0]

    # Clip to valid 0–100 range
    clipped = float(np.clip(prediction, 0.0, 100.0))

    if abs(prediction - clipped) > 0.01:
        st.warning(
            f"Raw model output was `{prediction:.2f}` — "
            f"clipped to the valid range [0, 100]. "
            f"This can happen when input values are outside the training distribution."
        )

    # Result card
    st.success(f"### Predicted Deep Sleep: **{clipped:.1f}%**")
    st.progress(clipped / 100, text=f"{clipped:.1f}% of 100%")

    # Interpretation
    if clipped < 13:
        level, advice = "🔴 Low",    "Below the healthy threshold (13–23%). Consider reducing alcohol and improving sleep consistency."
    elif clipped <= 23:
        level, advice = "🟢 Normal", "Within the healthy range (13–23%). Keep up your current sleep habits!"
    else:
        level, advice = "🔵 High",   "Above average deep sleep. This is generally positive but check if your data is within expected ranges."

    st.info(f"**{level}** — {advice}")

    # Input summary
    with st.expander("Show input summary"):
        summary_df = pd.DataFrame({
            "Feature": [FEATURE_CONFIG[f]["label"] if f in FEATURE_CONFIG else f for f in feature_names],
            "Value":   [input_values[f] for f in feature_names],
        })
        st.dataframe(summary_df.set_index("Feature"), use_container_width=True)


# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption("Built with SVR (Support Vector Regression) · Deployed with Streamlit")
