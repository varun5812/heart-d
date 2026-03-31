from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="HeartScope AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_artifact(filename: str):
    return joblib.load(BASE_DIR / filename)


@st.cache_resource
def load_assets():
    model = load_artifact("knn_heart_model.pkl")
    scaler = load_artifact("heart_scaler.pkl")
    columns = load_artifact("heart_columns.pkl")

    if callable(columns):
        columns = columns()
    elif hasattr(columns, "tolist"):
        columns = columns.tolist()
    elif isinstance(columns, dict):
        columns = list(columns.keys())
    else:
        columns = list(columns)

    return model, scaler, columns


model, scaler, expected_columns = load_assets()

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 120, 120, 0.35), transparent 28%),
            radial-gradient(circle at top right, rgba(103, 232, 249, 0.30), transparent 24%),
            linear-gradient(135deg, #fff7ed 0%, #fff1f2 34%, #eefcff 68%, #f0fdf4 100%);
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero-card, .glass-card, .result-card {
        border-radius: 24px;
        padding: 1.4rem 1.5rem;
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.7);
        box-shadow: 0 20px 60px rgba(15, 23, 42, 0.10);
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(255,255,255,0.72));
        margin-bottom: 1rem;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.74);
        min-height: 100%;
    }
    .result-card {
        color: #0f172a;
        background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(255,255,255,0.78));
    }
    .eyebrow {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: linear-gradient(90deg, #fb7185, #f59e0b);
        color: white;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .hero-title {
        font-size: 3rem;
        line-height: 1.05;
        font-weight: 800;
        color: #111827;
        margin: 0 0 0.6rem 0;
    }
    .hero-copy {
        font-size: 1.05rem;
        color: #334155;
        max-width: 760px;
        margin-bottom: 0;
    }
    .mini-stat {
        background: rgba(255,255,255,0.78);
        border: 1px solid rgba(255,255,255,0.75);
        border-radius: 20px;
        padding: 1rem;
        text-align: center;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.7);
    }
    .mini-stat h4 {
        margin: 0;
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
    }
    .mini-stat p {
        margin: 0.3rem 0 0;
        font-size: 1.4rem;
        font-weight: 800;
        color: #0f172a;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }
    .section-copy {
        color: #475569;
        margin-bottom: 1rem;
        font-size: 0.95rem;
    }
    .risk-pill {
        display: inline-block;
        padding: 0.45rem 0.85rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.88rem;
        margin-bottom: 0.75rem;
    }
    .risk-high {
        background: rgba(251, 113, 133, 0.16);
        color: #be123c;
    }
    .risk-low {
        background: rgba(34, 197, 94, 0.16);
        color: #15803d;
    }
    .probability-bar {
        width: 100%;
        height: 14px;
        border-radius: 999px;
        overflow: hidden;
        background: rgba(148, 163, 184, 0.18);
        margin: 0.75rem 0 0.35rem;
    }
    .probability-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #22c55e, #f59e0b, #f43f5e);
    }
    div[data-testid="stButton"] > button {
        width: 100%;
        border-radius: 16px;
        border: none;
        background: linear-gradient(90deg, #ef4444, #f59e0b, #06b6d4);
        color: white;
        font-weight: 800;
        padding: 0.8rem 1rem;
        box-shadow: 0 12px 30px rgba(239, 68, 68, 0.25);
    }
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-testid="stSlider"] {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="eyebrow">AI Heart Screening</div>
        <div class="hero-title">A brighter, faster way to check heart risk.</div>
        <p class="hero-copy">
            Enter a few clinical indicators and get an instant machine learning assessment with a cleaner,
            more modern experience designed for both desktop and mobile screens.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

top_stats = st.columns(4)
for col, label, value in [
    (top_stats[0], "Model", "KNN ML"),
    (top_stats[1], "Inputs", "11 UI fields"),
    (top_stats[2], "Features", f"{len(expected_columns)} encoded"),
    (top_stats[3], "Experience", "Render Ready"),
]:
    with col:
        st.markdown(
            f"""
            <div class="mini-stat">
                <h4>{label}</h4>
                <p>{value}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

left, right = st.columns([1.2, 0.8], gap="large")

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Patient Profile</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Use the form below to create a structured risk snapshot.</div>',
        unsafe_allow_html=True,
    )

    demo_a, demo_b = st.columns(2)
    with demo_a:
        age = st.slider("Age", 18, 100, 40)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    with demo_b:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, 0.1)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
        predict_clicked = st.button("Analyze Heart Risk")

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How to Read This</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-copy">
            This tool estimates risk patterns from training data. It supports awareness, not diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Best for quick screening before a deeper clinical review.")
    st.success("Low risk means the model sees fewer warning patterns in the provided values.")
    st.warning("High risk means the model sees a stronger pattern associated with heart disease.")
    st.markdown('</div>', unsafe_allow_html=True)


def build_input_frame() -> pd.DataFrame:
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1,
    }
    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    return input_df.reindex(columns=expected_columns, fill_value=0)


if predict_clicked:
    input_df = build_input_frame()
    scaled_input = scaler.transform(input_df)
    prediction = int(model.predict(scaled_input)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(scaled_input)[0][1]) * 100

    risk_label = "High Risk" if prediction == 1 else "Low Risk"
    pill_class = "risk-high" if prediction == 1 else "risk-low"
    headline = (
        "The model detected a stronger heart disease risk pattern."
        if prediction == 1
        else "The model detected a lower-risk pattern for the submitted values."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="risk-pill {pill_class}">{risk_label}</div>', unsafe_allow_html=True)
    st.subheader("Prediction Result")
    st.write(headline)

    metric_cols = st.columns(3)
    metric_cols[0].metric("Age", age)
    metric_cols[1].metric("Resting BP", resting_bp)
    metric_cols[2].metric("Max HR", max_hr)

    if probability is not None:
        st.markdown(
            f"""
            <div style="margin-top:0.75rem;font-weight:700;color:#0f172a;">Estimated disease probability: {probability:.1f}%</div>
            <div class="probability-bar">
                <div class="probability-fill" style="width:{probability:.1f}%;"></div>
            </div>
            <div style="color:#64748b;font-size:0.92rem;">
                Model confidence is directional and should be used alongside clinical judgement.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if prediction == 1:
        st.error("Consider follow-up with a medical professional for a more complete evaluation.")
    else:
        st.success("The submitted profile looks relatively safer to the model, but regular checkups still matter.")

    st.markdown('</div>', unsafe_allow_html=True)
