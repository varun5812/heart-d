from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Heart Risk Assessment",
    page_icon=":material/favorite:",
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
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    #MainMenu,
    footer {
        display: none !important;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 228, 230, 0.95), transparent 22%),
            radial-gradient(circle at top right, rgba(219, 234, 254, 0.95), transparent 24%),
            linear-gradient(180deg, #f8fbff 0%, #ffffff 35%, #fff8f5 100%);
        color: #172033;
    }
    .block-container {
        max-width: 1120px;
        padding-top: 0.8rem;
        padding-bottom: 3rem;
    }
    .shell {
        padding-top: 0.2rem;
    }
    .hero {
        display: grid;
        grid-template-columns: 1.15fr 0.85fr;
        gap: 1.25rem;
        align-items: stretch;
        margin-bottom: 1.4rem;
    }
    .hero-copy-card,
    .hero-metric-card,
    .form-card,
    .side-card,
    .result-card {
        background: rgba(255, 255, 255, 0.86);
        border: 1px solid rgba(226, 232, 240, 0.85);
        border-radius: 24px;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.07);
        backdrop-filter: blur(10px);
    }
    .hero-copy-card {
        padding: 2rem;
    }
    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        background: #fff1f2;
        color: #dc2626;
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 3.1rem;
        line-height: 1.02;
        font-weight: 800;
        letter-spacing: -0.04em;
        color: #14213d;
        max-width: 560px;
        margin-bottom: 0.9rem;
    }
    .hero-text {
        color: #5b667d;
        font-size: 1rem;
        line-height: 1.7;
        max-width: 540px;
        margin-bottom: 1.1rem;
    }
    .hero-note {
        color: #7a869d;
        font-size: 0.86rem;
    }
    .hero-metric-card {
        padding: 1.7rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        text-align: center;
        background: linear-gradient(180deg, #fff5f5 0%, #fffafb 100%);
    }
    .metric-heart {
        font-size: 3rem;
        color: #ef4444;
        line-height: 1;
    }
    .metric-score {
        font-size: 2.6rem;
        font-weight: 800;
        color: #dc2626;
        margin-top: 0.75rem;
    }
    .metric-label {
        color: #69758a;
        font-size: 0.92rem;
        margin-top: 0.3rem;
    }
    .content-grid {
        display: grid;
        grid-template-columns: 1.35fr 0.7fr;
        gap: 1.25rem;
        align-items: start;
    }
    .form-card {
        padding: 1.45rem;
    }
    .side-stack {
        display: grid;
        gap: 1rem;
    }
    .side-card {
        padding: 1.2rem;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #14213d;
        margin-bottom: 0.2rem;
    }
    .section-copy {
        color: #6b768c;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    .mini-title {
        font-size: 1rem;
        font-weight: 800;
        color: #14213d;
        margin-bottom: 0.35rem;
    }
    .mini-copy {
        color: #6b768c;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .info-list {
        display: grid;
        gap: 0.7rem;
        margin-top: 0.85rem;
    }
    .info-item {
        padding: 0.8rem 0.9rem;
        border-radius: 16px;
        background: #f8fafc;
        border: 1px solid #edf2f7;
    }
    .info-item strong {
        display: block;
        color: #1f2a44;
        margin-bottom: 0.2rem;
        font-size: 0.92rem;
    }
    .info-item span {
        color: #6b768c;
        font-size: 0.87rem;
    }
    .result-card {
        margin-top: 1rem;
        padding: 1.35rem 1.45rem;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 800;
        margin-bottom: 0.8rem;
    }
    .risk-high {
        background: #fff1f2;
        color: #be123c;
    }
    .risk-low {
        background: #ecfdf3;
        color: #15803d;
    }
    .prob-track {
        width: 100%;
        height: 12px;
        border-radius: 999px;
        background: #e8eef5;
        overflow: hidden;
        margin: 0.7rem 0 0.35rem;
    }
    .prob-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #22c55e, #f59e0b, #ef4444);
    }
    div[data-testid="stForm"] {
        border: 0 !important;
    }
    div[data-testid="stButton"] > button,
    div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        min-height: 3.2rem;
        border-radius: 16px;
        border: none;
        background: linear-gradient(180deg, #ff2020 0%, #e11d2e 100%);
        color: white;
        font-size: 1rem;
        font-weight: 800;
        box-shadow: 0 16px 32px rgba(225, 29, 46, 0.22);
    }
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        border-radius: 14px;
    }
    @media (max-width: 960px) {
        .hero,
        .content-grid {
            grid-template-columns: 1fr;
        }
        .hero-title {
            font-size: 2.4rem;
        }
    }
    </style>
    <div class="shell"></div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-copy-card">
            <div class="eyebrow">&#9825; Heart Risk Screening</div>
            <div class="hero-title">Clean, guided heart disease assessment.</div>
            <div class="hero-text">
                A simpler interface for entering clinical details and getting an instant machine learning prediction.
                The design is focused on clarity, trust, and a smoother user flow.
            </div>
            <div class="hero-note">Informational support only. This tool does not replace a licensed medical professional.</div>
        </div>
        <div class="hero-metric-card">
            <div class="metric-heart">&#9825;</div>
            <div class="metric-score">91%</div>
            <div class="metric-label">Model accuracy on evaluation data</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="content-grid">', unsafe_allow_html=True)
st.markdown('<div class="form-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Medical Parameters</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-copy">Please enter the patient information carefully. Each field includes a small info helper for quick reference.</div>',
    unsafe_allow_html=True,
)

with st.form("prediction_form"):
    row1 = st.columns(3, gap="medium")
    with row1[0]:
        age = st.number_input("Age", min_value=18, max_value=100, value=55, step=1, help="Age of the patient in years.")
    with row1[1]:
        sex = st.selectbox("Sex", ["M", "F"], help="Biological sex used in the trained model input.")
    with row1[2]:
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"], help="ATA: atypical angina, NAP: non-anginal pain, TA: typical angina, ASY: asymptomatic.")

    row2 = st.columns(3, gap="medium")
    with row2[0]:
        resting_bp = st.number_input("Resting BP (mmHg)", min_value=80, max_value=220, value=130, step=1, help="Resting blood pressure measured in mmHg.")
    with row2[1]:
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=650, value=240, step=1, help="Serum cholesterol level in milligrams per deciliter.")
    with row2[2]:
        fasting_bs = st.selectbox("Fasting Blood Sugar", [0, 1], help="0 means 120 mg/dL or below, 1 means above 120 mg/dL.")

    row3 = st.columns(3, gap="medium")
    with row3[0]:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], help="Result category from the resting ECG test.")
    with row3[1]:
        max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150, step=1, help="Maximum heart rate reached during exercise.")
    with row3[2]:
        exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"], help="Y if angina appears during exercise, otherwise N.")

    row4 = st.columns(3, gap="medium")
    with row4[0]:
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=6.5, value=1.5, step=0.1, help="ST depression induced by exercise relative to rest.")
    with row4[1]:
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], help="Slope of the peak exercise ST segment.")
    with row4[2]:
        selected_vessels = st.selectbox("Major Vessels", ["0", "1", "2", "3"], help="Reference-only field for UI completeness.")

    row5 = st.columns(3, gap="medium")
    with row5[0]:
        selected_thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"], help="Reference-only field for UI completeness.")
    with row5[1]:
        st.markdown("")
    with row5[2]:
        st.markdown("")

    submitted = st.form_submit_button("Get Prediction")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="side-stack">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="side-card">
        <div class="mini-title">How To Use</div>
        <div class="mini-copy">Fill in the clinical values, review the options using the info helpers, then submit to generate the risk assessment.</div>
        <div class="info-list">
            <div class="info-item"><strong>Fast screening</strong><span>Instant model response after submission.</span></div>
            <div class="info-item"><strong>Clear feedback</strong><span>Risk state, confidence-style bar, and summary text.</span></div>
            <div class="info-item"><strong>Better guidance</strong><span>Each field includes lightweight explanatory help.</span></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="side-card">
        <div class="mini-title">Model Snapshot</div>
        <div class="mini-copy">This app currently uses a K-Nearest Neighbors model with <strong>{len(expected_columns)}</strong> encoded input features.</div>
        <div class="info-list">
            <div class="info-item"><strong>Algorithm</strong><span>K-Nearest Neighbors classifier</span></div>
            <div class="info-item"><strong>Inputs in form</strong><span>11 primary medical inputs</span></div>
            <div class="info-item"><strong>Use case</strong><span>Informational heart disease risk screening</span></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)
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


if submitted:
    input_df = build_input_frame()
    scaled_input = scaler.transform(input_df)
    prediction = int(model.predict(scaled_input)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(scaled_input)[0][1]) * 100

    risk_label = "High Risk" if prediction == 1 else "Low Risk"
    badge_class = "risk-high" if prediction == 1 else "risk-low"
    headline = (
        "The model identified a stronger heart disease risk pattern for this profile."
        if prediction == 1
        else "The model identified a lower-risk pattern for this profile."
    )

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="risk-badge {badge_class}">{risk_label}</div>', unsafe_allow_html=True)
    st.subheader("Prediction Result")
    st.write(headline)

    metrics = st.columns(4)
    metrics[0].metric("Age", int(age))
    metrics[1].metric("Resting BP", int(resting_bp))
    metrics[2].metric("Max Heart Rate", int(max_hr))
    metrics[3].metric("Major Vessels", selected_vessels)

    if probability is not None:
        st.markdown(
            f"""
            <div style="margin-top:0.65rem;font-weight:700;color:#172033;">Estimated disease probability: {probability:.1f}%</div>
            <div class="prob-track"><div class="prob-fill" style="width:{probability:.1f}%;"></div></div>
            <div style="color:#6b768c;font-size:0.92rem;">This percentage is directional model output and should be interpreted with medical context.</div>
            """,
            unsafe_allow_html=True,
        )

    if prediction == 1:
        st.error("This result suggests elevated risk. Please consider follow-up with a healthcare professional.")
    else:
        st.success("This result suggests a lower-risk pattern, though regular medical advice still matters.")

    st.caption(f"Additional reference inputs captured: Thalassemia = {selected_thal}, Major Vessels = {selected_vessels}.")
    st.markdown('</div>', unsafe_allow_html=True)

st.caption("This tool is for informational purposes only and should not replace professional medical advice.")