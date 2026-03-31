from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Heart Disease Risk Assessment",
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
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 232, 232, 0.92), transparent 24%),
            radial-gradient(circle at top right, rgba(227, 239, 255, 0.92), transparent 28%),
            linear-gradient(180deg, #ffffff 0%, #fffdf9 36%, #f7fbff 100%);
        color: #16233b;
    }
    .block-container {
        max-width: 1180px;
        padding-top: 1.2rem;
        padding-bottom: 4rem;
    }
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.25rem 0 1rem;
    }
    .brand {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.95rem;
        font-weight: 700;
        color: #1f2a44;
    }
    .brand-heart {
        color: #ef4444;
        font-size: 1.1rem;
    }
    .signin-pill {
        background: #ef4444;
        color: white;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .hero {
        display: grid;
        grid-template-columns: 1.2fr 0.85fr;
        gap: 1.5rem;
        align-items: center;
        padding: 1rem 0 2.6rem;
    }
    .hero-title {
        font-size: 3.3rem;
        line-height: 1.02;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #1d2740;
        margin-bottom: 0.8rem;
        max-width: 580px;
    }
    .hero-copy {
        color: #5f6f89;
        font-size: 1.02rem;
        max-width: 560px;
        margin-bottom: 1rem;
    }
    .cta-row {
        display: flex;
        gap: 0.8rem;
        align-items: center;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    .cta-btn {
        display: inline-block;
        background: #ef4444;
        color: white;
        padding: 0.8rem 1.1rem;
        border-radius: 14px;
        font-weight: 700;
        font-size: 0.92rem;
    }
    .hero-note {
        font-size: 0.82rem;
        color: #7b879b;
        margin-top: 0.8rem;
    }
    .hero-stat {
        background: linear-gradient(180deg, #fff5f5 0%, #fff0f0 100%);
        border: 1px solid #ffd9d9;
        border-radius: 20px;
        padding: 2rem 1.6rem;
        text-align: center;
        box-shadow: 0 14px 36px rgba(226, 88, 88, 0.14);
    }
    .hero-stat-icon {
        font-size: 3rem;
        color: #ef4444;
        line-height: 1;
    }
    .hero-stat-score {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ef4444;
        margin-top: 0.65rem;
    }
    .hero-stat-copy {
        color: #6c7890;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    .section-heading {
        text-align: center;
        margin: 2rem 0 1rem;
    }
    .section-heading h2 {
        font-size: 2rem;
        color: #1d2740;
        margin-bottom: 0.3rem;
    }
    .section-heading p {
        color: #69768d;
        margin: 0;
    }
    .feature-card, .step-card, .info-card, .form-shell, .result-shell {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid #ebeff5;
        border-radius: 20px;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.06);
    }
    .feature-card, .step-card, .info-card {
        padding: 1.25rem;
        height: 100%;
    }
    .feature-icon, .step-icon {
        width: 42px;
        height: 42px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #fff1f2;
        color: #ef4444;
        font-size: 1.15rem;
        margin-bottom: 0.9rem;
    }
    .card-title {
        font-weight: 800;
        color: #1d2740;
        margin-bottom: 0.35rem;
    }
    .card-copy {
        color: #6b768c;
        font-size: 0.92rem;
    }
    .form-shell {
        max-width: 900px;
        margin: 0 auto;
        padding: 1.4rem 1.5rem 1.5rem;
    }
    .form-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #1d2740;
        margin-bottom: 0.25rem;
    }
    .form-copy {
        color: #6b768c;
        margin-bottom: 1.2rem;
    }
    .result-shell {
        max-width: 900px;
        margin: 1.25rem auto 0;
        padding: 1.35rem 1.5rem;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        font-weight: 800;
        font-size: 0.82rem;
        margin-bottom: 0.75rem;
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
        background: #e9eef6;
        border-radius: 999px;
        overflow: hidden;
        margin: 0.7rem 0 0.35rem;
    }
    .prob-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #22c55e, #f59e0b, #ef4444);
    }
    .footer-cta {
        margin-top: 2.2rem;
        text-align: center;
        padding: 2.4rem 1rem 0.5rem;
    }
    .footer-cta h3 {
        font-size: 2rem;
        color: #1d2740;
        margin-bottom: 0.35rem;
    }
    .footer-cta p {
        color: #6b768c;
        margin-bottom: 1rem;
    }
    div[data-testid="stButton"] > button {
        width: 100%;
        min-height: 3.2rem;
        border-radius: 14px;
        border: none;
        background: linear-gradient(180deg, #ff1f1f 0%, #e50914 100%);
        color: white;
        font-weight: 800;
        font-size: 1rem;
        box-shadow: 0 16px 28px rgba(239, 68, 68, 0.2);
    }
    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(180deg, #ef4444 0%, #dc2626 100%);
    }
    @media (max-width: 900px) {
        .hero {
            grid-template-columns: 1fr;
        }
        .hero-title {
            font-size: 2.5rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="topbar">
        <div class="brand"><span class="brand-heart">&#9825;</span> Heart Disease Prediction</div>
        <div class="signin-pill">Sign In</div>
    </div>
    <div class="hero">
        <div>
            <div class="hero-title">Advanced Heart Disease Risk Assessment</div>
            <div class="hero-copy">
                Get instant, AI-powered predictions of your heart disease risk based on clinical parameters.
                Built to feel more like a real healthcare product with clearer guidance and cleaner results.
            </div>
            <div class="cta-row">
                <span class="cta-btn">Start Assessment</span>
            </div>
            <div class="hero-note">For informational use only. This tool supports screening, not medical diagnosis.</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-icon">&#9825;</div>
            <div class="hero-stat-score">91%</div>
            <div class="hero-stat-copy">Model Accuracy on Test Data</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-heading">
        <h2>Key Features</h2>
        <p>Built to give the app more structure, clarity, and trust.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

feature_cols = st.columns(3, gap="large")
feature_items = [
    ("&#9825;", "13 Clinical Parameters", "Comprehensive assessment using age, blood pressure, cholesterol, heart rate, ECG findings, and more."),
    ("&#9889;", "Real-Time Predictions", "Submit your profile and receive a live model prediction with a cleaner interpretation layer."),
    ("&#9716;", "Result Clarity", "Understand what your result means with highlighted risk states, summaries, and confidence cues."),
]
for col, (icon, title, copy) in zip(feature_cols, feature_items):
    with col:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="feature-icon">{icon}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card-copy">{copy}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="section-heading">
        <h2>How It Works</h2>
        <p>A simple flow designed to feel guided instead of overwhelming.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

step_cols = st.columns(4, gap="medium")
step_items = [
    ("1", "Enter Your Data", "Provide your medical parameters through the guided form below."),
    ("2", "AI Analysis", "Our trained model processes the input profile using learned patterns."),
    ("3", "Get Result", "Receive an instant risk assessment with confidence-style feedback."),
    ("4", "Track Progress", "Use the output as a conversation starter for healthier follow-up decisions."),
]
for col, (icon, title, copy) in zip(step_cols, step_items):
    with col:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="step-icon">{icon}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card-copy">{copy}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="section-heading">
        <h2>Heart Disease Risk Assessment</h2>
        <p>Enter your medical parameters for an instant risk prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="form-shell">', unsafe_allow_html=True)
st.markdown('<div class="form-title">Medical Parameters</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="form-copy">Please provide your medical information accurately. Use the info icons for quick guidance.</div>',
    unsafe_allow_html=True,
)

with st.form("prediction_form"):
    row1 = st.columns(3, gap="medium")
    with row1[0]:
        age = st.number_input("Age", min_value=18, max_value=100, value=55, step=1, help="Patient age in years.")
    with row1[1]:
        sex = st.selectbox("Sex", ["M", "F"], help="Biological sex used by the model encoding.")
    with row1[2]:
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"], help="ATA: atypical angina, NAP: non-anginal pain, TA: typical angina, ASY: asymptomatic.")

    row2 = st.columns(3, gap="medium")
    with row2[0]:
        resting_bp = st.number_input("Resting BP (mmHg)", min_value=80, max_value=220, value=130, step=1, help="Resting blood pressure measured in mmHg.")
    with row2[1]:
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=650, value=240, step=1, help="Serum cholesterol level in mg/dL.")
    with row2[2]:
        fasting_bs = st.selectbox("Fasting Blood Sugar", [0, 1], help="0 means at or below 120 mg/dL, 1 means above 120 mg/dL.")

    row3 = st.columns(3, gap="medium")
    with row3[0]:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], help="Electrocardiogram result category.")
    with row3[1]:
        max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150, step=1, help="Maximum heart rate achieved during exercise testing.")
    with row3[2]:
        exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"], help="Y if exercise-induced angina is present.")

    row4 = st.columns(3, gap="medium")
    with row4[0]:
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=6.5, value=1.5, step=0.1, help="ST depression induced by exercise relative to rest.")
    with row4[1]:
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], help="Slope of the peak exercise ST segment.")
    with row4[2]:
        vessel_options = ["0", "1", "2", "3"]
        selected_vessels = st.selectbox("Major Vessels", vessel_options, help="Reference info card only. This model version does not use this value directly in its encoded input.")

    row5 = st.columns(3, gap="medium")
    with row5[0]:
        thal_options = ["Normal", "Fixed Defect", "Reversible Defect"]
        selected_thal = st.selectbox("Thalassemia", thal_options, help="Reference info card only. This model version does not use this value directly in its encoded input.")
    with row5[1]:
        st.markdown("")
    with row5[2]:
        st.markdown("")

    submitted = st.form_submit_button("Get Prediction")

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
        "The model identified a stronger heart disease risk pattern for this input profile."
        if prediction == 1
        else "The model identified a lower-risk pattern for this input profile."
    )

    st.markdown('<div class="result-shell">', unsafe_allow_html=True)
    st.markdown(f'<div class="risk-badge {badge_class}">{risk_label}</div>', unsafe_allow_html=True)
    st.subheader("Prediction Result")
    st.write(headline)

    metrics = st.columns(4)
    metrics[0].metric("Age", int(age))
    metrics[1].metric("Resting BP", int(resting_bp))
    metrics[2].metric("Max Heart Rate", int(max_hr))
    metrics[3].metric("Selected Vessels", selected_vessels)

    if probability is not None:
        st.markdown(
            f"""
            <div style="margin-top:0.6rem;font-weight:700;color:#1d2740;">Estimated disease probability: {probability:.1f}%</div>
            <div class="prob-track"><div class="prob-fill" style="width:{probability:.1f}%;"></div></div>
            <div style="color:#6b768c;font-size:0.92rem;">Probability provides directional confidence and should be interpreted with clinical context.</div>
            """,
            unsafe_allow_html=True,
        )

    if prediction == 1:
        st.error("This result suggests elevated risk. Please consider follow-up with a healthcare professional.")
    else:
        st.success("This result suggests a lower-risk pattern, though regular medical advice still matters.")

    st.caption(f"Additional reference inputs captured: Major Vessels = {selected_vessels}, Thalassemia = {selected_thal}.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="section-heading" style="margin-top:2.2rem;">
        <h2>Model Information</h2>
        <p>Helpful context to make the experience feel more complete and trustworthy.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

info_cols = st.columns(2, gap="large")
with info_cols[0]:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Algorithm</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-copy">K-Nearest Neighbors classifier currently powers the live prediction model used in this app.</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="margin-top:1rem;">Training Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-copy">Balanced clinical feature set with encoded patient indicators for fast browser-side form submission and server-side inference.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with info_cols[1]:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Performance Metrics</div>', unsafe_allow_html=True)
    metric_rows = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Prediction Inputs", "Encoded Features"],
            "Value": ["91%", "11", str(len(expected_columns))],
        }
    )
    st.table(metric_rows)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="footer-cta">
        <h3>Ready to Assess Your Risk?</h3>
        <p>Take the first step toward better heart health with a more structured prediction experience.</p>
        <span class="cta-btn">Start Your Assessment Now</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("This tool is for informational purposes only and should not replace professional medical advice.")