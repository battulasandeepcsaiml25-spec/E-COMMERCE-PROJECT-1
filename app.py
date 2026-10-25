
# ==========================================
# PURCHASE CONVERSION ADVISOR
# Premium Streamlit UI/UX
# ==========================================

import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------
st.set_page_config(
    page_title="Purchase Conversion Advisor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------
# CUSTOM CSS
# ------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #080d1b;
    --panel: #10182b;
    --panel2: #141f36;
    --border: rgba(148, 163, 184, 0.17);
    --text: #f1f5f9;
    --muted: #94a3b8;
    --blue: #60a5fa;
    --cyan: #22d3ee;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 15% 0%, rgba(37,99,235,.17), transparent 35%),
        radial-gradient(ellipse at 90% 20%, rgba(6,182,212,.09), transparent 30%),
        var(--bg);
    color: var(--text);
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10182b 0%, #0b1120 100%);
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

.sidebar-brand {
    padding: 18px 14px;
    border: 1px solid var(--border);
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(37,99,235,.18), rgba(34,211,238,.06));
    margin-bottom: 22px;
}

.brand-icon {
    font-size: 34px;
    margin-bottom: 8px;
}

.brand-title {
    font-size: 19px;
    font-weight: 800;
    letter-spacing: -0.5px;
}

.brand-subtitle {
    color: #94a3b8;
    font-size: 12px;
    margin-top: 6px;
    line-height: 1.6;
}

/* Hero */
.hero {
    position: relative;
    overflow: hidden;
    padding: 34px;
    border-radius: 25px;
    border: 1px solid rgba(96,165,250,.24);
    background:
        linear-gradient(120deg, rgba(30,64,175,.34), rgba(15,23,42,.93) 62%),
        #10182b;
    box-shadow: 0 18px 60px rgba(0,0,0,.20);
    margin-bottom: 24px;
}

.hero:after {
    content: "";
    position: absolute;
    width: 220px;
    height: 220px;
    right: -60px;
    top: -90px;
    border-radius: 50%;
    background: rgba(34,211,238,.10);
    filter: blur(3px);
    pointer-events: none;
}

.eyebrow {
    color: #7dd3fc;
    text-transform: uppercase;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 12px;
}

.hero h1 {
    color: #f8fafc;
    font-size: clamp(30px, 4vw, 44px);
    font-weight: 800;
    letter-spacing: -1.7px;
    line-height: 1.15;
    margin: 0 0 12px 0;
}

.hero p {
    color: #bdc9dc;
    font-size: 15px;
    line-height: 1.8;
    max-width: 680px;
    margin: 0;
}

.hero-badge {
    display: inline-block;
    margin-top: 22px;
    padding: 8px 12px;
    border-radius: 999px;
    border: 1px solid rgba(34,211,238,.25);
    background: rgba(34,211,238,.08);
    color: #a5f3fc;
    font-size: 12px;
    font-weight: 600;
}

/* Section headings */
.section-heading {
    color: #f8fafc;
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -.5px;
    margin: 10px 0 5px 0;
}

.section-subheading {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 20px;
    line-height: 1.7;
}

/* Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(16,24,43,.78);
    border: 1px solid var(--border);
    border-radius: 20px;
    transition: border-color .25s ease, transform .25s ease,
                box-shadow .25s ease;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(96,165,250,.48);
    box-shadow: 0 12px 30px rgba(0,0,0,.15);
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #141f36, #10182b);
    border: 1px solid var(--border);
    padding: 18px;
    border-radius: 17px;
    transition: transform .25s ease, border-color .25s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: rgba(96,165,250,.5);
}

div[data-testid="stMetricLabel"] {
    color: #a8b7cc;
    font-size: 12px;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #f8fafc;
    font-weight: 800;
}

/* Input labels */
div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] p {
    color: #dbeafe !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

/* Number inputs */
div[data-testid="stNumberInput"] input {
    background: #0b1222 !important;
    color: #f8fafc !important;
    border: 1px solid #293751 !important;
    border-radius: 11px !important;
    min-height: 46px;
    transition: border-color .2s ease, box-shadow .2s ease;
}

div[data-testid="stNumberInput"] input:focus {
    border-color: #60a5fa !important;
    box-shadow: 0 0 0 3px rgba(96,165,250,.13) !important;
}

/* Buttons */
.stButton > button {
    border-radius: 12px;
    min-height: 48px;
    font-weight: 700;
    border: 1px solid rgba(96,165,250,.45);
    background: linear-gradient(110deg, #2563eb, #0891b2);
    color: white;
    transition: transform .2s ease, box-shadow .2s ease, filter .2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 9px 25px rgba(37,99,235,.28);
    filter: brightness(1.12);
    border-color: #7dd3fc;
}

.stButton > button:active {
    transform: scale(.99);
}

/* Progress */
div[data-testid="stProgressBar"] > div {
    background: #26354f;
    border-radius: 999px;
    height: 12px;
}

div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #2563eb, #22d3ee);
    border-radius: 999px;
}

/* Result card */
.result-card {
    padding: 26px;
    border-radius: 20px;
    background: linear-gradient(140deg, #14213b, #0e1729);
    border: 1px solid rgba(96,165,250,.28);
    margin: 8px 0 18px 0;
}

.result-label {
    color: #9fb2cb;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
}

.result-number {
    font-size: clamp(38px, 5vw, 58px);
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1.25;
    margin: 8px 0;
}

.result-caption {
    color: #a8b7cc;
    font-size: 13px;
    line-height: 1.7;
}

.status-pill {
    display: inline-block;
    padding: 8px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 800;
    margin-top: 8px;
}

.status-high {
    background: rgba(16,185,129,.13);
    border: 1px solid rgba(16,185,129,.35);
    color: #6ee7b7;
}

.status-medium {
    background: rgba(245,158,11,.13);
    border: 1px solid rgba(245,158,11,.35);
    color: #fcd34d;
}

.status-low {
    background: rgba(248,113,113,.12);
    border: 1px solid rgba(248,113,113,.3);
    color: #fca5a5;
}

.small-note {
    color: #8494ad;
    font-size: 11px;
    line-height: 1.7;
}

hr {
    border-color: var(--border);
    margin: 25px 0;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 14px;
}

/* Footer */
.footer {
    text-align: center;
    padding: 22px 0 8px 0;
    color: #64748b;
    font-size: 11px;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem 1rem 2rem 1rem;
    }
    .hero {
        padding: 24px;
        border-radius: 19px;
    }
    .hero h1 {
        letter-spacing: -1px;
    }
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------
# LOAD MODEL AND DATA
# ------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("purchase_conversion_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("selected_features_dataset.csv")


try:
    model = load_model()
    df = load_data()

    if "Revenue" not in df.columns:
        st.error("The dataset must contain a 'Revenue' column.")
        st.stop()

    feature_names = df.drop(columns=["Revenue"]).columns.tolist()

except FileNotFoundError as error:
    st.error(
        f"Required file not found: {error.filename}. "
        "Keep app.py, purchase_conversion_model.pkl, and "
        "selected_features_dataset.csv in the same folder."
    )
    st.stop()

except Exception as error:
    st.error(f"Could not load the model or dataset: {error}")
    st.stop()


# ------------------------------------------
# SIDEBAR
# ------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon">🛒</div>
        <div class="brand-title">Purchase Advisor</div>
        <div class="brand-subtitle">
            AI-powered shopping session analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Navigation")
    st.caption("Use the main dashboard to run a prediction.")

    st.markdown("---")
    st.markdown("### Model information")
    st.markdown("**Task:** Binary classification")
    st.markdown("**Output:** Purchase probability")
    st.markdown("**Features:** Selected dataset features")

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. Enter session details.
    2. Click **Predict Purchase**.
    3. Review the probability.
    4. Check the conversion priority.
    """)

    st.markdown("---")
    st.markdown(
        '<div class="small-note">'
        'Predictions are estimates from the loaded model. '
        'They are not guarantees of actual customer behavior.'
        '</div>',
        unsafe_allow_html=True
    )


# ------------------------------------------
# HERO SECTION
# ------------------------------------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">Machine Learning · Decision Support</div>
    <h1>Understand the next<br>purchase opportunity.</h1>
    <p>
        Analyze shopping-session behavior and estimate the likelihood
        of a purchase. Use the conversion priority to help interpret
        the model's prediction.
    </p>
    <div class="hero-badge">✦ &nbsp; Purchase intention prediction</div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------
# OVERVIEW METRICS
# ------------------------------------------
st.markdown('<div class="section-heading">Your workspace</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-subheading">'
    'Enter session details to generate a model-based estimate.'
    '</div>',
    unsafe_allow_html=True
)

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("Dataset sessions", f"{len(df):,}")

with metric2:
    st.metric("Model features", len(feature_names))

with metric3:
    st.metric("Input controls", "03")


# ------------------------------------------
# USER INPUT SECTION
# ------------------------------------------
st.markdown("---")
st.markdown(
    '<div class="section-heading">Configure a shopping session</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-subheading">'
    'Adjust the three session values below. Other model features '
    'will use their dataset median, as in your original implementation.'
    '</div>',
    unsafe_allow_html=True
)

with st.container(border=True):
    st.markdown("#### 🧾 Session details")

    input_col1, input_col2, input_col3 = st.columns(3)

    with input_col1:
        pages = st.number_input(
            "Product-related pages",
            min_value=0,
            value=10,
            step=1,
            help="Number of product-related pages visited."
        )

    with input_col2:
        duration = st.number_input(
            "Product page duration (seconds)",
            min_value=0.0,
            value=300.0,
            step=10.0,
            help="Total time spent on product-related pages."
        )

    with input_col3:
        page_value = st.number_input(
            "Page value",
            min_value=0.0,
            value=10.0,
            step=1.0,
            help="PageValues feature used by the trained model."
        )

    st.markdown("")
    predict_clicked = st.button(
        "✦  Predict Purchase",
        type="primary",
        use_container_width=True
    )


# ------------------------------------------
# PREDICTION
# ------------------------------------------
if predict_clicked:
    # Start with dataset medians for all model features.
    input_data = df[feature_names].median(numeric_only=True).to_dict()

    # Add any non-numeric feature medians where needed.
    for feature in feature_names:
        if feature not in input_data:
            mode_values = df[feature].mode(dropna=True)
            if not mode_values.empty:
                input_data[feature] = mode_values.iloc[0]
            else:
                input_data[feature] = 0

    # Update the three user-provided values.
    feature_mapping = {
        "ProductRelated": pages,
        "ProductRelated_Duration": duration,
        "PageValues": page_value
    }

    for feature, value in feature_mapping.items():
        if feature in input_data:
            input_data[feature] = value

    # Ensure the feature order matches the training data.
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Make prediction.
    try:
        prediction = model.predict(input_df)[0]

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            # Support integer or string representations of class labels.
            positive_class_index = next(
                (
                    i for i, class_value in enumerate(classes)
                    if str(class_value).strip().lower()
                    in ("1", "true", "yes")
                ),
                None
            )

            if positive_class_index is None:
                st.error(
                    "Could not identify the positive purchase class "
                    "in the model's class labels."
                )
                st.stop()

            purchase_probability = (
                float(probabilities[positive_class_index]) * 100
            )

        else:
            st.error(
                "This model does not support predict_proba(). "
                "A probability display cannot be generated."
            )
            st.stop()

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.stop()

    # --------------------------------------
    # RESULT AND PRIORITY
    # --------------------------------------
    if purchase_probability >= 70:
        priority = "High"
        priority_class = "status-high"
        priority_description = "Higher predicted purchase probability"
    elif purchase_probability >= 40:
        priority = "Medium"
        priority_class = "status-medium"
        priority_description = "Moderate predicted purchase probability"
    else:
        priority = "Low"
        priority_class = "status-low"
        priority_description = "Lower predicted purchase probability"

    st.markdown("---")
    st.markdown(
        '<div class="section-heading">Prediction report</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-subheading">'
        'Here is the result for the shopping session you entered.'
        '</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2 = st.columns([1.25, 1])

    with result_col1:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Purchase probability</div>
                <div class="result-number">{purchase_probability:.2f}%</div>
                <div class="result-caption">
                    Estimated probability returned by the trained model.
                </div>
                <div class="status-pill {priority_class}">
                    {priority} conversion priority
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("**Probability scale**")
        st.progress(
            min(max(int(round(purchase_probability)), 0), 100)
        )
        st.caption(
            f"{purchase_probability:.2f}% estimated probability · "
            f"{100 - purchase_probability:.2f}% estimated non-purchase probability"
        )

    with result_col2:
        with st.container(border=True):
            st.markdown("#### Prediction status")

            if prediction == 1 or str(prediction).lower() in ("true", "yes"):
                st.success("🟢 Purchase Likely")
                st.write(
                    "The model predicts that this session belongs "
                    "to the purchase class."
                )
            else:
                st.warning("🟠 Purchase Unlikely")
                st.write(
                    "The model predicts that this session belongs "
                    "to the non-purchase class."
                )

            st.markdown("---")
            st.markdown("#### Conversion priority")
            st.markdown(
                f'<div class="status-pill {priority_class}">'
                f'{priority} Priority</div>',
                unsafe_allow_html=True
            )
            st.caption(priority_description)

    # Summary of entered values.
    st.markdown("#### Session summary")
    summary1, summary2, summary3 = st.columns(3)

    with summary1:
        st.metric("Product-related pages", f"{pages:g}")

    with summary2:
        st.metric("Product page duration", f"{duration:g} sec")

    with summary3:
        st.metric("Page value", f"{page_value:g}")

    st.info(
        "Interpretation: conversion priority is assigned using the "
        "70% and 40% probability thresholds configured in this app. "
        "These thresholds are decision-support rules, not model classes."
    )

    st.caption(
        "Important: this is a model-based estimate, not a guarantee "
        "of customer purchase behavior. Results depend on the trained "
        "model and the features supplied to it."
    )

else:
    # Empty state before the first prediction.
    st.markdown("---")
    with st.container(border=True):
        st.markdown("### ✨ Ready to analyze a session?")
        st.write(
            "Enter the shopping-session details above and click "
            "**Predict Purchase** to see the probability, prediction "
            "status, and conversion priority."
        )


# ------------------------------------------
# FOOTER
# ------------------------------------------
st.markdown("""
<div class="footer">
    PURCHASE CONVERSION ADVISOR &nbsp;·&nbsp;
    MACHINE LEARNING PROJECT<br>
    Model estimates are intended for decision support only.
</div>
""", unsafe_allow_html=True)
