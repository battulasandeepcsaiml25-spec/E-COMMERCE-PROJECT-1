import streamlit as st
import pandas as pd
import joblib
# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Purchase Conversion Checker",
    page_icon="🛒",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1120, #111c30);
    color: #f8fafc;
}
.block-container {
    padding-top: 2rem;
    max-width: 1400px;
}
.hero {
    padding: 28px;
    border-radius: 22px;
    background: linear-gradient(120deg, #4338ca, #7c3aed, #2563eb);
    margin-bottom: 24px;
    color: white;
}
.hero h1 { font-size: 36px; margin-bottom: 8px; }
.hero p { color: #e0e7ff; font-size: 16px; }
.panel {
    background: #172337;
    border: 1px solid #2a3a52;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 15px;
}
.buy-card {
    background: linear-gradient(145deg, #103b32, #172d35);
    border: 1px solid #23856c;
    border-radius: 18px;
    padding: 24px;
}
.no-buy-card {
    background: linear-gradient(145deg, #482431, #302336);
    border: 1px solid #a44b62;
    border-radius: 18px;
    padding: 24px;
}
.big-number { font-size: 38px; font-weight: 800; }
.muted { color: #a9b8cd; font-size: 14px; }
.section-title { color: #e2e8f0; font-size: 23px; font-weight: 700; }
.stButton button {
    width: 100%;
    height: 52px;
    border-radius: 12px;
    border: 0;
    color: white;
    font-weight: 700;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
}
div[data-testid="stMetric"] {
    background: #172337;
    padding: 16px;
    border-radius: 14px;
    border: 1px solid #2a3a52;
}
</style>
""", unsafe_allow_html=True)
# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("purchase_prediction_model.pkl")
    selector = joblib.load("feature_selector.pkl")
    columns = joblib.load("feature_columns.pkl")
    return model, selector, columns
try:
    model, selector, feature_columns = load_artifacts()
except Exception as e:
    st.error(f"Could not load the model files: {e}")
    st.stop()
# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🛒 Purchase Intelligence AI</h1>
    <p>Understand visitor behaviour. Predict conversions.
    Make smarter e-commerce decisions.</p>
</div>
""", unsafe_allow_html=True)
# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚙️ Dashboard")
    st.caption("ONLINE PURCHASE CONVERSION ADVISOR")
    st.divider()
    st.markdown("### 🎯 Prediction settings")
    st.info(
        "Enter visitor session details to estimate "
        "purchase and non-purchase probabilities."
    )
    st.markdown("**Model:** Trained classifier")
    st.markdown("**Task:** Binary classification")
    st.markdown("**Output:** Two complementary probabilities")
# ---------------- INPUT PANEL ----------------
st.markdown('<div class="section-title">👤 Visitor Session Details</div>',
            unsafe_allow_html=True)
st.caption("Adjust the values below, then run the analysis.")
with st.container(border=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        product_related = st.slider(
            "🛍 Product pages viewed", 0, 100, 10
        )
        product_related_duration = st.slider(
            "⏱ Product page time (seconds)", 0, 5000, 300
        )
        page_values = st.slider(
            "💰 Page value", 0.0, 100.0, 5.0, step=0.5
        )
    with col2:
        bounce_rates = st.slider(
            "🚪 Bounce rate", 0.0, 1.0, 0.02, step=0.01
        )
        exit_rates = st.slider(
            "📤 Exit rate", 0.0, 1.0, 0.05, step=0.01
        )
        visitor_type = st.selectbox(
            "👤 Visitor type",
            ["Returning_Visitor", "New_Visitor", "Other"]
        )
    with col3:
        administrative = st.number_input(
            "Administrative pages", 0, 100, 0
        )
        informational = st.number_input(
            "Informational pages", 0, 100, 0
        )
        month = st.selectbox(
            "📅 Month",
            ["nov", "feb", "mar", "may", "jun",
            "jul", "aug", "sep", "oct", "dec"]
        )
        weekend = st.toggle("📆 Weekend session")
    st.markdown("")
    predict = st.button(
        "✨ Analyse Visitor Behaviour",
        use_container_width=True
    )
# ---------------- PREDICTION ----------------
if predict:
    user_data = pd.DataFrame([{
        "Administrative": administrative,
        "Administrative_Duration": 0,
        "Informational": informational,
        "Informational_Duration": 0,
        "ProductRelated": product_related,
        "ProductRelated_Duration": product_related_duration,
        "BounceRates": bounce_rates,
        "ExitRates": exit_rates,
        "PageValues": page_values,
        "SpecialDay": 0,
        "Month": month,
        "OperatingSystems": 2,
        "Browser": 2,
        "Region": 1,
        "TrafficType": 2,
        "VisitorType": visitor_type,
        "Weekend": weekend
    }])
    # Use the same encoding and feature order as training.
    user_data = pd.get_dummies(user_data, dtype=int)
    user_data = user_data.reindex(
        columns=feature_columns, fill_value=0
    )
    try:
        selected_data = selector.transform(user_data)
        # One prediction; both probabilities come from it.
        probabilities = model.predict_proba(selected_data)[0]
        classes = list(model.classes_)
        buy_index = classes.index(1)
        no_buy_index = classes.index(0)
        buy_probability = float(probabilities[buy_index])
        no_buy_probability = float(probabilities[no_buy_index])
        prediction = model.predict(selected_data)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()
    st.divider()
    st.markdown(
        '<div class="section-title">📊 AI Prediction Results</div>',
        unsafe_allow_html=True
    )
    st.caption(
        "Both outcomes are calculated from the same model evaluation."
    )
    # BOTH OUTCOME CARDS SHOWN SIMULTANEOUSLY
    buy_col, no_buy_col = st.columns(2)
    with buy_col:
        st.markdown(f"""
        <div class="buy-card">
            <div style="font-size:35px">🟢</div>
            <h2 style="color:#a7f3d0">Will Buy</h2>
            <div class="big-number" style="color:#6ee7b7">
                {buy_probability:.2%}
            </div>
            <p class="muted">Estimated purchase probability</p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(buy_probability)
        if prediction == 1:
            st.success("Most likely outcome: Purchase")
        else:
            st.info("Alternative outcome: Purchase")
    with no_buy_col:
        st.markdown(f"""
        <div class="no-buy-card">
            <div style="font-size:35px">🔴</div>
            <h2 style="color:#fecdd3">Will Not Buy</h2>
            <div class="big-number" style="color:#fda4af">
                {no_buy_probability:.2%}
            </div>
            <p class="muted">Estimated non-purchase probability</p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(no_buy_probability)
        if prediction == 0:
            st.error("Most likely outcome: No purchase")
        else:
            st.info("Alternative outcome: No purchase")
    # ---------------- ANALYSIS ----------------
    st.markdown("### 📈 Conversion Overview")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Predicted outcome",
                "Will Buy" if prediction == 1 else "Will Not Buy")
    with m2:
        st.metric("Purchase probability", f"{buy_probability:.2%}")
    with m3:
        st.metric("Non-purchase probability", f"{no_buy_probability:.2%}")
    st.markdown("### 🎯 Conversion Segment")
    if buy_probability >= 0.70:
        st.success("🟢 High conversion likelihood")
    elif buy_probability >= 0.40:
        st.warning("🟡 Medium conversion likelihood")
    else:
        st.error("🔴 Low conversion likelihood")
    # ---------------- RECOMMENDATIONS ----------------
    st.markdown("### 💡 Recommended Actions")
    action1, action2 = st.columns(2)
    with action1:
        st.markdown('<div class="buy-card">',
                    unsafe_allow_html=True)
        st.markdown("#### 🟢 If the visitor buys")
        st.markdown("""
        - Make checkout fast and simple.
        - Recommend relevant complementary products.
        - Offer suitable bundles or loyalty benefits.
        - Avoid unnecessary checkout interruptions.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    with action2:
        st.markdown('<div class="no-buy-card">',
                    unsafe_allow_html=True)
        st.markdown("#### 🔴 If the visitor does not buy")
        st.markdown("""
        - Improve product recommendations.
        - Make shipping costs and returns clear.
        - Consider a relevant offer where appropriate.
        - Reduce navigation and checkout friction.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    # ---------------- VISITOR SUMMARY ----------------
    st.markdown("### 📋 Visitor Session Summary")
    summary = pd.DataFrame({
        "Feature": [
            "Product pages viewed",
            "Product page duration",
            "Page value",
            "Bounce rate",
            "Exit rate",
            "Visitor type",
            "Administrative pages",
            "Informational pages",
            "Month",
            "Weekend session"
        ],
        "Value": [
            product_related,
            product_related_duration,
            page_values,
            bounce_rates,
            exit_rates,
            visitor_type,
            administrative,
            informational,
            month,
            weekend
        ]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.caption(
        "Note: Probabilities are model estimates, not guarantees. "
        "The recommendations are general decision-support suggestions."
    )
else:
    st.markdown("""
    <div class="panel">
        <h3>🚀 Ready to analyse a visitor?</h3>
        <p class="muted">
        Enter session details above and click
        <b>Analyse Visitor Behaviour</b> to see both predicted outcomes,
        their probabilities, and suggested actions.
        </p>
    </div>
    """, unsafe_allow_html=True)
