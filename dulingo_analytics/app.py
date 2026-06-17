import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set clean, modern page layout configuration
st.set_page_config(page_title="App Store Intelligence Dashboard", layout="wide")

st.title("📊 App Store Intelligence Dashboard")
st.caption("Predictive Strategy Engine for Product Managers and Mobile Publishers (Duolingo Dataset Analysis)")
st.markdown("---")

# Load serialized model assets safely
@st.cache_resource
def load_assets():
    try:
        with open('src/feature_classification/sentiment_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('src/feature_classification/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_assets()

if model is None or scaler is None:
    st.error("⚠️ Physical model assets not found! Please check that 'sentiment_model.pkl' and 'scaler.pkl' are inside 'src/feature_classification/'")
    st.stop()

# --- PUBLISHER METRICS PANEL ---
st.subheader("🎯 Real-Time Update Risk Predictor")
st.markdown("Simulate how a new app update or monetization change will affect your user satisfaction metrics.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 💰 Price Configuration")
    monetization_active = st.toggle("Introduce New Paywall / More Ads", value=False)
    mentions_money = 1 if monetization_active else 0

with col2:
    st.markdown("### 📱 Product Configuration")
    layout_changed = st.toggle("Modify Core UI Layout / Feature Flow", value=True)
    mentions_update = 1 if layout_changed else 0

with col3:
    st.markdown("### 👥 Review Metadata")
    review_length = st.slider("Simulated Review Text Length (Words)", min_value=5, max_value=120, value=45)
    thumbs_up = st.slider("Target Community Upvotes (Social Weight)", min_value=0, max_value=50, value=5)

# CRITICAL FIX: Ensure the column names and order match your training array exactly
input_features = pd.DataFrame([{
    'mentions_money': float(mentions_money),
    'mentions_update': float(mentions_update),
    'review_length': float(review_length),
    'thumbsUpCount': float(thumbs_up),
    'is_weekend': 0.0
}])

# Execute prediction safely with correct transformations
try:
    scaled_input = scaler.transform(input_features)
    satisfaction_probability = model.predict_proba(scaled_input)[0][1]
    churn_risk_probability = 1.0 - satisfaction_probability
except Exception as e:
    st.error(f"Transformation Error: {e}")
    st.stop()

# --- OUTCOME & STRATEGIC RECOMMENDATION DISPLAY ---
st.markdown("---")
st.subheader("📈 Publisher Optimization Report")

res_col1, res_col2 = st.columns(2)

with res_col1:
    st.metric(
        label="Predicted User Satisfaction Rate", 
        value=f"{satisfaction_probability * 100:.2f}%"
    )
    st.progress(float(satisfaction_probability))

with res_col2:
    st.metric(
        label="Predicted User Churn Risk Window", 
        value=f"{churn_risk_probability * 100:.2f}%"
    )
    st.progress(float(churn_risk_probability))

# Actionable Strategies for App Publishers
st.markdown("### 💡 Automated Strategic Intervention Plans")
if churn_risk_probability > 0.4:
    st.warning("⚠️ **High Store Optimization Risk:** The combined impact of your updates will trigger visibility penalties.")
    st.markdown("""
    * **Developer Action:** Roll out a localized fallback menu within 48 hours to mitigate user layout confusion.
    * **Marketing Action:** Deploy an in-app reward notification giving active segments 3 days of ad-free access to prevent storefront rating decay.
    """)
else:
    st.success("✅ **Stable Store Optimization Window:** Predicted feedback parameters remain within the positive organic growth threshold.")
    st.markdown("""
    * **Marketing Action:** Automatically prompt users in this segment with store rating links to accelerate organic 5-star review collection.
    """)