import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DC Crime Insight",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
# FIX: Explicitly set text colors (color: #...) to prevent White-on-White/Blue issues in Dark Mode
st.markdown("""
    <style>
    /* Force main background slightly off-white for contrast in Light Mode */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* STYLE FOR BUTTONS */
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white !important; /* Force white text on red button */
        font-weight: bold;
        border-radius: 8px;
        height: 50px;
        transition: all 0.3s;
        border: none;
    }
    .stButton>button:hover {
        background-color: #d43f3f;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* STYLE FOR METRIC CARD (Prediction Box) */
    .metric-container {
        background-color: #ffffff; /* Always white background */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 20px;
    }
    /* Critical Fix: Force text to be dark inside the white card */
    .metric-container h2 {
        color: #333333 !important;
        margin-bottom: 5px;
        font-size: 1.2rem;
    }
    .metric-container h1 {
        color: #ff4b4b !important;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }
    .metric-container p {
        color: #666666 !important;
        margin-top: 10px;
    }

    /* STYLE FOR SAFETY TIP BOX */
    .safety-tip {
        background-color: #e3f2fd; /* Light Blue Background */
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #2196f3;
        margin-top: 20px;
    }
    /* Critical Fix: Force text to be Dark Blue inside the light blue box */
    .safety-tip p, .safety-tip strong {
        color: #0d47a1 !important; /* Dark Blue Text */
        margin: 0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load("optimized_model.pkl")
        options = joblib.load("app_options.pkl")
        return model, options
    except FileNotFoundError:
        st.error("⚠️ Files not found! Please run 'train_optimized.py' first.")
        st.stop()

model, options = load_resources()

# --- HEADER ---
st.title("🛡️ DC Crime Insight")
st.markdown("**AI-Powered Safety & Risk Assessment Tool**")
st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("📍 Location")
    selected_cluster = st.selectbox(
        "Neighborhood Cluster", 
        options["clusters"], 
        index=0
    )
    
    st.subheader("📅 Date & Time")
    selected_day = st.selectbox("Day of Week", options["days"])
    selected_month = st.selectbox("Month", options["months"])
    selected_hour = st.slider("Hour of Day (24h)", 0, 23, 12, format="%d:00")

    st.markdown("---")
    predict_btn = st.button("Analyze Risk 🚀")

# --- MAIN CONTENT ---
if predict_btn:
    # 1. Prepare Input
    input_data = pd.DataFrame([{
        "NEIGHBORHOOD_CLUSTER": selected_cluster,
        "HOUR_OF_DAY": selected_hour,
        "DAY_OF_WEEK": selected_day,
        "MONTH_NAME": selected_month
    }])

    # 2. Get Prediction
    prediction = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]
    classes = model.classes_

    # 3. Layout Results
    col1, col2 = st.columns([1, 2])

    with col1:
        # High-Impact Metric Card
        max_prob = np.max(probs) * 100
        
        # We use explicit class names to target the CSS above
        st.markdown(f"""
        <div class="metric-container">
            <h2>⚠️ Primary Risk</h2>
            <h1>{prediction}</h1>
            <p>Confidence: <strong>{max_prob:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # Dynamic Safety Tips
        tip = "Stay alert and keep valuables hidden."
        if "THEFT" in prediction:
            tip = "🔒 <strong>Tip:</strong> Ensure your vehicle is locked and valuables are out of sight. Avoid leaving bags unattended."
        elif "ROBBERY" in prediction:
            tip = "👀 <strong>Tip:</strong> Stay in well-lit areas. Avoid using your phone while walking alone at night."
        elif "ASSAULT" in prediction:
            tip = "🏃 <strong>Tip:</strong> Travel in groups if possible. Trust your instincts and avoid conflict."
        
        # We put this in a paragraph tag to pick up the CSS color rule
        st.markdown(f'<div class="safety-tip"><p>{tip}</p></div>', unsafe_allow_html=True)

    with col2:
        # Plotly Chart
        risk_df = pd.DataFrame({
            "Crime Type": classes,
            "Probability": probs * 100
        }).sort_values(by="Probability", ascending=True)

        # Enhance Chart Colors to match theme
        fig = px.bar(
            risk_df, 
            x="Probability", 
            y="Crime Type", 
            orientation='h',
            title="📊 Probability Breakdown",
            text_auto='.1f',
            height=400
        )
        fig.update_traces(marker_color='#ff4b4b', textfont_size=12)
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Likelihood (%)", 
            yaxis_title=None,
            font=dict(family="Arial", size=14)
        )
        # Update text color in chart for dark mode compatibility
        fig.update_layout(font_color="grey") 
        
        st.plotly_chart(fig, use_container_width=True)

else:
    # --- LANDING PAGE ---
    st.subheader("Welcome to Crime Insight")
    st.info("👈 **Start by selecting a Neighborhood and Time on the sidebar.**")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("### 🗺️ Select Area")
        st.write("Choose from DC's neighborhood clusters to pinpoint the analysis.")
    with col_b:
        st.markdown("### ⏰ Set Time")
        st.write("Crime trends change by hour and season. Input specific times for accuracy.")
    with col_c:
        st.markdown("### 🔍 Get Insights")
        st.write("Our AI analyzes historical patterns to predict the most likely risks.")
    
    st.markdown("---")
    st.caption("Disclaimer: This tool uses historical data for educational purposes and does not predict future events with certainty.")
