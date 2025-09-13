import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Medical Device Failure Predictor",
    page_icon="🩺",
    layout="wide"
)

# --- Asset Loading ---
# This function loads both the trained model and the pre-calculated feature importances.
@st.cache_resource
def load_assets():
    """Loads the model and feature importances from pickle files."""
    try:
        with open('model/final_model.pkl', 'rb') as f:
            pipeline = pickle.load(f)
    except FileNotFoundError:
        pipeline = None

    try:
        with open('model/feature_importances.pkl', 'rb') as f:
            importances_df = pickle.load(f)
    except FileNotFoundError:
        importances_df = None
        
    return pipeline, importances_df

# Load the assets when the app starts.
pipeline, importances_df = load_assets()

# --- Pre-populated Dropdown Options ---
# These lists make the app self-contained and fast.
risk_class_options = ['Class I', 'Class II', 'Class III', 'Not Classified', 'Class IIa', 'Class IIb', 'Class IV', 'Class A', 'Class B', 'Class C', 'Class D']
country_options = ['USA', 'DEU', 'CAN', 'GBR', 'AUS', 'FRA', 'ITA', 'ESP', 'CHE', 'JPN'] 
manufacturer_options = [
    'Medtronic', 'Abbott', 'Boston Scientific', 'Johnson & Johnson', 
    'Siemens Healthineers', 'Philips', 'Stryker', 'BD', 'GE Healthcare', 
    'Zimmer Biomet', 'Baxter', 'Smith & Nephew'
]

# --- App UI ---
st.title("🩺 Medical Device Failure Predictor")
st.markdown("Enter event details to predict the recall likelihood.")

# Only show the main content if the model has loaded successfully.
if pipeline is not None:
    # Use a form for a clean user experience.
    with st.form("prediction_form"):
        st.header("Event Details")
        col1, col2 = st.columns(2)
        with col1:
            risk_class = st.selectbox('Risk Class', options=sorted(risk_class_options))
            country = st.selectbox('Country', options=sorted(country_options))
        with col2:
            manufacturer = st.selectbox('Manufacturer', options=sorted(manufacturer_options))
        
        reason_text = st.text_area('Reason Text for the Event', 
                                   'A software update caused the device to malfunction during testing, leading to inaccurate readings.', 
                                   height=150)
        
        submit_button = st.form_submit_button(label='Analyze and Predict Risk')

    # This block of code runs only when the user clicks the submit button.
    if submit_button:
        # Create a DataFrame with the exact column names the model was trained on.
        input_data = pd.DataFrame([[risk_class, country, manufacturer, reason_text]],
                                  columns=['risk_class', 'country_event', 'manufacturer_name', 'reason'])
        
        # Get the probability of a "Failure" (class 1).
        prediction_proba = pipeline.predict_proba(input_data)[0][1]
        
        st.markdown("---")
        st.header("Prediction Result")
        
        # Display the result in a metric card.
        st.metric("Risk of Recall (Failure)", f"{prediction_proba*100:.2f}%")
        if prediction_proba > 0.6:
            st.error("Status: High Risk")
        elif prediction_proba > 0.3:
            st.warning("Status: Moderate Risk")
        else:
            st.success("Status: Low Risk")

    # --- Explanation Section (Replaced SHAP with Feature Importance Chart) ---
    st.markdown("---")
    st.header("What the Model Considers Important")
    st.markdown("This chart shows the top features the model uses to make predictions *in general*. Features related to the `reason` text are often the most influential.")

    # Display the chart if the importance data loaded correctly.
    if importances_df is not None:
        top_features = importances_df.head(15)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(top_features['feature'], top_features['importance'], color='#007BFF')
        ax.invert_yaxis() # Display the most important feature at the top.
        ax.set_title("Top 15 Most Important Features", fontsize=16)
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Feature importance data ('feature_importances.pkl') not found.")

else:
    st.error("The application cannot start because the model file ('final_model.pkl') is missing.")

