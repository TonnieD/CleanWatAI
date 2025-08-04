import streamlit as st
import sklearn
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

PIPELINE = joblib.load('models/water_quality_pipeline.pkl')

all_demo_cases = [
    {'name': 'Select a scenario matching your area', 'description': '', 'expected': '', 'inputs': {'pH': 7.4, 'TEMP': 22.0, 'EC': 350.0}},
    {'name': 'Clean Borehole Water', 'description': 'A properly maintained borehole with good mineral balance.', 'expected': 'Safe', 'inputs': {'pH': 7.4, 'TEMP': 21.0, 'EC': 350.0}},
    {'name': 'Urban River Contamination', 'description': 'River water downstream from a dense urban area.', 'expected': 'Action Required', 'inputs': {'pH': 7.9, 'TEMP': 26.0, 'EC': 1800.0}},
    {'name': 'Rift Valley Mineral Spring', 'description': 'Geothermal spring with high natural mineral content.', 'expected': 'Action Required', 'inputs': {'pH': 8.8, 'TEMP': 35.0, 'EC': 2200.0}},
    {'name': 'High-Altitude Forest Stream', 'description': 'Cold, pristine stream in a protected forest.', 'expected': 'Safe', 'inputs': {'pH': 7.1, 'TEMP': 16.0, 'EC': 250.0}},
    {'name': 'Livestock Watering Dam', 'description': 'Community dam with high livestock concentration.', 'expected': 'Action Required', 'inputs': {'pH': 8.3, 'TEMP': 29.0, 'EC': 1900.0}},
    {'name': 'Industrial Wastewater Discharge', 'description': 'Water downstream from a factory with chemical runoff.', 'expected': 'Action Required', 'inputs': {'pH': 5.8, 'TEMP': 32.0, 'EC': 2800.0}},
    {'name': 'Shallow Well (Farming Area)', 'description': 'Hand-dug well in an area with heavy fertilizer use.', 'expected': 'Action Required', 'inputs': {'pH': 8.4, 'TEMP': 24.0, 'EC': 1400.0}},
    {'name': 'Municipal Tap Water', 'description': 'Treated municipal water supply.', 'expected': 'Safe', 'inputs': {'pH': 7.2, 'TEMP': 22.0, 'EC': 420.0}},
    {'name': 'Acid Mine Drainage', 'description': 'Runoff from an abandoned mine with heavy metals.', 'expected': 'Action Required', 'inputs': {'pH': 4.2, 'TEMP': 28.0, 'EC': 3500.0}},
    {'name': 'Premium Bottled Water', 'description': 'Commercial bottled water from a reputable brand.', 'expected': 'Safe', 'inputs': {'pH': 7.0, 'TEMP': 20.0, 'EC': 180.0}}
]
case_names = [case['name'] for case in all_demo_cases]
DEFAULT_ENCODED_VALUE = 0 

if 'ph' not in st.session_state:
    st.session_state.ph = all_demo_cases[0]['inputs']['pH']
    st.session_state.temp = all_demo_cases[0]['inputs']['TEMP']
    st.session_state.ec = all_demo_cases[0]['inputs']['EC']
    st.session_state.description = all_demo_cases[0]['description']

def update_state_from_selection():
    selected_case_name = st.session_state.case_selection
    for case in all_demo_cases:
        if case['name'] == selected_case_name:
            st.session_state.ph = case['inputs']['pH']
            st.session_state.temp = case['inputs']['TEMP']
            st.session_state.ec = case['inputs']['EC']
            st.session_state.description = case['description']
            break

# Download necessary NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Dependent functions for the nlp_pipeline
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')

#cleaning + lemmatization function
def clean_text(text):
    # Lowercase
    text = text.lower()
    
    # Handle negations (combine with next word if possible)
    text = re.sub(r"\b(no|not|never)\s+(\w+)", r"no_\2", text)

    # Remove punctuation except underscores (used in negation)
    text = re.sub(r"[^\w\s_]", "", text)

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Lemmatize and remove stopwords (except 'no_x' preserved words)
    cleaned = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if (token in stop_words and token.startswith('no_')) or (token not in stop_words and len(token) > 2)
    ]

    return " ".join(cleaned)

def clean_texts(texts):
    return [clean_text(text) for text in texts]


# Models and pipeline loading
model_path = Path(__file__).parent / "models" / "nlp_pipeline.pkl"
nlp_pipeline = joblib.load(model_path) # NLP model

env_model_path = Path(__file__).parent / "models" / "environmental.pkl"
model = joblib.load(env_model_path) # Environmental model

# Config
st.set_page_config(
    page_title="CleanWatAI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Header section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("💧 CleanWatAI")
    st.subheader("What do you want to know about your local water quality?")

st.markdown("<br><br><br>", unsafe_allow_html=True)

# TextBox and Select Boxes
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    user_text = st.text_area(label="Describe what you want to know", height=150)

    col1, col2, col3 = st.columns(3)
    with col1:
        color = st.selectbox("Water Color", options=["", "Clear", "Brown", "Green", "Other"])
    with col2:
        clarity = st.selectbox("Clarity", options=["", "Clear", "Murky"])
    with col3:
        odor = st.selectbox("Odor", options=["", "None", "Chemical", "Sewage", "Other"])

    col1, col2, col3 = st.columns(3)
    with col1:
        rain = st.selectbox("Recent Rain", options=["", "No recent rain", "Light rain", "Heavy rain"])
    with col2:
        activity = st.selectbox("Nearby Activity", options=["", "Residential", "Industrial", "Agricultural", "None"])
    with col3:
        infrastructure = st.selectbox("Infrastructure", options=["", "Good condition", "Needs repair", "Unknown"])
    context_parts = []

    if color:
        context_parts.append(f"The water appears {color.lower()} in color.")
    if clarity:
        context_parts.append(f"It is {clarity.lower()} in clarity.")
    if odor:
        context_parts.append(f"It has a {odor.lower()} odor.")

    if rain:
        context_parts.append(f"There was {rain.lower()}.")
    if activity:
        context_parts.append(f"The area nearby is {activity.lower()}.")
    if infrastructure:
        context_parts.append(f"The infrastructure is (in) {infrastructure.lower()}.")

    # Combine original input + context
    combined_description = user_text.strip() + " " + " ".join(context_parts)

    # Edited text area with combined description from the select boxes and user input
    edited_description = st.text_area("📝 Final Input to the Model (Editable)", value=combined_description, height=200)

    col1, col2, col3 = st.columns(3)
    with col3:
        if st.button("Submit", type="primary", use_container_width=True):
            # Only runs when Submit is clicked
            if not edited_description or edited_description.strip() == "":
                st.warning("Please describe your concern in the text area above.")
            else:
                prediction = nlp_pipeline.predict([edited_description])[0]
                probability = nlp_pipeline.predict_proba([edited_description])[0][prediction]

                # Map prediction to label
                label_map = {0: "Safe", 1: "Unsafe"}
                prediction_label = label_map[prediction]

                # Display result with appropriate style
                if prediction_label == "Safe":
                    st.success(f"✅ Water is predicted to be SAFE.\nConfidence: {probability:.2%}")
                else:
                    st.error(f"⚠️ Water is predicted to be UNSAFE.\nConfidence: {probability:.2%}")



st.markdown("<br><br><br>", unsafe_allow_html=True)

# Dashboard sections
col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1],gap="large")
with col2:
    with st.container():
        st.subheader("Quick Insights")
        st.text("")
        st.selectbox("Select Location", options=["Nairobi Central", "Nairobi West", "Nairobi East"])

        st.text("Monitoring Stations: 8 active")
        st.text("")
        st.text("Current Status:")
        st.text("⚠️ Moderate Risk")
        st.text("")
        st.text("Trend: ↗️ Slight increase")
        st.text("Last Updated: Today 01:03")
with col3:
    with st.container():
        st.subheader("Latest Alerts")
        st.text("")
        st.text("🦠 Microbial contamination")
        st.text("Kiambiu area • 2h ago")
        st.text("")
        st.text("🌊 High turbidity levels")
        st.text("Industrial zone • 3h ago")
        st.text("")
        st.text("⚗️ Chemical levels elevated")
        st.text("Industrial zone • 1d ago")
        st.text("")
        st.button("View All", key="view_alerts")
with col4:
    with st.container():
        st.subheader("Reports & Analytics")
        st.text("")
        st.text("📈 Water Quality Trends")
        st.text("Urban Areas • Jul 15")
        st.text("")
        st.text("🌍 Regional Analysis")
        st.text("East Africa • Jun 30")
        st.text("")
        st.text("🏙️ Infrastructure Assessment")
        st.text("CBD Systems • Jun 22")
        st.text("")
        st.button("View All", key="view_reports")
        
st.markdown("<br><br><br>", unsafe_allow_html=True)
        
# Geospatial Map Section
with st.container(border=True):
    st.markdown("<br>", unsafe_allow_html=True)
    
    header_cols = st.columns([.5, 10, .5]) 
    with header_cols[1]:
        st.header("🗺️ Water Contamination Risk Map")
        st.text("Explore water quality risks in your area")
    
    st.text("")  
    st.text("") 
    
    map_cols = st.columns([.5, 10, .5])
    with map_cols[1]:
        with st.container(border=True):
            st.text("")
            
        #Load data
        df = pd.read_csv("data/processed/environmental.csv")
        df = df.dropna(subset=["latitude", "longitude"])
        df.dropna(axis=1, how="all", inplace=True)

        #Add location name
        df["location_name"] = (
            df["clean_adm3"].fillna("") + ", " +
            df["clean_adm2"].fillna("") + ", " +
            df["clean_adm1"].fillna("")
        )

        # Predict risk
        df["predicted_risk"] = model.predict(df)
        
        def risk_label(r):
            return {
                0: "🟢 Safe Quality",
                1: "🟡 Low Risk",
                2: "🟠 Medium Risk",
                3: "🔴 High Risk"
            }.get(r, "Unknown")

        def risk_color(r):
            return {
                0: [0, 255, 0, 160],
                1: [255, 255, 0, 160],
                2: [255, 165, 0, 160],
                3: [255, 0, 0, 160]
            }.get(r, [128, 128, 128, 160])

        #print("📦 Available columns in df:", df.columns.tolist())
        #print(df.head())

        df["risk_label"] = df["predicted_risk"].apply(risk_label)
        df["color"] = df["predicted_risk"].apply(risk_color)
        df["risk_label_clean"] = df["risk_label"].replace({
            "🔴 High Risk": "High Risk",
            "🟠 Medium Risk": "Medium Risk",
            "🟡 Low Risk": "Low Risk",
            "🟢 Safe Quality": "Safe Quality"
        })

        df["quality_score"] = (1 - df["predicted_risk"] / 3 * 0.75) * 100
        df["quality_score"] = df["quality_score"].round(1)

        df["risk_level"] = df["risk_label_clean"]

        if "contamination_type" not in df.columns:
            df["contamination_type"] = "Not available"


        available_risks = sorted(df["risk_label"].unique().tolist())
        selected_risks = st.multiselect("Filter by risk level:", available_risks, default=available_risks)
        filtered_df = df[df["risk_label"].isin(selected_risks)]

        # Handle case where no points are selected — show blank map
        if filtered_df.empty:
            st.info("No risk levels selected. Displaying a blank global map.")

            # Set global view
            view_state = pdk.ViewState(
                latitude=0.0,
                longitude=0.0,
                zoom=1,
                pitch=0
            )

            # Set empty DataFrame for rendering
            filtered_df = pd.DataFrame(columns=["latitude", "longitude", "location_name", "color", "risk_label"])
        else:
            view_state = pdk.ViewState(
                latitude=filtered_df["latitude"].mean(),
                longitude=filtered_df["longitude"].mean(),
                zoom=6,
                pitch=0
            )

        # Build map layer
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=filtered_df,
            get_position='[longitude, latitude]' if not filtered_df.empty else None,
            get_fill_color="color" if not filtered_df.empty else None,
            get_radius=3000,
            pickable=True,
        )

        # Tooltip definition
        tooltip = {
            "html": "<b>{location_name}</b><br/>"
                    "Risk: {risk_label}<br/>"
                    "Lat: {latitude}<br/>Lon: {longitude}",
            "style": {"backgroundColor": "white", "color": "black"}
        }

        # Show map
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip
        ))

        # Point selector and detail box
        if not filtered_df.empty:
            selected_point = st.selectbox(
                "Select a monitoring point for details:",
                options=filtered_df['location_name'].tolist()
            )

            if selected_point:
                point_data = filtered_df[filtered_df['location_name'] == selected_point].iloc[0]

                with st.container(border=True):
                    st.subheader("Point Details")
                    st.text("")

                    detail_col1, detail_col2 = st.columns(2)

                    with detail_col1:
                        st.metric("Quality Score", f"{point_data.get('quality_score', 'N/A')}/100")
                        st.text(f"Risk Level: {point_data.get('risk_level', 'Unknown')}")

                    with detail_col2:
                        st.text(f"Contamination: {point_data.get('contamination_type', 'Not available')}")
                        st.text(f"Location: {point_data['latitude']:.4f}, {point_data['longitude']:.4f}")

                    st.text("")
                    st.text("")
                    st.text("")

            
            # Add bottom padding
            st.text("")
    
    st.text("")  # Vertical spacing
    st.text("")  # Additional vertical spacing
    
    subheader_cols = st.columns([.5, 10, .5])  
    with subheader_cols[1]:
        with st.container(border=True):
            
            st.header("Anomaly Check for Water Quality based on Chemical Content")
            col1, col2 = st.columns([1, 1], gap="large")

            with col1:
                st.selectbox(
                    "Load a Pre-built Scenario",
                    options=case_names,
                    key='case_selection',
                    on_change=update_state_from_selection
                )

                st.subheader("Sensor Inputs")
                ph_val = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=st.session_state.ph, step=0.1)
                temp_val = st.number_input("Temperature (°C)", min_value=0.0, max_value=100.0, value=st.session_state.temp, step=0.5)
                ec_val = st.number_input("Electrical Conductivity (µS/cm)", min_value=0.0, max_value=10000.0, value=st.session_state.ec, step=10.0)

                if st.button("Check Risk", type="primary", use_container_width=True):
                    inputs_dict = {
                        'pH': ph_val,
                        'TEMP': temp_val,
                        'EC': ec_val,
                        'station_encoded': DEFAULT_ENCODED_VALUE
                    }
                    inputs_df = pd.DataFrame([inputs_dict])
                    risk_score = PIPELINE.predict_risk(inputs_df)[0]
                    st.session_state.risk_score = risk_score
                    st.session_state.last_case_checked = st.session_state.case_selection

            with col2:
                if 'risk_score' in st.session_state:
                    with st.container(border=True):
                        risk_score = st.session_state.risk_score
                        last_case_name = st.session_state.last_case_checked
                        
                        # Find the expected outcome for the tested case
                        expected_outcome = "N/A"
                        for case in all_demo_cases:
                            if case['name'] == last_case_name:
                                expected_outcome = case['expected']
                                break
                        
                        st.subheader(f"Results")
                        predicted_outcome = "Action Required" if risk_score > 0.5 else "Safe"
                        
                        st.metric(label="Chemically Predicted Risk Score", value=f"{risk_score:.3f}")

                        if predicted_outcome == "Action Required":
                            st.error(f"Verdict: {predicted_outcome}")
                        else:
                            st.success(f"Verdict: {predicted_outcome}")

                        # Display the check only if it was a pre-built scenario
                        if expected_outcome != "N/A" and expected_outcome != '':
                            st.markdown("---")
                            emoji = "✅" if predicted_outcome == expected_outcome else "❌"
                            st.write(f"**Expected:** {expected_outcome} | **Prediction Matches Expected:** {emoji}")

    st.text("")  # Vertical spacing
    st.text("")  # Additional vertical spacing
    
    monitoring_header_cols = st.columns([.5, 10, .5])
    with monitoring_header_cols[1]:
        st.subheader("Monitoring Data")
    
    monitoring_outer_cols = st.columns([.5, 10, .5])
    with monitoring_outer_cols[1]:
        with st.container(border=True):
            st.text("")
            tab1, tab2 = st.tabs(["Coverage Statistics", "Data Sources"])
            
            with tab1:
                total_points = len(filtered_df)
                st.metric("Total Points", total_points)
                st.text("Coverage: 95% of region") 
                st.text("Updated: Today")
            
            with tab2:
                st.text("Sources:")
                st.text("• WPDx (Water Point Data Exchange)")
                st.text("• Google Earth Engine")
                st.text("• Field Observations")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# 📊 Latest Public Water Data Section
st.header("📊 Latest Public Water Data")
data_cols = st.columns([.5, 10, .5])
with data_cols[1]:
    st.text("Showing the most recent water point data with model-predicted risk levels")
    st.text("Last updated: August 3, 2025")
    st.text("")

    with st.container(border=True):
        data_tab1, data_tab2, data_tab3, data_tab4 = st.tabs(["All Data", "Functional Status", "Risk Analysis", "Quality Trend"])
        
        # TAB 1: ALL DATA
        with data_tab1:
            st.dataframe(
                df,
                column_config={
                    "water_point_id": "Water Point ID",
                    "country_name": "Country",
                    "region": "Region",
                    "district": "District",
                    "water_source": "Water Source",
                    "status": st.column_config.SelectboxColumn(
                        "Status",
                        help="Operational status of the water point",
                        width="medium",
                        options=[
                            "Functional",
                            "Non-functional",
                            "Needs repair"
                        ]
                    ),
                    "date_installed": st.column_config.DateColumn(
                        "Installation Date",
                        format="MMM DD, YYYY",
                    ),
                    "latest_record": st.column_config.DateColumn(
                        "Last Updated",
                        format="MMM DD, YYYY",
                    ),
                    "water_quality": "Water Quality",
                    "risk_score": st.column_config.ProgressColumn(
                        "Risk Score",
                        help="Higher score = lower risk",
                        format="%d",
                        min_value=0,
                        max_value=100,
                    ),
                    "risk_level": "Risk Level"
                },
                hide_index=True,
                use_container_width=True
            )

        # TAB 2: FUNCTIONAL STATUS
        def clean_status(status):
            status = status.strip().lower()

            if "non-functional" in status:
                return "Non-Functional"
            elif "functional" in status:
                return "Functional"
            elif "abandoned" in status or "decommissioned" in status:
                return "Decommissioned"
            else:
                return "Unknown"

        df['status_clean'] = df['status_clean'].fillna("Unknown").apply(clean_status)

        with data_tab2:
            status_counts = df['status_clean'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']

            
            st.bar_chart(status_counts, x='Status', y='Count')
            
            st.text("Filter data by status:")
            status_filter = st.multiselect(
                "Select status to view:",
                options=df['status_clean'].unique(),
                default=df['status_clean'].unique()
            )
            
            filtered_status_data = df[df['status_clean'].isin(status_filter)]
            st.dataframe(filtered_status_data, hide_index=True, use_container_width=True)

        # TAB 3: RISK ANALYSIS
        with data_tab3:
            risk_counts = df['risk_level'].value_counts().reset_index()
            risk_counts.columns = ['Risk Level', 'Count']

            st.text("Average Risk Score by Region")
            region_risk = df.groupby('clean_adm1')['risk_score'].mean().reset_index()
            region_risk.columns = ['Region', 'Average Risk Score']

            st.bar_chart(region_risk, x='Region', y='Average Risk Score')

            st.text("Filter data by risk level:")
            risk_filter = st.multiselect(
                "Select risk level to view:",
                options=df['risk_level'].unique(),
                default=df['risk_level'].unique()
            )

            filtered_risk_data = df[df['risk_level'].isin(risk_filter)]
            st.dataframe(filtered_risk_data, hide_index=True, use_container_width=True)

        # TAB 4: QUALITY TREND
        with data_tab4:
            st.text("Water Quality Trend Over Time")
            st.text("")

            # ✅ Step 1: Define the new column (do this outside any widget)
            df['point_id'] = df['location_name'].fillna('Unnamed') + ' (' + df['latitude'].astype(str) + ', ' + df['longitude'].astype(str) + ')'

            # ✅ Step 2: Use it in the selectbox
            selected_water_point = st.selectbox(
                "Select a water point to view quality trend:",
                options=df['point_id'].tolist()
            )


            if selected_water_point:
                point_data = df[df['point_id'] == selected_water_point].iloc[0]

                st.text(f"30-day quality trend for {selected_water_point}")
                st.text("")

                days = 30
                dates = pd.date_range(end=pd.Timestamp.now(), periods=days)

                base_quality = point_data['risk_score']
                noise = np.random.normal(0, 5, days)
                trend = np.linspace(-10, 10, days)

                quality_trend = np.clip(base_quality + trend + noise, 0, 100)

                trend_data = pd.DataFrame({
                    'date': dates,
                    'quality': quality_trend
                })

                st.line_chart(trend_data.set_index('date'))

                avg_quality = quality_trend.mean()
                st.text(f"Average Quality Score: {avg_quality:.1f}/100")

                if trend[-1] > trend[0]:
                    st.text("Trend: ↗️ Improving")
                elif trend[-1] < trend[0]:
                    st.text("Trend: ↘️ Declining")
                else:
                    st.text("Trend: → Stable")

        # CSV download
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Full Water Data (CSV)",
            data=csv,
            file_name="cleanwat_ai_water_data.csv",
            mime="text/csv"
        )

        st.text("")
        st.text("")
        with st.container(border=True):
            st.caption("© 2025 CleanWaterAI. Data sourced from WPDx and other public datasets.")
