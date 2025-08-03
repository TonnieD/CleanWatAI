# Imports
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns
import ee
import os

# Streamlit import and config 
st.set_page_config(page_title="💧 CleanWat AI – Water Contamination Risk Map", layout="wide")


# Earth Engine Authentication
SERVICE_ACCOUNT = 'cleanwatai@ee-dianasituation.iam.gserviceaccount.com'
KEY_PATH = 'service_key.json'

@st.cache_resource
def init_ee():
    try:
        credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_PATH)
        ee.Initialize(credentials)
        return True
    except Exception as e:
        st.error(f"Earth Engine authentication failed: {e}")
        return False

if not init_ee():
    st.stop()
else:
    st.success("✅ Earth Engine authenticated successfully")


# Streamlit Configuration
st.markdown("Explore water quality risks in your area")

# Load Model and Data
model = joblib.load("../models/environmental.pkl")

df = pd.read_csv("../data/processed/environmental.csv")
df = df.dropna(subset=["latitude", "longitude"])
df.dropna(axis=1, how="all", inplace=True)

df["location_name"] = (
    df["clean_adm3"].fillna("") + ", " +
    df["clean_adm2"].fillna("") + ", " +
    df["clean_adm1"].fillna("")
)

df["predicted_risk"] = model.predict(df)


# Risk Formatting
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

df["risk_label"] = df["predicted_risk"].apply(risk_label)
df["color"] = df["predicted_risk"].apply(risk_color)
df["risk_label_clean"] = df["risk_label"].replace({
    "🔴 High Risk": "High Risk",
    "🟠 Medium Risk": "Medium Risk",
    "🟡 Low Risk": "Low Risk",
    "🟢 Safe Quality": "Safe Quality"
})


# Risk Level Filter
available_risks = sorted(df["risk_label"].unique().tolist())
selected_risks = st.multiselect("Filter by risk level:", available_risks, default=available_risks)
filtered_df = df[df["risk_label"].isin(selected_risks)]


# Interactive Map
view_state = pdk.ViewState(
    latitude=filtered_df["latitude"].mean(),
    longitude=filtered_df["longitude"].mean(),
    zoom=6,
    pitch=0
)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=filtered_df,
    get_position='[longitude, latitude]',
    get_fill_color="color",
    get_radius=3000,
    pickable=True,
)

tooltip = {
    "html": "<b>{location_name}</b><br/>"
            "Risk: {risk_label}<br/>"
            "Lat: {latitude}<br/>Lon: {longitude}",
    "style": {"backgroundColor": "white", "color": "black"}
}

st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=view_state,
    layers=[layer],
    tooltip=tooltip
))


# Sidebar Point Selection
st.sidebar.title("📍 Select Monitoring Point")
df["location_label"] = df.index.astype(str)
selected_label = st.sidebar.selectbox("Choose water point:", df["location_label"])
selected_index = int(selected_label)
selected_row = df.loc[selected_index]

st.sidebar.markdown("### Point Details")
st.sidebar.markdown(f"**Predicted Risk:** {selected_row['risk_label']}")
st.sidebar.text(f"Latitude: {selected_row['latitude']:.4f}")
st.sidebar.text(f"Longitude: {selected_row['longitude']:.4f}")
st.sidebar.text(f"Location: {selected_row['location_name']}")


# Summary Tabs
st.subheader("📊 Latest Public Water Data")
st.markdown("Showing the most recent water point data from the Water Point Data Exchange (WPDx) for Kenya")
st.markdown("Last updated: July 25, 2025")

tab1, tab2, tab3 = st.tabs(["All Data", "Functional Status", "Risk Analysis"])

with tab1:
    st.dataframe(
        df,
        column_order=["location_name", "latitude", "longitude", "risk_label"] +
                     [col for col in df.columns if col not in ["location_name", "latitude", "longitude", "risk_label"]]
    )
    st.download_button("Download WPDX Kenya Data (CSV)", df.to_csv(index=False), file_name="environmental_full_data.csv")

with tab2:
    st.markdown("### Functional Status")
    if "status_clean" in df.columns:
        fig1, ax1 = plt.subplots()
        df["status_clean"].value_counts().plot(kind="bar", ax=ax1, color="skyblue")
        ax1.set_ylabel("Count")
        ax1.set_title("Functional Status of Water Points")
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig1)

        status_options = df["status_clean"].dropna().unique().tolist()
        selected_status = st.selectbox("Filter by functional status:", status_options)
        filtered_status_df = df[df["status_clean"] == selected_status]
        st.dataframe(filtered_status_df[["location_name", "latitude", "longitude", "risk_label", "status_clean"]])
        st.download_button("Download Data", filtered_status_df.to_csv(index=False), file_name="functional_filtered.csv")
    else:
        st.info("Functional status data not available.")

with tab3:
    st.markdown("### 📈 Risk Score by Region")
    region_scores = df.groupby("clean_adm1")["predicted_risk"].mean().sort_values().head(10)
    fig2, ax2 = plt.subplots()
    region_scores.plot(kind="bar", ax=ax2, colormap="coolwarm")
    ax2.set_ylabel("Average Predicted Risk Score")
    ax2.set_xlabel("Region")
    ax2.set_title("Risk Score by Region")
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig2)

    risk_filter_options = df["risk_label_clean"].unique().tolist()
    selected_risk_filter = st.multiselect("Filter by risk level:", risk_filter_options, default=risk_filter_options)
    filtered_risk_df = df[df["risk_label_clean"].isin(selected_risk_filter)]
    st.dataframe(filtered_risk_df[["location_name", "latitude", "longitude", "risk_label", "clean_adm1"]])
    st.download_button("Download Risk Data", filtered_risk_df.to_csv(index=False), file_name="risk_filtered.csv")


# Footer
st.markdown("---")
st.markdown("GitHub: [CleanWatAI](https://github.com/TonnieD/CleanWatAI)")
st.markdown("© 2025 CleanWatAI. Data sourced from Water Point Data Exchange (WPDx).")
