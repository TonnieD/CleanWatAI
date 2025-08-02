import streamlit as st
import pandas as pd
import numpy as np

# Config
st.set_page_config(
    page_title="CleanWater AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Header section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("💧 CleanWater AI", width= "stretch")
    st.subheader("What do you want to know about your local water quality?")

st.markdown("<br><br><br>", unsafe_allow_html=True)

# TextBox and Select Boxes
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.text_area(label="Descsribe what you want to know", height=150)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.selectbox("Water Color", options=["", "Clear", "Brown", "Green", "Other"])
    with col2:
        st.selectbox("Clarity", options=["", "Clear", "Murky"])
    with col3:
        st.selectbox("Odor", options=["", "None", "Chemical", "Sewage", "Other"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.selectbox("Recent Rain", options=["", "No recent rain", "Light rain", "Heavy rain"])
    with col2:
        st.selectbox("Nearby Activity", options=["", "Residential", "Industrial", "Agricultural", "None"])
    with col3:
        st.selectbox("Infrastructure", options=["", "Good condition", "Needs repair", "Unknown"])
    
    col1, col2, col3 = st.columns(3)
    with col3:
        st.button("Submit", type="primary", use_container_width=True)

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
            
            np.random.seed(42)  
        nairobi_lat, nairobi_lon = -1.286389, 36.817223
        num_points = 20
        
        water_data = pd.DataFrame({
            'lat': nairobi_lat + np.random.normal(0, 0.05, num_points),
            'lon': nairobi_lon + np.random.normal(0, 0.05, num_points),
            'quality_score': np.random.randint(0, 100, num_points),
            'location_name': [f"Location {i+1}" for i in range(num_points)],
            'contamination_type': np.random.choice(
                ['Microbial', 'Chemical', 'Sediment', 'None'], 
                num_points
            )
        })
        
        def get_risk_level(score):
            if score < 25:
                return "High Risk 🔴"
            elif score < 50:
                return "Medium Risk 🟠"
            elif score < 75:
                return "Low Risk 🟡"
            else:
                return "Safe Quality 🟢"
        
        water_data['risk_level'] = water_data['quality_score'].apply(get_risk_level)
        
        def get_color(level):
            if "High" in level:
                return [255, 0, 0, 160] 
            elif "Medium" in level:
                return [255, 165, 0, 160] 
            elif "Low" in level:
                return [255, 255, 0, 160] 
            else:
                return [0, 255, 0, 160]  
        
        water_data['color'] = water_data['risk_level'].apply(get_color)
        
        risk_filter = st.multiselect(
            "Filter by risk level:",
            options=["High Risk 🔴", "Medium Risk 🟠", "Low Risk 🟡", "Safe Quality 🟢"],
            default=["High Risk 🔴", "Medium Risk 🟠", "Low Risk 🟡", "Safe Quality 🟢"]
        )
        
        filtered_data = water_data[water_data['risk_level'].isin(risk_filter)]
        st.map(filtered_data, size=20, color="color")
        selected_point = st.selectbox(
            "Select a monitoring point for details:",
            options=filtered_data['location_name'].tolist()
        )
        
        if selected_point:
            point_data = filtered_data[filtered_data['location_name'] == selected_point].iloc[0]
            
            with st.container(border=True):
                st.subheader("Point Details")
                st.text("")
                
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.metric("Quality Score", f"{point_data['quality_score']}/100")
                    st.text(f"Risk Level: {point_data['risk_level']}")
                
                with detail_col2:
                    st.text(f"Contamination: {point_data['contamination_type']}")
                    st.text(f"Location: {point_data['lat']:.4f}, {point_data['lon']:.4f}")
                
                st.text("")
                st.text("")
                st.text("")
            
            # Add bottom padding
            st.text("")
    
    st.text("")  # Vertical spacing
    st.text("")  # Additional vertical spacing
    
    subheader_cols = st.columns([.5, 10, .5])  # Reverted to 0.5 padding
    with subheader_cols[1]:
        st.subheader("Showing water quality monitoring data for Nairobi Central")

    outer_cols = st.columns([.5, 10, .5])  # Consistent 0.5 padding
    with outer_cols[1]:
        row1_cols = st.columns([6, 6])
        with row1_cols[0]:
            with st.expander("🔴 High Risk", expanded=True):
                st.text("Areas with severe contamination")
                st.text("Immediate action required")
        
        with row1_cols[1]:
            with st.expander("🟠 Medium Risk", expanded=True):
                st.text("Areas with concerning levels")
                st.text("Regular monitoring needed")
        
        row2_cols = st.columns([6, 6])
        with row2_cols[0]:
            with st.expander("🟡 Low Risk", expanded=True):
                st.text("Areas with minor issues")
                st.text("Routine testing recommended")
        
        with row2_cols[1]:
            with st.expander("🟢 Safe Quality", expanded=True):
                st.text("Areas meeting all standards")
                st.text("No action required")

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
                st.text("Total Points: 150")
                st.text("Coverage: 95% of region") 
                st.text("Updated: Today")
            
            with tab2:
                st.text("Sources:")
                st.text("• WPDx (Water Point Data Exchange)")
                st.text("• Google Earth Engine")
                st.text("• Field Observations")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# Latest Public Data Section
st.header("📊 Latest Public Water Data")
data_cols = st.columns([.5, 10, .5])
with data_cols[1]:
    st.text("Showing the most recent water point data from the Water Point Data Exchange (WPDx) for Kenya")
    st.text("Last updated: July 25, 2025")
    st.text("")
    
    def get_wpdx_kenya():
        np.random.seed(42)
        
        data = {
            'water_point_id': [f'KE{i:05d}' for i in range(1, 11)],
            'country': ['Kenya'] * 10,
            'region': np.random.choice(['Central', 'Nairobi', 'Eastern', 'Rift Valley'], 10),
            'district': np.random.choice(['Nairobi Central', 'Kiambu', 'Machakos', 'Nakuru'], 10),
            'water_source': np.random.choice(['Borehole', 'Spring', 'River', 'Well'], 10),
            'status': np.random.choice(['Functional', 'Non-functional', 'Needs repair'], 10, p=[0.6, 0.3, 0.1]),
            'installation_date': pd.date_range(start='2010-01-01', end='2023-12-31', periods=10),
            'last_updated': pd.date_range(start='2025-01-01', end='2025-07-25', periods=10),
            'water_quality': np.random.choice(['Safe', 'Unsafe', 'Unknown'], 10, p=[0.7, 0.2, 0.1]),
            'risk_score': np.random.randint(0, 100, 10)
        }
        
        def status_color(val):
            if val == 'Functional':
                return 'background-color: rgba(0, 255, 0, 0.2)'
            elif val == 'Non-functional':
                return 'background-color: rgba(255, 0, 0, 0.2)'
            else:
                return 'background-color: rgba(255, 255, 0, 0.2)'
        
        df = pd.DataFrame(data)
        
        def get_risk_level(score):
            if score < 25:
                return "High Risk"
            elif score < 50:
                return "Medium Risk"
            elif score < 75:
                return "Low Risk"
            else:
                return "Safe"
        
        df['risk_level'] = df['risk_score'].apply(get_risk_level)
        
        return df
    
    # Container with border for visual consistency
    with st.container(border=True):
        wpdx_data = get_wpdx_kenya()
        
        data_tab1, data_tab2, data_tab3, data_tab4 = st.tabs(["All Data", "Functional Status", "Risk Analysis", "Quality Trend"])
    
    with data_tab1:
        st.dataframe(
            wpdx_data,
            column_config={
                "water_point_id": "Water Point ID",
                "country": "Country",
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
                "installation_date": st.column_config.DateColumn(
                    "Installation Date",
                    format="MMM DD, YYYY",
                ),
                "last_updated": st.column_config.DateColumn(
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
    
    with data_tab2:
        status_counts = wpdx_data['status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        st.bar_chart(
            status_counts,
            x='Status',
            y='Count'
        )
        
        st.text("Filter data by status:")
        status_filter = st.multiselect(
            "Select status to view:",
            options=wpdx_data['status'].unique(),
            default=wpdx_data['status'].unique()
        )
        
        filtered_status_data = wpdx_data[wpdx_data['status'].isin(status_filter)]
        st.dataframe(filtered_status_data, hide_index=True, use_container_width=True)
    
    with data_tab3:
        risk_counts = wpdx_data['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        
        st.text("Average Risk Score by Region")
        region_risk = wpdx_data.groupby('region')['risk_score'].mean().reset_index()
        region_risk.columns = ['Region', 'Average Risk Score']
        
        st.bar_chart(
            region_risk,
            x='Region',
            y='Average Risk Score'
        )
        
        st.text("Filter data by risk level:")
        risk_filter = st.multiselect(
            "Select risk level to view:",
            options=wpdx_data['risk_level'].unique(),
            default=wpdx_data['risk_level'].unique()
        )
        
        filtered_risk_data = wpdx_data[wpdx_data['risk_level'].isin(risk_filter)]
        st.dataframe(filtered_risk_data, hide_index=True, use_container_width=True)
    
    with data_tab4:
        st.text("Water Quality Trend Over Time")
        st.text("")
        
        selected_water_point = st.selectbox(
            "Select a water point to view quality trend:",
            options=wpdx_data['water_point_id'].tolist(),
            key="trend_water_point"
        )
        
        if selected_water_point:
            point_data = wpdx_data[wpdx_data['water_point_id'] == selected_water_point].iloc[0]
            
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

    csv = wpdx_data.to_csv(index=False)
    st.download_button(
        label="Download WPDX Kenya Data (CSV)",
        data=csv,
        file_name="wpdx_kenya_data.csv",
        mime="text/csv"
    )
    
    st.text("")
    st.text("")
    
    with st.container(border=True):
        st.caption("© 2025 CleanWaterAI. Data sourced from Water Point Data Exchange (WPDx).")
            
        
        



