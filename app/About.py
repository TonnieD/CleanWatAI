import streamlit as st
from PIL import Image
from pathlib import Path
import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


st.set_page_config(page_title="About | CleanWatAI", page_icon="💧", layout="wide")


st.title("About this Project")
st.caption("A practical data app showcasing end-to-end data science skills.")

# Base path (adjust if your assets are deeper, e.g., in "assets/images/")
ASSETS_DIR = Path(__file__).parent / "images"

# Individual image paths
about_img = ASSETS_DIR / "about.jpeg"
mission_img = ASSETS_DIR / "mission.jpeg"
vision_img = ASSETS_DIR / "vision.jpeg"

# Team avatars
team_members = {
    "Diana": ASSETS_DIR / "diana.jpeg",
    "Phanela": ASSETS_DIR / "phanela.jpeg",
    "Lewis": ASSETS_DIR / "lewis.jpeg",
    "Margaret": ASSETS_DIR / "maggie.jpeg",
    "Anthony": ASSETS_DIR / "anthony.jpeg",
}

# Social/contact icons
social_icons = {
    "Email": ASSETS_DIR / "email.jpeg",
    "Phone": ASSETS_DIR / "phone.jpeg",
    "Twitter": ASSETS_DIR / "twitter.jpeg",
    "LinkedIn": ASSETS_DIR / "linkedin.jpeg"
}

# --- About Section ---
st.markdown("## 🧼 About CleanWatAI")
col1, col2 = st.columns([1, 2])
with col1:
    st.image(about_img, use_container_width=True)
with col2:
    st.markdown(
        """
        <div style='font-size: 22px; line-height: 1.6'>
            Water is life. Yet for millions, that life is silently threatened every 
            day by contaminated sources, failing infrastructure, and overlooked 
            early signs.  
            At CleanWatAI, we set out to change that — by teaching machines to 
            listen when people speak about water.  
            <br><br>
            We are a team of data scientists who believe that Artificial Intelligence 
            shouldn't just be smart — it should be <i>human-aware</i>.  
            CleanWatAI was born from a simple but powerful idea: that hidden within 
            scattered news reports around the world are stories that warn us — if only 
            we had the tools to hear them.  
            <br><br>
            At the heart of CleanWatAI is our predictive engine for assessing water 
            point contamination risk. By combining environmental data, infrastructure 
            reports, and machine learning models, we identify high-risk areas before 
            crises unfold. Our goal is to provide communities, NGOs, and policymakers 
            with early warnings, enabling faster response, resource prioritization, and 
            ultimately — safer water for all.  
            <br><br>
            Our project doesn’t just predict — it <i>prevents</i>.  
            It gives a voice to forgotten communities and empowers decision-makers 
            with clarity before disaster strikes.
        </div>
        """,
        unsafe_allow_html=True
    )




st.markdown("---")

# --- Mission Section ---
st.markdown("## 🎯 Mission")
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
        <div style='font-size: 24px; line-height: 1.6'>
        To harness the power of Natural Language Processing and data science to detect,
        visualize, and prevent water contamination risks — empowering communities and organizations
        with early, actionable insights.
        </div>
        """, unsafe_allow_html=True)
with col2:
    st.image(mission_img, use_container_width=True)

st.markdown("---")

# --- Meet the Team Section ---
st.subheader("👨‍👩‍👧 Meet the Team")

cols = st.columns(len(team_members))

for col, (name, image_path) in zip(cols, team_members.items()):
    try:
        with col:
            st.image(Image.open(image_path), caption=name, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading image for {name}: {e}")

st.markdown("---")

# --- Vision Section ---
st.markdown("## 🔮 Vision")
col1, col2 = st.columns([1, 2])
with col1:
    st.image(vision_img, use_container_width=True)
with col2:
    st.markdown("""
        <div style='font-size: 24px; line-height: 1.6'>
        A world where no community is left vulnerable to water-related dangers because warnings were missed, unheard, or too late.  
        A future where Artificial Intelligence doesn’t just predict outcomes — it <i>protects lives</i>.
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# --- Contact Section ---
st.markdown("## 📞 Contact Us")
contact_cols = st.columns(4)
for i, (label, icon_path) in enumerate(social_icons.items()):
    with contact_cols[i]:
        st.image(icon_path, width=40)
        st.caption(label)

st.markdown("---")


with st.container(border=True):
        st.caption("© 2025 CleanWaterAI. Data sourced from WPDx and other public datasets.")