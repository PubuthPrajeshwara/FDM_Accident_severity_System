import streamlit as st
import json
import requests
from streamlit_extras.switch_page_button import switch_page
from streamlit_lottie import st_lottie


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_hello = load_lottieurl("https://lottie.host/19a7fa19-7af0-439a-ae95-6da2ba7bc1db/Qf8JaxfzMn.json")

# Set page configuration
st.set_page_config(
    page_title="XGBoost Prediction Web App",
    page_icon=":guardsman:",
    layout="wide"
)

# CSS to hide the sidebar and its toggle arrow
hide_sidebar_style = """
    <style>
    [data-testid="stSidebar"] {
        display: none;
    }
    [data-testid="collapsedControl"] {
        display: none;
    }
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)


# Hide the Streamlit header and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .header {
        margin-top: -3.5%;
        font-size: 70px;
        font-weight: bold;
        color: #FFD700; /* Gold color for the text */
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 15px 100px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        margin-top: 20px;
        transition: background-color 0.3s, transform 0.3s;
    }
    div.stButton > button:hover {
        background-color: #3e8e41;
        transform: scale(1.05);
        color: white;
    }
    </style>
    <div class="header">
        Welcome to Accident Severity Prediction System
    </div>
    """, unsafe_allow_html=True
)

# Home Page Content
st.title("Predict the severity of accidents based on various input factors.")

st_lottie(
    lottie_hello,
    reverse=False,
    quality="medium",
    height= 500,
    key="hello")

# Create the interactive button and handle page navigation
if st.button("Get Started"):
    switch_page("prediction")