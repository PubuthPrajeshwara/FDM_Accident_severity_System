import streamlit as st
import joblib
import numpy as np
import pandas as pd
import datetime
import json
import requests
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from streamlit_lottie import st_lottie
import path

path_to_model = './xgb_model.joblib'

with open(path_to_model, 'rb') as file:
    model = joblib.load(file)

feature_names = joblib.load('./feature_names.pkl')

# Load the label encoder
label_encoders = {
    'Day_of_week': joblib.load('./Day_of_week_encoder.pkl'),
    'Age_band_of_driver': joblib.load('./Age_band_of_driver_encoder.pkl'),
    'Educational_level': joblib.load('./Educational_level_encoder.pkl'),
    'Driving_experience': joblib.load('./Driving_experience_encoder.pkl'),
    'Road_allignment': joblib.load('./Road_allignment_encoder.pkl'),
    'Time': joblib.load('./Time_encoder.pkl')
}

scaler = joblib.load('./scaler.pkl')

onehot_encoder = joblib.load('./onehot_encoder.pkl')

target_encoder = joblib.load('./Accident_severity_encoder.pkl')

# Define a prediction function
def make_prediction(input_data):
    # Convert input_data to DataFrame for consistency
    input_df = pd.DataFrame([input_data], columns=[
        'Time', 'Day_of_week', 'Number_of_vehicles_involved', 'Number_of_casualties', 
        'Age_band_of_driver', 'Sex_of_driver', 'Educational_level', 'Vehicle_driver_relation', 
        'Driving_experience', 'Type_of_vehicle', 'Owner_of_vehicle', 'Area_accident_occured', 
        'Lanes_or_Medians', 'Road_allignment', 'Types_of_Junction', 'Road_surface_type', 
        'Road_surface_conditions', 'Light_conditions', 'Weather_conditions', 'Type_of_collision', 
        'Vehicle_movement', 'Pedestrian_movement', 'Cause_of_accident'
    ])

    #st.write("Column types before encoding:", input_df.dtypes)
    #st.write("time = ", input_df["Time"])

    # Transform label-encoded columns using pre-trained label encoders
    for col, le in label_encoders.items():
        try:
            input_df[col] = le.transform(input_df[col])
        except ValueError as e:
            st.write(f"Error encoding column {col}: {e}")
    
    # Define columns that need one-hot encoding
    one_hot_columns = [
        'Sex_of_driver', 'Vehicle_driver_relation', 'Type_of_vehicle', 'Owner_of_vehicle',
        'Area_accident_occured', 'Lanes_or_Medians', 'Types_of_Junction', 'Road_surface_type',
        'Road_surface_conditions', 'Light_conditions', 'Weather_conditions', 'Type_of_collision',
        'Vehicle_movement', 'Pedestrian_movement', 'Cause_of_accident'
    ]
    
    # Separate the columns to be one-hot encoded
    categorical_data = input_df[one_hot_columns]

    # Apply the pre-trained OneHotEncoder to the categorical columns
    try:
        one_hot_encoded_data = onehot_encoder.transform(categorical_data)
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded_data, columns=onehot_encoder.get_feature_names_out(one_hot_columns))
    except Exception as e:
        st.write(f"Error during one-hot encoding: {e}")
        return None

    # Drop the original categorical columns from the input dataframe
    input_df = input_df.drop(columns=one_hot_columns)
    
    # Concatenate the one-hot encoded columns back to the input dataframe
    input_df = pd.concat([input_df, one_hot_encoded_df], axis=1)

    # Ensure the input is in the same format as the model's training data
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Change 0 to 0.0 and 1 to 1.0 for float64 columns
    input_df = input_df.astype('int64')

    #st.write("Encoded Input DataFrame:", input_df)

    # Apply the same scaling as during training
    input_df_scaled = scaler.transform(input_df)

    #st.write(f"Shape of input_df: {input_df_scaled.shape}")
    #st.write("Encoded Input DataFrame:", input_df_scaled)

    # Make prediction
    prediction = model.predict(input_df_scaled)
    return prediction

# Set page configuration
st.set_page_config(
    page_title="XGBoost Prediction Web App",
    page_icon=":guardsman:",
    layout="wide"
)

# Custom CSS for styling the prediction box
st.markdown("""
    <style>
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    .prediction-box {
        animation: fadeIn 1s ease-in-out;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 25px;
        font-weight: bold;
        border: 2px solid;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        margin: 20px auto;  /* Centering the box */
        width: 50%;  /* You can adjust this width as needed */
    }
    .fatal-injury {
        border-color: red;
        background: linear-gradient(to right, rgba(255, 0, 0, 0.5), rgba(255, 76, 76, 0.3)); /* Red gradient for fatal injury */
    }
    .serious-injury {
        border-color: orange;
        background: linear-gradient(to right, rgba(255, 140, 0, 0.5), rgba(255, 165, 0, 0.3)); /* Orange gradient for serious injury */
    }
    .slight-injury {
        border-color: yellow;
        background: linear-gradient(to right, rgba(255, 215, 0, 0.5), rgba(255, 255, 0, 0.3)); /* Yellow gradient for slight injury */
    }
    .prediction-box h3 {
        margin-bottom: 10px;
        font-size: 25px;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

# Header Styling
st.markdown(
    """
    <style>
    .header {
        margin-top: -3.5%;
        font-size: 45px;
        font-weight: bold;
        color: #FFD700;  /* Gold color for the text */
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, rgba(30,31,41,1) 0%, rgba(75,0,130,1) 100%);
        border-radius: 15px; /* Rounded corners */
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2); /* Subtle shadow */
        width: 100%; /* Full width */
        position: absolute; /* Ensure it stays in place */
        z-index: 10; /* Bring to front */
    }
    </style>
    <div class="header">
        Welcome to Accident Severity Prediction System
    </div>
    """, unsafe_allow_html=True
)

# Hide the Streamlit header and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_hello = load_lottieurl("https://lottie.host/d5613d92-aea6-4307-8405-e88a0392eef0/1wOJeXkVoD.json")

# Streamlit UI
st.title("The Pattern Seekers's predictions")

st_lottie(
    lottie_hello,
    reverse=False,
    quality="medium",
    height= 300,
    key="hello")

with st.form("prediction_form", clear_on_submit=False):
    with st.container():
        # Organize inputs in columns
        col1, spacer, col2 = st.columns([1, 0.1, 1])

        with col1:
            # Driver Information
            st.subheader("Driver Information")
            age_band_of_driver = st.selectbox("Age Band of Driver", ['18-30', '31-50', 'Under 18', 'Over 51', 'Unknown'])
            sex_of_driver = st.selectbox("Sex of Driver", ['Male', 'Female', 'Unknown'])
            educational_level = st.selectbox("Educational Level", ['Above high school', 'Junior high school', 'Elementary school', 'High school', 'Unknown', 'Illiterate', 'Writing & reading'])
            driving_experience = st.selectbox("Driving Experience", ['1-2yr', 'Above 10yr', '5-10yr', '2-5yr', 'No Licence', 'Below 1yr', 'Unknown', 'na'])

            # Accident Details
            st.subheader("Accident Details")
            time_input = st.time_input("Select Time")  # Time input in HH:MM:SS format
            time_str = time_input.strftime("%H:%M:%S")  # Convert to HH:MM:SS string format
            time_obj = datetime.datetime.strptime(time_str, "%H:%M:%S").time()
            total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
            total_seconds_str = str(total_seconds) + ".0"

            day_of_week = st.selectbox("Day of Week", ['Monday', 'Sunday', 'Friday', 'Wednesday', 'Saturday', 'Thursday', 'Tuesday'])
            Number_of_vehicles_involved = st.number_input("Number of vehicles involved", step=1, format="%i")
            Number_of_casualties = st.number_input("Number of casualties", step=1, format="%i")
            cause_of_accident = st.selectbox("Cause of Accident", ['Moving Backward', 'Overtaking', 'Changing lane to the left','No distancing', 'Changing lane to the right', 'Overloading', 'No priority to vehicle', 'No priority to pedestrian', 'Driving under influence', 'Ignoring traffic signal', 'Turning left', 'Turning right', 'Not adhering to speed limit', 'Driving carelessly', 'Speeding', 'Not adhering to lane discipline','Driving at high speed', 'Other'])

            # Vehicle Information
            st.subheader("Vehicle Information")
            type_of_vehicle = st.selectbox("Type of Vehicle", ['Automobile', 'Public (> 45 seats)', 'Lorry (41?100Q)', 'Public (13?45 seats)', 'Lorry (11?40Q)', 'Long lorry', 'Public (12 seats)', 'Taxi', 'Pick up upto 10Q', 'Stationwagen', 'Ridden horse', 'Other', 'Bajaj', 'Turbo', 'Motorcycle', 'Special vehicle', 'Bicycle'])
            owner_of_vehicle = st.selectbox("Owner of Vehicle", ['Owner', 'Governmental', 'Organization', 'Other'])
            vehicle_driver_relation = st.selectbox("Vehicle Driver Relation", ['Employee', 'Unknown', 'Owner', 'Other'])

        with col2:

            # Environmental Factors
            st.subheader("Environmental Factors")
            area_accident_occurred = st.selectbox("Area of Accident Occurred", ['Residential areas', 'Office areas', 'Recreational areas', 'Industrial areas', 'Other', 'Unknown'])
            lanes_or_medians = st.selectbox("Lanes or Medians", ['Undivided Two way', 'Double carriageway (median)', 'One way', 'Two-way (divided with solid lines road marking)', 'Two-way (divided with broken lines road marking)', 'Unknown'])
            road_alignment = st.selectbox("Road Alignment", ['Tangent road with flat terrain', 'Tangent road with mild grade and flat terrain', 'Escarpments', 'Tangent road with rolling terrain', 'Gentle horizontal curve', 'Tangent road with mountainous terrain', 'Steep grade downward with mountainous terrain', 'Sharp reverse curve', 'Steep grade upward with mountainous terrain'])
            types_of_junction = st.selectbox("Types of Junction", ['No junction', 'Y Shape', 'Crossing', 'O Shape', 'Other', 'Unknown', 'T Shape', 'X Shape'])
            road_surface_type = st.selectbox("Road Surface Type", ['Asphalt roads', 'Earth roads', 'Gravel roads', 'Other'])
            road_surface_conditions = st.selectbox("Road Surface Conditions", ['Dry', 'Wet or damp', 'Snow', 'Flood over 3cm. deep'])
            light_conditions = st.selectbox("Light Conditions", ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting', 'Darkness - lights unlit'])
            weather_conditions = st.selectbox("Weather Conditions", ['Normal', 'Raining', 'Raining and Windy', 'Cloudy', 'Other', 'Windy', 'Snow', 'Unknown', 'Fog or mist'])

            # Collision Information
            st.subheader("Collision Information")
            type_of_collision = st.selectbox("Type of Collision", ['Collision with roadside-parked vehicles', 'Vehicle with vehicle collision', 'Collision with roadside objects', 'Collision with animals', 'Other', 'Rollover', 'Fall from vehicles', 'Collision with pedestrians', 'With Train', 'Unknown'])

            # Vehicle Movement and Pedestrian Movement
            st.subheader("Movement Information")
            vehicle_movement = st.selectbox("Vehicle Movement", ['Going straight', 'U-Turn', 'Moving Backward', 'Turnover', 'Waiting to go', 'Getting off', 'Reversing', 'Unknown', 'Parked', 'Stopping', 'Overtaking', 'Other', 'Entering a junction'])
            pedestrian_movement = st.selectbox("Pedestrian Movement", ['Not a Pedestrian', "Crossing from driver's nearside", 'Crossing from nearside - masked by parked or stationary vehicle', 'Unknown or other', 'Crossing from offside - masked by parked or stationary vehicle', 'In carriageway, stationary - not crossing (standing or playing)', 'Walking along in carriageway, back to traffic', 'Walking along in carriageway, facing traffic', 'In carriageway, stationary - not crossing (standing or playing) - masked by parked or stationary vehicle'])

    # Submit button for the form
    submit_button = st.form_submit_button(label="Predict",use_container_width=True)

    if submit_button:
        # Prepare the input data for prediction
        input_data = [total_seconds_str, day_of_week, Number_of_vehicles_involved, Number_of_casualties, age_band_of_driver, sex_of_driver, educational_level,
                    vehicle_driver_relation, driving_experience, type_of_vehicle, owner_of_vehicle,
                    area_accident_occurred, lanes_or_medians, road_alignment, types_of_junction, road_surface_type,
                    road_surface_conditions, light_conditions, weather_conditions, type_of_collision, vehicle_movement,
                    pedestrian_movement, cause_of_accident]

        # Make prediction with loading spinner
        with st.spinner("Making prediction... Please wait."):
            # Make prediction
            prediction = make_prediction(input_data)

        # Decode the prediction if it's not None
        if prediction is not None:
            decoded_prediction = target_encoder.inverse_transform(prediction)
            #st.write(f"Input Data: {input_data}")
            #st.write(f"Prediction: {prediction}")
            # Determine the CSS class based on the prediction
            if decoded_prediction[0] == 'Fatal injury':
                css_class = 'fatal-injury'
            elif decoded_prediction[0] == 'Serious Injury':
                css_class = 'serious-injury'
            elif decoded_prediction[0] == 'Slight Injury':
                css_class = 'slight-injury'
            else:
                css_class = ''

            st.markdown(f"""
            <div class="prediction-box {css_class}">
                <h3>Prediction Result</h3>
                Accident Severity : {decoded_prediction[0]}
            </div>
            """, unsafe_allow_html=True)

# Footer Styling
st.markdown(
    """
    <style>
    .footer {
        left: 0;
        bottom: 0;
        width: 100%;
        background: linear-gradient(90deg, rgba(30,31,41,1) 0%, rgba(75,0,130,1) 100%);
        color: #FFD700;  /* Gold color for the text */
        text-align: center;
        padding: 15px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 15px 15px 0 0;  /* Rounded top corners */
        box-shadow: 0px -2px 8px rgba(0, 0, 0, 0.2); /* Subtle shadow */
    }
    </style>
    <div class="footer">
        © 2024 The Pattern Seekers. All rights reserved.
    </div>
    """, unsafe_allow_html=True
)