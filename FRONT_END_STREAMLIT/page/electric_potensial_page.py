import streamlit as st
import json
import requests
import io
from PIL import Image
import streamlit as st
from PIL import Image
from tempfile import NamedTemporaryFile
from PIL import Image
import io
import webbrowser

LINK_BACKEND = "http://127.0.0.1:8000"

def tutorial_guide():
    st.title("Estimating Photovoltaic Electric Value using Streamlit")
    
    # Add your guide content here
    st.write("Welcome to the second part of the tutorial! In this section, we will estimate the potential photovoltaic electric value based on the total area of segmented masking area obtained from the Prediction Image Page.")

    coordinates = "<span style='font-size:1.5rem;color:green'><b>Longitude, Latitude</b></span>"
    st.markdown(f"Coordinate {coordinates} value to process in the Prediction Potential Photovoltaic Page. (Google Collab)", unsafe_allow_html=True)

    step1_header = "<span style='font-size:1.5rem;color:#FF7F7F'><b>Step 1: Input Coordinates</b></span>"
    step2_header = "<span style='font-size:1.5rem;color:yellow'><b>Step 2: Changing Map Data</b></span>"
    step3_header = "<span style='font-size:1.5rem;color:orange'><b>Step 3: Getting GHI Values</b></span>"

    step1 = "<span style='font-size:1rem;color:#FF7F7F'><b>Enter the coordinates of your desired location. These coordinates will be used to generate solar potential data for that location (top left arrow in the image).</b></span>"
    step2 = "<span style='font-size:1rem;color:yellow'><b>Navigate to the visualization section and locate the map display. Switch the map data representation from per year to per day to observe solar potential changes on a daily basis (middle right arrow in the image).</b></span>"
    step3 = "<span style='font-size:1rem;color:orange'><b>Navigate to the GHI (Global Horizontal Irradiation) section and get the value (kWh/m^2/day) (middle bottom right arrow in the image).</b></span>"
    
    st.markdown(f"{step1_header}", unsafe_allow_html=True)
    st.markdown(f"{step1}", unsafe_allow_html=True)
    # Display image from assets folder
    st.image("assets/tutorial-input-coordinate.jpg", caption="Tutorial How to Get Centroid Data", use_column_width=True)
    st.markdown(f"{step2_header}", unsafe_allow_html=True)
    st.markdown(f"{step2}", unsafe_allow_html=True)
    st.markdown(f"{step3_header}", unsafe_allow_html=True)
    st.markdown(f"{step3}", unsafe_allow_html=True)

    # Add a button
    if st.button("To proceed to determine the value of Global Solar Radiation (GSR) and complete the steps, Click Me"):
        # When the button is clicked, open the browser
        st.success("Redirecting to Global Solar Atlas...")
        webbrowser.open_new_tab("https://globalsolaratlas.info/map")

    # Display image from assets folder
    st.image("assets/tutorial_gsr.jpg", caption="Tutorial How to Get GSR Data", use_column_width=True)
    
def input_totalArea_GSR():
    # Create a text input field for float variables
    st.write("To begin, let's input  centroid we obtained from Prediction Page (Google Collab)")
    centroid = st.text_input("Centroid (longitude, latitude):")

    st.write("To begin, let's use the total area we obtained from Prediction Page")
    total_area_input = st.text_input("Total Area Field (m^2):")

    st.write("To begin, let's use the GSR we obtained from Global Solar Atlas Guide")
    gsr_input = st.text_input("GSR Field (kWh/m^2 per day):")
    # Create a button to trigger processing
    if st.button("Input Total Area and GSR Value"):
        try:
            # Try to convert the input to a float
            float_total_area_input = float(total_area_input)
            st.write("Total Area you entered:", float_total_area_input)
        except:
            # If the input cannot be converted to a float, handle the error
            st.error("Please enter a valid number or float value on Total Area Field.")

        try:
            # Try to convert the input to a float
            float_gsr_input = float(gsr_input)
            st.write("GSR you entered (day average):", float_gsr_input)
        except:
            # If the input cannot be converted to a float, handle the error
            st.error("Please enter a valid number or float value on GSR Field.")
        # st.session_state['step_2_done'] = True
        return True, float_total_area_input, float_gsr_input, centroid
    else:
        return False, 0.0, 0.0, ""

def count_photovoltaic_potential(total_area, gsr_monthly, centroid):
    st.write("")
    st.write("Now, let's proceed with the calculations to estimate the photovoltaic electricity generation based on monthly solar data.")
    st.write("The solar radiation you entered (monthly average):", gsr_monthly, "[Average GSR * 30]")
   
    input = {"total_area": float(total_area), "gsr": float(gsr_monthly), "centroid": str(centroid)}
    res = requests.post(url = LINK_BACKEND+"/estimate_photovoltaic_electric", data=json.dumps(input))
    # st.subheader(f"Response from API = {res.text}")
    Cr = res.json()["Cr"]
    Energy = res.json()["Energy"]
    st.warning(f"The peak power of a PV system or panel (Cr): {Cr} KwP")
    st.success(f"The estimated electrical energy output is {Energy} kWh per month.")

    st.write("The assumption underlying the calculation is centered on the rated capacity of each individual solar module, denoted as CM. In this context, it is assumed that each module has a fixed rated capacity of 200 peak Watts (Wp), reflecting its maximum power output under standardized test conditions. Additionally, the assumption considers the physical dimensions of the modules, where the width and height of each module are specified as 1,487 meters by 0.992 meters, respectively. These assumptions are integral to accurately estimating the module rated capacity (Cr) and subsequently, the energy yield of the photovoltaic system. By accounting for these assumptions, the calculation provides a tailored assessment of the system's potential energy production, aiding in the planning and implementation of solar energy projects.")
    return Energy

def show_page():
    st.sidebar.markdown("# Prediction Potential Photovoltaic Page ❄️")

    # Call the function to display tutorial guide
    tutorial_guide()

    # Initialize trigger variable
    trigger = False

    try:
        # Get the trigger and input values
        trigger, total_area, gsr, centroid = input_totalArea_GSR()
    except:
        st.write("Error Input Value (Can't Continue To Next Process)")

    if centroid == "":
        st.error("Please Input Centroid Value (From Google Collab for Coordinate -> Link in Prediction Page)")
    if trigger and centroid != "":
        gsr_monthly = gsr*30
        with st.spinner('Predicting...'):
            result = count_photovoltaic_potential(total_area, gsr_monthly, centroid)




