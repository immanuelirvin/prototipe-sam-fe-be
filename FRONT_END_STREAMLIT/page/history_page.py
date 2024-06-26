import streamlit as st
import json
import requests
import io
from PIL import Image
import streamlit as st
from tempfile import NamedTemporaryFile
import io
import webbrowser
import pandas as pd

LINK_BACKEND = "http://127.0.0.1:8000"

def list_all_history():
    with st.spinner('Fetching from ElephantSQL Clouds...'):
        res = requests.get(url=LINK_BACKEND+"/get_all_history_data")
        
        if res.status_code == 200:  # Check if the request was successful
            data = res.json()  # Parse the JSON response
            
            if data:
                st.success("History in the Database")
                # Convert data to a pandas DataFrame for better display
                # remove id primary key, only get total area, gsr, energy
                data_display = [[item[1], item[2], item[3], item[4].replace('T', ' '), item[5]] for item in data]
                
                # Convert to DataFrame
                df = pd.DataFrame(data_display, columns=["Total Area", "GSR Monthly", "Energy", "Date In", "Centroid"], index=pd.RangeIndex(start=1, stop=len(data)+1))
                st.dataframe(df)
            else:
                st.write("No data available.")
        else:
            st.write("Error: Unable to retrieve data.")

def show_page():
    st.sidebar.markdown("# History of Result from Prediction Potential Photovoltaic Page ❄️")

    # Call the function to display tutorial guide
    list_all_history()




