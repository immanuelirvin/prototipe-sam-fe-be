# command terminal : streamlit run frontend.py
import streamlit as st
import json
import requests
import io
from PIL import Image
import streamlit as st
from PIL import Image
from rasterio.plot import show
from tempfile import NamedTemporaryFile
from PIL import Image
import io
import os
from page import predict_page, electric_potensial_page, history_page
def page3():
    st.markdown("# Page 3 🎉")
    st.sidebar.markdown("# Page 3 🎉")

def main():
    # Get the current directory of the script
    st.set_page_config(layout="wide")
    page_names_dictionary = {
        "Prediction Image": predict_page.show_page,
        "Prediction Potential Photovoltaic": electric_potensial_page.show_page,
        "History of Prediction Potential Photovoltaic": history_page.show_page,
    }
    selected_page = st.sidebar.selectbox("Select a page", page_names_dictionary.keys())
    page_names_dictionary[selected_page]()

main()

