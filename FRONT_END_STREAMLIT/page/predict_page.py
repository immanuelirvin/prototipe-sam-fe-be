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
    st.title("Calculate Total Area of Segmented Masking Area using Streamlit")
    
    # Add your guide content here
    st.write("Welcome to this comprehensive guide on calculating the total area of segmented masking area using Streamlit! This tutorial will walk you through the process step by step.")
    
    step1_header = "<span style='font-size:1.5rem;color:#FF7F7F'><b>Step 1: Setting up Dependencies in Google Colab</b></span>"
    step2_header = "<span style='font-size:1.5rem;color:yellow'><b>Step 2: Creating an Interactive Map</b></span>"
    step3_header = "<span style='font-size:1.5rem;color:orange'><b>Step 3: Downloading a Sample Image</b></span>"

    step1 = "<span style='font-size:1rem;color:#FF7F7F'><b>First, install the necessary tools and software in Google Colab to ensure your code runs smoothly. This typically involves running a few commands to install libraries or packages needed for your project. Setting up dependencies beforehand helps guarantee that your code will work seamlessly when shared with others on Google Colab (estimated 1 minute install depedency).</b></span>"
    step2 = "<span style='font-size:1rem;color:yellow'><b>Next, design an interactive map interface in Google Colab. Start by importing libraries that enable map creation and interaction. Then, add features like a tool to draw rectangles and an option to delete layers to make sure users can easily clear all drawn elements to fix mistakes. These features enhance user experience and make map creation and editing more efficient (to enlarge the interactive map, use the mouse to scroll).</b></span>"
    step3 = "<span style='font-size:1rem;color:orange'><b>Lastly, enable users to download a sample image onto Google Colab. Add functionality to initiate the download process and specify where the image should be saved within the Colab environment. Include a waiting period to ensure the image generation process completes before downloading. Allow flexibility in naming the downloaded image, and store it in the Colab folder for easy access throughout the session. This simplifies workflow and enhances productivity for users (use the sample image in application).</b></span>"
    
    st.markdown(f"{step1_header}", unsafe_allow_html=True)
    st.markdown(f"{step1}", unsafe_allow_html=True)
    st.image("assets/google_collab_step1.jpg", caption="Setting up Dependencies", use_column_width=True)
    st.markdown(f"{step2_header}", unsafe_allow_html=True)
    st.markdown(f"{step2}", unsafe_allow_html=True)
    st.image("assets/google_collab_step2.jpg", caption="Creating an Interactive Map", use_column_width=True)
    st.markdown(f"{step3_header}", unsafe_allow_html=True)
    st.markdown(f"{step3}", unsafe_allow_html=True)
    st.image("assets/google_collab_step3.jpg", caption="Downloading a Sample Image", use_column_width=True)

    # Add a button
    if st.button("To proceed to Google Collabs (get data.tif) and complete the steps, Click Me"):
        # When the button is clicked, open the browser
        st.success("Redirecting to Google Drive...")
        webbrowser.open_new_tab("https://drive.google.com/file/d/1-Px4csEa14Ec1QPoYvaCz0-Y_4ycxX19/view?usp=sharing")
        # Display image from assets folder

def step1():
    # Retrieve bounding box coordinates again inside the button callback
    uploaded_file = st.file_uploader(label="", type=["tif", "tiff"])

    if uploaded_file is not None:
        files = {"tif_file": uploaded_file.getvalue()}
    
        # Make a POST request to the FastAPI endpoint
        res = requests.post(url=LINK_BACKEND+"/show_input_image", files=files, stream=True)

        if res.status_code == 200:
            # Display the streamed image
            st.image(res.content, caption='Input Image', use_column_width=True)
        else:
            st.error(f"Error: {res.status_code} - {res.text}")

        with NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
            image = uploaded_file.name
            # print("Uploaded file name with extension:", image)

        return True, image
    else:
        st.error("Image not yet uploaded")
        return False, ""
        

def step2(image):
    with st.spinner('Predicting...'):
        try:
            params = {
                "rgb_file_name": image,
                "masking_file_name": "test"
            }
            res_prediction_sam = requests.post(url = LINK_BACKEND+"/prediction_sam", data = json.dumps(params))

            # Running Predict SAM
            if res_prediction_sam.status_code == 200:
                # Running Regularization
                res_regularize_gan = requests.post(url = LINK_BACKEND+"/regularize_gan", data = json.dumps(params))
                if res_regularize_gan.status_code == 200:

                    # Plot the result based on user selections
                    print("Success Regularized")
                else:
                    st.error(f"Error: Image width and height too big, Please use smaller area of data")
            else:
                st.error(f"Error: {res_prediction_sam.status_code} - {res_prediction_sam.text}")
        except:
            print("ERROR /prediction_sam")

def plot_prediction_result(show_real=True, show_mask_gan=True):
    # Define query parameters based on user preferences
    params = {
        "show_real": show_real,
        "show_mask_gan": show_mask_gan
    }

    response = requests.get(LINK_BACKEND+"/plot_image_overlay", params=params)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Read the image bytes from the response content
        img_bytes = io.BytesIO(response.content)
        
        # Display the image using Streamlit
        st.image(Image.open(img_bytes), caption='Segmentation Prediction Result')
    else:
        st.error("Failed to fetch plot image. Error code: {}".format(response.status_code))

def plot_prediction_vector_shp_result(show_real=True, show_mask_gan=True):
    # Define query parameters based on user preferences
    params = {
        "show_real": show_real,
        "show_mask_gan": show_mask_gan
    }

    response = requests.get(LINK_BACKEND+"/plot_vector_result", params=params)
    response_total_area = requests.get(LINK_BACKEND+"/get_total_area")
    # Check if the request was successful
    if response.status_code == 200:
        # Read the image bytes from the response content
        img_bytes = io.BytesIO(response.content)
        
        # Display the image using Streamlit
        st.image(Image.open(img_bytes), caption='Vector Masking Area')

        if response_total_area.status_code == 200:
            total_area = response_total_area.json()["total_area"]
            st.success("Total area from segmented mask into shp (vector): "+str(total_area)+" (m^2)")
            st.markdown(f"Please Copy/Save the total area value <span style='font-size:1.5rem;color:green'><b>{total_area} (m^2)</b></span> to process in the Prediction Potential Photovoltaic Page.", unsafe_allow_html=True)
    else:
        st.error("Failed to fetch plot image. Error code: {}".format(response.status_code))


def show_page():
    st.sidebar.markdown("# Prediction Image Page 🎈")

    tutorial_guide()
    # Button to trigger bounding box retrieval
    trigger, image = step1()
    if trigger:
        if st.button("Step 2 (Prediction Segmentation Rooftop Mask Area)"):
            step2(image)
            st.session_state['step_2_done'] = True  # Set flag to indicate Step 2 is done

    if st.session_state.get('step_2_done', False):  # Check if Step 2 is done
        plot_prediction_result()
        if st.button("Step 3 (Counting Total Area on Selected Mask Extract to Shp)"):
            # plot_prediction_vector_shp_result()
            st.session_state['step_3_done'] = True  # Set flag to indicate Step 3 is done

    if st.session_state.get('step_3_done', False):  # Check if Step 3 is done
        plot_prediction_vector_shp_result()
        # if st.button("Step 4"):
            
        #     st.warning("STEP 4 BRO")
        #     st.session_state['step_3_done'] = False  # Reset flag to indicate Step 3 is not done anymore

