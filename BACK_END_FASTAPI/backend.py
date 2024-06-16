# uvicorn backend:app --reload
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from tempfile import NamedTemporaryFile
from fastapi.responses import Response
from pydantic import BaseModel
from database import connect_db_cloud
from samgeo import tms_to_geotiff
from samgeo.text_sam import LangSAM
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from typing import List
from projectRegularization.regularize import run_regularize
from fastapi.responses import StreamingResponse
from fastapi import BackgroundTasks
import geopandas as gpd
import matplotlib.pyplot as plt
import io
import os

# VARIABLES
current_dir = os.path.dirname(os.path.abspath(__file__))
REGULARIZATION_FOLDER_PATH = f"{current_dir}/projectRegularization/test_data/"

print("asdasd:",REGULARIZATION_FOLDER_PATH)
REGULARIZATION_RGB = REGULARIZATION_FOLDER_PATH+"rgb/"
REGULARIZATION_MASK = REGULARIZATION_FOLDER_PATH+"seg/"
REGULARIZATION_OUTPUT = REGULARIZATION_FOLDER_PATH+"reg_output/"
REGULARIZATION_SHP_OUTPUT = REGULARIZATION_FOLDER_PATH+"shp_output/"
IMAGE_NAME = "input.tif"
IMAGE_SHP_NAME = "input.shp"
total_area = 0.0

# INPUT AND OUTPUT RETURN (HANYA UNTUK TEMPLATE MEMPERMUDAH PENULISAN)
class OutputBase(BaseModel):
    result: str
    text: str

class Db_Input(BaseModel):
    total_area : float
    gsr : float

app = FastAPI()

# ALL FUNCTION API FOR PAGE 1
# INITIALIZE LangSAM MODEL
sam = None

def get_langsam_model():
    global sam
    if sam is None:
        sam = LangSAM()
    return sam

async def run_regularize_async():
    # Call the synchronous run_regularize function
    result = run_regularize()
    return True

def overlay_images(image, overlay_image, alpha=0.5):
    # Resize overlay image to match base image dimensions
    overlay_image = cv2.resize(overlay_image, (image.shape[1], image.shape[0]))
    
    # Blend the images
    overlaid_image = cv2.addWeighted(image, 1 - alpha, overlay_image, alpha, 0)
    
    return overlaid_image

@app.post("/show_input_image")
def show_input_image(tif_file: UploadFile = File(...)):
    # Upload Local FastAPI BackEnd
    with open(REGULARIZATION_RGB+IMAGE_NAME, "wb") as f:
        f.write(tif_file.file.read())
    
    # Read the uploaded TIFF file
    with open(REGULARIZATION_RGB+IMAGE_NAME, "rb") as f:
        tiff_data = f.read()

    # Return the TIFF data as a streaming response
    return StreamingResponse(iter([tiff_data]), media_type="image/tiff")

@app.post("/prediction_sam")
def prediction_sam():
    # model sam + grounding dino model
    sam = get_langsam_model()
    text_prompt = "house"

    # predict
    sam.predict(REGULARIZATION_RGB+IMAGE_NAME, text_prompt, box_threshold=0.26, text_threshold=0.4)
    
    # save file
    sam.show_anns(
        cmap='Greys_r',
        add_boxes=False,
        alpha=1,
        title='Automatic Segmentation of Roofs',
        blend=False,
        output=REGULARIZATION_MASK+IMAGE_NAME,
    )

    # Read the uploaded TIFF file
    with open(REGULARIZATION_MASK+IMAGE_NAME, "rb") as f:
        tiff_data = f.read()

    # Return the TIFF data as a streaming response
    return StreamingResponse(iter([tiff_data]), media_type="image/tiff")

@app.post("/regularize_gan")
async def regularize_gan():
    # Wait for the result of run_regularize_async
    process_gan = await run_regularize_async()

    return OutputBase(result="Success",text="Regularization Done")

@app.get("/get_total_area")
def get_total_area():
    shapefile_path = REGULARIZATION_SHP_OUTPUT+IMAGE_SHP_NAME
    # Read the Shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)
    # Create a new column 'new_column' with values equal to the index
    gdf['id'] = gdf.index

    # Calculate the area for each polygon and create a new column 'area'
    gdf['area'] = gdf.geometry.area

    # Print the GeoDataFrame with the new 'area' column
    print("GeoDataFrame with Areas:")
    print(gdf)

    # Calculate the total sum of the areas
    total_area = gdf['area'].sum()
    return {
        "total_area": total_area
    }

@app.get("/plot_vector_result")
async def generate_plot_vector(
    show_real: bool = Query(True, description="Show real image"),
    show_mask_gan: bool = Query(True, description="Show mask GAN image"),
):
    # model sam + grounding dino model
    sam = get_langsam_model()

    # create shp vector
    sam.raster_to_vector(REGULARIZATION_OUTPUT+IMAGE_NAME, REGULARIZATION_SHP_OUTPUT+IMAGE_SHP_NAME)

    shapefile_path = REGULARIZATION_SHP_OUTPUT+IMAGE_SHP_NAME
    # Read the Shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)
    # Create a new column 'new_column' with values equal to the index
    gdf['id'] = gdf.index

    # Calculate the area for each polygon and create a new column 'area'
    gdf['area'] = gdf.geometry.area

    # Plot the GeoDataFrame with different colors based on the "value" column
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, column="id", legend=True, categorical=True, cmap="tab20")
    ax.axis('off')

    # Create a buffer to save the plot
    buf = io.BytesIO()

    # Save the plot
    plt.savefig(buf, format='png')
    plt.close()

    # Convert the plot to bytes
    buf.seek(0)

    # Return the plot as a streaming response
    return StreamingResponse(buf, media_type="image/png")

@app.get("/plot_image_overlay")
async def generate_plot_image_overlay(
    show_real: bool = Query(True, description="Show real image"),
    show_mask_gan: bool = Query(True, description="Show mask GAN image"),
):
    # Load the real image
    real_image = cv2.imread(REGULARIZATION_RGB + IMAGE_NAME)

    # Load the mask GAN image
    mask_image_gan = cv2.imread(REGULARIZATION_OUTPUT + IMAGE_NAME)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 10))

    real_predict_image = overlay_images(real_image, mask_image_gan)

    # Hide the axes
    ax.axis('off')

    # Create a buffer to save the plot
    buf = io.BytesIO()

    # Save the plot
    plt.imshow(cv2.cvtColor(real_predict_image, cv2.COLOR_BGR2RGB))
    plt.savefig(buf, format='png')
    plt.close()

    # Convert the plot to bytes
    buf.seek(0)

    # Return the plot as a streaming response
    return StreamingResponse(buf, media_type="image/png")

# ALL FUNCTION API FOR PAGE 2
@app.post("/estimate_photovoltaic_electric")
def estimate_photovoltaic_electric(input : Db_Input):
    # Rumus
    # Cr = (Cm/1000)*(RCR*total_area/(1.487*0.992))
    Cr = (200/1000)*(0.85*input.total_area/(1.487*0.992))
    # print("hasil Cr (kWp):",Cr)

    # Rumus
    # Energy = Cr*(GSR)*D
    Energy = Cr*(input.gsr)*0.75
    # print("hasil Energy listrik (kWh):",Energy,"per monthly")

    # Establish a connection (already done in your code)
    conn = connect_db_cloud()
    cursor = conn.cursor()
    # Define the SQL INSERT statement dynamically with parameters
    insert_data_sql = f"INSERT INTO history (total_area, gsr, energy) VALUES ('{input.total_area}', '{input.gsr}', '{Energy}');"

    # Execute the SQL statement with parameters
    cursor.execute(insert_data_sql)
    # Commit the changes to the database
    conn.commit()

    # Close the cursor and the connection
    cursor.close()
    conn.close()

    return {
        "Cr": Cr,
        "Energy": Energy
    }

# ALL FUNCTION API FOR PAGE 3
@app.get("/get_all_history_data")
def get_all_data():
    # Establish a connection (already done in your code)
    conn = connect_db_cloud()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM history;")
    data = cursor.fetchall()
    
    cursor.close()
    conn.close()
    return data