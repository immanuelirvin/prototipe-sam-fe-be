## Training on Google Collab and Code Reference

| Name | Link |
| ------ | ------ |
| Training UNET | https://colab.research.google.com/drive/1apZ_ONt5VLUbEXI-jWcDxdXlTDLlllLw?authuser=1#scrollTo=BrPNLCTVPwBE |
| Training SAM | https://colab.research.google.com/drive/1SDHMyp0Ok9lzHXt0IU_r74JOwi9k9Bb4?authuser=1 |
| Regularization Polygon | https://colab.research.google.com/drive/1zYUwW00G5N9FOCrAPJAcqQKD9-58AyzK?usp=sharing |

## How to running application

How to running streamlit frontend, running the code below inside `FRONT_END_STREAMLIT/frontend.py`

```sh
streamlit run frontend.py
```

How to running FastAPI backend, running the code below inside `BACK_END_FASTAPI/backend.py`

```sh
uvicorn backend:app --reload
```

## Installation

Installation requirements for frontend
```sh
streamlit
requests
Pillow
rasterio
```

Installation requqirements for backend

```sh
segment-geospatial
torch
fastapi
uvicorn
pydantic
asyncpg
starlette-context
python-jose
passlib
click
geopandas
matplotlib
numpy
opencv-python
psycopg2
torch
tqdm
torchvision
scikit-image
gdal
```

for gdal library sometimes error problem so I provide the wheel inside folder `BACK_END_FASTAPI\gdal (library import gdal)/GDAL-3.4.3-cp39-cp39-win_amd64.whl`
