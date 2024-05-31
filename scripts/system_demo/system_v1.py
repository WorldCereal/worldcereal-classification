# %%

# ADD WORLDCEREAL LOGO in markdown
# ![](./WorldCereal_logo.png)
# ADD ESA LOGO
# ![](./ESA_logo.png)

# ADD TITLE: WorldCereal System v1 Demo

# DATE: ...

# AUTHORS: ...

# ADD consortium logos
# ![](./Consortium_logos.png)

'''
This notebook contains a demo of the WorldCereal system v1
and all its functionalities.

Content:
- Define a region of interest
- Generate a cropland and crop type product
    using the default WorldCereal models
- Exploring available reference data in the RDM
- Contributing reference data to RDM
- Requesting reference data from RDM
- Launching point extractions for obtaining satellite input data 
- Launching catboost model training based on the extracted points
- Perform inference run with the newly trained model
    (and/or the default cropland model)

'''

# ADD TABLE OF CONTENTS USING MARKDOWN
# Notebook outline:
# - [1. Introduction](#1.-Introduction)

# %%
# IMPORTS
import geopandas as gpd
import requests
from shapely.geometry import shape, Polygon

from worldcereal.utils.map import get_ui_map

RDM_API = 'https://ewoc-rdm-api.iiasa.ac.at'

# %%
# WorldCereal user authentication

# BEFORE PROCEEDING, A USER SHOULD CREATE AN ACCOUNT ON VITO'S TERRASCOPE PLATFORM:
# https://sso.terrascope.be/auth/realms/terrascope/login-actions/registration?client_id=drupal-terrascope&tab_id=2MybIFKQHdo&execution=67e5ef09-bc23-4344-b099-4e710a86e68a&kc_locale=en

username = input('Enter your Terrascope username: ')
password = input('Enter your Terrascope password: ')

url = 'https://sso.vgt.vito.be/auth/realms/terrascope/protocol/openid-connect/token'
payload = {'grant_type': 'password',
           'client_id': 'worldcereal-rdm',
           'username': username,
           'password': password}

token = requests.post(url, data=payload)

tokentype = token.json()['token_type']
accessToken = token.json()['access_token']
headers = {
    'Authorization': f'{tokentype} {accessToken}',
}

# %% Define a region of interest
m, dc = get_ui_map()
m

# Now use the interactive map to draw a rectangle around your region of interest...

# %%
# retrieve bounding box from drawn rectangle
obj = dc.last_draw
if obj.get('geometry') is not None:
    poly = Polygon(shape(obj.get('geometry')))
    bbox = poly.bounds
else:
    raise ValueError('Please first draw a rectangle '
                     'on the map before proceeding.')
print(f'Your area of interest: {bbox}')

# %%
# Generate default cropland and crop type products for
# the region of interest

# Perform inference run with the default cropland model
start_date = '2021-01-01'
end_date = '2021-12-31'
aoi = bbox
cropland_result = run_inference(aoi, start_date, end_date, product='cropland')
# download and visualize map

# Perform inference run with default crop type model
croptype_result = run_inference(aoi, start_date, end_date, product='maize')
# download and visualize map

# use the cropland product to mask the previously generated custom crop type product
croptype_fin = mask_product(croptype_result, cropland_result)


# %%
# Exploring available reference data in the RDM

# Demo on RDM API Phase I --> https://github.com/WorldCereal/ewoc_rdm_demo_api/blob/8548a65e04350a9a881fc4d78f54a170d6809c2c/rdmApiDemo.ipynb#L22
# Functionalities of RDM API Phase II --> https://ewoc-rdm-api.iiasa.ac.at/swagger/index.html

# Check full list of available collections
collectionResponse = requests.get(f'{RDM_API}/collections', headers=headers)
collections = collectionResponse.json()
col_ids = [x['collectionId'] for x in collections['items']]
print(f'Available collections: {col_ids}')

# Now we check which collections intersect with our AOI
bbox_str = f'Bbox={bbox[0]}&Bbox={bbox[1]}&Bbox={bbox[2]}&Bbox={bbox[3]}'
colSearchUrl = f'{RDM_API}/collections/search?{bbox_str}'
colSearchResponse = requests.get(colSearchUrl, headers=headers)
test = colSearchResponse.json()
print('The following collections intersect with your AOI:')
for i, col in enumerate(test):
    print()
    print(f'Collection {i+1}: {col["collectionId"]}')


# %%
# Contributing reference data to RDM

# Here we should have a function that allows a user to upload a zip file containing the reference data
# If possible, harmonization should be done on the server side, not by the user

# check whether data has been successfully ingested...
colSearchResponse = requests.get(colSearchUrl, headers=headers)
cols_aoi = colSearchResponse.json()
print('The following collections intersect with your AOI:')
for i, col in enumerate(test):
    print()
    print(f'Collection {i+1}: {col["collectionId"]}')

# %%
# Now that our reference data has been added, we launch a request to extract all reference points and polygons intersecting our region of interest
# --> this should return the selected points/polygons as a geojson file/object (one per collection)

responses = []
for col in cols_aoi:
    itemSearchCollectionId = col['collectionId']
    print(
        f'Extracting reference data from collection {itemSearchCollectionId}')
    itemSearchUrl = f'{RDM_API}/collections/{itemSearchCollectionId}/items?{bbox_str}'
    itemSearchResponse = requests.get(itemSearchUrl, headers=headers)
    responses.append(itemSearchResponse.json())

responses


# %%
# Next, we convert polygons to points (by taking the centroid) and combine all reference data into one geoparquet file


# %%
# Launch point extractions for obtaining satellite input data
points = gpd.read_file('data/points.geoparquet')

# THIS WORKFLOW IS SPLIT INTO TWO:
#     - 1. Extracting the satellite data for the points, up to monthly composites (pre-processing done)
#     - 2. (If needed) finetuning presto + applying presto to generate the 128 classification features

training_df = get_training_data(points)

# FYI, for now we have a basic pipeline, here -->
# https://github.com/Open-EO/openeo-gfmap/blob/issue97-basic-pipeline/examples/basic_pipeline/basic_pipeline.ipynb

# %%
# Train catboost model

link_to_model = train_catboost_model(training_df)

# Upload model to artifactory (should be automatically done by openeo)


# %%
# Perform inference run with the newly trained model

aoi = bbox
start_date = '2021-03-01'
end_date = '2021-10-01'

result = run_inference(aoi, start_date, end_date, model=link_to_model)

# --> https://github.com/WorldCereal/worldcereal-classification/blob/kvt_mvp_inferenceUDF/minimal_wc_presto/backend_inference_example_openeo.ipynb
