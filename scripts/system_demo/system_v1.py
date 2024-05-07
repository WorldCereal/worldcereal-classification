# %%
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

# Now use the interactive map to draw a rectangle around the region of interest

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

# NEEDS INFERENCE RUN


# %%
# Exploring available reference data in the RDM
collectionResponse = requests.get(f'{RDM_API}/collections', headers=headers)
collections = collectionResponse.json()
col_ids = [x['collectionId'] for x in collections['items']]
print(f'Available collections: {col_ids}')

# Now we check which collections intersect our AOI
bbox_str = f'Bbox={bbox[0]}&Bbox={bbox[1]}&Bbox={bbox[2]}&Bbox={bbox[3]}'
colSearchUrl = f'{RDM_API}/collections/search?{bbox_str}'
colSearchResponse = requests.get(colSearchUrl, headers=headers)
test = colSearchResponse.json()


# %%


# %%
# --> this should return the selected points as a geojson file/object...


# %%


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


# %%
# Perform inference run with the newly trained model

link_to_model = ''
aoi =
start_date =
end_date =


# %%
# Perform inference run with the default cropland model


# use the cropland product to mask the previously generated custom crop type product
