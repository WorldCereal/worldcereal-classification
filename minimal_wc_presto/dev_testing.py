#%%
from pathlib import Path  

from pyproj import Transformer
import numpy as np

import requests
import xarray as xr


#%% GET DEPENDENCIES
import urllib
# Generate absolute path for the dependencies folder
dependencies_dir = Path.cwd() / 'dependencies'
dependencies_dir.mkdir(exist_ok=True, parents=True)

base_url = 'https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies'
dependency_name = "wc_presto_onnx_dependencies.zip"

# Download and extract the model file
modelfile_url = f"{base_url}/{dependency_name}"
modelfile, _ = urllib.request.urlretrieve(modelfile_url, filename=dependencies_dir / Path(modelfile_url).name)
#shutil.unpack_archive(modelfile, extract_dir=dependencies_dir)

#Add the model directory to system path if it's not already there
#abs_path = str(dependencies_dir / Path(modelfile_url).name.split('.zip')[0])
#sys.path.append(abs_path)

# Get Data
#url = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/belgium_good_2020-12-01_2021-11-30.nc"
#filename = "belgium_good_2020-12-01_2021-11-30.nc"

#with requests.get(url, stream=True) as r:
#    r.raise_for_status()
#    with open(filename, 'wb') as f:
#        for chunk in r.iter_content(chunk_size=8192):
#            f.write(chunk)

#%%

# Read the file into xarray
ds = xr.open_dataset('data/belgium_good_2020-12-01_2021-11-30.nc')


arr = ds.drop('crs').to_array(dim='bands')
arr[:,:,50:,50:] = np.nan 
orig_dims = list(arr.dims)
map_dims = arr.shape[2:]

#%% Get Presto
from mvp_wc_presto.world_cereal_inference import get_presto_features

#bands: 19, t: 12y, : 100x: 100y
data_url  = 'https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/belgium_good_2020-12-01_2021-11-30.nc'
# Fetch the data from the URL
response = requests.get(data_url)

#10000,128
presto_path = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/presto.pt"
features = get_presto_features(arr, presto_path) 

#10000,
from mvp_wc_presto.world_cereal_inference import  classify_with_catboost

CATBOOST_PATH = 'https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/wc_catboost.onnx'
classification = classify_with_catboost(features, CATBOOST_PATH)

#%%

#%%plot output
import matplotlib.pyplot as plt

transformer = Transformer.from_crs(f"EPSG:{4326}", "EPSG:4326", always_xy=True)
longitudes, latitudes = transformer.transform(arr.x, arr.y)
classification = np.flip(classification.reshape(map_dims),axis = 0)
classification = np.expand_dims(np.expand_dims(classification, axis=0),axis = 0)
output = xr.DataArray(classification, dims=orig_dims)

output = output.to_numpy().squeeze()
plt.imshow(output)

output.shape
# %%
