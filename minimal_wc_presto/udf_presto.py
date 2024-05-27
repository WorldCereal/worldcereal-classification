import logging
import urllib.request
import shutil
from pathlib import Path
import sys
import functools
import xarray as xr
from typing import Dict
import numpy as np
from pyproj import Transformer


def _setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

@functools.lru_cache(maxsize=6)
def extract_dependencies(base_url: str, dependency_name: str):

    # Generate absolute path for the dependencies folder
    dependencies_dir = Path.cwd() / 'dependencies'

    # Create the directory if it doesn't exist
    dependencies_dir.mkdir(exist_ok=True, parents=True)


    # Download and extract the model file
    modelfile_url = f"{base_url}/{dependency_name}"
    modelfile, _ = urllib.request.urlretrieve(modelfile_url, filename=dependencies_dir / Path(modelfile_url).name)
    shutil.unpack_archive(modelfile, extract_dir=dependencies_dir)

    # Add the model directory to system path if it's not already there
    abs_path = str(dependencies_dir / Path(modelfile_url).name.split('.zip')[0])

    return(abs_path)


def apply_datacube(cube: xr.DataArray, context:Dict) -> xr.DataArray:
    
    logger = _setup_logging() 

    # shape and indiches for output
    orig_dims = list(cube.dims)
    map_dims = cube.shape[2:]

    logger.info("Unzipping dependencies")
    base_url = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/"
    dependency_name = "wc_presto_onnx_dependencies.zip"

    logger.info("Appending depencency")
    dep_dir = extract_dependencies(base_url, dependency_name)
    

    #directly add a path to the older pandas version
    sys.path.append(str(dep_dir))
    sys.path.append(str(dep_dir) + '/pandas')

    from dependencies.wc_presto_onnx_dependencies.mvp_wc_presto.world_cereal_inference import get_presto_features

    logger.info("Reading in required libs")

    logger.info("Extracting presto features")
    PRESTO_PATH = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/presto.pt"
    features = get_presto_features(cube, PRESTO_PATH)

    # go to 128,1,100,100
    presto_dim = map_dims + (128,)    
    features = features.reshape(presto_dim)
    features = np.expand_dims(features, axis = 0)
    features = np.transpose(features, (3, 0, 1, 2))


    transformer = Transformer.from_crs(f"EPSG:{4326}", "EPSG:4326", always_xy=True)
    longitudes, latitudes = transformer.transform(cube.x, cube.y)

    
    output = xr.DataArray(features, dims=orig_dims, coords={'y': longitudes, 'x': latitudes})
    return output
















