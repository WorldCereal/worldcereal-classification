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

@functools.lru_cache(maxsize=25)
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
    

    # The below is required to avoid flipping of the result
    # when running on OpenEO backend!
    cube = cube.transpose("bands", "t", "x", "y")

    # Handle NaN values in Presto compatible way
    cube = cube.fillna(65535)

    logger.info("Unzipping dependencies")
    #base_url = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/"
    base_url  = "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies"
    dependency_name = "wc_presto_onnx_dependencies.zip"

    logger.info("Appending depencency")
    dep_dir = extract_dependencies(base_url, dependency_name)
    sys.path.append(str(dep_dir))

    #directly add a path to the older pandas version
    from dependencies.wc_presto_onnx_dependencies.mvp_wc_presto.world_cereal_inference import get_presto_features


    logger.info("Extracting presto features")
    PRESTO_PATH = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/presto.pt"
    output = get_presto_features(cube, PRESTO_PATH)

 
    return output
















