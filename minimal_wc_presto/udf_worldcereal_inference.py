import logging
import urllib.request
import shutil
from pathlib import Path
import sys
import functools
import xarray as xr
from typing import Dict
from openeo.metadata import  CollectionMetadata


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


def apply_metadata(input_metadata:CollectionMetadata, context:dict) -> CollectionMetadata:

    xstep = input_metadata.get('x','step')
    ystep = input_metadata.get('y','step')


    new_metadata = {
          "x": {"type": "spatial", "axis": "x", "step": xstep, "reference_system": 4326},
          "y": {"type": "spatial", "axis": "y", "step": ystep, "reference_system": 4326},
    }
    return CollectionMetadata(new_metadata)


def apply_datacube(cube: xr.DataArray, context:Dict) -> xr.DataArray:
    
    logger = _setup_logging() 


# Install PyTorch using pip

    orig_dims = list(cube.dims)
    orig_dims.remove("t")

    logger.info("Unzipping dependencies")
    base_url = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/"
    dependency_name = "wc_presto_onnx_dependencies.zip"

    logger.info("Appending depencency")
    dep_dir = extract_dependencies(base_url, dependency_name)
    sys.path.append(str(dep_dir))


    from dependencies.wc_presto_onnx_dependencies.mvp_wc_presto.world_cereal_inference import get_presto_features, classify_with_catboost

    logger.info("Reading in required libs")

    logger.info("Extracting presto features")
    PRESTO_PATH = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/presto.pt"
    features = get_presto_features(cube, PRESTO_PATH)

    logger.info("Catboost classification")
    CATBOOST_PATH = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/wc_catboost.onnx"
    classification = classify_with_catboost(features, orig_dims, CATBOOST_PATH)


    return classification













