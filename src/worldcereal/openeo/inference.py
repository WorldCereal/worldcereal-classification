"""Model inference on Presto feature for binary classication"""

import xarray as xr

from openeo_gfmap.inference.model_inference import ModelInference

class CroplandClassifier(ModelInference):
    import functools

    CATBOOST_PATH = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/wc_catboost.onnx"  # NOQA
    BASE_URL = "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies"  # NOQA
    DEPENDENCY_NAME = "wc_presto_onnx_dependencies.zip"

    def __init__(self):
        import logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(WorldCerealInference.__name__)

    @classmethod
    @functools.lru_cache(maxsize=6)
    def extract_dependencies(cls, base_url: str, dependency_name: str):
        from pathlib import Path
        import urllib.request
        import shutil

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
    
    def output_labels(self) -> list:
        return ["classification"]

    def execute(self, inarr: xr.DataArray) -> xr.DataArray:
        import sys
    
        # shape and indiches for output
        inarr = inarr.transpose('bands', 'x', 'y')

        # Unzip de dependencies on the backend
        self.logger.info("Unzipping dependencies")
        dep_dir = self.extract_dependencies(self.BASE_URL, self.DEPENDENCY_NAME)

        self.logger.info("Adding dependencies")
        sys.path.append(str(dep_dir))

        from dependencies.wc_presto_onnx_dependencies.mvp_wc_presto.world_cereal_inference import classify_with_catboost

        # Run catboost classification
        self.logger.info("Catboost classification")
        classification = classify_with_catboost(inarr, self.CATBOOST_PATH)
        self.logger.info("Done")

        return classification
