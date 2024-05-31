"""Feature computer GFMAP compatible to compute Presto embeddings."""

import xarray as xr
from openeo_gfmap.features.feature_extractor import PatchFeatureExtractor


class PrestoFeatureExtractor(PatchFeatureExtractor):
    """Feature extractor to use Presto model to compute embeddings.
    This will generate a datacube with 128 bands, each band representing a
    feature from the Presto model.
    """

    import functools
    from pathlib import Path
    from typing import Tuple

    PRESTO_PATH = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/presto.pt"  # NOQA
    BASE_URL = "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies"  # NOQA
    DEPENDENCY_NAME = "wc_presto_onnx_dependencies.zip"

    GFMAP_BAND_MAPPING = {
        "S2-L2A-B02": "B02",
        "S2-L2A-B03": "B03",
        "S2-L2A-B04": "B04",
        "S2-L2A-B05": "B05",
        "S2-L2A-B06": "B06",
        "S2-L2A-B07": "B07",
        "S2-L2A-B08": "B08",
        "S2-L2A-B8A": "B8A",
        "S2-L2A-B11": "B11",
        "S2-L2A-B12": "B12",
        "S1-SIGMA0-VH": "VH",
        "S1-SIGMA0-VV": "VV",
        "COP-DEM": "DEM",
        "A5-tmean": "temperature-mean",
        "A5-precip": "precipitation-flux",
    }

    def __init__(self):
        """
        Initializes the PrestoFeatureExtractor object, starting a logger.
        """
        import logging

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(PrestoFeatureExtractor.__name__)

    @classmethod
    @functools.lru_cache(maxsize=6)
    def extract_dependencies(cls, base_url: str, dependency_name: str):
        """Extract the dependencies from the given URL. Unpacking a zip
        file in the current working directory.
        """
        import shutil
        import urllib.request
        from pathlib import Path

        # Generate absolute path for the dependencies folder
        dependencies_dir = Path.cwd() / "dependencies"

        # Create the directory if it doesn't exist
        dependencies_dir.mkdir(exist_ok=True, parents=True)

        # Download and extract the model file
        modelfile_url = f"{base_url}/{dependency_name}"
        modelfile, _ = urllib.request.urlretrieve(
            modelfile_url, filename=dependencies_dir / Path(modelfile_url).name
        )
        shutil.unpack_archive(modelfile, extract_dir=dependencies_dir)

        # Add the model directory to system path if it's not already there
        abs_path = str(
            dependencies_dir / Path(modelfile_url).name.split(".zip")[0]
        )  # NOQA

        return abs_path

    def output_labels(self) -> list:
        """Returns the output labels from this UDF, which is the output labels
        of the presto embeddings"""
        return [f"presto_ft_{i}" for i in range(128)]

    def execute(self, inarr: xr.DataArray) -> xr.DataArray:
        import sys

        if self.epsg is None:
            raise ValueError(
                "EPSG code is required for Presto feature extraction, but was "
                "not correctly initialized."
            )

        # The below is required to avoid flipping of the result
        # when running on OpenEO backend!
        inarr = inarr.transpose("bands", "t", "x", "y")

        # Change the band names
        new_band_names = [
            self.GFMAP_BAND_MAPPING.get(b.item(), b.item()) for b in inarr.bands
        ]
        inarr = inarr.assign_coords(bands=new_band_names)

        # Handle NaN values in Presto compatible way
        inarr = inarr.fillna(65535)

        # Unzip de dependencies on the backend
        self.logger.info("Unzipping dependencies")
        deps_dir = self.extract_dependencies(self.BASE_URL, self.DEPENDENCY_NAME)

        self.logger.info("Appending dependencies")
        sys.path.append(str(deps_dir))

        from dependencies.wc_presto_onnx_dependencies.mvp_wc_presto.world_cereal_inference import (
            get_presto_features,
        )

        self.logger.info("Extracting presto features")
        features = get_presto_features(inarr, self.PRESTO_PATH, self.epsg)
        return features
