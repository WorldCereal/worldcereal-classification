"""Model inference on Presto feature for binary classication"""

import xarray as xr
from openeo.udf import XarrayDataCube
from openeo_gfmap.inference.model_inference import ModelInference


class CroptypeClassifier(ModelInference):
    """Multi-class crop classifier using ONNX to load a catboost model.

    The classifier use the embeddings computed from the Presto Feature
    Extractor.

    Interesting UDF parameters:
    - classifier_url: A public URL to the ONNX classification model. Default is
      the public Presto model.
    """

    import functools

    import numpy as np

    CATBOOST_CROPLAND_PATH = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/wc_catboost.onnx"  # NOQA
    CATBOOST_CROPTYPE_PATH = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ss-wc-ft-ct-30D_test_CROPTYPE9.onnx"  # NOQA
    PRESTO_CROPLAND_MODEL_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/presto.pt"  # NOQA
    PRESTO_CROPTYPE_MODEL_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ss-wc-ft-ct-30D_test.pt"  # NOQA"
    PRESTO_WHL_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/dependencies/presto_worldcereal-0.1.1-py3-none-any.whl"
    BASE_URL = "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies"  # NOQA
    DEPENDENCY_NAME = "worldcereal_deps.zip"

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
        "AGERA5-TMEAN": "temperature-mean",
        "AGERA5-PRECIP": "precipitation-flux",
    }

    def __init__(self):
        super().__init__()

        self.onnx_session = None

    @functools.lru_cache(maxsize=6)
    def unpack_presto_wheel(self, wheel_url: str, destination_dir: str) -> list:
        import urllib.request
        import zipfile
        from pathlib import Path

        # Downloads the wheel file
        modelfile, _ = urllib.request.urlretrieve(
            wheel_url, filename=Path.cwd() / Path(wheel_url).name
        )
        with zipfile.ZipFile(modelfile, "r") as zip_ref:
            zip_ref.extractall(destination_dir)
        return destination_dir

    def dependencies(self) -> list:
        return []  # Disable the dependencies from PIP install

    def output_labels(self) -> list:
        return ["classification", "probability"]

    def predict_croptype(self, features: np.ndarray) -> np.ndarray:
        """
        Predicts labels using the provided features array.
        """
        import numpy as np

        if self.onnx_session is None:
            raise ValueError("Model has not been loaded. Please load a model first.")

        # Prepare input data for ONNX model
        outputs = self.onnx_session.run(None, {"features": features})

        # Apply LUT: TODO:this needs an update!
        LUT = {
            "barley": 1,
            "maize": 2,
            "millet_sorghum": 3,
            "other_crop": 4,
            "rapeseed_rape": 5,
            "soy_soybeans": 6,
            "sunflower": 7,
            "wheat": 8,
        }

        # Extract classes as INTs and probability of winning class values
        labels = np.zeros((len(outputs[0]),), dtype=np.uint16)
        probabilities = np.zeros((len(outputs[0]),), dtype=np.uint8)
        for i, (label, prob) in enumerate(zip(outputs[0], outputs[1])):
            labels[i] = LUT[label]
            probabilities[i] = int(prob[label] * 100)

        return np.stack([labels, probabilities], axis=0)

    def predict_cropland(self, features: np.ndarray) -> np.ndarray:
        """
        Predicts labels using the provided features array.
        """
        import numpy as np

        if self.onnx_session is None:
            raise ValueError("Model has not been loaded. Please load a model first.")

        # Prepare input data for ONNX model
        outputs = self.onnx_session.run(None, {"features": features})

        # Threshold for binary conversion
        threshold = 0.5

        # Extract all prediction values and convert them to binary labels
        prediction_values = [sublist["True"] for sublist in outputs[1]]
        binary_labels = np.array(prediction_values) >= threshold
        binary_labels = binary_labels.astype("uint8")

        prediction_values = np.array(prediction_values) * 100.0
        prediction_values = np.round(prediction_values).astype("uint8")

        return np.stack([binary_labels, prediction_values], axis=0)

    def execute(self, inarr: xr.DataArray) -> xr.DataArray:
        import sys

        if self.epsg is None:
            raise ValueError(
                "EPSG code is required for Presto feature extraction, but was "
                "not correctly initialized."
            )
        presto_cropland_model_url = self._parameters.get(
            "presto_cropland_model_url", self.PRESTO_CROPLAND_MODEL_URL
        )
        presto_croptype_model_url = self._parameters.get(
            "presto_croptype_model_url", self.PRESTO_CROPTYPE_MODEL_URL
        )
        presto_wheel_url = self._parameters.get("presto_wheel_url", self.PRESTO_WHL_URL)

        ignore_dependencies = self._parameters.get("ignore_dependencies", False)
        if ignore_dependencies:
            self.logger.info(
                "`ignore_dependencies` flag is set to True. Make sure that "
                "Presto and its dependencies are available on the runtime "
                "environment"
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
        if not ignore_dependencies:
            self.logger.info("Unzipping dependencies")
            deps_dir = self.extract_dependencies(self.BASE_URL, self.DEPENDENCY_NAME)
            self.logger.info("Unpacking presto wheel")
            deps_dir = self.unpack_presto_wheel(presto_wheel_url, deps_dir)

            self.logger.info("Appending dependencies")
            sys.path.append(str(deps_dir))

        # --------------------------
        # PRESTO FEATURE COMPUTATION

        from presto.inference import (  # pylint: disable=import-outside-toplevel
            get_presto_features,
        )

        batch_size = self._parameters.get("batch_size", 1024)

        self.logger.info("Extracting presto cropland features")
        cropland_features = get_presto_features(
            inarr, presto_cropland_model_url, self.epsg, batch_size=batch_size
        )
        self.logger.info("Extracting presto croptype features")
        croptype_features = get_presto_features(
            inarr, presto_croptype_model_url, self.epsg, batch_size=batch_size
        )

        # --------------------------
        # CATBOOST INFERENCE

        cropland_classifier_url = self._parameters.get(
            "cropland_classifier_url", self.CATBOOST_CROPLAND_PATH
        )
        croptype_classifier_url = self._parameters.get(
            "croptype_classifier_url", self.CATBOOST_CROPTYPE_PATH
        )

        # shape and indices for output ("xy", "bands")
        x_coords, y_coords = inarr.x.values, inarr.y.values

        # First cropland
        cropland_features = (
            cropland_features.transpose("bands", "x", "y")
            .stack(xy=["x", "y"])
            .transpose()
        )
        self.onnx_session = self.load_ort_session(cropland_classifier_url)
        self.logger.info(
            "Catboost cropland classification with input shape: %s",
            cropland_features.shape,
        )
        cropland_classification = self.predict_cropland(cropland_features.values)
        cropland_classification = xr.DataArray(
            cropland_classification.reshape((2, len(x_coords), len(y_coords))),
            dims=["bands", "x", "y"],
            coords={
                "bands": ["classification", "probability"],
                "x": x_coords,
                "y": y_coords,
            },
        )
        self.logger.info(
            "Classification done with shape: %s", cropland_classification.shape
        )

        # Then Croptype
        croptype_features = (
            croptype_features.transpose("bands", "x", "y")
            .stack(xy=["x", "y"])
            .transpose()
        )
        self.onnx_session = self.load_ort_session(croptype_classifier_url)
        self.logger.info(
            "Catboost croptype classification with input shape: %s",
            croptype_features.shape,
        )
        croptype_classification = self.predict_croptype(croptype_features.values)
        self.logger.info(
            "Classification done with shape: %s", croptype_classification.shape
        )

        croptype_classification = xr.DataArray(
            croptype_classification.reshape((2, len(x_coords), len(y_coords))),
            dims=["bands", "x", "y"],
            coords={
                "bands": ["classification", "probability"],
                "x": x_coords,
                "y": y_coords,
            },
        )

        # Mask croptype for cropland
        croptype_classification = croptype_classification.where(
            cropland_classification.sel(bands="classification") == 1, 0
        )

        return croptype_classification

    def _execute(self, cube: XarrayDataCube, parameters: dict) -> XarrayDataCube:
        # Disable S1 rescaling (decompression) by default
        if parameters.get("rescale_s1", None) is None:
            parameters.update({"rescale_s1": False})

        arr = cube.get_array().transpose("bands", "t", "y", "x")
        arr = self._common_preparations(arr, parameters)
        if self._parameters.get("rescale_s1", True):
            arr = self._rescale_s1_backscatter(arr)

        arr = self.execute(arr).transpose("bands", "y", "x")
        return XarrayDataCube(arr)
