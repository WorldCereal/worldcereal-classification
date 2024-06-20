"""Model inference on Presto feature for binary classication"""

import xarray as xr
from openeo_gfmap.inference.model_inference import ModelInference


class CroplandClassifier(ModelInference):
    """Binary crop-land classifier using ONNX to load a catboost model.

    The classifier use the embeddings computed from the Presto Feature
    Extractor.

    Interesting UDF parameters:
    - classifier_url: A public URL to the ONNX classification model. Default is
      the public Presto model.
    """

    import numpy as np

    CATBOOST_PATH = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/wc_catboost.onnx"  # NOQA

    def __init__(self):
        super().__init__()

        self.onnx_session = None

    def dependencies(self) -> list:
        return []  # Disable the dependencies from PIP install

    def output_labels(self) -> list:
        return ["classification", "probability"]

    def predict(self, features: np.ndarray) -> np.ndarray:
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
        classifier_url = self._parameters.get("classifier_url", self.CATBOOST_PATH)

        # shape and indices for output ("xy", "bands")
        x_coords, y_coords = inarr.x.values, inarr.y.values
        inarr = inarr.transpose("bands", "x", "y").stack(xy=["x", "y"]).transpose()

        self.onnx_session = self.load_ort_session(classifier_url)

        # Run catboost classification
        self.logger.info("Catboost classification with input shape: %s", inarr.shape)
        classification = self.predict(inarr.values)
        self.logger.info("Classification done with shape: %s", inarr.shape)

        classification = xr.DataArray(
            classification.reshape((2, len(x_coords), len(y_coords))),
            dims=["bands", "x", "y"],
            coords={
                "bands": ["classification", "probability"],
                "x": x_coords,
                "y": y_coords,
            },
        )

        return classification
