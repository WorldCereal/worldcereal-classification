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

        # Get the prediction labels
        binary_labels = (outputs[0] == "True").astype(np.uint8).reshape((-1, 1))

        # Extract all probabilities
        all_probabilities = np.round(
            np.array([[x["False"], x["True"]] for x in outputs[1]]) * 100.0
        ).astype(np.uint8)
        max_probability = np.max(all_probabilities, axis=1, keepdims=True)

        return np.concatenate([binary_labels, max_probability], axis=1).transpose()

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

        classification_da = xr.DataArray(
            classification.reshape((2, len(x_coords), len(y_coords))),
            dims=["bands", "x", "y"],
            coords={
                "bands": [
                    "classification",
                    "probability",
                ],
                "x": x_coords,
                "y": y_coords,
            },
        )

        return classification_da


class CroptypeClassifier(ModelInference):
    """Multi-class crop classifier using ONNX to load a catboost model.

    The classifier use the embeddings computed from the Presto Feature
    Extractor.

    Interesting UDF parameters:
    - classifier_url: A public URL to the ONNX classification model. Default is
      the public Presto model.
    - lookup_table: A dictionary mapping class names to class labels, ordered by
      model probability output. This is required for the model to map the output
      probabilities to class names.
    """

    import numpy as np

    CATBOOST_PATH = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ss-wc-ft-ct-30D_test_CROPTYPE9.onnx"  # NOQA

    def __init__(self):
        super().__init__()

        self.onnx_session = None

    def dependencies(self) -> list:
        return []  # Disable the dependencies from PIP install

    def output_labels(self) -> list:
        class_names = self._parameters["lookup_table"].keys()

        return ["classification", "probability"] + [
            f"probability_{name}" for name in class_names
        ]

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predicts labels using the provided features array.
        """
        import numpy as np

        # Classes names to codes
        lookup_table = self._parameters.get("lookup_table", None)

        if lookup_table is None:
            raise ValueError(
                "Lookup table is not defined. Please provide lookup_table in the UDFs parameters."
            )

        if self.onnx_session is None:
            raise ValueError("Model has not been loaded. Please load a model first.")

        # Prepare input data for ONNX model
        outputs = self.onnx_session.run(None, {"features": features})

        # Extract classes as INTs and probability of winning class values
        labels = np.zeros((len(outputs[0]),), dtype=np.uint16)
        probabilities = np.zeros((len(outputs[0]),), dtype=np.uint8)
        for i, (label, prob) in enumerate(zip(outputs[0], outputs[1])):
            labels[i] = lookup_table[label]
            probabilities[i] = int(round(prob[label] * 100))

        # Extract per class probabilities
        output_probabilities = []
        for output_px in outputs[1]:
            output_probabilities.append(list(output_px.values()))

        output_probabilities = (
            (np.array(output_probabilities) * 100).round().astype(np.uint8)
        )

        return np.hstack(
            [labels[:, np.newaxis], probabilities[:, np.newaxis], output_probabilities]
        ).transpose()

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

        output_labels = self.output_labels()

        classification_da = xr.DataArray(
            classification.reshape((len(output_labels), len(x_coords), len(y_coords))),
            dims=["bands", "x", "y"],
            coords={
                "bands": output_labels,
                "x": x_coords,
                "y": y_coords,
            },
        )

        return classification_da
