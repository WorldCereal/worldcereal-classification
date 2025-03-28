"""Model inference on Presto feature for binary classication"""

import xarray as xr
from openeo_gfmap.inference.model_inference import ModelInference


class CropClassifier(ModelInference):
    """Binary or multi-class crop classifier using ONNX to load a catboost model.

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

    def __init__(self):
        super().__init__()

        self.onnx_session = None

    def dependencies(self) -> list:
        return []  # Disable the dependencies from PIP install

    def output_labels(self) -> list:
        lut = self._parameters.get("lookup_table", None)
        if lut is None:
            raise ValueError("Lookup table is not defined.")
        lut_sorted = {k: v for k, v in sorted(lut.items(), key=lambda item: item[1])}
        class_names = lut_sorted.keys()

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

        lut_sorted = {
            k: v for k, v in sorted(lookup_table.items(), key=lambda item: item[1])
        }

        if self.onnx_session is None:
            raise ValueError("Model has not been loaded. Please load a model first.")

        # Prepare input data for ONNX model
        outputs = self.onnx_session.run(None, {"features": features})

        # Extract classes as INTs and probability of winning class values
        labels = np.zeros((len(outputs[0]),), dtype=np.uint16)
        probabilities = np.zeros((len(outputs[0]),), dtype=np.uint8)
        for i, (label, prob) in enumerate(zip(outputs[0], outputs[1])):
            labels[i] = lut_sorted[label]
            probabilities[i] = int(round(prob[label] * 100))

        # Extract per class probabilities
        output_probabilities = []
        for output_px in outputs[1]:
            output_probabilities.append(
                [output_px[label] for label in lut_sorted.keys()]
            )

        output_probabilities = (
            (np.array(output_probabilities) * 100).round().astype(np.uint8)
        )

        return np.hstack(
            [labels[:, np.newaxis], probabilities[:, np.newaxis], output_probabilities]
        ).transpose()

    def execute(self, inarr: xr.DataArray) -> xr.DataArray:

        if "classifier_url" not in self._parameters:
            raise ValueError('Missing required parameter "classifier_url"')
        classifier_url = self._parameters.get("classifier_url")
        self.logger.info(f'Loading classifier model from "{classifier_url}"')

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
