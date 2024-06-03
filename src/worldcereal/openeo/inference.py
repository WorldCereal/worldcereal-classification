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

    CATBOOST_PATH = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/wc_catboost.onnx"  # NOQA
    BASE_URL = "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies"  # NOQA
    DEPENDENCY_NAME = "wc_presto_onnx_dependencies.zip"

    def output_labels(self) -> list:
        return ["classification"]

    def execute(self, inarr: xr.DataArray) -> xr.DataArray:
        import sys

        classifier_url = self._parameters.get("classifier_url", self.CATBOOST_PATH)

        # shape and indiches for output
        inarr = inarr.transpose("bands", "x", "y")

        # Unzip de dependencies on the backend
        self.logger.info("Unzipping dependencies")
        dep_dir = self.extract_dependencies(self.BASE_URL, self.DEPENDENCY_NAME)

        self.logger.info("Adding dependencies")
        sys.path.append(str(dep_dir))

        from dependencies.wc_presto_onnx_dependencies.mvp_wc_presto.world_cereal_inference import (
            classify_with_catboost,
        )

        # Run catboost classification
        self.logger.info("Catboost classification")
        classification = classify_with_catboost(inarr, classifier_url)
        self.logger.info("Done")

        return classification
