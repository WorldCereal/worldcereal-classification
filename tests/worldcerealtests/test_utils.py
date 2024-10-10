import numpy as np
from catboost import CatBoostClassifier
from openeo_gfmap.backend import cdse_connection

from worldcereal.utils.models import load_model_onnx
from worldcereal.utils.upload import deploy_model


def test_deploy_model():
    """Simple test to deploy a CatBoost model and load it back."""
    model = CatBoostClassifier(iterations=10).fit(X=[[1, 2], [3, 4]], y=[0, 1])
    presigned_uri = deploy_model(cdse_connection(), model)
    model = load_model_onnx(presigned_uri)

    # Compare model predictions with the original targets
    np.testing.assert_array_equal(
        model.run(None, {"features": [[1, 2], [3, 4]]})[0], [0, 1]
    )
