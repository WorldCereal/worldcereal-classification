import copy
import functools
import sys
import requests
import xarray as xr

import numpy as np
from scipy.signal import convolve2d

from openeo.udf.udf_data import UdfData
from openeo.udf import XarrayDataCube
from openeo.metadata import CollectionMetadata

sys.path.append("onnx_deps")
import onnxruntime as ort  # noqa: E402


EXCLUDED_VALUES = [254, 255, 65535]
NODATA = 255

@functools.lru_cache(maxsize=6)
def lut_from_url(model_url: str) -> dict:
    """Method to extract lookup table from model URL.
    Loads the model, validates it and extracts LUT from the model metadata.

    Parameters
    ----------
    model_url : str
        Public URL to the ONNX classification model.
    """

    # Load the model
    response = requests.get(model_url, timeout=120)
    model = ort.InferenceSession(response.content)

    # Validate the model
    metadata = model.get_modelmeta().custom_metadata_map

    if "class_params" not in metadata:
        raise ValueError("Could not find class names in the model metadata.")

    class_params = eval(metadata["class_params"], {"__builtins__": None}, {})

    if "class_names" not in class_params:
        raise ValueError("Could not find class names in the model metadata.")

    if "class_to_label" not in class_params:
        raise ValueError("Could not find class to labels in the model metadata.")

    # Load model LUT
    lut = dict(zip(class_params["class_names"], class_params["class_to_label"]))
    sorted_lut = {k: v for k, v in sorted(lut.items(), key=lambda item: item[1])}

    return sorted_lut



def get_output_labels(sorted_lut: dict, parameters: dict) -> list:
    if parameters.get("keep_class_probs", False):
        return ["classification", "probability"] + [
            f"probability_{name}" for name in sorted_lut.keys()
        ]
    return ["classification", "probability"]


def majority_vote(
    base_labels: xr.DataArray,
    max_probabilities: xr.DataArray,
    kernel_size: int,
) -> xr.DataArray:
    """Majority vote is performed using a sliding local kernel.
    For each pixel, the voting of a final class is done by counting
    neighbours values.
    Pixels that have one of the specified excluded values are
    excluded in the voting process and are unchanged.

    The prediction probabilities are reevaluated by taking, for each pixel,
    the average of probabilities of the neighbors that belong to the winning class.
    (For example, if a pixel was voted to class 2 and there are three
    neighbors of that class, then the new probability is the sum of the
    old probabilities of each pixels divided by 3)

    Parameters
    ----------
    base_labels : xr.DataArray
        The original predicted classification labels.
    max_probabilities : xr.DataArray
        The original probabilities of the winning class (ranging between 0 and 100).
    kernel_size : int
        The size of the kernel used for the neighbour around the pixel.

    Returns
    -------
    xr.DataArray
        The cleaned classification labels and associated probabilities.
    """



    prediction = base_labels.values
    probability = max_probabilities.values

    # As the probabilities are in integers between 0 and 100,
    # we use uint16 matrices to store the vote scores
    assert (
        kernel_size <= 25
    ), f"Kernel value cannot be larger than 25 (currently: {kernel_size}) because it might lead to scenarios where the 16-bit count matrix is overflown"

    # Build a class mapping, so classes are converted to indexes and vice-versa
    unique_values = set(np.unique(prediction))
    unique_values = sorted(unique_values - set(EXCLUDED_VALUES))  # type: ignore
    index_value_lut = [(k, v) for k, v in enumerate(unique_values)]

    counts = np.zeros(
        shape=(*prediction.shape, len(unique_values)), dtype=np.uint16
    )
    probabilities = np.zeros(
        shape=(*probability.shape, len(unique_values)), dtype=np.uint16
    )

    # Iterates for each classes
    for cls_idx, cls_value in index_value_lut:
        # Take the binary mask of the interest class, and multiply by the probabilities
        class_mask = ((prediction == cls_value) * probability).astype(np.uint16)

        # Set to 0 the class scores where the label is excluded
        for excluded_value in EXCLUDED_VALUES:
            class_mask[prediction == excluded_value] = 0

        # Binary class mask, used to count HOW MANY neighbours pixels are used for this class
        binary_class_mask = (class_mask > 0).astype(np.uint16)

        # Creates the kernel
        kernel = np.ones(shape=(kernel_size, kernel_size), dtype=np.uint16)

        # Counts around the window the sum of probabilities for that given class
        counts[:, :, cls_idx] = convolve2d(class_mask, kernel, mode="same")

        # Counts the number of neighbors pixels that voted for that given class
        class_voters = convolve2d(binary_class_mask, kernel, mode="same")
        # Remove the 0 values because might create divide by 0 issues
        class_voters[class_voters == 0] = 1

        probabilities[:, :, cls_idx] = np.divide(
            counts[:, :, cls_idx], class_voters
        )

    # Initializes output array
    aggregated_predictions = np.zeros(
        shape=(counts.shape[0], counts.shape[1]), dtype=np.uint16
    )
    # Initializes probabilities output array
    aggregated_probabilities = np.zeros(
        shape=(counts.shape[0], counts.shape[1]), dtype=np.uint16
    )

    if len(unique_values) > 0:
        # Takes the indices that have the biggest scores
        aggregated_predictions_indices = np.argmax(counts, axis=2)

        # Get the new probabilities of the predictions
        aggregated_probabilities = np.take_along_axis(
            probabilities,
            aggregated_predictions_indices.reshape(
                *aggregated_predictions_indices.shape, 1
            ),
            axis=2,
        ).squeeze()

        # Check which pixels have a counts value equal to 0
        no_score_mask = np.sum(counts, axis=2) == 0

        # convert back to values from indices
        for cls_idx, cls_value in index_value_lut:
            aggregated_predictions[aggregated_predictions_indices == cls_idx] = (
                cls_value
            )
            aggregated_predictions = aggregated_predictions.astype(np.uint16)

        aggregated_predictions[no_score_mask] = NODATA
        aggregated_probabilities[no_score_mask] = NODATA

    # Setting excluded values back to their original values
    for excluded_value in EXCLUDED_VALUES:
        aggregated_predictions[prediction == excluded_value] = excluded_value
        aggregated_probabilities[prediction == excluded_value] = excluded_value

    return xr.DataArray(
        np.stack((aggregated_predictions, aggregated_probabilities)),
        dims=["bands", "y", "x"],
        coords={
            "bands": ["classification", "probability"],
            "y": base_labels.y,
            "x": base_labels.x,
        },
    )


def smooth_probabilities(
    base_labels: xr.DataArray, class_probabilities: xr.DataArray
) -> xr.DataArray:
    """Performs gaussian smoothing on the class probabilities. Requires the
    base labels to keep the pixels that are excluded away from smoothing.
    """

    base_labels_vals = base_labels.values
    probabilities_vals = class_probabilities.values

    excluded_mask = np.in1d(
        base_labels_vals.reshape(-1),
        EXCLUDED_VALUES,
    ).reshape(*base_labels_vals.shape)

    conv_kernel = np.array([[1, 2, 1], [2, 3, 2], [1, 2, 1]], dtype=np.int16)

    for class_idx in range(probabilities_vals.shape[0]):
        probabilities_vals[class_idx] = (
            convolve2d(
                probabilities_vals[class_idx],
                conv_kernel,
                mode="same",
                boundary="symm",
            )
            / conv_kernel.sum()
        )
        probabilities_vals[class_idx][excluded_mask] = 0

    # Sum of probabilities should be 1, cast to uint16
    probabilities_vals = np.round(
        probabilities_vals / probabilities_vals.sum(axis=0) * 100.0
    ).astype("uint16")

    return xr.DataArray(
        probabilities_vals,
        coords=class_probabilities.coords,
        dims=class_probabilities.dims,
    )

def reclassify(
    base_labels: xr.DataArray,
    base_max_probs: xr.DataArray,
    probabilities: xr.DataArray,
) -> xr.DataArray:

    base_labels_vals = base_labels.values
    base_max_probs_vals = base_max_probs.values

    excluded_mask = np.in1d(
        base_labels_vals.reshape(-1),
        EXCLUDED_VALUES,
    ).reshape(*base_labels_vals.shape)

    new_labels_vals = np.argmax(probabilities.values, axis=0)
    new_max_probs_vals = np.max(probabilities.values, axis=0)

    new_labels_vals[excluded_mask] = base_labels_vals[excluded_mask]
    new_max_probs_vals[excluded_mask] = base_max_probs_vals[excluded_mask]

    return xr.DataArray(
        np.stack((new_labels_vals, new_max_probs_vals)),
        dims=["bands", "y", "x"],
        coords={
            "bands": ["classification", "probability"],
            "y": base_labels.y,
            "x": base_labels.x,
        },
    )


def execute(inarr: xr.DataArray, parameters: dict) -> xr.DataArray:

    if "classifier_url" not in parameters:
        raise ValueError('Missing required parameter "classifier_url"')
    classifier_url = parameters.get("classifier_url")

    lookup_table = lut_from_url(classifier_url)

    if parameters.get("method") == "smooth_probabilities":
        # Cast to float for more accurate gaussian smoothing
        class_probabilities = (
            inarr.isel(bands=slice(2, None)).astype("float32") / 100.0
        )

        # Peform probability smoothing
        class_probabilities = smooth_probabilities(
            inarr.sel(bands="classification"), class_probabilities
        )

        # Reclassify
        new_labels = reclassify(
            inarr.sel(bands="classification"),
            inarr.sel(bands="probability"),
            class_probabilities,
        )

        # Re-apply labels
        class_labels = list(lookup_table.values())
        # create a final labels array with same dimensions as new_labels
        final_labels = xr.full_like(new_labels, fill_value=float("nan"))
        for idx, label in enumerate(class_labels):
            final_labels.loc[{"bands": "classification"}] = xr.where(
                new_labels.sel(bands="classification") == idx,
                label,
                final_labels.sel(bands="classification"),
            )
        new_labels.sel(bands="classification").values = final_labels.sel(
            bands="classification"
        ).values

        # Append the per-class probabalities if required
        if parameters.get("keep_class_probs", False):
            new_labels = xr.concat([new_labels, class_probabilities], dim="bands")

    elif parameters.get("method") == "majority_vote":

        kernel_size = parameters.get("kernel_size", 5)

        new_labels = majority_vote(
            inarr.sel(bands="classification"),
            inarr.sel(bands="probability"),
            kernel_size=kernel_size,
        )

        # Append the per-class probabalities if required
        if parameters.get("keep_class_probs", False):
            class_probabilities = inarr.isel(bands=slice(2, None))
            new_labels = xr.concat([new_labels, class_probabilities], dim="bands")

    else:
        raise ValueError(
            f"Unknown post-processing method: {parameters.get('method')}"
        )

    return new_labels


# Below comes the actual UDF part

# Apply the Inference UDF
def apply_udf_data(udf_data: UdfData) -> UdfData:
    """This is the actual openeo UDF that will be executed by the backend."""

    cube = udf_data.datacube_list[0]
    parameters = copy.deepcopy(udf_data.user_context)

    arr = cube.get_array().transpose("bands", "y", "x")
    arr = execute(
        inarr=arr,
        parameters=parameters,
    ).transpose("bands", "y", "x")

    cube = XarrayDataCube(arr)

    udf_data.datacube_list = [cube]

    return udf_data

# Change band names, since the target labels are parameterized in the UDF
def apply_metadata(metadata: CollectionMetadata, context: dict) -> CollectionMetadata:

    lut_sorted = lut_from_url(context["classifier_url"])

    return metadata.rename_labels(dimension="bands", target=get_output_labels(lut_sorted, context))
