import xarray as xr
from openeo_gfmap.inference.model_inference import ModelInference


class PostProcessor(ModelInference):
    """Perform post-processing from the model outputs. Expects an input cube
    with 2 + N bands, where N is the number of classes. The first band is the
    prediction value, the second class is the max probability and the rest are
    the per-class probabilities.

    Interesting UDF parameters:
    is_binary: bool
        If the postprocessing is applied on a binary classification model (cropland)
        or a multi-class classification model (croptype). Default is False.
    lookup_table: Optional[dict]
        Required if is_binary is False. A lookup table to map the class names
        to class labels, ordered by model output.
    """

    EXCLUDED_VALUES = [254, 255, 65535]

    def output_labels(self) -> list:
        if self._parameters.get("keep_class_probs", False):
            return ["classification", "probability"] + [
                f"probability_{name}"
                for name in self._parameters["lookup_table"].keys()
            ]
        return ["classification", "probability"]

    def dependencies(self) -> list:
        return []

    @classmethod
    def smooth_probabilities(
        cls, base_labels: xr.DataArray, class_probabilities: xr.DataArray
    ) -> xr.DataArray:
        """Performs gaussian smoothing on the class probabilities. Requires the
        base labels to keep the pixels that are excluded away from smoothing.
        """
        import numpy as np
        from scipy.signal import convolve2d

        base_labels_vals = base_labels.values
        probabilities_vals = class_probabilities.values

        excluded_mask = np.in1d(
            base_labels_vals.reshape(-1),
            cls.EXCLUDED_VALUES,
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

        return xr.DataArray(
            probabilities_vals,
            coords=class_probabilities.coords,
            dims=class_probabilities.dims,
        )

    @classmethod
    def reclassify(
        cls,
        base_labels: xr.DataArray,
        base_max_probs: xr.DataArray,
        probabilities: xr.DataArray,
    ) -> xr.DataArray:
        import numpy as np

        base_labels_vals = base_labels.values
        base_max_probs_vals = base_max_probs.values

        excluded_mask = np.in1d(
            base_labels_vals.reshape(-1),
            cls.EXCLUDED_VALUES,
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

    def execute(self, inarr: xr.DataArray) -> xr.DataArray:
        # Make some checks on the bands
        if self._parameters.get("is_binary", False):
            # Cast to float for more accurate gaussian smoothing
            probability = inarr.sel(bands="probability").astype("float32") / 100.0
            classification = inarr.sel(bands="classification").astype("uint8")

            # Get the per-class probabilities from the max probability and the
            # classification result.
            true_prob = xr.where(classification > 0, probability, 1.0 - probability)
            false_prob = 1.0 - true_prob

            class_probabilities = xr.concat(
                [false_prob, true_prob], dim="bands"
            ).assign_coords(bands=["probability_False", "probability_True"])

        else:
            # Cast to float for more accurate gaussian smoothing
            class_probabilities = (
                inarr.isel(bands=slice(2, None)).astype("float32") / 100.0
            )

        # Peform probability smoothing
        class_probabilities = PostProcessor.smooth_probabilities(
            inarr.sel(bands="classification"), class_probabilities
        )

        # Cast back to uint16, sum of probabilities should be 100
        class_probabilities = (class_probabilities * 100.0).round().astype("uint16")

        # Reclassify
        new_labels = PostProcessor.reclassify(
            inarr.sel(bands="classification"),
            inarr.sel(bands="probability"),
            class_probabilities,
        )

        # Re-apply labels
        if not self._parameters.get("is_binary", True):
            lookup_table = self._parameters.get("lookup_table")
            class_labels = list(lookup_table.values())
            for idx, label in enumerate(class_labels):
                new_labels.loc[{"bands": "classification"}] = xr.where(
                    new_labels.sel(bands="classification") == idx,
                    label,
                    new_labels.sel(bands="classification"),
                )

        # Append the per-class probabalities if required
        if self._parameters.get("keep_class_probs", False):
            new_labels = xr.concat([new_labels, class_probabilities], dim="bands")

        return new_labels
