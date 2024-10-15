import xarray as xr
from openeo_gfmap.inference.model_inference import ModelInference


class PostProcessor(ModelInference):
    """Perform post-processing from the model outputs. Expects an input cube
    with 2 + N bands, where N is the number of classes. The first band is the
    prediction value, the second class is the max probability and the rest are
    the per-class probabilities.

    Interesting UDF parameters:
    lookup_table: Optional[dict]
        A lookup table to map the class names to class labels, ordered by model output.
    """

    EXCLUDED_VALUES = [254, 255, 65535]
    NODATA = 255

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
    def majority_vote(
        cls,
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

        import numpy as np
        from scipy.signal import convolve2d

        prediction = base_labels.values
        probability = max_probabilities.values

        # As the probabilities are in integers between 0 and 100,
        # we use uint16 matrices to store the vote scores
        assert (
            kernel_size <= 25
        ), f"Kernel value cannot be larger than 25 (currently: {kernel_size}) because it might lead to scenarios where the 16-bit count matrix is overflown"

        # Build a class mapping, so classes are converted to indexes and vice-versa
        unique_values = set(np.unique(prediction))
        unique_values = sorted(unique_values - set(cls.EXCLUDED_VALUES))  # type: ignore
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
            for excluded_value in cls.EXCLUDED_VALUES:
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

            # Get the new confidence score for the indices
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

            aggregated_predictions[no_score_mask] = cls.NODATA
            aggregated_probabilities[no_score_mask] = cls.NODATA

        # Setting excluded values back to their original values
        for excluded_value in cls.EXCLUDED_VALUES:
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

        # Sum of probabilities should be 1, cast to uint16
        probabilities_vals = np.round(
            probabilities_vals / probabilities_vals.sum(axis=0) * 100.0
        ).astype("uint16")

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

        if self._parameters.get("method") == "smooth_probabilities":
            # Cast to float for more accurate gaussian smoothing
            class_probabilities = (
                inarr.isel(bands=slice(2, None)).astype("float32") / 100.0
            )

            # Peform probability smoothing
            class_probabilities = PostProcessor.smooth_probabilities(
                inarr.sel(bands="classification"), class_probabilities
            )

            # Reclassify
            new_labels = PostProcessor.reclassify(
                inarr.sel(bands="classification"),
                inarr.sel(bands="probability"),
                class_probabilities,
            )

            # Re-apply labels
            lookup_table = self._parameters.get("lookup_table")
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
            if self._parameters.get("keep_class_probs", False):
                new_labels = xr.concat([new_labels, class_probabilities], dim="bands")

        elif self._parameters.get("method") == "majority_vote":

            kernel_size = self._parameters.get("kernel_size")

            new_labels = PostProcessor.majority_vote(
                inarr.sel(bands="classification"),
                inarr.sel(bands="probability"),
                kernel_size=kernel_size,
            )

            # Append the per-class probabalities if required
            if self._parameters.get("keep_class_probs", False):
                class_probabilities = inarr.isel(bands=slice(2, None))
                new_labels = xr.concat([new_labels, class_probabilities], dim="bands")

        else:
            raise ValueError(
                f"Unknown post-processing method: {self._parameters.get('method')}"
            )

        return new_labels
