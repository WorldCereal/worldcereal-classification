"""Feature computer GFMAP compatible to compute Presto embeddings."""

# /// script
# dependencies = [
#   "presto @ https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/dependencies/presto_worldcereal-0.0.1-py3-none-any.whl"
# ]
# ///
import xarray as xr
from openeo.udf import XarrayDataCube
from openeo_gfmap.features.feature_extractor import PatchFeatureExtractor


class PrestoFeatureExtractor(PatchFeatureExtractor):
    """Feature extractor to use Presto model to compute per-pixel embeddings.
    This will generate a datacube with 128 bands, each band representing a
    feature from the Presto model.

    Interesting UDF parameters:
    - presto_url: A public URL to the Presto model file. A default Presto
      version is provided if the parameter is left undefined.
    - rescale_s1: Is specifically disabled by default, as the presto
      dependencies already take care of the backscatter decompression. If
      specified, should be set as `False`.
    """

    PRESTO_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/presto.pt"  # NOQA
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
        "AGERA5-TMEAN": "temperature-mean",
        "AGERA5-PRECIP": "precipitation-flux",
    }

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
        presto_url = self._parameters.get("presto_url", self.PRESTO_URL)

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

        from presto.inference import get_presto_features

        self.logger.info("Extracting presto features")
        features = get_presto_features(inarr, presto_url, self.epsg)
        return features

    def _execute(self, cube: XarrayDataCube, parameters: dict) -> XarrayDataCube:
        # Disable S1 rescaling (decompression) by default
        if parameters.get("rescale_s1", None) is None:
            parameters.update({"rescale_s1": False})
        return super()._execute(cube, parameters)
