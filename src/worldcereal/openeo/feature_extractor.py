"""Feature computer GFMAP compatible to compute Presto embeddings."""

import xarray as xr
from openeo.udf import XarrayDataCube
from openeo.udf.debug import inspect
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

    import functools

    PRESTO_MODEL_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/presto.pt"  # NOQA
    PRESTO_WHL_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/dependencies/presto_worldcereal-0.1.3-py3-none-any.whl"
    BASE_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/dependencies/"  # NOQA
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
        "COP-DEM": "elevation",
        "AGERA5-TMEAN": "temperature-mean",
        "AGERA5-PRECIP": "precipitation-flux",
    }

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

    def output_labels(self) -> list:
        """Returns the output labels from this UDF, which is the output labels
        of the presto embeddings"""
        return [f"presto_ft_{i}" for i in range(128)]

    def _compute_slope(self, inarr: xr.DataArray) -> xr.DataArray:
        """Computes the slope using the richdem library. The input array should
        have the following bands: 'elevation' And no time dimension. Returns a
        new DataArray containing the new `slope` band.
        """
        from richdem import (  # pylint: disable=import-outside-toplevel
            TerrainAttribute,
            rdarray,
        )
        
        dem = inarr.sel(bands="elevation").values
        dem_array = rdarray(dem, no_data=65535)
        slope = TerrainAttribute(dem_array, attrib="slope_riserun")
        return xr.DataArray(
            slope[None, :, :],
            dims=("bands", "x", "y"),
            coords={"bands": ["slope"], "x": inarr.x, "y": inarr.y},
        )

    def execute(self, inarr: xr.DataArray) -> xr.DataArray:
        import sys

        if self.epsg is None:
            raise ValueError(
                "EPSG code is required for Presto feature extraction, but was "
                "not correctly initialized."
            )
        presto_model_url = self._parameters.get(
            "presto_model_url", self.PRESTO_MODEL_URL
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

        from presto.inference import (  # pylint: disable=import-outside-toplevel
            get_presto_features,
        )

        slope = self._compute_slope(inarr.isel(t=0))
        slope = slope.expand_dims({"t": inarr.t}, axis=0).astype("float32")

        inarr = xr.concat([inarr.astype("float32"), slope], dim="bands")
        inspect(inarr.rio, "Rioxarray accesor: ")
        inspect(inarr.rio.bounds, "Rioxarray bounds: ")

        batch_size = self._parameters.get("batch_size", 256)

        self.logger.info("Extracting presto features")
        features = get_presto_features(
            inarr, presto_model_url, self.epsg, batch_size=batch_size
        )
        return features

    def _execute(self, cube: XarrayDataCube, parameters: dict) -> XarrayDataCube:
        # Disable S1 rescaling (decompression) by default
        if parameters.get("rescale_s1", None) is None:
            parameters.update({"rescale_s1": False})
        return super()._execute(cube, parameters)
