"""Feature computer GFMAP compatible to compute Presto embeddings."""

import numpy as np
import xarray as xr
from openeo_gfmap.features.feature_extractor import PatchFeatureExtractor
from pyproj import Transformer


class PrestoFeatureExtractor(PatchFeatureExtractor):
    """Feature extractor to use Presto model to compute embeddings.
    This will generate a datacube with 128 bands, each band representing a
    feature from the Presto model.
    """

    import functools
    from pathlib import Path
    from typing import Tuple

    CATBOOST_PATH = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/wc_catboost.onnx"  # NOQA
    PRESTO_PATH = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/presto.pt"  # NOQA
    BASE_URL = "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies"  # NOQA
    DEPENDENCY_NAME = "wc_presto_onnx_dependencies.zip"

    _NODATAVALUE = 65535

    BAND_MAPPING = {
        "B02": "B2",
        "B03": "B3",
        "B04": "B4",
        "B05": "B5",
        "B06": "B6",
        "B07": "B7",
        "B08": "B8",
        "B8A": "B8A",
        "B11": "B11",
        "B12": "B12",
        "VH": "VH",
        "VV": "VV",
        "precipitation-flux": "total_precipitation",
        "temperature-mean": "temperature_2m",
    }

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
        "A5-tmean": "temperature-mean",
        "A5-precip": "precipitation-flux",
    }

    def __init__(self):
        """
        Initializes the PrestoFeatureExtractor object, starting a logger.
        """
        import logging

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(PrestoFeatureExtractor.__name__)
        self.model = None  # To be initialized within the OpenEO environment

    @classmethod
    def _preprocess_band_values(
        cls, values: np.ndarray, presto_band: str
    ) -> np.ndarray:
        """
        Preprocesses the band values based on the given presto_val.

        Args:
            values (np.ndarray): Array of band values to preprocess.
            presto_val (str): Name of the band for preprocessing.

        Returns:
            np.ndarray: Preprocessed array of band values.
        """
        if presto_band in ["VV", "VH"]:
            # Convert to dB
            values = 20 * np.log10(values) - 83
        elif presto_band == "total_precipitation":
            # Scale precipitation and convert mm to m
            values = values / (100 * 1000.0)
        elif presto_band == "temperature_2m":
            # Remove scaling
            values = values / 100
        return values

    @classmethod
    def _extract_eo_data(cls, inarr: xr.DataArray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts EO data and mask arrays from the input xarray.DataArray.

        Args:
            inarr (xr.DataArray): Input xarray.DataArray containing EO data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing EO data array and mask array.
        """
        num_pixels = len(inarr.x) * len(inarr.y)
        num_timesteps = len(inarr.t)

        eo_data = np.zeros(
            (num_pixels, num_timesteps, len(BANDS))
        )  # pylint: disable=E0602
        mask = np.zeros(
            (num_pixels, num_timesteps, len(BANDS_GROUPS_IDX))
        )  # pylint: disable=E0602

        for org_band, presto_band in cls.BAND_MAPPING.items():
            if org_band in inarr.coords["bands"]:
                values = rearrange(  # pylint: disable=E0602
                    inarr.sel(bands=org_band).values, "t x y -> (x y) t"
                )
                idx_valid = values != cls._NODATAVALUE
                values = cls._preprocess_band_values(values, presto_band)
                eo_data[
                    :, :, BANDS.index(presto_band)
                ] = values  # pylint: disable=E0602
                mask[:, :, IDX_TO_BAND_GROUPS[presto_band]] += ~idx_valid

        return eo_data, mask

    @staticmethod
    def _extract_latlons(inarr: xr.DataArray, epsg: int) -> np.ndarray:
        """
        Extracts latitudes and longitudes from the input xarray.DataArray.

        Args:
            inarr (xr.DataArray): Input xarray.DataArray containing spatial coordinates.
            epsg (int): EPSG code for coordinate reference system.

        Returns:
            np.ndarray: Array containing extracted latitudes and longitudes.
        """
        # EPSG:4326 is the supported crs for presto
        lon, lat = np.meshgrid(inarr.x, inarr.y)
        transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(lon, lat)
        latlons = rearrange(
            np.stack([lat, lon]), "c x y -> (x y) c"
        )  # pylint: disable=E0602

        #  2D array where each row represents a pair of latitude and longitude coordinates.
        return latlons

    @staticmethod
    def _extract_months(inarr: xr.DataArray) -> np.ndarray:
        """
        Calculate the start month based on the first timestamp in the input array,
        and create an array of the same length filled with that start month value.

        Parameters:
        - inarr: xarray.DataArray or numpy.ndarray
            Input array containing timestamps.

        Returns:
        - months: numpy.ndarray
            Array of start month values, with the same length as the input array.
        """
        num_instances = len(inarr.x) * len(inarr.y)

        start_month = (
            inarr.t.values[0].astype("datetime64[M]").astype(int) % 12 + 1
        ) - 1

        months = np.ones((num_instances)) * start_month
        return months

    def _create_dataloader(
        self,
        eo: np.ndarray,
        dynamic_world: np.ndarray,
        months: np.ndarray,
        latlons: np.ndarray,
        mask: np.ndarray,
    ):
        """
        Create a PyTorch DataLoader for encoding features.

        Args:
            eo_data (np.ndarray): Array containing Earth Observation data.
            dynamic_world (np.ndarray): Array containing dynamic world data.
            latlons (np.ndarray): Array containing latitude and longitude coordinates.
            inarr (xr.DataArray): Input xarray.DataArray.
            mask (np.ndarray): Array containing masking data.

        Returns:
            DataLoader: PyTorch DataLoader for encoding features.
        """

        # pylint: disable=E0602
        dl = DataLoader(
            TensorDataset(
                torch.from_numpy(eo).float(),
                torch.from_numpy(dynamic_world).long(),
                torch.from_numpy(latlons).float(),
                torch.from_numpy(months).long(),
                torch.from_numpy(mask).float(),
            ),
            batch_size=8192,
            shuffle=False,
        )
        # pylint: enable=E0602

        return dl

    @classmethod
    def _create_presto_input(
        cls, inarr: xr.DataArray, epsg: int = 4326
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        eo_data, mask = cls._extract_eo_data(inarr)
        latlons = cls._extract_latlons(inarr, epsg)
        months = cls._extract_months(inarr)
        dynamic_world = np.ones((eo_data.shape[0], eo_data.shape[1])) * (
            DynamicWorld2020_2021.class_amount  # pylint: disable=E0602
        )

        return (
            S1_S2_ERA5_SRTM.normalize(eo_data),  # pylint: disable=E0602
            dynamic_world,
            months,
            latlons,
            np.repeat(mask, BAND_EXPANSION, axis=-1),  # pylint: disable=E0602
        )

    def _get_encodings(self, dl) -> np.ndarray:
        """
        Get encodings from DataLoader.

        Args:
            dl (DataLoader): PyTorch DataLoader containing data for encoding.

        Returns:
            np.ndarray: Array containing encoded features.
        """

        all_encodings = []

        for x, dw, latlons, month, variable_mask in dl:
            x_f, dw_f, latlons_f, month_f, variable_mask_f = [
                t.to(device)
                for t in (x, dw, latlons, month, variable_mask)  # pylint: disable=E0602
            ]

            with torch.no_grad():  # pylint: disable=E0602
                encodings = (
                    self.model.encoder(
                        x_f,
                        dynamic_world=dw_f.long(),
                        mask=variable_mask_f,
                        latlons=latlons_f,
                        month=month_f,
                    )
                    .cpu()
                    .numpy()
                )

            all_encodings.append(encodings)

        return np.concatenate(all_encodings, axis=0)

    def extract_presto_features(
        self, inarr: xr.DataArray, epsg: int = 4326
    ) -> np.ndarray:
        """General function to prepare the input data, generate a data loader,
        initialize the model, perform the inference and return the features.
        """
        eo, dynamic_world, months, latlons, mask = self._create_presto_input(
            inarr, epsg
        )
        dl = self._create_dataloader(eo, dynamic_world, months, latlons, mask)

        features = self._get_encodings(dl)
        features = rearrange(  # pylint: disable=E0602
            features, "(x y) c -> x y c", x=len(inarr.x), y=len(inarr.y)
        )
        ft_names = [f"presto_ft_{i}" for i in range(128)]
        features = xr.DataArray(
            features,
            coords={"x": inarr.x, "y": inarr.y, "bands": ft_names},
            dims=["x", "y", "bands"],
        )

        return features

    @classmethod
    @functools.lru_cache(maxsize=6)
    def extract_dependencies(cls, base_url: str, dependency_name: str):
        """Extract the dependencies from the given URL. Unpacking a zip
        file in the current working directory.
        """
        import shutil
        import sys
        import urllib.request
        from pathlib import Path

        # Generate absolute path for the dependencies folder
        dependencies_dir = Path.cwd() / "dependencies"

        # Create the directory if it doesn't exist
        dependencies_dir.mkdir(exist_ok=True, parents=True)

        # Download and extract the model file
        modelfile_url = f"{base_url}/{dependency_name}"
        modelfile, _ = urllib.request.urlretrieve(
            modelfile_url, filename=dependencies_dir / Path(modelfile_url).name
        )
        shutil.unpack_archive(modelfile, extract_dir=dependencies_dir)

        # Add the model directory to system path if it's not already there
        abs_path = str(
            dependencies_dir / Path(modelfile_url).name.split(".zip")[0]
        )  # NOQA

        # Append the dependencies
        sys.path.append(str(abs_path))
        sys.path.append(str(abs_path) + "/pandas")

    def get_presto_features(self, inarr: xr.DataArray, presto_path: str) -> np.ndarray:
        """
        Extracts features from input data using Presto.

        Args:
            inarr (xr.DataArray): Input data as xarray DataArray.
            presto_path (str): Path to the pretrained Presto model.

        Returns:
            xr.DataArray: Extracted features as xarray DataArray.
        """
        self.logger.info("Loading presto model.")
        presto_model = Presto.load_pretrained_artifactory(  # pylint: disable=E0602
            presto_url=presto_path, strict=False
        )
        self.model = presto_model
        self.logger.info("Presto model loaded sucessfully. Extracting features.")

        # Get the local EPSG code
        features = self.extract_presto_features(inarr, epsg=self.epsg)
        self.logger.info("Features extracted.")
        # features = self.extract_presto_features(inarr, epsg=32631)  # TODO remove hardcoded
        return features

    def output_labels(self) -> list:
        """Returns the output labels from this UDF, which is the output labels
        of the presto embeddings"""
        return [f"presto_ft_{i}" for i in range(128)]

    def execute(self, inarr: xr.DataArray) -> xr.DataArray:
        # The below is required to avoid flipping of the result
        # when running on OpenEO backend!
        inarr = inarr.transpose("bands", "t", "x", "y")

        # Change the band names
        new_band_names = [
            self.GFMAP_BAND_MAPPING.get(b.item(), b.item()) for b in inarr.bands
        ]
        inarr = inarr.assign_coords(bands=new_band_names)

        self.logger.info("Input data shape: %s", inarr.shape)
        for band in inarr.bands:
            self.logger.info(
                "Input data null values for band %s -> %s",
                band,
                inarr.sel(bands=band).isnull().sum().item(),
            )

        # Handle NaN values in Presto compatible way
        inarr = inarr.fillna(65535)

        self.logger.info(
            "After filling NaN values, total input data null values: %s",
            inarr.isnull().sum().item(),
        )

        # Unzip de dependencies on the backend
        self.logger.info("Unzipping dependencies")
        self.extract_dependencies(self.BASE_URL, self.DEPENDENCY_NAME)

        # pylint: disable=E0401
        # pylint: disable=C0401
        # pylint: disable=C0415
        # pylint: disable=W0601
        # pylint: disable=W0603
        # pylint: disable=reportMissingImports
        ##########################################################################
        global requests, torch, BANDS, BANDS_GROUPS_IDX, NORMED_BANDS
        global S1_S2_ERA5_SRTM, DynamicWorld2020_2021, BAND_EXPANSION
        global IDX_TO_BAND_GROUPS, BAND_EXPANSION, Presto, device, rearrange
        global DataLoader, TensorDataset

        import requests
        import torch
        from dependencies.wc_presto_onnx_dependencies.mvp_wc_presto.dataops import (
            BANDS,
            BANDS_GROUPS_IDX,
            NORMED_BANDS,
            S1_S2_ERA5_SRTM,
            DynamicWorld2020_2021,
        )
        from dependencies.wc_presto_onnx_dependencies.mvp_wc_presto.masking import (
            BAND_EXPANSION,
        )
        from dependencies.wc_presto_onnx_dependencies.mvp_wc_presto.presto import Presto
        from dependencies.wc_presto_onnx_dependencies.mvp_wc_presto.utils import device
        from einops import rearrange
        from torch.utils.data import DataLoader, TensorDataset

        ##########################################################################
        # pylint: enable=E0401
        # pylint: enable=C0401
        # pylint: enable=C0415
        # pylint: enable=W0601
        # pylint: enable=W0603
        # pylint: enable=reportMissingImports
        # Index to band groups mapping
        IDX_TO_BAND_GROUPS = {
            NORMED_BANDS[idx]: band_group_idx
            for band_group_idx, (_, val) in enumerate(BANDS_GROUPS_IDX.items())
            for idx in val
        }

        self.logger.info("Extracting presto features")
        features = self.get_presto_features(inarr, self.PRESTO_PATH)
        return features
