import functools
import logging
import shutil
import sys
import urllib.request
from pathlib import Path
from functools import lru_cache


def _setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

def extract_dependencies(base_url: str, dependency_name: str):

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
    abs_path = str(dependencies_dir / Path(modelfile_url).name.split(".zip")[0])  # NOQA

    # Append the dependencies
    sys.path.append(str(abs_path))

BASE_URL = "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies"  # NOQA
DEPENDENCY_NAME = "wc_presto_onnx_dependencies.zip"
extract_dependencies(BASE_URL, DEPENDENCY_NAME)


import onnxruntime
import requests
from typing import Dict, Tuple
import numpy as np
import xarray as xr
from pyproj import Transformer
from einops import rearrange
import torch
from torch.utils.data import DataLoader, TensorDataset

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

class CatBoostPredictor:
    
    #singleton
    _instance = None

    def __new__(cls, model_url: str):
        if cls._instance is None:
            cls._instance = super(CatBoostPredictor, cls).__new__(cls)
            cls._instance.model_url = model_url
            cls._instance.onnx_session = cls._load_model_from_url(model_url)
        return cls._instance

    @staticmethod
    @lru_cache(maxsize=None)
    def _load_model_from_url(model_url: str):
        """
        Load the ONNX model from the specified URL.
        """
        response = requests.get(model_url)
        model_content = response.content

        # Load the dependency into an InferenceSession
        onnx_session = onnxruntime.InferenceSession(model_content)
        return onnx_session

    def _inference(self, features: np.ndarray) -> np.ndarray:
        """
        Predict labels using the provided features array.

        Args:
            features (np.ndarray): 2D array containing the features.

        Returns:
            np.ndarray: Array containing the predicted binary labels.
        """
        if self.onnx_session is None:
            raise ValueError("Model has not been loaded. Please load a model first.")

        # Prepare input data for ONNX model
        outputs = self.onnx_session.run(None, {"features": features})

        # Threshold for binary conversion
        threshold = 0.5

        # Extract all prediction values and convert them to binary labels
        prediction_values = [sublist["True"] for sublist in outputs[1]]
        binary_labels = np.array(prediction_values) >= threshold
        binary_labels = binary_labels.astype(int)

        return binary_labels

    def classify(self, features: xr.DataArray) -> xr.DataArray:
        """
        Classifies features using the WorldCereal ONNX model.

        Args:
            features (xr.DataArray): Features to be classified [x, y, fts].

        Returns:
            xr.DataArray: Classified data as xarray DataArray.
        """
        # Stack the features and transpose for feeding to ONNX model
        stacked_features = features.stack(xy=["x", "y"]).transpose()

        # Predict the labels
        predictions = self._inference(stacked_features.values)

        # Convert predictions to xarray DataArray
        predictions = (
            xr.DataArray(predictions, coords={"xy": stacked_features.xy}, dims=["xy"])
            .unstack()
            .expand_dims(dim="bands")
        )

        return predictions

class PrestoEmbeddingExtractor:
        
        #singleton
        _instance = None

        def __new__(cls, model_url: str):
            if cls._instance is None:
                cls._instance = super(PrestoEmbeddingExtractor, cls).__new__(cls)
                cls._instance.model_url = model_url
                cls._instance.model = cls._load_model_with_cache(model_url)
            return cls._instance

        @classmethod
        def _load_model_with_cache(cls, model_url: str):
            """
            Load the Presto model with caching to avoid reloading the same model multiple times.

            Args:
                model_url (str): The URL to the Presto model.

            Returns:
                Presto: The loaded Presto model.
            """
            if not hasattr(cls, '_model_cache'):
                cls._model_cache = {}
            if model_url not in cls._model_cache:
                cls._model_cache[model_url] = Presto.load_pretrained_artifactory(presto_url=model_url, strict=False)
            return cls._model_cache[model_url]
            
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

        # Index to band groups mapping
        IDX_TO_BAND_GROUPS = {
            NORMED_BANDS[idx]: band_group_idx
            for band_group_idx, (_, val) in enumerate(BANDS_GROUPS_IDX.items())
            for idx in val
        }

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

            eo_data = np.zeros((num_pixels, num_timesteps, len(BANDS)))
            mask = np.zeros((num_pixels, num_timesteps, len(BANDS_GROUPS_IDX)))

            for org_band, presto_band in cls.BAND_MAPPING.items():
                if org_band in inarr.coords["bands"]:
                    values = rearrange(
                        inarr.sel(bands=org_band).values, "t x y -> (x y) t"
                    )
                    idx_valid = values != cls._NODATAVALUE
                    values = cls._preprocess_band_values(values, presto_band)
                    eo_data[:, :, BANDS.index(presto_band)] = values
                    mask[:, :, cls.IDX_TO_BAND_GROUPS[presto_band]] += ~idx_valid

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
            transformer = Transformer.from_crs(
                f"EPSG:{epsg}", "EPSG:4326", always_xy=True
            )
            lon, lat = transformer.transform(lon, lat)
            latlons = rearrange(np.stack([lat, lon]), "c x y -> (x y) c")

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
        
        @classmethod
        def _create_dataloader(
            self,
            eo: np.ndarray,
            dynamic_world: np.ndarray,
            months: np.ndarray,
            latlons: np.ndarray,
            mask: np.ndarray,
        ) -> DataLoader:
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

            return dl

        def _create_presto_input(
            cls, inarr: xr.DataArray, epsg: int = 4326
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            eo_data, mask = cls._extract_eo_data(inarr)
            latlons = cls._extract_latlons(inarr, epsg)
            months = cls._extract_months(inarr)
            dynamic_world = np.ones((eo_data.shape[0], eo_data.shape[1])) * (
                DynamicWorld2020_2021.class_amount
            )

            return (
                S1_S2_ERA5_SRTM.normalize(eo_data),
                dynamic_world,
                months,
                latlons,
                np.repeat(mask, BAND_EXPANSION, axis=-1),
            )

        def _get_encodings(self, dl: DataLoader) -> np.ndarray:
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
                    t.to(device) for t in (x, dw, latlons, month, variable_mask)
                ]

                with torch.no_grad():
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

        def get_presto_embedding(
            self, inarr: xr.DataArray, epsg: int = 4326
        ) -> xr.DataArray:
            eo, dynamic_world, months, latlons, mask = self._create_presto_input(
                inarr, epsg
            )
            dl = self._create_dataloader(eo, dynamic_world, months, latlons, mask)

            features = self._get_encodings(dl)
            features = rearrange(
                features, "(x y) c -> x y c", x=len(inarr.x), y=len(inarr.y)
            )
            ft_names = [f"presto_ft_{i}" for i in range(128)]
            features = xr.DataArray(
                features,
                coords={"x": inarr.x, "y": inarr.y, "bands": ft_names},
                dims=["x", "y", "bands"],
            )

            return features

logger = _setup_logging()

def apply_datacube(cube: xr.DataArray, context: Dict) -> xr.DataArray:

    CATBOOST_PATH = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/wc_catboost.onnx"  # NOQA
    PRESTO_PATH = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/presto.pt"  # NOQA

    # The below is required to avoid flipping of the result
    # when running on OpenEO backend!
    cube = cube.transpose("bands", "t", "x", "y")
    # Handle NaN values in Presto compatible way
    cube = cube.fillna(65535)

    # Run presto feature extraction
    logger.info("Extracting presto features")
    presto_embeddor = PrestoEmbeddingExtractor(PRESTO_PATH)
    features = presto_embeddor.get_presto_embedding(cube, 32631)

    # Run catboost classification
    logger.info("Catboost classification")
    catboost_predictor = CatBoostPredictor(CATBOOST_PATH)

    # Assuming `features` is your xr.DataArray with the required data
    classified_data = catboost_predictor.classify(features)

    # Add time dimension
    classification = classified_data.expand_dims(dim="t")

    return classification
