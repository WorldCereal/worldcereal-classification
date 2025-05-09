"""Contants for the WorldCereal STAC products."""

from enum import Enum

import pystac


class ExtractionCollection(Enum):
    """Collections that can be extracted in the extraction scripts."""

    PATCH_SENTINEL1 = "PATCH_SENTINEL1"
    PATCH_SENTINEL2 = "PATCH_SENTINEL2"
    PATCH_METEO = "PATCH_METEO"
    PATCH_WORLDCEREAL = "PATCH_WORLDCEREAL"
    POINT_WORLDCEREAL = "POINT_WORLDCEREAL"
    PATCH_TO_POINT = "PATCH_TO_POINT"


# Define the sentinel 1 asset
SENTINEL1_ASSET = pystac.extensions.item_assets.AssetDefinition(
    {
        "gsd": 20,
        "title": "Sentinel1",
        "description": "Sentinel-1 bands",
        "type": "application/x-netcdf",
        "roles": ["data"],
        "proj:shape": [32, 32],
        "raster:bands": [
            {"name": "S1-SIGMA0-VV"},
            {
                "name": "S1-SIGMA0-VH",
            },
        ],
        "cube:variables": {
            "S1-SIGMA0-VV": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S1-SIGMA0-VH": {"dimensions": ["time", "y", "x"], "type": "data"},
        },
        "eo:bands": [
            {
                "name": "S1-SIGMA0-VV",
                "common_name": "VV",
            },
            {
                "name": "S1-SIGMA0-VH",
                "common_name": "VH",
            },
        ],
    }
)


# Define the sentinel 2 asset
SENTINEL2_ASSET = pystac.extensions.item_assets.AssetDefinition(
    {
        "gsd": 10,
        "title": "Sentinel2",
        "description": "Sentinel-2 bands",
        "type": "application/x-netcdf",
        "roles": ["data"],
        "proj:shape": [64, 64],
        "raster:bands": [
            {"name": "S2-L2A-B01"},
            {"name": "S2-L2A-B02"},
            {"name": "S2-L2A-B03"},
            {"name": "S2-L2A-B04"},
            {"name": "S2-L2A-B05"},
            {"name": "S2-L2A-B06"},
            {"name": "S2-L2A-B07"},
            {"name": "S2-L2A-B08"},
            {"name": "S2-L2A-B8A"},
            {"name": "S2-L2A-B09"},
            {"name": "S2-L2A-B11"},
            {"name": "S2-L2A-B12"},
            {"name": "S2-L2A-SCL"},
            {"name": "S2-L2A-SCL_DILATED_MASK"},
            {"name": "S2-L2A-DISTANCE-TO-CLOUD"},
        ],
        "cube:variables": {
            "S2-L2A-B01": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S2-L2A-B02": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S2-L2A-B03": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S2-L2A-B04": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S2-L2A-B05": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S2-L2A-B06": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S2-L2A-B07": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S2-L2A-B08": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S2-L2A-B8A": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S2-L2A-B09": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S2-L2A-B11": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S2-L2A-B12": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S2-L2A-SCL": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S2-L2A-SCL_DILATED_MASK": {
                "dimensions": ["time", "y", "x"],
                "type": "data",
            },
            "S2-L2A-DISTANCE-TO-CLOUD": {
                "dimensions": ["time", "y", "x"],
                "type": "data",
            },
        },
        "eo:bands": [
            {
                "name": "S2-L2A-B01",
                "common_name": "coastal",
                "center_wavelength": 0.443,
                "full_width_half_max": 0.027,
            },
            {
                "name": "S2-L2A-B02",
                "common_name": "blue",
                "center_wavelength": 0.49,
                "full_width_half_max": 0.098,
            },
            {
                "name": "S2-L2A-B03",
                "common_name": "green",
                "center_wavelength": 0.56,
                "full_width_half_max": 0.045,
            },
            {
                "name": "S2-L2A-B04",
                "common_name": "red",
                "center_wavelength": 0.665,
                "full_width_half_max": 0.038,
            },
            {
                "name": "S2-L2A-B05",
                "common_name": "rededge",
                "center_wavelength": 0.704,
                "full_width_half_max": 0.019,
            },
            {
                "name": "S2-L2A-B06",
                "common_name": "rededge",
                "center_wavelength": 0.74,
                "full_width_half_max": 0.018,
            },
            {
                "name": "S2-L2A-B07",
                "common_name": "rededge",
                "center_wavelength": 0.783,
                "full_width_half_max": 0.028,
            },
            {
                "name": "S2-L2A-B08",
                "common_name": "nir",
                "center_wavelength": 0.842,
                "full_width_half_max": 0.145,
            },
            {
                "name": "S2-L2A-B8A",
                "common_name": "nir08",
                "center_wavelength": 0.865,
                "full_width_half_max": 0.033,
            },
            {
                "name": "S2-L2A-B09",
                "common_name": "nir09",
                "center_wavelength": 0.945,
                "full_width_half_max": 0.033,
            },
            {
                "name": "S2-L2A-B11",
                "common_name": "swir16",
                "center_wavelength": 1.61,
                "full_width_half_max": 0.143,
            },
            {
                "name": "S2-L2A-B12",
                "common_name": "swir16",
                "center_wavelength": 1.61,
                "full_width_half_max": 0.143,
            },
            {
                "name": "S2-L2A-SCL",
                "common_name": "swir16",
                "center_wavelength": 1.61,
                "full_width_half_max": 0.143,
            },
            {
                "name": "S2-L2A-SCL_DILATED_MASK",
            },
            {
                "name": "S2-L2A-DISTANCE-TO-CLOUD",
            },
        ],
    }
)

METEO_ASSET = pystac.extensions.item_assets.AssetDefinition({})


COLLECTION_IDS = {
    ExtractionCollection.PATCH_SENTINEL1: "SENTINEL1-EXTRACTION",
    ExtractionCollection.PATCH_SENTINEL2: "sentinel2-EXTRACTION",
    ExtractionCollection.PATCH_METEO: "METEO-EXTRACTION",
    ExtractionCollection.PATCH_WORLDCEREAL: "WORLDCEREAL-INPUTS",
}

COLLECTION_DESCRIPTIONS = {
    ExtractionCollection.PATCH_SENTINEL1: "Sentinel1 GRD data extraction.",
    ExtractionCollection.PATCH_SENTINEL2: "Sentinel2 L2A data extraction.",
    ExtractionCollection.PATCH_METEO: "Meteo data extraction.",
    ExtractionCollection.PATCH_WORLDCEREAL: "WorldCereal preprocessed inputs extraction.",
}

CONSTELLATION_NAMES = {
    ExtractionCollection.PATCH_SENTINEL1: "sentinel1",
    ExtractionCollection.PATCH_SENTINEL2: "sentinel2",
    ExtractionCollection.PATCH_METEO: "agera5",
    ExtractionCollection.PATCH_WORLDCEREAL: "worldcereal",
}

ITEM_ASSETS = {
    ExtractionCollection.PATCH_SENTINEL1: {"sentinel1": SENTINEL1_ASSET},
    ExtractionCollection.PATCH_SENTINEL2: {"sentinel2": SENTINEL2_ASSET},
    ExtractionCollection.PATCH_METEO: {"agera5": METEO_ASSET},
    ExtractionCollection.PATCH_WORLDCEREAL: None,
}

COLLECTION_REGEXES = {
    ExtractionCollection.PATCH_SENTINEL1: r"^S1-SIGMA0-10m_(?:ASCENDING|DESCENDING)_(.*)_[0-9]{4,5}_([0-9]{4}-[0-9]{2}-[0-9]{2}(?:_)?){2}.nc$",
    ExtractionCollection.PATCH_SENTINEL2: r"^S2-L2A-10m_(.*)_[0-9]{4,5}_([0-9]{4}-[0-9]{2}-[0-9]{2}(?:_)?){2}.nc$",
}
