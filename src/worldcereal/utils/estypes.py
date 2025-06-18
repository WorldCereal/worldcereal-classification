"""
This module defines the base and specific types used to represent WorldCereal products
that are stored in an Elasticsearch index. It supports dynamic subclass resolution
based on the product type field and includes utilities for fetching and updating these records.
@author: Dieter Wens (dieter.wens@vito.be)
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Type, Union
from elasticsearch import Elasticsearch, NotFoundError

from shapely.geometry.base import BaseGeometry
from shapely.geometry import shape
from shapely.geometry.geo import mapping
from shapely.ops import unary_union

import geopandas as gpd


class BaseWorldCerealType(ABC):
    DEFAULT_ES_ENDPOINT = "https://es-apps-dev.vgt.vito.be/"
    DEFAULT_INDEX = "worldcereal"
    _registry: Dict[str, Type["BaseWorldCerealType"]] = {}

    def __init__(self,
                 _id: str,
                 creation_date: datetime,
                 modified_date: datetime,
                 version: str,
                 geospatial_coverage: BaseGeometry,
                 pipelines_status: Dict[str, Any],
                 properties: Optional[Dict[str, Any]] = None):

        self._id = _id
        self.creation_date = creation_date
        self.modified_date = modified_date
        self.version = version
        self._geospatial_coverage = geospatial_coverage
        self.pipelines_status = pipelines_status
        self.properties = properties or {}

    @abstractmethod
    @property
    def product_type(self) -> str:
        pass

    @classmethod
    def get_by_id(cls, _id: str) -> "BaseWorldCerealType":
        es = cls.get_es_instance()
        index = cls.get_index()

        try:
            doc = es.get(index=index, id=_id)
            src = doc["_source"]
            product_type = src["product_type"]

            if product_type not in cls._registry:
                raise KeyError(f'Could not find product type "{product_type}" in registry.')

            subclass = cls._registry[product_type]

            return subclass(
                _id=_id,
                creation_date=datetime.fromisoformat(src["creation_date"]),
                modified_date=datetime.fromisoformat(src["modified_date"]),
                version=src["version"],
                geospatial_coverage=shape(src["geospatial_coverage"]),
                pipelines_status=src.get("pipelines_status", {}),
                properties=src.get("properties", {})
            )
        except NotFoundError as e:
            return None

    @classmethod
    def get_by_type(cls, product_type) -> List["BaseWorldCerealType"]:
        """
            TODO: get all products of a certain type.
        """
        pass

    def update(self):
        """
        Create or overwrite the document in Elasticsearch.
        """
        es = self.get_es_instance()
        index = self.get_index()

        # Assemble full document body
        doc = {
            "product_type": self.product_type,
            "creation_date": self.creation_date.isoformat(),
            "modified_date": datetime.now().isoformat(),
            "geospatial_coverage": mapping(self.geospatial_coverage),
            "pipelines_status": self.pipelines_status,
            "properties": self.properties
        }

        # Index the document (create or replace)
        es.index(index=index, id=self._id, body=doc)

    def delete(self):
        """
        TODO: Delete the doc on elasticsearch
        """

    def set_pipeline_status(self, pipeline: str, status: str):
        """
        TODO: set processing status of a pipeline
        TODO: make Enum for statuses
        """

    @staticmethod
    def get_es_instance():
        """
            Get elasticsearch instance

            TODO:
                Make configurable via env vars.
        """
        es_endpoint = BaseWorldCerealType.DEFAULT_ES_ENDPOINT
        es = Elasticsearch(es_endpoint)
        return es

    @staticmethod
    def get_index():
        """
            Get elasticsearch index

            TODO:
                Make configurable via env vars.
        """

        return BaseWorldCerealType.DEFAULT_INDEX

    @classmethod
    def register_subclass(cls, product_type: str):
        def decorator(subclass):
            cls._registry[product_type] = subclass
            return subclass
        return decorator

    @property
    def geospatial_coverage(self) -> BaseGeometry:
        return self._geospatial_coverage

    @geospatial_coverage.setter
    def geospatial_coverage(self, geospatial_coverage: BaseGeometry):
        if isinstance(geospatial_coverage, BaseGeometry):
            self._geospatial_coverage = geospatial_coverage
        elif isinstance(geospatial_coverage, dict):
            self._geospatial_coverage = shape(geospatial_coverage)


@BaseWorldCerealType.register_subclass("WORDCEREAL_TRAINING_DATA")
class WordCerealTrainingData(BaseWorldCerealType):
    def __init__(self,
                 _id: str,
                 creation_date: datetime,
                 modified_date: datetime,
                 version: str,
                 geospatial_coverage: BaseGeometry,
                 pipelines_status: Dict[str, Any],
                 properties: Optional[Dict[str, Any]] = None):

        super().__init__(
            _id=_id,
            creation_date=creation_date,
            modified_date=modified_date,
            version=version,
            geospatial_coverage=geospatial_coverage,
            pipelines_status=pipelines_status,
            properties=properties
        )

    @property
    def product_type(self) -> str:
        return "WORDCEREAL_TRAINING_DATA"

    @property
    def filepath(self) -> str:
        return Path(self.properties["filepath"])

    @filepath.setter
    def filepath(self, filepath: Union[str, Path]):
        self.properties["filepath"] = str(filepath)

    @property
    def gp_entries(self) -> Dict[str, Any]:
        return self.properties["gp_entries"]

    @gp_entries.setter
    def gp_entries(self, entries: Dict[str, Any]):
        self.properties["gp_entries"] = entries

    @classmethod
    def from_geoparquet(cls, geoparquet_file: Union[str, Path], version: str, allow_update=False):
        """
        Create an instance from a GeoParquet file.

        Parameters
        ----------
        geoparquet_file : str or Path
            Path to the input GeoParquet file.
        version : str
            Version string used for generating the object ID.
        allow_update : bool, optional
            If True, allows overwriting an existing instance with the same ID. Default is False.

        Returns
        -------
        cls
            An instance of the class initialized from the GeoParquet file.

        Raises
        ------
        ValueError
            If an instance with the generated ID already exists and `allow_update` is False.
        """

        geoparquet_file = Path(geoparquet_file)
        _id = f"TD_{geoparquet_file.stem}_{version}"

        if cls.get_by_id(_id) is not None and not allow_update:
            raise ValueError(f'ID "{_id}" already exists. Enable '
                             f'allow_update to overwrite with new GeoParquet data.')

        gdf = gpd.read_parquet(geoparquet_file)

        # Combine geometry
        geoms = gdf.geometry.values
        combined: BaseGeometry = mapping(unary_union(geoms))

        properties = dict()
        properties['filepath'] = str(geoparquet_file)
        properties['gp_entries'] = gdf.to_dict(orient="records")

        return cls(
            _id=_id,
            creation_date=datetime.now(),
            modified_date=datetime.now(),
            version=version,
            geospatial_coverage=combined,
            pipelines_status={},
            properties=properties
        )













