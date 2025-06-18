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


class BaseWorldCerealType(ABC):
    DEFAULT_ES_ENDPOINT = "https://es-apps-dev.vgt.vito.be/"
    DEFAULT_INDEX = "worldcereal"
    _registry: Dict[str, Type["BaseWorldCerealType"]] = {}

    def __init__(self,
                 id: str,
                 product_type: str,
                 upload_date: datetime,
                 geospatial_coverage: BaseGeometry,
                 pipelines_status: Dict[str, Any],
                 properties: Optional[Dict[str, Any]] = None):

        self.id = id
        self._product_type = product_type
        self.upload_date = upload_date
        self.geospatial_coverage = geospatial_coverage
        self.pipelines_status = pipelines_status
        self.properties = properties or {}

    @property
    def product_type(self) -> str:
        return self._product_type

    @classmethod
    def get_by_id(cls, id: str) -> "BaseWorldCerealType":
        es = cls.get_es_instance()
        index = cls.get_index()

        try:
            doc = es.get(index=index, id=id)
            src = doc["_source"]
            product_type = src["product_type"]

            if product_type not in cls._registry:
                raise KeyError(f'Could not find product type "{product_type}" in registry.')

            subclass = cls._registry[product_type]

            return subclass(
                id=id,
                upload_date=datetime.fromisoformat(src["upload_date"]),
                geospatial_coverage=shape(src["geospatial_coverage"]),
                pipelines_status=src.get("pipelines_status", {}),
                properties=src.get("properties", {})
            )
        except NotFoundError as e:
            raise e

    def update(self):
        """
        Create or overwrite the document in Elasticsearch.
        """
        es = self.get_es_instance()
        index = self.get_index()

        # Assemble full document body
        doc = {
            "product_type": self.product_type,
            "upload_date": self.upload_date.isoformat(),
            "geospatial_coverage": mapping(self.geospatial_coverage),
            "pipelines_status": self.pipelines_status,
            "properties": self.properties
        }

        # Index the document (create or replace)
        es.index(index=index, id=self.id, body=doc)

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


@BaseWorldCerealType.register_subclass("WORDCEREAL_TRAINING_DATA")
class WordCerealTrainingData(BaseWorldCerealType):
    def __init__(self,
                 id: str,
                 upload_date: datetime,
                 geospatial_coverage: BaseGeometry,
                 pipelines_status: Dict[str, Any],
                 properties: Optional[Dict[str, Any]] = None):
        super().__init__(
            id=id,
            product_type="WORDCEREAL_TRAINING_DATA",
            upload_date=upload_date,
            geospatial_coverage=geospatial_coverage,
            pipelines_status=pipelines_status,
            properties=properties
        )

    @property
    def filepath(self) -> str:
        return Path(self.properties["filepath"])

    @filepath.setter
    def filepath(self, filepath: Union[str, Path]):
        self.properties["filepath"] = str(filepath)

    @property
    def entries(self) -> Dict[str, Any]:
        return self.properties["entries"]

    @entries.setter
    def entries(self, entries: Dict[str, Any]):
        self.properties["entries"] = entries






