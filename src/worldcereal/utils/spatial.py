""" Definitions of spatial context, either point-based or spatial"""

from dataclasses import dataclass
from typing import Union

from geojson import GeoJSON
from shapely.geometry import Polygon, box

# TODO: once gfmap is on PyPI we need to import the class from there
# instead of duplicating it here.


@dataclass
class BoundingBoxExtent:
    """Definition of a bounding box as accepted by OpenEO

    Contains the minx, miny, maxx, maxy coordinates expressed as east, south
    west, north. The EPSG is also defined.
    """

    west: float
    south: float
    east: float
    north: float
    epsg: int = 4326

    def __dict__(self):
        return {
            "west": self.west,
            "south": self.south,
            "east": self.east,
            "north": self.north,
            "crs": f"EPSG:{self.epsg}",
            "srs": f"EPSG:{self.epsg}",
        }

    def __iter__(self):
        return iter(
            [
                ("west", self.west),
                ("south", self.south),
                ("east", self.east),
                ("north", self.north),
                ("crs", f"EPSG:{self.epsg}"),
                ("srs", f"EPSG:{self.epsg}"),
            ]
        )

    def to_geometry(self) -> Polygon:
        return box(self.west, self.south, self.east, self.north)

    def to_geojson(self) -> GeoJSON:
        return self.to_geometry().__geo_interface__


SpatialContext = Union[GeoJSON, BoundingBoxExtent, str]
