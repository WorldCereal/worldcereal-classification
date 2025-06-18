import argparse
from pathlib import Path

import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.geometry import mapping


# class GeoparquetFile:
#     def __init__(self, geoparquet_file: Path):
#         self.geoparquet_file = geoparquet_file
#         self.gdf = None
#
#         self._read()
#
#     def _read(self):
#         self.gdf = gpd.read_parquet(self.geoparquet_file)
#
#     def to_dict(self) -> dict:
#         return self.gdf.to_dict(orient='records')
#
#     def combine_geometries(self) -> BaseGeometry:
#         geoms = self.gdf.geometry.values
#         combined: BaseGeometry = unary_union(geoms)
#         return combined


def main(geoparquet_file: Path):
    # TODO : use util.esttypes


if __name__ == "__main__":

    # Argparsing
    parser = argparse.ArgumentParser(description="Extract data from a collection")
    parser.add_argument(
        "geoparquet_file", type=Path, help="The path to the geoparquet"
    )
    args = parser.parse_args()

    main(args.geoparquet_file)

