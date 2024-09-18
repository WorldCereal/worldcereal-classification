"""From a pickle object containing the paths of all the patches, performs a
batch job to build the catalogue containing all the patches."""

import argparse
import json
import logging
import pickle
import re
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, List

import mepsy  # Installable from the repository https://git.vito.be/projects/TAP-VEGTEAM/repos/mepsy/browse
import pystac
import xarray as xr
from openeo_gfmap.stac.constants import (
    LICENSE,
    LICENSE_LINK,
    STAC_EXTENSIONS,
    SUMMARIES,
)
from pyproj import CRS, Transformer
from shapely import box, to_geojson
from shapely.ops import transform

from worldcereal.stac.constants import (
    COLLECTION_DESCRIPTIONS,
    COLLECTION_IDS,
    COLLECTION_REGEXES,
    CONSTELLATION_NAMES,
    ITEM_ASSETS,
    ExtractionCollection,
)

# Logger used for the pipeline
builder_log = logging.getLogger("catalogue_builder")

builder_log.setLevel(level=logging.INFO)

stream_handler = logging.StreamHandler()
builder_log.addHandler(stream_handler)

formatter = logging.Formatter("%(asctime)s|%(name)s|%(levelname)s:  %(message)s")
stream_handler.setFormatter(formatter)


def _init_collection(collection: ExtractionCollection) -> pystac.Collection:
    builder_log.info("Initializing the catalogue for the paths.")

    stac_collection = pystac.Collection(
        id=COLLECTION_IDS[collection],
        description=COLLECTION_DESCRIPTIONS[collection],
        extent=None,
    )
    stac_collection.license = LICENSE
    stac_collection.license_link = LICENSE_LINK
    stac_collection.stac_extensions = STAC_EXTENSIONS

    stac_collection.extra_fields["summaries"] = SUMMARIES.get(
        CONSTELLATION_NAMES[collection], pystac.summaries.Summaries({})
    ).to_dict()

    item_asset_extension = pystac.extensions.item_assets.ItemAssetsExtension.ext(
        stac_collection, add_if_missing=True
    )
    item_asset_extension.item_assets = ITEM_ASSETS[collection]

    return stac_collection


def _parse_item(path: Path, collection: ExtractionCollection) -> pystac.Item:
    try:
        ds = xr.open_dataset(path)
    except Exception as _:
        builder_log.error("Failed to open the dataset at %s.", path)
        return None

    attributes = ds.attrs

    if "sample_id" not in attributes:
        # Attempting to parse the sample_id from the path name
        file_name = Path(path).name
        # Find the sample_id from the regex
        match = re.match(COLLECTION_REGEXES[collection], file_name)
        if not match:
            builder_log.error("Failed to parse the sample_id from %s.", path)
            return None

        if collection == ExtractionCollection.SENTINEL1:
            sample_id = match.group(2)
        else:
            sample_id = match.group(1)
    else:
        sample_id = attributes["sample_id"]

    epsg = CRS.from_wkt(ds.crs.crs_wkt).to_epsg()

    transformer = Transformer.from_crs(
        CRS.from_epsg(epsg), CRS.from_epsg(4326), always_xy=True
    )

    if collection == ExtractionCollection.SENTINEL1:
        if "orbit_state" not in attributes:
            # Attempting to parse the orbit state from the path
            file_name = Path(path).name
            match = re.match(COLLECTION_REGEXES[collection], file_name)
            if not match:
                builder_log.error("Failed to parse the orbit state from %s.", path)
                return None

            orbit_state = match.group(1)
        else:
            orbit_state = attributes["orbit_state"]

        sample_id = f"{sample_id}_{orbit_state}"

    utm_bounds = (
        ds.x.min().item(),
        ds.y.min().item(),
        ds.x.max().item(),
        ds.y.max().item(),
    )
    # Transform the bounds to lat/lon
    latlon_bounds = transform(transformer.transform, box(*utm_bounds)).bounds

    item = pystac.Item(
        id=sample_id,
        datetime=None,
        start_datetime=ds.t.min().values.astype("datetime64[ms]").astype(datetime),
        end_datetime=ds.t.max().values.astype("datetime64[ms]").astype(datetime),
        geometry=json.loads(to_geojson(box(*latlon_bounds))),
        bbox=list(latlon_bounds),
        properties={
            "proj:epsg": epsg,
        },
    )

    item.extra_fields["epsg"] = epsg

    if collection == ExtractionCollection.SENTINEL1:
        item.properties["orbit_state"] = orbit_state

    constellation_name = CONSTELLATION_NAMES[collection]

    if ITEM_ASSETS[collection] is None:
        raise ValueError("No assets defined for the collection.")

    collection_assets: dict[str, Any] = ITEM_ASSETS[collection]  # type: ignore
    asset_definition = collection_assets[constellation_name]  # type: ignore

    asset = asset_definition.create_asset(
        href=Path(path).absolute().as_posix(),
    )

    asset.media_type = "application/x-netcdf"
    asset.title = sample_id
    asset.extra_fields["proj:shape"] = list(ds.rio.shape)
    asset.extra_fields["proj:epsg"] = epsg
    asset.extra_fields["proj:bbox"] = list(utm_bounds)

    item.add_asset(sample_id, asset)

    return item


def parse_collection(
    paths: List[Path], output_folder: Path, collection: ExtractionCollection
) -> None:
    stac_collection = _init_collection(collection)

    failed_paths = []

    first_item = None

    builder_log.info("Parsing %s paths...", len(paths))
    for path in paths:
        item = _parse_item(path, collection)
        if item is not None:
            if first_item is None:
                first_item = item
            stac_collection.add_item(item)
        else:
            failed_paths.append(path)

    builder_log.info(
        "Finished parsing paths. %s items were created, while %s FAILED.",
        len(list(stac_collection.get_all_items())),
        len(failed_paths),
    )

    if first_item is None:
        builder_log.error("No items were created for the collection.")
        return

    stac_collection.update_extent_from_items()

    # Save the collection as a pickle file
    output_path = output_folder / f"{first_item.id}_collection.pkl"
    builder_log.info("Saving the collection as a pickle file: %s", output_path)

    with open(output_path, "wb") as f:
        pickle.dump(stac_collection, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuilds the catalogue from a list of paths to extracted patches."
    )

    parser.add_argument(
        "collection",
        type=ExtractionCollection,
        choices=list(ExtractionCollection),
        help="The collection to extract",
    )
    parser.add_argument(
        "input_paths",
        type=Path,
        help="The path to the pickle file containing the paths to the patches.",
    )
    parser.add_argument(
        "output_folder",
        type=Path,
        help="Path where the collection files will be saved.",
    )
    parser.add_argument(
        "mepsy_config_file", type=Path, help="The path to the mepsy configuration file."
    )
    parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=5_000,
        help="The size of the patches to process per execution.",
    )
    parser.add_argument(
        "-l", "--local", action="store_true", help="If set, will run the job locally."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="If set, will print the logs."
    )

    args = parser.parse_args()

    # Load the paths from the pickle file
    builder_log.info("Loading the paths from the pickle file.")
    with open(args.input_paths, "rb") as f:
        paths = pickle.load(f)

    # Split the list of paths in sublists from the chunk size
    paths = [
        paths[i : i + args.chunk_size] for i in range(0, len(paths), args.chunk_size)
    ]

    if args.local:
        builder_log.info("Subsampling the patches paths as the job is running locally.")
        paths = paths[:1000]
        parse_collection(
            paths, output_folder=args.output_folder, collection=args.collection
        )
        exit()

    builder_log.info("Configuring the Mepsy App.")
    app_config = dict(
        app_name=f"CatalogueBuilder {args.collection.value}",
        driver_memory=2,
        executor_memory=2,
        max_executors=20,
        queue="default",
        config_path=args.mepsy_config_file,
        local=args.local,
        verbose=args.verbose,
    )

    mep = mepsy.SparkApp(**app_config)

    builder_log.info("Processing %s patches...", len(paths))

    mep.parallelize(
        partial(
            parse_collection,
            output_folder=args.output_folder,
            collection=args.collection,
        ),
        paths,
    )
