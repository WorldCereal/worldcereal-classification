"""Split catalogue by the local UTM projection of the products to be utilisable
by OpenEO processes."""

import argparse
import logging
import pickle
from pathlib import Path

from openeo_gfmap.utils.split_stac import split_collection_by_epsg
from tqdm import tqdm

# Logger used for the pipeline
builder_log = logging.getLogger("catalogue_splitter")

builder_log.setLevel(level=logging.INFO)

stream_handler = logging.StreamHandler()
builder_log.addHandler(stream_handler)

formatter = logging.Formatter("%(asctime)s|%(name)s|%(levelname)s:  %(message)s")
stream_handler.setFormatter(formatter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Splits the catalogue by the local UTM projection of the products."
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="The path to the folder containing the collection files.",
    )
    parser.add_argument(
        "output_folder",
        type=Path,
        help="The path where to save the splitted STAC collections to.",
    )

    args = parser.parse_args()

    if not args.input_folder.exists():
        raise FileNotFoundError(f"The input folder {args.input_folder} does not exist.")

    if not args.output_folder.exists():
        raise FileNotFoundError(
            f"The output folder {args.output_folder} does not exist."
        )

    builder_log.info("Loading the catalogues from the directory %s", args.input_folder)
    # List the catalogues in the input folder
    catalogues = []
    for catalogue_path in tqdm(args.input_folder.glob("*.pkl")):
        with open(catalogue_path, "rb") as file:
            catalogue = pickle.load(file)
            try:
                catalogue.strategy
            except AttributeError:
                setattr(catalogue, "strategy", None)
            catalogues.append(catalogue)

    builder_log.info("Loaded %s catalogues. Merging them...", len(catalogues))

    merged_catalogue = None
    for catalogue_path in tqdm(catalogues):
        if merged_catalogue is None:
            merged_catalogue = catalogue_path
        else:
            merged_catalogue.add_items(catalogue_path.get_all_items())

    if merged_catalogue is None:
        raise ValueError("No catalogues found in the input folder.")

    builder_log.info("Merged catalogues into one. Updating the extent...")
    merged_catalogue.update_extent_from_items()

    with open("temp_merged_catalogue.pkl", "wb") as file:
        pickle.dump(merged_catalogue, file)

    builder_log.info("Splitting the catalogue by the local UTM projection...")
    split_collection_by_epsg(merged_catalogue, args.output_folder)
