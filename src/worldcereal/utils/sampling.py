"""
Sampling utilities for stratified reference data sampling.

This module contains functions for distance and class-balanced sampling of reference data,
copied from worldcereal-referencedata to avoid cross-repository dependencies.
"""

from typing import Optional, List

import h3
import geopandas as gpd
import numpy as np
from sklearn.neighbors import BallTree


DEFAULT_SEED: int = 42
LC_CODES = [
    "cropland_unspecified",
    "temporary_crops",
    "non_cropland_incl_perennial",
    "non_cropland_excl_perennial",
    "herbaceous_vegetation",
    "grasslands",
    "wetlands",
    "non_cropland_mixed",
    "shrubland",
    "trees_unspecified",
    "trees_broadleaved",
    "trees_coniferous",
    "trees_mixed",
    "bare_sparsely_vegetated",
    "built_up",
    "open_water",
]


def distance_and_class_balanced_sampling(
    df: gpd.GeoDataFrame,
    distance_threshold_meters: Optional[int],
    sampling_count_per_class: dict,
) -> list:
    """
    Samples points ensuring minimum distance between all samples and max samples per crop class,
    prioritizing rarest classes first, selecting samples across H3 L5 cells,
    and removing spatial neighbors after each selection.
    Now using `sample_id` as unique identifier instead of relying on dataframe index.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    # Reproject to metric CRS
    df = df.to_crs("EPSG:3857")

    if not all(df.geometry.geom_type == "Point"):
        df = df.copy()
        df["centroid_geometry"] = df.geometry.centroid
        geometry_column = "centroid_geometry"
    else:
        geometry_column = "geometry"

    coords = np.array([[geom.x, geom.y] for geom in df[geometry_column]])
    tree = BallTree(coords, metric="euclidean")

    rng = np.random.default_rng(seed=DEFAULT_SEED)

    sample_ids = df["sample_id"].values
    sample_id_to_pos = {sid: pos for pos, sid in enumerate(sample_ids)}

    available_sample_ids = set(sample_ids)
    selected_sample_ids = []

    crop_counts = df[df["sample_id"].isin(available_sample_ids)][
        "sampling_ewoc_code"
    ].value_counts()
    sorted_classes = crop_counts.sort_values(ascending=True).index.tolist()

    for crop_class in sorted_classes:
        selected_in_crop = []

        max_samples_per_class = sampling_count_per_class.get(crop_class, 0)
        if max_samples_per_class <= 0:
            continue

        while len(selected_in_crop) < max_samples_per_class:
            # Update available samples of this crop
            crop_mask = df["sampling_ewoc_code"] == crop_class
            available_crop_sample_ids = available_sample_ids.intersection(
                df[crop_mask]["sample_id"]
            )

            if not available_crop_sample_ids:
                break  # No more available samples for this crop

            crop_df = df[df["sample_id"].isin(available_crop_sample_ids)]

            # Group by lower level H3 cell
            cell_groups = crop_df.groupby("h3_best_res_cell")

            picked_this_round = False

            for l5_cell, group in cell_groups:
                if len(selected_in_crop) >= max_samples_per_class:
                    break

                selected_row = group.sample(1, random_state=rng).iloc[0]
                selected_sample_id = selected_row["sample_id"]
                selected_in_crop.append(selected_sample_id)

                selected_sample_ids.append(selected_sample_id)

                # Remove neighbors
                selected_pos = sample_id_to_pos[selected_sample_id]
                radius = float(distance_threshold_meters or 0)
                neighbors = tree.query_radius(coords[[selected_pos]], r=radius)[0]
                neighbor_sample_ids = set(sample_ids[neighbors])

                available_sample_ids.difference_update(neighbor_sample_ids)

                picked_this_round = True

            if not picked_this_round:
                break

    # Return final selected samples
    return selected_sample_ids


def collect_samples_for_extraction(
    gdf: gpd.GeoDataFrame,
    sampling_count_per_class: dict,
    distance_threshold_meters: Optional[int] = None,
) -> list:
    """
    Collects sample IDs that should be extracted from a GeoDataFrame based on crop class balancing
    and optional minimum distance constraints.
    """

    samples_for_extraction = []

    # Assign lower resolution H3 cells
    max_samples_per_cell = max(sampling_count_per_class.values())
    if max_samples_per_cell <= 50:
        best_h3_res = 5
    else:
        best_h3_res = 6

    gdf["h3_best_res_cell"] = gdf.apply(
        lambda xx: h3.latlng_to_cell(
            xx["geometry"].centroid.y, xx["geometry"].centroid.x, best_h3_res
        ),
        axis=1,
    )

    # Treat each H3 L3 cell separately
    h3_cell_groups = gdf.groupby("h3_l3_cell")

    for h3_cell, h3_all_samples in h3_cell_groups:
        # Remove unknown classes
        h3_all_samples = h3_all_samples[
            h3_all_samples["sampling_ewoc_code"] != "unknown"
        ]
        # ignore samples with confidence scores = 0
        h3_all_samples = h3_all_samples[h3_all_samples["quality_score_ct"] > 0]
        h3_all_samples = h3_all_samples[h3_all_samples["quality_score_lc"] > 0]
        if h3_all_samples.empty:
            continue

        # Unified sampling for all crops with optional distance constraint
        selected_samples = distance_and_class_balanced_sampling(
            h3_all_samples,
            distance_threshold_meters=distance_threshold_meters,
            sampling_count_per_class=sampling_count_per_class,
        )
        samples_for_extraction.extend(selected_samples)

    return samples_for_extraction


def get_sampling_ewoc_codes(gdf, legend) -> gpd.GeoDataFrame:
    """
    Assigns sampling ewoc codes to the GeoDataFrame based on the legend.
    """
    # Prepare the legend
    legend["ewoc_code"] = legend["ewoc_code"].str.replace("-", "").astype(int)
    legend.set_index("ewoc_code", inplace=True)

    # Assign sampling ewoc codes based on the legend
    gdf["sampling_ewoc_code"] = gdf["ewoc_code"].map(legend["sampling_label"])

    # check whether there are any nodata values in sampling_ewoc_code
    if gdf["sampling_ewoc_code"].isna().sum() > 0:
        raise ValueError(
            "There are samples with no sampling_ewoc_code, cannot continue!"
        )

    return gdf


def run_sampling(
    gdf,
    legend,
    max_samples_lc: int = 30,
    max_samples_ct: int = 30,
    sampling_distance: int = 1500,
    custom_lc_codes: Optional[List[str]] = None,
) -> gpd.GeoDataFrame:
    """
    Run stratified sampling on the input GeoDataFrame based on the sampling ewoc codes.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame containing the samples to be sampled.
        Required attributes = [ewoc_code, sample_id, h3_l3_cell, geometry]
    legend : pd.DataFrame
        Legend DataFrame containing the ewoc_code to sampling_label mapping.
    max_samples_lc : int, optional
        Maximum number of samples per H3 L3 cell to be extracted for LC codes, by default 30.
    max_samples_ct : int, optional
        Maximum number of samples per H3 L3 cell to be extracted for CT codes, by default 30.
    sampling_distance : int, optional
        Minimum distance between samples in meters, by default 1500 m.
    custom_lc_codes : list, optional
        Custom list of LC codes to use instead of the default LC_CODES.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with 'extract' column updated (1 for selected, 0 for non-selected).
    """

    # get sampling ewoc codes
    gdf = get_sampling_ewoc_codes(gdf, legend)

    # prepare for sampling
    gdf["extract"] = 0

    # Construct a dict with the sampling ewoc codes and the corresponding samples per class to take
    sampling_count = {}
    unique_classes = gdf["sampling_ewoc_code"].unique()
    if custom_lc_codes is not None:
        lc_codes = custom_lc_codes
    else:
        lc_codes = LC_CODES
    for class_name in unique_classes:
        # Check if the class is in the lc_codes list
        if class_name in lc_codes:
            sampling_count[class_name] = max_samples_lc
        else:
            sampling_count[class_name] = max_samples_ct

    # For LC codes, sample max_samples_lc samples per class
    samples = collect_samples_for_extraction(
        gdf,
        sampling_count_per_class=sampling_count,
        distance_threshold_meters=sampling_distance,
    )
    gdf.loc[
        (gdf["sample_id"].isin(samples) & (gdf["extract"] == 0)),
        "extract",
    ] = 1

    # perform check on sampling
    if gdf["extract"].sum() == 0:
        raise ValueError("No samples were selected for extraction, cannot continue!")

    return gdf
