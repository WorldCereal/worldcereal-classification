import copy
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import geopandas as gpd
import openeo
import pandas as pd
import pystac_client
from loguru import logger
from openeo.processes import ProcessBuilder, eq, if_, not_, or_
from openeo_gfmap import Backend, BackendContext, BoundingBoxExtent, TemporalContext
from openeo_gfmap.preprocessing.compositing import mean_compositing, median_compositing
from openeo_gfmap.preprocessing.sar import (
    compress_backscatter_uint16,
    decompress_backscatter_uint16,
)
from openeo_gfmap.utils.catalogue import UncoveredS1Exception, select_s1_orbitstate_vvvh
from pandas.core.dtypes.dtypes import CategoricalDtype
from shapely import wkb
from shapely.geometry import MultiPolygon, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

from worldcereal.extract.point_worldcereal import REQUIRED_ATTRIBUTES
from worldcereal.extract.utils import S2_GRID, upload_geoparquet_artifactory
from worldcereal.rdm_api import RdmInteraction
from worldcereal.rdm_api.rdm_interaction import RDM_DEFAULT_COLUMNS
from worldcereal.utils.refdata import gdf_to_points

STAC_ENDPOINT_S1 = (
    "https://stac.openeo.vito.be/collections/worldcereal_sentinel_1_patch_extractions"
)

STAC_ENDPOINT_S2 = (
    "https://stac.openeo.vito.be/collections/worldcereal_sentinel_2_patch_extractions"
)

STAC_ENDPOINT_MONTHLY_METEO = (
    "https://stac.openeo.vito.be/collections/agera5_monthly_composite"
)

STAC_ENDPOINT_DEKADAL_METEO = (
    "https://stac.openeo.vito.be/collections/agera5_dekadal_composite"
)

STAC_ENDPOINT_SLOPE_TERRASCOPE = (
    "https://stac.openeo.vito.be/collections/COPERNICUS30_DEM_SLOPE_TERRASCOPE"
)

# Due to a bug on openEO side (https://github.com/Open-EO/openeo-geopyspark-driver/issues/1153)
# We have to provide here ALL bands in ALPHABETICAL order!
S2_BANDS = [
    "S2-L2A-B01",
    "S2-L2A-B02",
    "S2-L2A-B03",
    "S2-L2A-B04",
    "S2-L2A-B05",
    "S2-L2A-B06",
    "S2-L2A-B07",
    "S2-L2A-B08",
    "S2-L2A-B09",
    "S2-L2A-B11",
    "S2-L2A-B12",
    "S2-L2A-B8A",
    "S2-L2A-DISTANCE-TO-CLOUD",
    "S2-L2A-SCL",
    "S2-L2A-SCL_DILATED_MASK",
]

S2_BANDS_SELECTED = [
    "S2-L2A-B02",
    "S2-L2A-B03",
    "S2-L2A-B04",
    "S2-L2A-B05",
    "S2-L2A-B06",
    "S2-L2A-B07",
    "S2-L2A-B08",
    "S2-L2A-B8A",
    "S2-L2A-B11",
    "S2-L2A-B12",
    "S2-L2A-SCL",
    "S2-L2A-SCL_DILATED_MASK",
]


def label_points_centroid(
    gdf: gpd.GeoDataFrame, epsg: Optional[int] = None
) -> gpd.GeoDataFrame:
    """
    Sample points from the centroid of the input GeoDataFrame.
    """
    if epsg is not None:
        gdf = gdf.to_crs(epsg=epsg)

    gdf["geometry"] = gdf.centroid

    if epsg is not None:
        gdf = gdf.to_crs(epsg=4326)

    return gdf


def get_label_points(
    row: pd.Series,
    ground_truth_file: Optional[Union[Path, str]] = None,
    only_flagged_samples: bool = False,
) -> gpd.GeoDataFrame:
    """
    Retrieve label points for a given row from STAC collections and RDM API.

    Parameters
    ----------
    row : pd.Series
        The row containing ref_id, epsg, start_date, and end_date.
    ground_truth_file : Optional[Union[Path, str]], optional
        The path to the ground truth file. If provided, this file will
        be queried for getting the ground truth. If not, the RDM will
        be used for the query.
    only_flagged_samples : bool, optional
        If True, only samples with extract flag >0 will be retrieved, no collateral samples.
        (This is useful for very large and dense datasets like USDA).

    Returns
    -------
    gpd.GeoDataFrame, bool, list[tuple[str | None, shapely.geometry.base.BaseGeometry]], dict[str, dict[str, int]], dict[str, dict[str, int]]
        A tuple containing:
        - gpd.GeoDataFrame: The sampled GeoDataFrame with label points.
        - bool: A flag indicating whether S1 extraction is disabled
                (True if no S1 sample_ids found).
        - list of (orbit_state, patch_geometry) for every S1 STAC patch
          matching (ref_id, epsg). Used downstream to classify samples by
          which orbit(s) actually cover them and to split jobs per orbit.
        - dict {sample_id: {orbit_state: epsg_int}} mapping each sample to
          the EPSG of its S1 patch per orbit. Used to detect UTM-zone
          boundary samples whose S1 patch is stored under a different EPSG
          than their S2 patch.
        - dict {sample_id: {orbit_state: size_bytes}} of S1 asset file sizes,
          a monotonic proxy for per-sample, per-orbit time-series density
          (~4 KB per timestep). Used to route ambiguous "BOTH" samples to
          the orbit that has more acquisitions for that specific sample.

    """

    # Find all items (i.e. patches) corresponding to the given ref_id and epsg.
    # Support both old-style `proj:epsg` (integer) and new-style `proj:code`
    # (string like "EPSG:32634") property names.
    ref_id = row["ref_id"]
    epsg = int(row["epsg"])
    stac_query_old = {
        "ref_id": {"eq": ref_id},
        "proj:epsg": {"eq": epsg},
    }
    stac_query_new = {
        "ref_id": {"eq": ref_id},
        "proj:code": {"eq": f"EPSG:{epsg}"},
    }
    client = pystac_client.Client.open("https://stac.openeo.vito.be/")

    logger.info("Querying S1/S2 STAC collections ...")

    # For S1 we deliberately drop the EPSG filter: samples near a UTM-zone
    # boundary can have their S1 patch indexed under a different EPSG than
    # their S2 patch, and we want to discover those so they don't silently
    # lose S1 data.
    s1_query_no_epsg = {"ref_id": {"eq": ref_id}}

    def _collect_items(collection: str, queries, capture_orbit: bool = False):
        """Search a STAC collection.

        ``queries`` is an iterable of STAC query dicts; results from each are
        deduplicated by item id.

        When `capture_orbit` is True, also returns:
        - a list of (orbit_state, geometry) records deduplicated by item id,
          used to classify samples by S1 orbit coverage downstream.
        - a dict {sample_id: {orbit_state: epsg_int}} that lets the caller
          discover boundary samples whose S1 patch lives in a UTM zone other
          than the one their S2 patch was indexed under.
        - a dict {sample_id: {orbit_state: size_bytes}} of asset file sizes,
          used as a cheap proxy for per-orbit time-series density. ~4 KB per
          S1 timestep, monotonic with the actual t-length, so the larger file
          is always the orbit with more acquisitions for that sample.
        """
        items: dict = {}
        seen_ids: set = set()
        orbit_records: List[Tuple[Optional[str], object]] = []
        sample_orbit_epsg: dict = {}
        sample_orbit_size: dict = {}
        for query in queries:
            search = client.search(collections=[collection], query=query)
            for item in search.items():
                if item.id in seen_ids:
                    continue
                seen_ids.add(item.id)
                sid = item.properties["sample_id"]
                geom = shape(item.geometry).buffer(1e-9)
                if sid not in items:
                    items[sid] = geom
                if capture_orbit:
                    orbit_state = item.properties.get("sat:orbit_state")
                    orbit_records.append((orbit_state, geom))
                    item_epsg: Optional[int] = None
                    if "proj:epsg" in item.properties:
                        item_epsg = int(item.properties["proj:epsg"])
                    elif "proj:code" in item.properties:
                        item_epsg = int(item.properties["proj:code"].split(":")[-1])
                    else:
                        for asset in item.assets.values():
                            asset_code = asset.extra_fields.get("proj:code")
                            asset_epsg = asset.extra_fields.get("proj:epsg")
                            if asset_code:
                                item_epsg = int(str(asset_code).split(":")[-1])
                                break
                            if asset_epsg is not None:
                                item_epsg = int(asset_epsg)
                                break
                    if item_epsg is not None:
                        sample_orbit_epsg.setdefault(sid, {})[orbit_state] = item_epsg
                    # Best-effort file-size lookup for per-sample orbit
                    # density signal. Silently skipped if the asset href is
                    # not a locally-accessible filesystem path.
                    for asset in item.assets.values():
                        try:
                            sz = os.path.getsize(asset.href)
                        except (OSError, TypeError):
                            continue
                        prev = sample_orbit_size.setdefault(sid, {}).get(orbit_state, 0)
                        if sz > prev:
                            sample_orbit_size[sid][orbit_state] = sz
                        break
        return (
            (items, orbit_records, sample_orbit_epsg, sample_orbit_size)
            if capture_orbit
            else items
        )

    items_s1, s1_patches, s1_sample_orbit_epsg, s1_sample_orbit_size = _collect_items(
        "worldcereal_sentinel_1_patch_extractions",
        queries=(s1_query_no_epsg,),
        capture_orbit=True,
    )
    items_s2 = _collect_items(
        "worldcereal_sentinel_2_patch_extractions",
        queries=(stac_query_old, stac_query_new),
    )
    logger.info(
        f"Found {len(items_s1)} S1 items and {len(items_s2)} S2 items"
    )

    # Find sample_ids which are present in either S2 or both STAC collections
    common_sample_ids = set(items_s1.keys()).intersection(set(items_s2.keys()))
    if len(common_sample_ids) == 0:
        logger.warning(
            "No common sample_ids found in S1 and S2 STAC collections. "
            "S1 extraction will be disabled from process graph!."
        )
        disable_s1 = True
    else:
        logger.info(f"Found {len(common_sample_ids)} common sample_ids in S1 and S2")
        disable_s1 = False

    s2_only_sample_ids = set(items_s2.keys()).difference(common_sample_ids)
    logger.info(f"Found {len(s2_only_sample_ids)} S2-only sample_ids")
    selected_sample_ids = common_sample_ids.union(s2_only_sample_ids)

    if len(selected_sample_ids) == 0:
        raise ValueError(
            "No sample_ids found in S1 or S2 STAC collections. "
            "Please check the extractions for this ref_id and epsg."
        )
    logger.info(f"Total selected sample_ids for extraction: {len(selected_sample_ids)}")

    # Build the spatial extent from the patch footprints.
    polygons = [items_s2[sample_id] for sample_id in selected_sample_ids]
    patches_multi = MultiPolygon(polygons)

    temporal_extent = TemporalContext(start_date=row.start_date, end_date=row.end_date)

    if ground_truth_file is not None:
        # Read the ground truth file row-group by row-group to keep memory
        # low (the full file can be multi-GB). For each row group we check
        # the h3_l3_cell statistics to skip irrelevant groups, then spatially
        # intersect with the actual patch footprints to capture collaterals.
        import pyarrow.parquet as pq
        from shapely import prepare

        logger.info(f"Reading ground truth from: {ground_truth_file}")

        # Build a spatial index over the patch footprints for fast lookups
        patch_tree = STRtree(polygons)
        prepare(patches_multi)

        # Extract unique h3_l3_cell values from the selected sample_ids.
        # sample_id format: "<ref_id>_<h3_l3_cell><index>"
        ref_id_prefix = row["ref_id"] + "_"
        h3_cells = set()
        for sid in selected_sample_ids:
            rest = sid[len(ref_id_prefix):]
            # h3 L3 cell ids are 15-char hex strings like "831e20fffffffff"
            if len(rest) >= 15:
                h3_cells.add(rest[:15])

        logger.info(f"Pre-filtering ground truth on {len(h3_cells)} H3 L3 cells")

        read_columns = [c for c in RDM_DEFAULT_COLUMNS if c != "ref_id"]
        pf = pq.ParquetFile(ground_truth_file)
        chunks = []

        for rg_idx in range(pf.metadata.num_row_groups):
            # Use row group statistics to skip groups with no matching h3 cells
            if h3_cells:
                rg = pf.metadata.row_group(rg_idx)
                skip = False
                for col_idx in range(rg.num_columns):
                    col = rg.column(col_idx)
                    if col.path_in_schema == "h3_l3_cell" and col.statistics and col.statistics.has_min_max:
                        rg_min, rg_max = col.statistics.min, col.statistics.max
                        if not any(rg_min <= c <= rg_max for c in h3_cells):
                            skip = True
                        break
                if skip:
                    continue

            table = pf.read_row_group(rg_idx, columns=read_columns)
            df = table.to_pandas()
            del table

            # Filter to matching h3 cells first (cheap, no geometry parsing)
            if h3_cells and "h3_l3_cell" in df.columns:
                df = df[df["h3_l3_cell"].isin(h3_cells)]

            if df.empty:
                continue

            # Apply temporal filter before geometry parsing (cheap)
            if "valid_time" in df.columns:
                df["valid_time"] = pd.to_datetime(df["valid_time"])
                df = df[
                    (df["valid_time"] >= temporal_extent.start_date)
                    & (df["valid_time"] <= temporal_extent.end_date)
                ]

            if df.empty:
                continue

            # Apply subset filter before geometry parsing (cheap)
            if only_flagged_samples and "extract" in df.columns:
                df = df[df["extract"] > 0]

            if df.empty:
                continue

            # Now parse geometry and do spatial intersection with actual patches
            df["geometry"] = df["geometry"].apply(lambda b: wkb.loads(bytes(b)))
            chunk_gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
            del df

            # Use STRtree for fast spatial lookup against patch footprints
            hit_indices = patch_tree.query(chunk_gdf.geometry, predicate="intersects")
            matched_rows = chunk_gdf.index[sorted(set(hit_indices[0]))]
            chunk_gdf = chunk_gdf.loc[matched_rows]

            if not chunk_gdf.empty:
                chunks.append(chunk_gdf)

            del chunk_gdf

        if chunks:
            gdf = gpd.GeoDataFrame(pd.concat(chunks, ignore_index=True))
        else:
            gdf = gpd.GeoDataFrame(columns=read_columns)

        gdf["ref_id"] = row["ref_id"]

        # Keep only default columns
        available_cols = [c for c in RDM_DEFAULT_COLUMNS if c in gdf.columns]
        gdf = gdf[available_cols]

        logger.info(f"Loaded {len(gdf)} samples (including collaterals) from ground truth file")
    else:
        # Fall back to RDM API query for non-file sources.
        gdf = RdmInteraction().get_samples(
            ref_ids=[row["ref_id"]],
            spatial_extent=patches_multi,
            temporal_extent=temporal_extent,
            include_private=True,
            ground_truth_file=None,
            subset=only_flagged_samples,
        )

    sampled_gdf = gdf_to_points(gdf)

    return (
        sampled_gdf,
        disable_s1,
        s1_patches,
        s1_sample_orbit_epsg,
        s1_sample_orbit_size,
    )


def generate_output_path_patch_to_point_worldcereal(
    root_folder: Path,
    geometry_index: int,
    row: pd.Series,
    asset_id: Optional[str] = None,
) -> Path:
    """
    Generate the output path for point extractions.

    Parameters
    ----------
    root_folder : Path
        Root folder where the output parquet file will be saved.
    geometry_index : int
        Index of the geometry. Always 0 for point extractions.
    row : pd.Series
        The current job row from the job manager.
    asset_id : str, optional
        Asset ID for compatibility with the job manager. Not used.

    Returns
    -------
    Path
        Path to the output parquet file.
    """

    epsg = row.epsg

    # Create the subfolder to store the output
    subfolder = root_folder / str(epsg)
    subfolder.mkdir(parents=True, exist_ok=True)

    # we may have multiple output files per s2_tile_id and need
    # a unique name so we use the job ID
    output_file = f"WORLDCEREAL_{root_folder.name}_{row.start_date}_{row.end_date}_{epsg}_{row.id}{row.out_extension}"

    return subfolder / output_file


def _has_s1_patches_for_cell_epsg(
    client: pystac_client.Client, ref_id: str, h3_cell: str, epsg: int
) -> bool:
    """Return True if the S1 STAC has any patches for the given
    (ref_id, h3_l3_cell, epsg).

    H3 cells near a UTM-zone boundary may have S1 patches stored under one UTM
    zone only. Without this check, the outer EPSG loop emits a job for the
    cell under the EPSG that has zero S1 patches, and the backend's
    load_stac then crashes on an empty list.
    """
    for prop, value in (("proj:code", f"EPSG:{epsg}"), ("proj:epsg", int(epsg))):
        search = client.search(
            collections=["worldcereal_sentinel_1_patch_extractions"],
            query={
                "ref_id": {"eq": ref_id},
                "h3_l3_cell": {"eq": h3_cell},
                prop: {"eq": value},
            },
            limit=1,
        )
        if any(True for _ in search.items()):
            return True
    return False


def _classify_samples_by_orbit(
    group_df: gpd.GeoDataFrame,
    s1_patches: List[Tuple[Optional[str], object]],
    s1_sample_orbit_size: Optional[Dict[str, Dict[str, int]]] = None,
) -> pd.Series:
    """Classify each sample by which S1 orbit(s) cover it.

    Returns a pd.Series aligned to ``group_df.index`` with values in
    {"ASCENDING", "DESCENDING", "BOTH", NaN}. NaN means no S1 patch covers
    the sample.

    When ``s1_sample_orbit_size`` is provided and a sample's geometry is
    covered by both orbits, the per-sample asset file sizes are used to
    pick the orbit with more acquisitions for that specific sample (file
    size is a monotonic proxy for t-length). Samples with tied or missing
    sizes stay tagged "BOTH" so the downstream CDSE-based resolver can
    still assign them.
    """
    if not s1_patches:
        return pd.Series([None] * len(group_df), index=group_df.index, dtype=object)

    orbits = [o for o, _ in s1_patches]
    tree = STRtree([g for _, g in s1_patches])
    hits = tree.query(group_df.geometry.values, predicate="intersects")

    orbit_sets: List[set] = [set() for _ in range(len(group_df))]
    for s_idx, p_idx in zip(hits[0], hits[1]):
        orbit_sets[s_idx].add(orbits[p_idx])

    sample_ids = group_df["sample_id"].tolist() if "sample_id" in group_df.columns else None
    size_lookup = s1_sample_orbit_size or {}
    refined_via_size = 0
    labels = []
    for i, o in enumerate(orbit_sets):
        if not o:
            labels.append(None)
        elif len(o) == 1:
            labels.append(next(iter(o)))
        else:
            # Sample geometry intersects both orbits. Prefer the orbit whose
            # OWN patch for this sample has the larger file (more timesteps);
            # only fall back to "BOTH" when sizes are unavailable or equal.
            picked = "BOTH"
            if sample_ids is not None:
                sizes = size_lookup.get(sample_ids[i], {})
                asc = sizes.get("ASCENDING", 0)
                desc = sizes.get("DESCENDING", 0)
                if asc > desc:
                    picked = "ASCENDING"
                    refined_via_size += 1
                elif desc > asc:
                    picked = "DESCENDING"
                    refined_via_size += 1
            labels.append(picked)
    if refined_via_size:
        logger.info(
            f"Per-sample file-size routing resolved {refined_via_size} BOTH samples."
        )
    return pd.Series(labels, index=group_df.index, dtype=object)


def _resolve_orbit_split(
    sample_orbit: pd.Series,
    s1_patches: List[Tuple[Optional[str], object]],
    group_df: gpd.GeoDataFrame,
    temporal_extent: TemporalContext,
) -> pd.Series:
    """Resolve the "BOTH" and no-coverage buckets into a concrete orbit label.

    The target orbit for ambiguous samples (BOTH / no-coverage) is picked by
    querying CDSE for actual S1 acquisition density via
    ``select_s1_orbitstate_vvvh``. WorldCereal STAC entries can exist for an
    orbit that has near-empty time series in a region (e.g. ASC over
    Madagascar), so STAC item counts alone are not a reliable signal.

    Falls back to STAC patch counts (ties -> ASCENDING) when CDSE is
    inconclusive or unreachable.
    """
    ambiguous_mask = (sample_orbit == "BOTH") | sample_orbit.isna()
    if not ambiguous_mask.any():
        return sample_orbit

    target: Optional[str] = None
    try:
        ambiguous_df = group_df.loc[ambiguous_mask]
        bbox = (
            ambiguous_df.to_crs(epsg=3857)
            .buffer(1)
            .to_crs(epsg=4326)
            .total_bounds
        )
        target = select_s1_orbitstate_vvvh(
            BackendContext(Backend.CDSE),
            BoundingBoxExtent(*bbox),
            temporal_extent,
        )
        logger.info(f"CDSE-selected orbit for ambiguous samples: {target}")
    except UncoveredS1Exception:
        logger.warning("CDSE reports no S1 coverage; falling back to STAC counts.")
    except Exception as e:
        logger.warning(f"CDSE orbit query failed ({e}); falling back to STAC counts.")

    if target is None:
        n_asc_only = (sample_orbit == "ASCENDING").sum()
        n_desc_only = (sample_orbit == "DESCENDING").sum()
        if n_asc_only or n_desc_only:
            target = "ASCENDING" if n_asc_only >= n_desc_only else "DESCENDING"
        else:
            asc_patches = sum(1 for o, _ in s1_patches if o == "ASCENDING")
            desc_patches = sum(1 for o, _ in s1_patches if o == "DESCENDING")
            target = "ASCENDING" if asc_patches >= desc_patches else "DESCENDING"

    return sample_orbit.replace("BOTH", target).fillna(target)


def create_job_dataframe_patch_to_point_worldcereal(
    ref_id,
    ground_truth_file=None,
    only_flagged_samples: bool = False,
    max_samples_per_job: Optional[int] = None,
):
    """
    Create a job dataframe for patch-to-point extractions.

    This function queries the STAC catalog to retrieve unique EPSG codes and temporal extents
    for the given `ref_id`. It also identifies ground truth samples and prepares the job
    dataframe for further processing.

    Parameters
    ----------
    ref_id : str
        Reference ID for the extraction.
    ground_truth_file : str, optional
        Path to a ground truth file. If not provided, the function queries RDM for ground truth.
    only_flagged_samples : bool, optional
        If True, only samples with extract flag >0 will be retrieved, no collateral samples.
        (This is useful for very large and dense datasets like USDA).
    max_samples_per_job : int, optional
        Maximum number of samples allowed per EPSG job before splitting into multiple H3 L3 cells.
        If None, additional splitting is disabled.

    Returns
    -------
    pd.DataFrame
        A dataframe containing job configurations for each EPSG zone.
    """

    client = pystac_client.Client.open("https://stac.openeo.vito.be/")

    stac_query = {
        "ref_id": {"eq": ref_id},
    }

    search = client.search(
        collections=["worldcereal_sentinel_2_patch_extractions"],
        query=stac_query,
    )

    # Get a list of EPSG codes that occur for this ref_id as we need
    # to run jobs per UTM zone.
    logger.info(f"Creating job dataframe for: {ref_id}")
    logger.info("Looking for unique EPSG codes in STAC collection ...")
    epsg_codes = {}
    for item in search.items():
        if "proj:epsg" in item.properties:
            epsg = int(item.properties["proj:epsg"])
            epsg_prop = "proj:epsg"
        elif "proj:code" in item.properties:
            epsg = int(item.properties["proj:code"].split(":")[-1])
            epsg_prop = "proj:code"
        else:
            logger.warning(
                f"Item {item.id} does not have 'proj:epsg' nor 'proj:code' property, skipping ..."
            )
            continue

        if epsg not in epsg_codes and epsg != 4038:
            logger.debug(f"Found EPSG: {epsg}")

            epsg_codes[epsg] = {
                "start_date": pd.to_datetime(item.properties["start_date"]),
                "end_date": pd.to_datetime(item.properties["end_date"]),
                "epsg_property": epsg_prop,
            }
        elif epsg != 4038:
            current_start_date = pd.to_datetime(item.properties["start_date"])
            current_end_date = pd.to_datetime(item.properties["end_date"])
            if current_start_date < epsg_codes[epsg]["start_date"]:
                epsg_codes[epsg]["start_date"] = current_start_date
            if current_end_date > epsg_codes[epsg]["end_date"]:
                epsg_codes[epsg]["end_date"] = current_end_date

    # Initialize job dataframe for patch to point
    rows = []

    logger.info(f"Found {len(epsg_codes)} unique EPSG codes in STAC collection.")

    for epsg in epsg_codes.keys():
        # We assume identical start and end date for the entire ref_id
        start_date = epsg_codes[epsg]["start_date"]
        end_date = epsg_codes[epsg]["end_date"]

        # Clamp end_date to last day of the last complete month to avoid compositing an incomplete month
        last_complete_month_end = pd.Timestamp.today().normalize().replace(
            day=1
        ) - pd.Timedelta(days=1)
        end_date = min(pd.Timestamp(end_date), last_complete_month_end)

        # ensure start date is 1st day of month, end date is last day of month
        start_date = start_date.replace(day=1)
        end_date = end_date.replace(day=1) + pd.offsets.MonthEnd(0)

        # Convert dates to string format
        start_date, end_date = (
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

        variables = {
            "backend_name": "terrascope",
            "out_prefix": "patch-to-point",
            "out_extension": ".geoparquet",
            "start_date": start_date,
            "end_date": end_date,
            "ref_id": ref_id,
            "ground_truth_file": ground_truth_file,
            "epsg": epsg,
            "epsg_property": epsg_codes[epsg]["epsg_property"],
            "geometry_url": None,
            "h3l3_cell": "",
        }
        rows.append(pd.Series(variables))

    job_df = pd.DataFrame(rows)
    if job_df.empty:
        logger.warning(f"No EPSG codes found for ref_id {ref_id}")
        return job_df

    job_df["geometry_url"] = job_df["geometry_url"].astype("string")
    job_df["h3l3_cell"] = job_df["h3l3_cell"].astype("string")

    final_rows: List[dict] = []

    # Now find matching ground truth by querying RDM
    for _, row in job_df.iterrows():
        row = row.copy()
        logger.info(f"Processing EPSG {row.epsg} for REF_ID {row.ref_id}")

        # Get the ground truth in the patches
        # Note that we can work around RDM by specifically providing a ground truth file
        logger.info("Finding ground truth samples ...")
        (
            gdf,
            disable_s1,
            s1_patches,
            s1_sample_orbit_epsg,
            s1_sample_orbit_size,
        ) = get_label_points(
            row,
            ground_truth_file=row["ground_truth_file"],
            only_flagged_samples=only_flagged_samples,
        )
        gdf["ref_id"] = (
            row.ref_id
        )  # Overwrite due to current bug in automatic assignment

        if gdf.empty:
            logger.warning(f"No samples found for {row.epsg} and {row.ref_id}")
            continue

        logger.info(f"Found {len(gdf)} samples for {row.epsg} and {row.ref_id}")

        # Keep essential attributes only (H3 identifiers are part of the defaults)
        gdf = gdf[RDM_DEFAULT_COLUMNS].copy()

        total_samples = len(gdf)
        logger.info(f"Total samples for EPSG {row.epsg}: {total_samples}")

        has_h3_column = "h3_l3_cell" in gdf.columns
        if not has_h3_column:
            logger.warning(
                "Column 'h3_l3_cell' missing in ground truth samples; "
                f"falling back to single job for EPSG {row.epsg}."
            )

        split_applied = (
            max_samples_per_job is not None
            and total_samples > max_samples_per_job
            and has_h3_column
        )

        groups: List[Tuple[Optional[str], gpd.GeoDataFrame]]
        if split_applied:
            logger.info(
                f"EPSG {row.epsg} exceeds threshold with {total_samples} samples. "
                "Applying further split by H3 L3 cells."
            )
            missing_cells = gdf["h3_l3_cell"].isna().sum()

            if missing_cells > 0:
                logger.warning(
                    "H3 L3 cell information missing for some samples; "
                    f"skipping split for EPSG {row.epsg}."
                )
                split_applied = False
                groups = [(None, gdf.copy())]
            else:
                grouped: List[Tuple[Optional[str], gpd.GeoDataFrame]] = []
                for cell_value, cell_df in gdf.groupby("h3_l3_cell", dropna=False):
                    grouped.append((cell_value, cell_df.copy()))
                groups = grouped
                logger.info(
                    f"Splitting EPSG {row.epsg} into {len(groups)} H3 L3 cells "
                    f"(threshold {max_samples_per_job}, {total_samples} samples)."
                )
        else:
            groups = [(None, gdf.copy())]

        for cell_value, group_df in groups:
            cell_str = None if cell_value is None or pd.isna(cell_value) else str(cell_value)
            h3l3_cell_value = (
                cell_str if split_applied and cell_str is not None else ""
            )

            # Skip cells with no S1 patches in the current EPSG group: the
            # per-cell load_stac would return an empty datacube and the backend
            # crashes on an empty list. Such cells are picked up by the outer
            # EPSG loop iteration where their S1 patches actually live.
            if split_applied and cell_str is not None:
                if not _has_s1_patches_for_cell_epsg(
                    client, row.ref_id, cell_str, int(row.epsg)
                ):
                    logger.warning(
                        f"Cell {cell_str}: no S1 patches in EPSG {row.epsg}; "
                        "skipping (handled by the EPSG group where its S1 "
                        "patches exist)."
                    )
                    continue

            # Decide orbit buckets. If S1 is disabled altogether, emit a single
            # bucket with orbit_state=None. Otherwise classify each sample by
            # which orbit(s) cover it and split into per-orbit jobs.
            cell_label = f"EPSG {row.epsg}" + (
                f" cell {cell_str}" if cell_str is not None else ""
            )
            if disable_s1 or not s1_patches:
                orbit_buckets: List[Tuple[Optional[str], gpd.GeoDataFrame]] = [
                    (None, group_df.copy())
                ]
                logger.info(f"{cell_label}: S1 disabled -> 1 job")
            else:
                sample_orbit = _classify_samples_by_orbit(
                    group_df, s1_patches, s1_sample_orbit_size
                )
                n_asc_only = int((sample_orbit == "ASCENDING").sum())
                n_desc_only = int((sample_orbit == "DESCENDING").sum())
                n_both = int((sample_orbit == "BOTH").sum())
                n_none = int(sample_orbit.isna().sum())
                sample_orbit = _resolve_orbit_split(
                    sample_orbit,
                    s1_patches,
                    group_df,
                    TemporalContext(row.start_date, row.end_date),
                )
                orbit_buckets = [
                    (orbit, group_df.loc[idx].copy())
                    for orbit, idx in sample_orbit.groupby(sample_orbit).groups.items()
                ]
                bucket_summary = ", ".join(
                    f"{o}={len(sub)}" for o, sub in orbit_buckets
                )
                logger.info(
                    f"{cell_label}: {n_asc_only} ASC-only, {n_desc_only} DESC-only, "
                    f"{n_both} BOTH, {n_none} no-coverage -> "
                    f"{len(orbit_buckets)} job(s) ({bucket_summary})"
                )

            for orbit, sub_df in orbit_buckets:
                # Determine S2 tiles (per sub-group so each job has its own upload)
                logger.info(f"Finding S2 tiles for orbit {orbit} ...")
                original_crs = sub_df.crs
                sub_df = sub_df.to_crs(epsg=3857)
                sub_df["centroid"] = sub_df.geometry.centroid

                sub_df = gpd.sjoin(
                    sub_df.set_geometry("centroid"),
                    S2_GRID[["tile", "geometry"]].to_crs(epsg=3857),
                    predicate="intersects",
                ).drop(columns=["index_right", "centroid"])
                sub_df = sub_df.set_geometry("geometry").to_crs(original_crs)

                # Set back the valid_time in the geometry as string
                sub_df["valid_time"] = sub_df.valid_time.dt.strftime("%Y-%m-%d")

                # Add other attributes we want to keep in the result
                logger.info(
                    f"Determined start and end date: {row.start_date} - {row.end_date}"
                )
                sub_df["start_date"] = row.start_date
                sub_df["end_date"] = row.end_date
                sub_df["lat"] = sub_df.geometry.y
                sub_df["lon"] = sub_df.geometry.x

                # Reset index for certain openEO compatibility
                sub_df = sub_df.reset_index(drop=True)

                # Upload the geoparquet file to Artifactory
                logger.info(
                    f"Deploying geoparquet file ({len(sub_df)} samples, "
                    f"orbit={orbit}) to Artifactory ..."
                )
                collection_suffix = f"{row.epsg}"
                if split_applied and cell_str is not None:
                    collection_suffix = f"{collection_suffix}_{cell_str}"
                if orbit is not None:
                    collection_suffix = f"{collection_suffix}_{orbit[:3]}"

                url = upload_geoparquet_artifactory(
                    sub_df, ref_id, collection=collection_suffix
                )

                # Discover S1 EPSGs for this bucket's samples. When samples
                # near a UTM-zone boundary have their S1 patch indexed under
                # an EPSG different from the job's S2 EPSG, the S1 STAC filter
                # has to accept the union of those EPSGs or those samples lose
                # their S1 data.
                s1_epsgs_set: set = set()
                for sid in sub_df["sample_id"]:
                    orbit_map = s1_sample_orbit_epsg.get(sid, {})
                    if orbit is not None and orbit in orbit_map:
                        s1_epsgs_set.add(orbit_map[orbit])
                    else:
                        s1_epsgs_set.update(orbit_map.values())
                s1_epsgs_set.discard(None)
                if not s1_epsgs_set:
                    s1_epsgs_set.add(int(row.epsg))
                s1_epsgs_str = ",".join(str(e) for e in sorted(s1_epsgs_set))
                if s1_epsgs_set != {int(row.epsg)}:
                    logger.info(
                        f"{cell_label} (orbit={orbit}): S1 patches span EPSGs "
                        f"{sorted(s1_epsgs_set)}; S2 job EPSG is {row.epsg}. "
                        "S1 STAC filter will accept the union."
                    )

                job_row_dict = row.to_dict()
                job_row_dict["h3l3_cell"] = h3l3_cell_value
                job_row_dict["orbit_state"] = orbit
                job_row_dict["geometry_url"] = url
                job_row_dict["s1_epsgs"] = s1_epsgs_str
                final_rows.append(job_row_dict)

    final_job_df = pd.DataFrame(final_rows)
    if final_job_df.empty:
        logger.warning(f"No valid jobs created for ref_id {ref_id}")
        return final_job_df

    final_job_df["geometry_url"] = final_job_df["geometry_url"].astype("string")
    final_job_df["h3l3_cell"] = final_job_df["h3l3_cell"].astype("string")

    # Remove rows without geometry URL as indication for jobs to skip
    final_job_df = final_job_df[final_job_df["geometry_url"].notna()]

    return final_job_df


def create_job_patch_to_point_worldcereal(
    row: pd.Series,
    connection: openeo.Connection,
    provider,
    connection_provider,
    job_options: dict,
    period="month",
    optical_mask_method: Literal[
        "mask_scl_dilation", "mask_scl_raw_values"
    ] = "mask_scl_dilation",
):
    """Creates an OpenEO BatchJob from the given row information."""

    # Assume row has the following fields: backend, start_date, end_date, epsg, ref_id and geometry_url

    # s1_orbit_state = row.get(
    #     "orbit_state", "DESCENDING"
    # )  # default to DESCENDING, same as for inference workflow

    # Currently, empty datacubes for NetCDF collections are not supported
    # so we have to manually take care of the no-S1 case.
    s1_orbit_state = row.get("orbit_state") if not row.isnull()["orbit_state"] else None

    temporal_extent = TemporalContext(start_date=row.start_date, end_date=row.end_date)

    # Optional set of S1 EPSGs to accept (handles UTM-zone boundary samples
    # whose S1 patch is indexed under a different EPSG than their S2 patch).
    s1_epsgs_raw = row.get("s1_epsgs")
    if s1_epsgs_raw is not None and not (isinstance(s1_epsgs_raw, float) and pd.isna(s1_epsgs_raw)) and str(s1_epsgs_raw):
        s1_epsgs = [int(e) for e in str(s1_epsgs_raw).split(",") if e]
    else:
        s1_epsgs = None

    # Get preprocessed cube from patch extractions
    logger.info(f"Creating cube with compositing window: {period}")
    cube = worldcereal_preprocessed_inputs_from_patches(
        connection,
        temporal_extent=temporal_extent,
        ref_id=row["ref_id"],
        epsg=int(row["epsg"]),
        s1_orbit_state=s1_orbit_state,
        period=period,
        optical_mask_method=optical_mask_method,
        epsg_property=row.get("epsg_property", "proj:epsg"),
        s1_epsgs=s1_epsgs,
    )

    # Do spatial aggregation
    point_geometries = connection.load_url(
        url=str(row["geometry_url"]), format="Parquet"
    )
    cube = cube.aggregate_spatial(geometries=point_geometries, reducer="mean")

    return cube.create_job(
        title=f"WorldCereal patch-to-point extraction for: {row['ref_id']} and epsg: {row['epsg']} (period: {period})",
        out_format="Parquet",
        job_options=job_options,
    )


def post_job_action_point_worldcereal(parquet_file):
    """
    Perform post-processing on the extracted parquet file.

    This function cleans and validates the extracted data, removes invalid samples,
    and ensures the data conforms to the required schema.

    Parameters
    ----------
    parquet_file : str or Path
        Path to the parquet file to be processed.

    Returns
    -------
    None
    """

    logger.info(f"Running post-job action for: {parquet_file}")
    gdf = gpd.read_parquet(parquet_file)

    # Convert the dates to datetime format
    gdf["timestamp"] = pd.to_datetime(gdf["date"])
    gdf.drop(columns=["date"], inplace=True)

    if "S1-SIGMA0-VH" not in gdf.columns or "S1-SIGMA0-VV" not in gdf.columns:
        logger.warning(
            "S1 bands not found in the extracted data. "
            "This probably due to disabling of S1 in patch-to-point."
            " Filling with nodata values."
        )
        gdf["S1-SIGMA0-VH"] = 65535
        gdf["S1-SIGMA0-VV"] = 65535

    # Convert band dtype to uint16 (temporary fix)
    # TODO: remove this step when the issue is fixed on the OpenEO backend
    bands = [
        "S2-L2A-B02",
        "S2-L2A-B03",
        "S2-L2A-B04",
        "S2-L2A-B05",
        "S2-L2A-B06",
        "S2-L2A-B07",
        "S2-L2A-B08",
        "S2-L2A-B8A",
        "S2-L2A-B11",
        "S2-L2A-B12",
        "S1-SIGMA0-VH",
        "S1-SIGMA0-VV",
        "elevation",
        "slope",
        "AGERA5-PRECIP",
        "AGERA5-TMEAN",
    ]
    gdf[bands] = gdf[bands].fillna(65535).astype("uint16")

    # Remove samples where S1 and S2 are completely nodata
    cols = [c for c in gdf.columns if "S2" in c or "S1" in c]
    orig_sample_nr = len(gdf["sample_id"].unique())
    nodata_rows = (gdf[cols] == 65535).all(axis=1)
    all_nodata_per_sample = (
        gdf.assign(nodata=nodata_rows).groupby("sample_id")["nodata"].all()
    )
    valid_sample_ids = all_nodata_per_sample[~all_nodata_per_sample].index
    removed_samples = orig_sample_nr - len(valid_sample_ids)
    if removed_samples > 0:
        logger.warning(
            f"Removed {removed_samples} samples with all S1 and S2 bands as nodata."
        )
        gdf = gdf[gdf["sample_id"].isin(valid_sample_ids)]

    # Do some checks and perform corrections
    assert (
        len(gdf["ref_id"].unique()) == 1
    ), f"There are multiple ref_ids in the dataframe: {gdf['ref_id'].unique()}"
    ref_id = gdf["ref_id"].iloc[0]
    year = int(ref_id.split("_")[0])
    gdf["year"] = year

    # Make sure we remove the timezone information from the timestamp
    gdf["timestamp"] = gdf["timestamp"].dt.tz_localize(None)

    # Select required attributes and cast to dtypes
    required_attributes = copy.deepcopy(REQUIRED_ATTRIBUTES)
    required_attributes["ref_id"] = CategoricalDtype(categories=[ref_id], ordered=False)
    gdf = gdf[required_attributes.keys()]
    gdf = gdf.astype(required_attributes)

    gdf.to_parquet(parquet_file, index=False)


def worldcereal_preprocessed_inputs_from_patches(
    connection,
    temporal_extent,
    ref_id: str,
    epsg: int,
    s1_orbit_state: Optional[str] = None,
    period: Optional[str] = "month",
    optical_mask_method: Literal[
        "mask_scl_dilation", "mask_scl_raw_values"
    ] = "mask_scl_dilation",
    epsg_property: str = "proj:epsg",
    s1_epsgs: Optional[List[int]] = None,
):
    assert period in ["month", "dekad"], "period must be either 'month' or 'dekad'"

    # Build the EPSG filter value: integer for old-style `proj:epsg`,
    # string like "EPSG:32634" for new-style `proj:code`.
    epsg_filter_value = f"EPSG:{epsg}" if epsg_property == "proj:code" else epsg

    # Decide whether to scope the S1 STAC query by EPSG. The openEO backend's
    # property filter only supports {eq, lte, gte, array_contains}, so when
    # this job needs S1 patches from multiple UTM zones (boundary samples),
    # we drop the EPSG filter altogether and rely on resample_spatial below
    # to bring everything into the target EPSG before merging with S2.
    s1_epsg_values = sorted(set(s1_epsgs)) if s1_epsgs else [epsg]
    multi_s1_epsg = len(s1_epsg_values) > 1
    single_s1_epsg_value = (
        f"EPSG:{s1_epsg_values[0]}" if epsg_property == "proj:code" else s1_epsg_values[0]
    )

    # TODO: move preprocessing to separate functions 'preprocess_cube_x(cube: openeo.DataCube) -> openeo.DataCube' which will be the same across the different extraction workflows
    s1_stac_property_filter = {
        "ref_id": lambda x: eq(x, ref_id),
        "sat:orbit_state": lambda x: eq(x, s1_orbit_state),
    }
    if not multi_s1_epsg:
        s1_stac_property_filter[epsg_property] = (
            lambda x, _v=single_s1_epsg_value: eq(x, _v)
        )

    s2_stac_property_filter = {
        "ref_id": lambda x: eq(x, ref_id),
        epsg_property: lambda x: eq(x, epsg_filter_value),
    }

    if s1_orbit_state is not None:
        s1_raw = connection.load_stac(
            url=STAC_ENDPOINT_S1,
            properties=s1_stac_property_filter,
            temporal_extent=[temporal_extent.start_date, temporal_extent.end_date],
            bands=["S1-SIGMA0-VH", "S1-SIGMA0-VV"],
        )
        s1_raw.result_node().update_arguments(featureflags={"allow_empty_cube": True})
        if multi_s1_epsg:
            # Patches loaded from multiple UTM zones must be reprojected onto a
            # single CRS before merge_cubes with S2 (which lives in `epsg`).
            s1_raw = s1_raw.resample_spatial(
                resolution=10.0, projection=epsg, method="bilinear"
            )
        s1 = decompress_backscatter_uint16(backend_context=None, cube=s1_raw)
        s1 = mean_compositing(s1, period=period)
        s1 = compress_backscatter_uint16(backend_context=None, cube=s1)
    else:
        logger.warning("No S1 orbit state provided, S1 extraction will be disabled.")

    s2_raw = connection.load_stac(
        url=STAC_ENDPOINT_S2,
        properties=s2_stac_property_filter,
        temporal_extent=[temporal_extent.start_date, temporal_extent.end_date],
        bands=S2_BANDS,
    ).filter_bands(S2_BANDS_SELECTED)

    def optimized_mask_precomputed(input: ProcessBuilder):
        """
        To be used as a callback to apply_dimension on the band dimension.
        It's an optimized way of masking, if the mask is already present in the cube.
        """
        mask_band = input.array_element(label="S2-L2A-SCL_DILATED_MASK")
        return if_(mask_band != 1, input)

    def optimized_mask_raw_scl_values(input: ProcessBuilder):
        """
        Using raw SCL values to mask invalid pixels and get less aggressive masking compared to precomputed masks with large erode/dilate radius.
        Valid pixels are those with SCL values not in [0,1,3,8,9,10,11].
        0: No data
        1: Saturated or defective
        3: Cloud shadows
        8: Medium probability cloud
        9: High probability cloud
        10: Thin cirrus
        11: Snow or ice
        """
        mask_band = input.array_element(label="S2-L2A-SCL")
        invalid = or_(mask_band == 0, mask_band == 1)
        invalid = or_(invalid, mask_band == 3)
        invalid = or_(invalid, mask_band == 8)
        invalid = or_(invalid, mask_band == 9)
        invalid = or_(invalid, mask_band == 10)
        invalid = or_(invalid, mask_band == 11)
        return if_(not_(invalid), input)

    if optical_mask_method == "mask_scl_dilation":
        s2 = s2_raw.apply_dimension(
            dimension="bands", process=optimized_mask_precomputed
        )
    elif optical_mask_method == "mask_scl_raw_values":
        s2 = s2_raw.apply_dimension(
            dimension="bands", process=optimized_mask_raw_scl_values
        )
    else:
        raise ValueError(
            f"Unknown optical_mask_method: {optical_mask_method}. "
            f"Supported methods are 'mask_scl_dilation' and 'mask_scl_raw_values'."
        )

    s2 = median_compositing(s2, period=period)
    s2 = s2.filter_bands(S2_BANDS_SELECTED[:-2])
    s2 = s2.linear_scale_range(0, 65534, 0, 65534)

    dem_raw = connection.load_collection("COPERNICUS_30", bands=["DEM"])
    dem_raw = dem_raw.resample_spatial(
        resolution=10.0, projection=epsg, method="bilinear"
    )
    dem = dem_raw.min_time()
    dem = dem.rename_labels(dimension="bands", target=["elevation"], source=["DEM"])

    slope = connection.load_stac(
        STAC_ENDPOINT_SLOPE_TERRASCOPE,
        bands=["Slope"],
    ).rename_labels(dimension="bands", target=["slope"])
    slope = slope.resample_spatial(resolution=10.0, projection=epsg, method="bilinear")
    # Client fix for CDSE, the openeo client might be unsynchronized with
    # the backend.
    if "t" not in slope.metadata.dimension_names():
        slope.metadata = slope.metadata.add_dimension("t", "2020-01-01", "temporal")
    slope = slope.min_time()

    copernicus = slope.merge_cubes(dem)
    copernicus = copernicus.linear_scale_range(0, 65534, 0, 65534)

    if period == "month":
        # Load precomposited monthly meteo data
        meteo_raw = connection.load_stac(
            url=STAC_ENDPOINT_MONTHLY_METEO,
            temporal_extent=[temporal_extent.start_date, temporal_extent.end_date],
            bands=["temperature-mean", "precipitation-flux"],
        )
    elif period == "dekad":
        # Load precomposited dekadal meteo data
        meteo_raw = connection.load_stac(
            url=STAC_ENDPOINT_DEKADAL_METEO,
            temporal_extent=[temporal_extent.start_date, temporal_extent.end_date],
            bands=["temperature-mean", "precipitation-flux"],
        )

    meteo = meteo_raw.resample_spatial(
        resolution=10.0, projection=epsg, method="bilinear"
    )
    meteo = meteo.rename_labels(
        dimension="bands",
        target=["AGERA5-TMEAN", "AGERA5-PRECIP"],
    )

    if s1_orbit_state is None:
        # If no S1 orbit state is provided, we disable S1
        cube = s2
    else:
        cube = s2.merge_cubes(s1)
    cube = cube.merge_cubes(meteo)
    cube = cube.merge_cubes(copernicus)

    return cube
