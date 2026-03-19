"""Perform cropland and croptype mapping inference using local execution of UDFs.

Make sure you test this script on Python version 3.9+, and have worldcereal
dependencies installed with the presto wheel file and its dependencies.

This script tests both cropland and croptype mapping workflows by calling
the UDF functions directly without running batch jobs on OpenEO.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
import xarray as xr
from dateutil.parser import parse
from loguru import logger
from prometheo.predictors import NODATAVALUE
from pyproj import CRS
from worldcereal.openeo.inference import (SeasonalInferenceEngine,
                                          get_expected_timesteps_from_artifact)
from worldcereal.openeo.parameters import DEFAULT_SEASONAL_MODEL_URL
from worldcereal.utils.models import load_model_artifact


def _parse_season_arg(season):
    """Extract (start_date_str, end_date_str) from various season representations.

    Accepts:
    - A ``TemporalContext``-like object with ``.start_date`` / ``.end_date``
    - A ``(start_str, end_str)`` tuple or list
    - A ``dict`` with keys ``start_date`` and ``end_date``
    """
    if hasattr(season, "start_date") and hasattr(season, "end_date"):
        return str(season.start_date), str(season.end_date)
    if isinstance(season, (tuple, list)) and len(season) == 2:
        return str(season[0]), str(season[1])
    if isinstance(season, dict):
        return str(season["start_date"]), str(season["end_date"])
    raise TypeError(
        f"Cannot interpret season argument of type {type(season).__name__}. "
        "Expected a TemporalContext, a (start, end) tuple, or a dict."
    )


def subset_ds_temporally(
    ds,
    season,
    min_coverage: float = 1.0,
    nodata_value: int = NODATAVALUE,
    max_timesteps: int = 12,
    prefer_tail: bool = True,
):
    """
    Subsets a dataset temporally based on a given season.

    This function extracts a subset of the dataset ``ds`` that matches the
    temporal context defined by ``season``.  The season provides a start and
    end date; the function ensures that the subset spans at most
    ``max_timesteps`` monthly slots, even if the raw season window is wider
    (e.g. 13 months for an annual season).

    When the season window contains more than ``max_timesteps`` months the
    window is trimmed.  By default (``prefer_tail=True``) the **first**
    month(s) are dropped — keeping the later months which tend to carry
    more discriminative crop phenology.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset containing a time dimension ``t`` to be subset.
    season : TemporalContext | tuple[str, str] | dict
        Season specification.  Accepted forms:

        * A ``TemporalContext`` (or any object with ``.start_date`` /
          ``.end_date`` string attributes).
        * A ``(start_date_str, end_date_str)`` tuple.
        * A ``dict`` with keys ``"start_date"`` and ``"end_date"``.
    min_coverage : float, optional
        Minimum fraction of the ``max_timesteps`` monthly slots that must
        be present in the dataset (default 1.0 = all months required).
        When the actual coverage is below this threshold a ``ValueError``
        is raised.  When coverage is at or above the threshold but some
        months are still missing, they are filled with *nodata_value*.
        Set to 0.0 to accept any coverage.  Analogous to
        ``eval_min_season_coverage`` in the training pipeline.
    nodata_value : int, optional
        Fill value for missing timestamps.  Default is ``NODATAVALUE``
        (65535).
    max_timesteps : int, optional
        Maximum number of monthly slots to keep (default 12).
    prefer_tail : bool, optional
        When the month sequence exceeds ``max_timesteps``, keep the **last**
        months (drop the head) if *True* (default), or keep the **first**
        months (drop the tail) if *False*.

    Returns
    -------
    xarray.Dataset
        Temporal subset with at most ``max_timesteps`` monthly slots.

    Raises
    ------
    ValueError
        If the fraction of available months is below ``min_coverage``.
    """

    start_str, end_str = _parse_season_arg(season)
    start_dt = parse(start_str)
    end_dt = parse(end_str)

    # Does the season wrap over year end?
    wrap = (end_dt.month, end_dt.day) <= (start_dt.month, start_dt.day)

    # Full month sequence for the season (may be >12 for annual seasons)
    months = (
        (list(range(start_dt.month, 13)) + list(range(1, end_dt.month + 1)))
        if wrap
        else list(range(start_dt.month, end_dt.month + 1))
    )

    # Trim to max_timesteps, preferring tail or head as requested
    if len(months) > max_timesteps:
        dropped = len(months) - max_timesteps
        if prefer_tail:
            trimmed = months[dropped:]
            logger.info(
                f"Season has {len(months)} months; dropping first {dropped} "
                f"month(s) {months[:dropped]} to keep {max_timesteps} (prefer_tail=True)."
            )
        else:
            trimmed = months[:max_timesteps]
            logger.info(
                f"Season has {len(months)} months; dropping last {dropped} "
                f"month(s) {months[max_timesteps:]} to keep {max_timesteps} (prefer_tail=False)."
            )
        months = trimmed

    t_index = ds.t.to_index()

    # Find the first year in ds that can provide the complete window
    selected = None
    for y in sorted(set(t_index.year)):
        if wrap:
            expected = [
                pd.Timestamp(datetime(y if m >= months[0] else y + 1, m, 1))
                for m in months
            ]
        else:
            expected = [pd.Timestamp(datetime(y, m, 1)) for m in months]
        if all(ts in t_index for ts in expected):
            selected = ds.sel(t=expected)
            break

    if selected is None:
        # Partial mode: find best year and check coverage against threshold
        y = min(t_index.year)
        if wrap:
            expected = [
                pd.Timestamp(datetime(y if m >= months[0] else y + 1, m, 1))
                for m in months
            ]
        else:
            expected = [pd.Timestamp(datetime(y, m, 1)) for m in months]
        present = [ts for ts in expected if ts in t_index]
        missing = [ts for ts in expected if ts not in t_index]
        coverage = len(present) / len(expected) if expected else 0.0
        if coverage < min_coverage:
            raise ValueError(
                f"Temporal coverage {coverage:.0%} ({len(present)}/{len(expected)} months) "
                f"is below min_coverage={min_coverage:.0%} for the season pattern."
            )
        if missing:
            logger.warning(
                f"Partial temporal subset: coverage={coverage:.0%} "
                f"({len(present)}/{len(expected)} months present); "
                f"missing {', '.join(ts.strftime('%Y-%m') for ts in missing)}; "
                f"filling with nodata_value={nodata_value}."
            )
        # Reindex will insert missing timestamps with nodata_value
        selected = ds.sel(t=present).reindex(t=expected, fill_value=nodata_value)

    return selected


def compute_temporal_subset_window(
    season_windows: Mapping[str, object],
) -> Optional[object]:
    """Derive a single season window that covers all requested seasons.

    Given a mapping of ``{season_id: (start_date_str, end_date_str)}``,
    this function returns a ``(earliest_start, latest_end)`` tuple that
    spans all seasons.  The caller should then pass this to
    ``subset_ds_temporally`` which will handle trimming to 12 months.

    Returns *None* when ``season_windows`` is empty or *None*.
    """
    if not season_windows:
        return None

    earliest_start: Optional[pd.Timestamp] = None
    latest_end: Optional[pd.Timestamp] = None

    for _sid, window in season_windows.items():
        start_str, end_str = _parse_season_arg(window)
        s = pd.Timestamp(start_str)
        e = pd.Timestamp(end_str)
        if earliest_start is None or s < earliest_start:
            earliest_start = s
        if latest_end is None or e > latest_end:
            latest_end = e

    if earliest_start is None or latest_end is None:
        return None

    return (earliest_start.strftime("%Y-%m-%d"), latest_end.strftime("%Y-%m-%d"))


def _clamp_season_windows_to_ds(
    season_windows: Mapping[str, object],
    ds: xr.Dataset,
) -> Dict[str, Tuple[str, str]]:
    """Clamp season windows so they don't extend beyond the dataset timestamps.

    After ``subset_ds_temporally`` trims the dataset, the original season
    windows may reference months that were dropped.  This function clamps
    each window's start/end to the dataset's actual timestamp range so that
    ``_build_masks_from_windows`` does not reject them.
    """
    t_vals = pd.to_datetime(ds.t.values)
    ds_start = t_vals.min()
    ds_end = t_vals.max()

    clamped: Dict[str, Tuple[str, str]] = {}
    for sid, window in season_windows.items():
        start_str, end_str = _parse_season_arg(window)
        s = pd.Timestamp(start_str)
        e = pd.Timestamp(end_str)
        clamped_s = max(s, ds_start)
        clamped_e = min(e, ds_end)
        if clamped_e < clamped_s:
            raise ValueError(
                f"Season '{sid}' window ({s.strftime('%Y-%m-%d')}, {e.strftime('%Y-%m-%d')}) "
                f"does not overlap with subset timestamps "
                f"({ds_start.strftime('%Y-%m-%d')} to {ds_end.strftime('%Y-%m-%d')}). "
                f"Check that the season windows match the available data."
            )
        elif clamped_s != s or clamped_e != e:
            logger.info(
                f"Clamped season '{sid}' window from "
                f"({s.strftime('%Y-%m-%d')}, {e.strftime('%Y-%m-%d')}) to "
                f"({clamped_s.strftime('%Y-%m-%d')}, {clamped_e.strftime('%Y-%m-%d')}) "
                f"to match subset timestamps."
            )
        clamped[sid] = (
            clamped_s.strftime("%Y-%m-%d"),
            clamped_e.strftime("%Y-%m-%d"),
        )
    return clamped


def reconstruct_dataset(arr: xr.DataArray, ds: xr.Dataset) -> xr.Dataset:
    """Reconstruct CRS attributes."""
    crs_attrs = ds["crs"].attrs
    x = ds.coords.get("x", None)
    y = ds.coords.get("y", None)

    # Build dataset with bands as separate variables
    new_ds = arr.assign_coords(bands=arr.bands.astype(str)).to_dataset(dim="bands")

    # Reset the coordinates
    new_ds = new_ds.assign_coords(x=x)
    new_ds["x"].attrs.setdefault("standard_name", "projection_x_coordinate")
    new_ds["x"].attrs.setdefault("units", "m")

    new_ds = new_ds.assign_coords(y=y)
    new_ds["y"].attrs.setdefault("standard_name", "projection_y_coordinate")
    new_ds["y"].attrs.setdefault("units", "m")

    # Assign CRS attributes to all data variables
    crs_name = "spatial_ref"
    new_ds[crs_name] = xr.DataArray(0, attrs=crs_attrs)

    for v in new_ds.data_vars:
        new_ds[v].attrs["grid_mapping"] = crs_name

    return new_ds


def build_postprocess_spec(
    *,
    enabled: bool,
    method: Optional[str],
    kernel_size: Optional[int],
) -> Optional[Dict[str, object]]:
    requested = enabled or method is not None or kernel_size is not None
    if not requested:
        return None

    spec: Dict[str, object] = {
        "enabled": enabled or method is not None or kernel_size is not None
    }
    if method:
        spec["method"] = method
    if kernel_size is not None:
        spec["kernel_size"] = kernel_size
    return spec


def attach_crs_metadata(
    result: Union[xr.Dataset, xr.DataArray], template: xr.Dataset
) -> Union[xr.Dataset, xr.DataArray]:
    crs_var = template.get("crs")
    coords = {}
    if "x" in template.coords:
        coords["x"] = template.coords["x"]
    if "y" in template.coords:
        coords["y"] = template.coords["y"]
    out = result.assign_coords(**coords) if coords else result

    if crs_var is None:
        return out

    crs_name = "spatial_ref"
    crs_data = xr.DataArray(0, attrs=crs_var.attrs)

    if isinstance(out, xr.Dataset):
        out[crs_name] = crs_data
        for name in out.data_vars:
            out[name].attrs.setdefault("grid_mapping", crs_name)
            out[name].encoding.pop("NETCDF_DIM_bands_VALUES", None)
        return out

    out = out.assign_coords({crs_name: crs_data})
    out.attrs.setdefault("grid_mapping", crs_name)
    return out


def run_seasonal_inference(
    ds_or_path: Union[xr.Dataset, str, Path],
    *,
    seasonal_model_zip: Union[str, Path] = DEFAULT_SEASONAL_MODEL_URL,
    landcover_head_zip: Optional[Union[str, Path]] = None,
    croptype_head_zip: Optional[Union[str, Path]] = None,
    season_ids: Optional[Sequence[str]] = None,
    season_windows: Optional[Mapping[str, Sequence[object]]] = None,
    export_class_probabilities: bool = False,
    enable_cropland_head: bool = True,
    enable_croptype_head: bool = True,
    enforce_cropland_gate: bool = True,
    batch_size: int = 2048,
    cache_root: Optional[Path] = None,
    device: str = "cpu",
    cropland_postprocess: Optional[Mapping[str, Any]] = None,
    croptype_postprocess: Optional[Mapping[str, Any]] = None,
    epsg: Optional[int] = None,
    fillna_value: int = NODATAVALUE,
    as_dataset: bool = True,
) -> Union[xr.Dataset, xr.DataArray]:
    """Run seasonal cropland/croptype inference locally with a single entrypoint."""

    if isinstance(ds_or_path, (str, Path)):
        ds = xr.open_dataset(ds_or_path)
    else:
        ds = ds_or_path

    ds = ds.fillna(fillna_value)

    # --- Resolve expected timesteps from model artifact ---
    artifact = load_model_artifact(seasonal_model_zip, cache_root=cache_root)
    expected_timesteps = get_expected_timesteps_from_artifact(artifact)
    if expected_timesteps is None:
        logger.warning(
            "Could not determine expected timesteps from model artifact; "
            "defaulting to 12."
        )
        expected_timesteps = 12
    logger.info(f"Model expects {expected_timesteps} timesteps.")

    # --- Temporal subsetting ---
    # Ensure the dataset is trimmed to match the model's training regime.
    # Feeding more timesteps than the model was trained with produces
    # out-of-distribution embeddings and garbage predictions.
    if season_windows:
        union_window = compute_temporal_subset_window(season_windows)
        if union_window is not None:
            t_count = ds.sizes.get("t", 0)
            if t_count > expected_timesteps:
                logger.info(
                    f"Temporal subsetting: input has {t_count} timesteps; "
                    f"subsetting to ≤{expected_timesteps} using union season window {union_window}."
                )
                ds = subset_ds_temporally(
                    ds,
                    union_window,
                    min_coverage=0.5,
                    nodata_value=fillna_value,
                    max_timesteps=expected_timesteps,
                    prefer_tail=True,
                )
                logger.info(
                    f"After temporal subset: {ds.sizes.get('t', 0)} timesteps "
                    f"({pd.to_datetime(ds.t.values[0]).strftime('%Y-%m')} to "
                    f"{pd.to_datetime(ds.t.values[-1]).strftime('%Y-%m')})."
                )
                # Clamp season_windows to the subset's actual timestamp range
                # so that _build_masks_from_windows does not reject windows
                # that now fall outside the trimmed data.
                season_windows = _clamp_season_windows_to_ds(season_windows, ds)
    elif ds.sizes.get("t", 0) > expected_timesteps:
        raise ValueError(
            f"Input has {ds.sizes['t']} timesteps but no season_windows were "
            f"provided for temporal subsetting.  Model expects {expected_timesteps} "
            f"timesteps; provide season_windows or pre-subset the data."
        )

    if epsg is None:
        if "crs" not in ds:
            raise ValueError(
                "EPSG not provided and dataset is missing a 'crs' variable."
            )
        epsg = CRS.from_wkt(ds.crs.attrs["spatial_ref"]).to_epsg()

    arr = ds.drop_vars("crs", errors="ignore").astype("uint16").to_array(dim="bands")

    engine = SeasonalInferenceEngine(
        seasonal_model_zip=seasonal_model_zip,
        landcover_head_zip=landcover_head_zip,
        croptype_head_zip=croptype_head_zip,
        cache_root=cache_root,
        device=device,
        batch_size=batch_size,
        season_ids=season_ids,
        export_class_probabilities=export_class_probabilities,
        enable_cropland_head=enable_cropland_head,
        enable_croptype_head=enable_croptype_head,
        cropland_postprocess=cropland_postprocess,
        croptype_postprocess=croptype_postprocess,
    )

    result = engine.infer(
        arr,
        epsg=epsg,
        season_windows=season_windows,
        season_ids=season_ids,
        enforce_cropland_gate=enforce_cropland_gate,
    )

    if device == "cuda" and hasattr(result, "cpu"):
        result = result.cpu()

    if as_dataset and isinstance(result, xr.DataArray):
        result = xr.Dataset(
            {
                str(b): result.sel(bands=b).drop_vars("bands")
                for b in result.bands.values
            }
        )

    return attach_crs_metadata(result, ds)


def classification_to_geotiff(
    classification: xr.DataArray,
    epsg: int,
    out_path: Path,
    class_map: Optional[Mapping[int, str]] = None,
) -> None:
    """Save classification DataArray as GeoTIFF.
    Parameters
    ----------
    classification : xr.DataArray
        Classification DataArray to save.
    epsg : int
        EPSG code for the CRS.
    out_path : Path
        Output path for the GeoTIFF file.
    class_map : Optional[Mapping[int, str]]
        Optional integer-to-class mapping (e.g. head_config["classes_list"]).
        Stored as metadata tags on croptype classification bands."""

    import json

    import rasterio
    # ignore import error for rioxarray if not used
    import rioxarray  # noqa: F401

    logger.info(f"Saving classification to GeoTIFF at: {out_path}")

    band_names = [str(b) for b in classification.bands.values]
    classification.rio.set_crs(f"epsg:{epsg}", inplace=True)
    classification.rio.to_raster(out_path)

    class_map_json = None
    if class_map:
        class_map_json = json.dumps({int(k): str(v) for k, v in class_map.items()})

    with rasterio.open(out_path, "r+") as dst:
        dst.descriptions = tuple(band_names)
        if class_map_json:
            # Store globally and on croptype classification bands for GIS tools.
            dst.update_tags(croptype_class_map=class_map_json)
            for idx, name in enumerate(band_names, start=1):
                if name.startswith("croptype_classification"):
                    dst.update_tags(idx, class_map=class_map_json)

    logger.info("Classification saved!")


def _reproject_bands_from_raster(
    src_path: Union[str, Path],
    out_path: Union[str, Path],
    band_list: list[int],
    resampling: str = "nearest",
    target_epsg: str = "EPSG:3857",
) -> None:
    """Reproject selected raster bands into a new GeoTIFF.

    Parameters
    ----------
    src_path : Union[str, Path]
        Source raster path.
    out_path : Union[str, Path]
        Output raster path.
    band_list : list[int]
        1-based band indices to extract and reproject in the given order.
    resampling : str, optional
        Resampling method name, by default "nearest".
    target_epsg : str, optional
        Target CRS (e.g. "EPSG:3857"), by default "EPSG:3857".
    """

    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "lanczos": Resampling.lanczos,
    }
    resampling_enum = resampling_map.get(resampling, Resampling.nearest)

    src_path = Path(src_path)
    out_path = Path(out_path)

    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError("Source raster is missing CRS metadata.")

        dst_crs = target_epsg
        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
        )

        meta = src.meta.copy()
        meta.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "count": len(band_list),
            }
        )

        with rasterio.open(out_path, "w", **meta) as dst:
            new_band = 1
            for band_idx in band_list:
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, new_band),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling_enum,
                )
                new_band += 1

    logger.info(f"Exported bands {band_list} to {target_epsg}: {out_path}")


def isolate_cropland_croptype_products(
    src_path: Union[str, Path],
    cropland_included: bool,
    resampling: str = "nearest",
    target_epsg: str = "EPSG:3857",
) -> dict:
    """Extract cropland/croptype products from a multi-band inference raster.

    When cropland is included, bands 1-2 are cropland (class + probability)
    and bands 4-5 are croptype (class + probability). When cropland is
    not included, bands 1-2 represent croptype instead.
    """

    src_path = Path(src_path)
    outdir = src_path.parent

    products = {}

    # First two bands are either cropland or crop type product.
    if cropland_included:
        dst_path = outdir / f"{src_path.stem}_cropland.tif"
        products["cropland"] = dst_path
    else:
        dst_path = outdir / f"{src_path.stem}_croptype.tif"
        products["croptype"] = dst_path

    _reproject_bands_from_raster(
        src_path,
        dst_path,
        [1, 2],
        resampling=resampling,
        target_epsg=target_epsg,
    )

    # If there was a cropland product, export croptype from bands 4 and 5.
    if cropland_included:
        dst_path = outdir / f"{src_path.stem}_croptype.tif"
        products["croptype"] = dst_path
        _reproject_bands_from_raster(
            src_path,
            dst_path,
            [4, 5],
            resampling=resampling,
            target_epsg=target_epsg,
        )

    return products
