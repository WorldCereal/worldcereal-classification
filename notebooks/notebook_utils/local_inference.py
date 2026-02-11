"""Perform cropland and croptype mapping inference using local execution of UDFs.

Make sure you test this script on Python version 3.9+, and have worldcereal
dependencies installed with the presto wheel file and its dependencies.

This script tests both cropland and croptype mapping workflows by calling
the UDF functions directly without running batch jobs on OpenEO.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import pandas as pd
import xarray as xr
from dateutil.parser import parse
from loguru import logger
from prometheo.predictors import NODATAVALUE
from pyproj import CRS

from worldcereal.openeo.inference import SeasonalInferenceEngine
from worldcereal.openeo.parameters import DEFAULT_SEASONAL_MODEL_URL


def subset_ds_temporally(
    ds, season, allow_partial: bool = False, nodata_value: int = NODATAVALUE
):
    """
    Subsets a dataset temporally based on a given season.
    This function extracts a subset of the dataset `ds` that matches the
    temporal context defined by the `season`. The `season` is expected to
    provide a start and end date, and the function ensures that the subset
    spans a complete 12-month window, even if the season wraps over the
    end of the year.
    Parameters:
    ----------
    ds : xarray.Dataset
        The input dataset containing a time dimension `t` to be subset.
    season : TemporalContext
        An object containing `start_date` and `end_date` as strings,
        representing the start and end of the season.
    allow_partial : bool, optional
        If False (default), behaves strictly and raises ValueError when no
        complete 12‑month window is found. If True, returns a 12‑month
        window for the earliest candidate year, filling missing timestamps
        with nodata_value and logging a warning.
    nodata_value : int, optional
        Value used to fill missing timestamps when allow_partial is True.
        Default is NODATAVALUE defined by Prometheo (65535).

    Returns:
    -------
    xarray.Dataset
        A subset of the input dataset `ds` that matches the 12-month
        temporal window defined by the `season`. If allow_partial is True
        and a full window is not available, missing months are inserted
        and filled with nodata_value.

    Raises:
    ------
    ValueError
        If no matching 12-month window is found and allow_partial=False.
    Notes:
    -----
    - The function assumes that the `t` dimension in the dataset is
      convertible to a pandas DatetimeIndex.
    - If the season wraps over the end of the year (e.g., starts in
      December and ends in February), the function handles this case
      by constructing the appropriate month sequence.
    """

    # Parse season (already a TemporalContext with string dates)
    start_dt = parse(season.start_date)
    end_dt = parse(season.end_date)

    # Does the season wrap over year end?
    wrap = (end_dt.month, end_dt.day) <= (start_dt.month, start_dt.day)

    # Month sequence for one full season (12 months)
    months = (
        (list(range(start_dt.month, 13)) + list(range(1, end_dt.month + 1)))
        if wrap
        else list(range(start_dt.month, end_dt.month + 1))
    )

    t_index = ds.t.to_index()

    # Find the first year in ds that can provide the complete 12‑month window
    selected = None
    for y in sorted(set(t_index.year)):
        if wrap:
            expected = [
                pd.Timestamp(datetime(y if m >= start_dt.month else y + 1, m, 1))
                for m in months
            ]
        else:
            expected = [pd.Timestamp(datetime(y, m, 1)) for m in months]
        if all(ts in t_index for ts in expected):
            selected = ds.sel(t=expected)
            break

    if selected is None:
        if not allow_partial:
            raise ValueError(
                "No matching 12-month window in ds for the season pattern."
            )
        # Partial mode: use earliest year to build expected sequence and fill gaps
        y = min(t_index.year)
        if wrap:
            expected = [
                pd.Timestamp(datetime(y if m >= start_dt.month else y + 1, m, 1))
                for m in months
            ]
        else:
            expected = [pd.Timestamp(datetime(y, m, 1)) for m in months]
        present = [ts for ts in expected if ts in t_index]
        missing = [ts for ts in expected if ts not in t_index]
        logger.warning(
            f"Partial temporal subset: missing {len(missing)} months "
            f"({', '.join([ts.strftime('%Y-%m') for ts in missing])}); "
            f"filling with nodata_value={nodata_value}."
        )
        # Reindex will insert missing timestamps with nodata_value
        selected = ds.sel(t=present).reindex(t=expected, fill_value=nodata_value)

    return selected


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
) -> None:
    """Save classification DataArray as GeoTIFF.
    Parameters
    ----------
    classification : xr.DataArray
        Classification DataArray to save.
    epsg : int
        EPSG code for the CRS.
    out_path : Path
        Output path for the GeoTIFF file."""

    # ignore import error for rioxarray if not used
    import rioxarray  # noqa: F401

    logger.info(f"Saving classification to GeoTIFF at: {out_path}")

    classification.rio.set_crs(f"epsg:{epsg}", inplace=True)
    classification.rio.to_raster(out_path)
