"""Integration test for `worldcereal_preprocessed_inputs` sensor-combination flags.

This script builds, submits and validates one openEO batch job per valid
combination of the `disable_s1` / `disable_s2` / `disable_meteo` / `disable_dem`
flags accepted by `worldcereal.openeo.preprocessing.worldcereal_preprocessed_inputs`.

For each combination, the resulting NetCDF is downloaded and inspected to make
sure it contains *exactly* the bands that correspond to the enabled sensors
(no missing bands, no unexpected leftovers from a disabled sensor).

WARNING
-------
This submits real batch jobs to the CDSE openEO backend and will consume
processing credits on your account. Keep the default small AOI / short
temporal extent, or pass `--dry-run` to only build and validate the process
graphs locally (no job is submitted).

Usage
-----
    python test_preprocessed_inputs_combinations.py
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

import xarray as xr
from loguru import logger
from openeo_gfmap import (
    Backend,
    BackendContext,
    BoundingBoxExtent,
    FetchType,
    TemporalContext,
)
from openeo_gfmap.backend import cdse_connection

from worldcereal.openeo.preprocessing import (
    WORLDCEREAL_BANDS,
    worldcereal_preprocessed_inputs,
)

# -----------------------------------------------------------------------------
# Defaults: kept intentionally small/short to minimize processing cost.
# -----------------------------------------------------------------------------
DEFAULT_SPATIAL_EXTENT = BoundingBoxExtent(
    west=44.432274, south=51.317362, east=44.462274, north=51.337362, epsg=4326
)
DEFAULT_TEMPORAL_EXTENT = TemporalContext("2021-06-01", "2021-07-31")
DEFAULT_TILE_SIZE = 64

DEFAULT_JOB_OPTIONS = {
    "driver-memory": "2g",
    "executor-memory": "1g",
    "executor-memoryOverhead": "1g",
    "python-memory": "2g",
}


@dataclass(frozen=True)
class Combination:
    """One valid (disable_s1, disable_s2, disable_meteo, disable_dem) setting."""

    disable_s1: bool
    disable_s2: bool
    disable_meteo: bool
    disable_dem: bool

    @property
    def name(self) -> str:
        parts = []
        for sensor in ("s1", "s2", "meteo", "dem"):
            enabled = not getattr(self, f"disable_{sensor}")
            parts.append(f"{sensor}-{'on' if enabled else 'off'}")
        return "_".join(parts)

    def expected_bands(self) -> Set[str]:
        expected: Set[str] = set()
        if not self.disable_s2:
            expected.update(WORLDCEREAL_BANDS["SENTINEL2"])
        if not self.disable_s1:
            expected.update(WORLDCEREAL_BANDS["SENTINEL1"])
        if not self.disable_dem:
            expected.update(WORLDCEREAL_BANDS["DEM"])
        if not self.disable_meteo:
            expected.update(WORLDCEREAL_BANDS["METEO"])
        return expected


def all_valid_combinations() -> List[Combination]:
    """Enumerate all 16 boolean combinations, dropping the 4 invalid ones
    where both S1 and S2 are disabled (no reference grid would remain)."""
    combos = []
    for s1, s2, meteo, dem in itertools.product([False, True], repeat=4):
        if s1 and s2:
            continue
        combos.append(
            Combination(disable_s1=s1, disable_s2=s2, disable_meteo=meteo, disable_dem=dem)
        )
    return combos


def build_cube(
    combo: Combination,
    connection,
    backend_context: BackendContext,
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    tile_size: int,
):
    """Build the preprocessed-inputs process graph for a given combination."""
    cube = worldcereal_preprocessed_inputs(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        fetch_type=FetchType.TILE,
        disable_s1=combo.disable_s1,
        disable_s2=combo.disable_s2,
        disable_meteo=combo.disable_meteo,
        disable_dem=combo.disable_dem,
        tile_size=tile_size,
    )
    return cube.filter_bbox(dict(spatial_extent))


def check_bands(nc_path: Path, expected_bands: Set[str]) -> bool:
    """Open a resulting NetCDF and check that its bands exactly match expectations."""
    ds = xr.open_dataset(nc_path)
    actual_bands = {var for var in ds.data_vars if var != "crs"}
    ds.close()

    missing = expected_bands - actual_bands
    unexpected = actual_bands - expected_bands

    if missing:
        logger.error(f"  Missing expected bands: {sorted(missing)}")
    if unexpected:
        logger.error(f"  Unexpected extra bands: {sorted(unexpected)}")

    return not missing and not unexpected


def run_combination(
    combo: Combination,
    connection,
    backend_context: BackendContext,
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    tile_size: int,
    outdir: Path,
    dry_run: bool,
) -> bool:
    logger.info(f"--- Combination: {combo.name} ---")
    expected_bands = combo.expected_bands()
    logger.info(f"Expected bands: {sorted(expected_bands)}")

    cube = build_cube(
        combo, connection, backend_context, spatial_extent, temporal_extent, tile_size
    )

    if dry_run:
        # Only validate that the process graph can be constructed.
        _ = cube.flat_graph()
        logger.info("Dry run: process graph built successfully, skipping job submission.")
        return True

    cube = cube.save_result(
        format="NetCDF",
        options={"filename_prefix": f"preprocessed-inputs_{combo.name}"},
    )

    job_outdir = outdir / combo.name
    job_outdir.mkdir(parents=True, exist_ok=True)

    job = cube.execute_batch(
        title=f"WorldCereal preprocessed inputs test - {combo.name}",
        outputfile=job_outdir / "result.nc",
        job_options=DEFAULT_JOB_OPTIONS,
    )
    logger.info(f"Job {job.job_id} finished, results downloaded to {job_outdir}")

    nc_files = sorted(job_outdir.glob("*.nc"))
    if not nc_files:
        logger.error("No NetCDF file found among downloaded results!")
        return False

    passed = True
    for nc_file in nc_files:
        logger.info(f"Checking bands in {nc_file.name}")
        if not check_bands(nc_file, expected_bands):
            passed = False

    return passed


if __name__ == "__main__":
    outdir = Path("./preprocessed_inputs_combo_test")
    tile_size = DEFAULT_TILE_SIZE
    dry_run = False
    outdir.mkdir(parents=True, exist_ok=True)

    backend_context = BackendContext(Backend.CDSE)
    # A connection is needed even for a dry run: it is used to build the
    # process graph (no data is actually fetched/started in dry-run mode).
    connection = cdse_connection()

    combinations = all_valid_combinations()
    logger.info(f"Testing {len(combinations)} valid sensor combinations.")

    results = {}
    for combo in combinations:
        try:
            results[combo.name] = run_combination(
                combo,
                connection,
                backend_context,
                DEFAULT_SPATIAL_EXTENT,
                DEFAULT_TEMPORAL_EXTENT,
                tile_size,
                outdir,
                dry_run,
            )
        except Exception:
            logger.exception(f"Combination {combo.name} raised an exception")
            results[combo.name] = False

    logger.info("=== Summary ===")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"{status}: {name}")
        all_passed = all_passed and passed

    if not all_passed:
        logger.error("One or more combinations failed the band check!")
        sys.exit(1)

    logger.info("All combinations passed the band check.")
