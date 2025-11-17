"""Mapping helpers for cropland, croptype and embeddings products.

Band naming conventions produced by the UDF (`inference.py`):

Single workflow (only cropland OR only croptype parameters passed to UDF):
    classification, probability, probability_<class>
    If save_intermediate: raw_<band> duplicates (e.g. raw_classification)

Combined workflow (croptype with cropland masking: both `croptype_params` &
`cropland_params` passed):
    croptype_<band>, cropland_<band>
    If save_intermediate: croptype_raw_<band>, cropland_raw_<band>
        Example: croptype_classification -> croptype_raw_classification

Important: Raw bands in the combined workflow do NOT duplicate the base prefix;
they simply replace the leading product prefix with <product>_raw_.

Simplification: We ignore any *save_intermediate* flags. If raw bands are
present we save them; the UDF only emits them when intermediate results were
requested upstream.
"""

from pathlib import Path
from typing import List

import openeo
from openeo import DataCube
from openeo_gfmap import TemporalContext
from openeo_gfmap.preprocessing.scaling import compress_uint16

from worldcereal.openeo.inference import apply_metadata
from worldcereal.parameters import (
    CropLandParameters,
    CropTypeParameters,
    EmbeddingsParameters,
    WorldCerealProductType,
)

NEIGHBORHOOD_SPEC = dict(
    size=[
        {"dimension": "x", "unit": "px", "value": 128},
        {"dimension": "y", "unit": "px", "value": 128},
    ],
    overlap=[
        {"dimension": "x", "unit": "px", "value": 0},
        {"dimension": "y", "unit": "px", "value": 0},
    ],
)


def _run_udf(inputs: DataCube, udf: openeo.UDF) -> DataCube:
    return inputs.apply_neighborhood(process=udf, **NEIGHBORHOOD_SPEC)


def _reduce_temporal_mean(cube: DataCube) -> DataCube:
    return cube.reduce_dimension(dimension="t", reducer="mean")


def _filename_prefix(
    product: WorldCerealProductType, temporal: TemporalContext, raw: bool = False
) -> str:
    suffix = "-raw" if raw else ""
    return f"{product.value}{suffix}_{temporal.start_date}_{temporal.end_date}"


def _save_result(cube: DataCube, prefix: str) -> DataCube:
    return cube.save_result(format="GTiff", options={"filename_prefix": prefix})


def _cropland_map(
    inputs: DataCube,
    temporal_extent: TemporalContext,
    cropland_parameters: CropLandParameters,
) -> List[DataCube]:
    """Produce cropland product from preprocessed inputs (single workflow).

    Saves final bands and any raw_* bands purely based on presence.
    """
    inference_udf = openeo.UDF.from_file(
        path=Path(__file__).resolve().parent / "inference.py",
        context=cropland_parameters.model_dump(),
    )
    classes = _run_udf(inputs, inference_udf)
    classes.metadata = apply_metadata(
        classes.metadata, cropland_parameters.model_dump()
    )
    classes = _reduce_temporal_mean(classes)
    classes = compress_uint16(classes)

    bands = classes.metadata.band_names
    result_cubes: List[DataCube] = []

    final_bands = [b for b in bands if not b.startswith("raw_")]
    if final_bands:
        final_cube = classes.filter_bands(final_bands)
        result_cubes.append(
            _save_result(
                final_cube,
                _filename_prefix(WorldCerealProductType.CROPLAND, temporal_extent),
            )
        )

    raw_bands = [b for b in bands if b.startswith("raw_")]
    if raw_bands:
        raw_cube = classes.filter_bands(raw_bands)
        result_cubes.append(
            _save_result(
                raw_cube,
                _filename_prefix(
                    WorldCerealProductType.CROPLAND, temporal_extent, raw=True
                ),
            )
        )

    return result_cubes


def _croptype_map(
    inputs: DataCube,
    temporal_extent: TemporalContext,
    croptype_parameters: CropTypeParameters,
    cropland_parameters: CropLandParameters,
) -> List[DataCube]:
    """Produce crop type product. Optionally includes cropland masking.
    Cropland mask final bands saved only if `croptype_parameters.save_mask` is True.
    """
    if croptype_parameters.mask_cropland:
        parameters = {
            "cropland_params": cropland_parameters.model_dump(),
            "croptype_params": croptype_parameters.model_dump(),
        }
    else:
        parameters = croptype_parameters.model_dump()

    inference_udf = openeo.UDF.from_file(
        path=Path(__file__).resolve().parent / "inference.py",
        context=parameters,
    )
    classes = _run_udf(inputs, inference_udf)
    classes.metadata = apply_metadata(classes.metadata, parameters)
    classes = _reduce_temporal_mean(classes)
    classes = compress_uint16(classes)

    bands = classes.metadata.band_names
    result_cubes: List[DataCube] = []

    if croptype_parameters.mask_cropland:
        # Prefixed croptype final and raw bands
        croptype_final_bands = [
            b for b in bands if b.startswith("croptype_") and "raw" not in b
        ]
        # Raw croptype bands (presence-based)
        raw_croptype_bands = [b for b in bands if b.startswith("croptype_raw_")]
    else:
        # Single workflow: unprefixed croptype bands
        croptype_final_bands = [b for b in bands if not b.startswith("raw_")]
        raw_croptype_bands = [b for b in bands if b.startswith("raw_")]

    # Final croptype
    croptype_cube = classes.filter_bands(croptype_final_bands).rename_labels(
        dimension="bands",
        target=[
            b.replace("croptype_", "") for b in croptype_final_bands
        ],  # Remove prefix
    )
    result_cubes.append(
        _save_result(
            croptype_cube,
            _filename_prefix(WorldCerealProductType.CROPTYPE, temporal_extent),
        )
    )

    # Raw croptype if present
    if raw_croptype_bands:
        raw_croptype_cube = classes.filter_bands(raw_croptype_bands).rename_labels(
            dimension="bands",
            target=[
                b.replace("croptype_", "") for b in raw_croptype_bands
            ],  # Remove prefix
        )
        result_cubes.append(
            _save_result(
                raw_croptype_cube,
                _filename_prefix(
                    WorldCerealProductType.CROPTYPE, temporal_extent, raw=True
                ),
            )
        )

    # Optional cropland mask & raw cropland bands
    if croptype_parameters.save_mask:
        cropland_final_bands = [
            b
            for b in bands
            if b.startswith("cropland_") and not b.startswith("cropland_raw_")
        ]
        cropland_cube = classes.filter_bands(cropland_final_bands).rename_labels(
            dimension="bands",
            target=[
                b.replace("cropland_", "") for b in cropland_final_bands
            ],  # Remove prefix
        )
        result_cubes.append(
            _save_result(
                cropland_cube,
                _filename_prefix(WorldCerealProductType.CROPLAND, temporal_extent),
            )
        )
        raw_cropland_bands = [b for b in bands if b.startswith("cropland_raw_")]
        if raw_cropland_bands:
            raw_cropland_cube = classes.filter_bands(raw_cropland_bands).rename_labels(
                dimension="bands",
                target=[
                    b.replace("cropland_", "") for b in raw_cropland_bands
                ],  # Remove prefix
            )
            result_cubes.append(
                _save_result(
                    raw_cropland_cube,
                    _filename_prefix(
                        WorldCerealProductType.CROPLAND, temporal_extent, raw=True
                    ),
                )
            )

    return result_cubes


def _embeddings_map(
    inputs: DataCube,
    temporal_extent: TemporalContext,  # temporal extent unused but kept for signature consistency
    embeddings_parameters: EmbeddingsParameters,
    scale_uint16: bool = True,
) -> DataCube:
    """Produce embeddings map using Prometheo feature extractor."""

    feature_udf = openeo.UDF.from_file(
        path=Path(__file__).resolve().parent / "feature_extractor.py",
        context=embeddings_parameters.feature_parameters.model_dump(),
    )
    embeddings = _run_udf(inputs, feature_udf)
    embeddings = _reduce_temporal_mean(embeddings)

    if scale_uint16:
        OFFSET = -6
        SCALE = 0.0002
        embeddings = (embeddings - OFFSET) / SCALE
        embeddings = embeddings.linear_scale_range(0, 65534, 0, 65534)

    return embeddings
