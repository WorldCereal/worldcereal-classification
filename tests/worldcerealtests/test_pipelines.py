import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from loguru import logger
from openeo_gfmap import BoundingBoxExtent, TemporalContext
from sklearn.utils.class_weight import compute_class_weight

from worldcereal.parameters import CropTypeParameters
from worldcereal.train.data import compute_embeddings_from_input_df
from worldcereal.train.downstream import TorchTrainer
from worldcereal.utils.refdata import (
    process_extractions_df,
    query_private_extractions,
    query_public_extractions,
)

SPATIAL_EXTENT = BoundingBoxExtent(
    west=4.63761, south=51.11649, east=4.73761, north=51.21649, epsg=4326
).to_geometry()


def test_custom_croptype_demo(WorldCerealPrivateExtractionsPath):
    """Test for a full custom croptype pipeline up and till the point of
    training a custom catboost classifier. This test uses both public and private
    extractions.
    """

    # Query private and public extractions

    private_df = query_private_extractions(
        "/data/worldcereal_data/EXTRACTIONS/WORLDCEREAL/WORLDCEREAL_PUBLIC_EXTRACTIONS/worldcereal_public_extractions.parquet",
        bbox_poly=SPATIAL_EXTENT,
        filter_cropland=True,
        buffer=250000,  # Meters
    )

    assert not private_df.empty, "Should have found private extractions"
    logger.info(
        (
            f"Found {private_df['sample_id'].nunique()} unique samples in the "
            f"private data, spread across {private_df['ref_id'].nunique()} "
            "unique reference datasets."
        )
    )

    public_df = query_public_extractions(
        SPATIAL_EXTENT,
        buffer=1000,  # Meters
        filter_cropland=True,
        query_collateral_samples=True,  # query collateral to resemble previous behavior
    )

    assert not public_df.empty, "Should have found public extractions"
    logger.info(
        (
            f"Found {public_df['sample_id'].nunique()} unique samples in the "
            f"public data, spread across {public_df['ref_id'].nunique()} "
            "unique reference datasets."
        )
    )

    # Concatenate extractions
    extractions_df = pd.concat([private_df, public_df])

    assert len(extractions_df) == len(public_df) + len(private_df)

    # Process the merged data
    processing_period = TemporalContext("2020-01-01", "2020-12-31")
    print(f"Shape of extractions_df: {extractions_df.shape}")
    training_df = process_extractions_df(extractions_df, processing_period)
    logger.info(f"training_df shape: {training_df.shape}")

    # Drop labels that occur infrequently for this test
    value_counts = training_df["ewoc_code"].value_counts()
    single_labels = value_counts[value_counts < 3].index.to_list()
    training_df = training_df[~training_df["ewoc_code"].isin(single_labels)]

    print("*" * 40)
    for c in training_df.columns:
        print(c)
    print("*" * 40)

    # Direct shape assert: if process_extractions_df changes, this may have to be updated
    aux_columns = [
        "ewoc_code",
        "lat",
        "quality_score_lc",
        "available_timesteps",
        "tile",
        "valid_position",
        "filename",
        "quality_score_ct",
        "year",
        "geometry",
        "extract",
        "h3_l3_cell",
        "geom_text",
        "lon",
        "ref_id",
        "start_date",
        "end_date",
        "irrigation_status",
        "valid_time",
        "label_full",
        "sampling_label",
    ]
    static_columns = ["DEM-alt-20m", "DEM-slo-20m"]
    feature_columns = [
        "OPTICAL-B02",
        "OPTICAL-B03",
        "OPTICAL-B04",
        "OPTICAL-B08",
        "OPTICAL-B05",
        "OPTICAL-B06",
        "OPTICAL-B07",
        "OPTICAL-B8A",
        "OPTICAL-B11",
        "OPTICAL-B12",
        "SAR-VV",
        "SAR-VH",
        "METEO-temperature_mean",
        "METEO-precipitation_flux",
    ]
    num_timesteps_expected = 16
    total_cols_expected = (
        len(aux_columns)
        + len(static_columns)
        + len(feature_columns) * num_timesteps_expected
    )
    # Make sure we have the expected columns and non-empty sample size
    assert training_df.shape[1] == total_cols_expected
    assert training_df.shape[0] > 0

    # We keep original ewoc_code for this test
    training_df["downstream_class"] = training_df["ewoc_code"]

    # Compute presto embeddings
    presto_model_url = CropTypeParameters().feature_parameters.presto_model_url

    # Load pretrained Presto model
    logger.info(f"Presto URL: {presto_model_url}")

    # Compute embeddings
    embeddings_df = compute_embeddings_from_input_df(
        training_df, presto_model_url, stratify_label="downstream_class"
    )

    logger.info("Presto embeddings computed.")

    # Train classifier
    trainer = TorchTrainer(
        embeddings_df,
        lr_grid=[1e-2],
        weight_decay_grid=[1e-5],
    )
    trainer.train()

    # Make predictions
    # _ = custom_downstream_model.predict(df[df["split"] == "test"][bands]).flatten()
