import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from loguru import logger
from openeo_gfmap import BoundingBoxExtent, TemporalContext
from prometheo.models import Presto
from prometheo.models.presto.wrapper import load_presto_weights
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from worldcereal.parameters import CropTypeParameters
from worldcereal.train.data import get_training_df
from worldcereal.train.datasets import SensorMaskingConfig, WorldCerealTrainingDataset
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
        WorldCerealPrivateExtractionsPath,
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
    presto_model = Presto()
    presto_model = load_presto_weights(presto_model, presto_model_url)

    # Split dataframe in cal/val
    samples_train, samples_test = train_test_split(
        training_df,
        test_size=0.2,
        random_state=42,
        stratify=training_df["downstream_class"],
    )

    # Initialize datasets
    samples_train, samples_test = samples_train.reset_index(), samples_test.reset_index()
    masking_config = SensorMaskingConfig(
        enable=True,
        s1_full_dropout_prob=0.05,
        s1_timestep_dropout_prob=0.1,
        s2_cloud_timestep_prob=0.1,
        s2_cloud_block_prob=0.05,
        s2_cloud_block_min=2,
        s2_cloud_block_max=3,
        meteo_timestep_dropout_prob=0.03,
        dem_dropout_prob=0.01
    )

    # Augmentations and repeats only on training set
    repeats = 2
    ds_train = WorldCerealTrainingDataset(
        samples_train,
        task_type="multiclass",
        augment=True,
        masking_config=masking_config,
        repeats=repeats
    )
    assert len(ds_train) == len(samples_train) * repeats

    # No augmentations on test set
    ds_test = WorldCerealTrainingDataset(
        samples_test,
        task_type="multiclass",
        augment=False,
        masking_config=SensorMaskingConfig(enable=False)
    )

    logger.info("Computing Presto embeddings on train set ...")
    df_train = get_training_df(
        ds_train,
        presto_model,
        batch_size=256,
        time_explicit=False,
    )
    logger.info("Computing Presto embeddings on test set ...")
    df_test = get_training_df(
        ds_test,
        presto_model,
        batch_size=256,
        time_explicit=False,
    )
    logger.info("Presto embeddings computed.")

    # Merging train and test embeddings to simulate behavior of full pipeline
    df_train["split"] = "train"
    df_test["split"] = "test"
    df = pd.concat([df_train, df_test]).reset_index(drop=True)

    # Train classifier
    eval_metric = "MultiClass"
    loss_function = "MultiClass"

    # Compute class weights based on training set
    logger.info("Computing class weights ...")
    class_weights = np.round(
        compute_class_weight(
            class_weight="balanced",
            classes=np.unique(samples_train["downstream_class"]),
            y=samples_train["downstream_class"],
        ),
        3,
    )
    class_weights = {
        k: v
        for k, v in zip(np.unique(samples_train["downstream_class"]), class_weights)
    }
    logger.info(f"Class weights: {class_weights}")

    sample_weights = np.ones((len(df["downstream_class"]),))
    for k, v in class_weights.items():
        sample_weights[df["downstream_class"] == k] = v
    df["weight"] = sample_weights

    # Define classifier
    custom_downstream_model = CatBoostClassifier(
        iterations=2000,  # Not too high to avoid too large model size
        depth=8,
        early_stopping_rounds=20,
        loss_function=loss_function,
        eval_metric=eval_metric,
        random_state=3,
        verbose=25,
        class_names=np.unique(samples_train["downstream_class"]),
    )

    # Setup dataset Pool
    bands = [f"presto_ft_{i}" for i in range(128)]
    calibration_data = Pool(
        data=df[df["split"] == "train"][bands],
        label=df[df["split"] == "train"]["downstream_class"],
        weight=df[df["split"] == "train"]["weight"],
    )
    eval_data = Pool(
        data=df[df["split"] == "test"][bands],
        label=df[df["split"] == "test"]["downstream_class"],
        weight=df[df["split"] == "test"]["weight"],
    )

    # Train classifier
    logger.info("Training CatBoost classifier ...")
    custom_downstream_model.fit(
        calibration_data,
        eval_set=eval_data,
    )

    # Make predictions
    _ = custom_downstream_model.predict(df[df["split"] == "test"][bands]).flatten()
