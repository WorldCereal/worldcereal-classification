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
from worldcereal.train.data import WorldCerealTrainingDataset, get_training_df
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
    # with new datasets added, number of samples may increase
    # if collateral samples are excluded, this will decrease drastically
    assert training_df.shape == (549, total_cols_expected)

    # We keep original ewoc_code for this test
    training_df["downstream_class"] = training_df["ewoc_code"]

    # Compute presto embeddings
    presto_model_url = CropTypeParameters().feature_parameters.presto_model_url

    # Load pretrained Presto model
    logger.info(f"Presto URL: {presto_model_url}")
    presto_model = Presto()
    presto_model = load_presto_weights(presto_model, presto_model_url)

    # Initialize dataset
    df = training_df.reset_index()

    ds = WorldCerealTrainingDataset(df, task_type="multiclass", augment=True)
    logger.info("Computing Presto embeddings ...")
    df = get_training_df(
        ds,
        presto_model,
        batch_size=256,
        time_explicit=True,  # Important for croptype
    )
    logger.info("Presto embeddings computed.")

    # Train classifier
    logger.info("Split train/test ...")
    samples_train, samples_test = train_test_split(
        df,
        test_size=0.2,
        random_state=3,
        stratify=df["downstream_class"],
    )

    eval_metric = "MultiClass"
    loss_function = "MultiClass"

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

    sample_weights = np.ones((len(samples_train["downstream_class"]),))
    sample_weights_val = np.ones((len(samples_test["downstream_class"]),))
    for k, v in class_weights.items():
        sample_weights[samples_train["downstream_class"] == k] = v
        sample_weights_val[samples_test["downstream_class"] == k] = v
    samples_train["weight"] = sample_weights
    samples_test["weight"] = sample_weights_val

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
        data=samples_train[bands],
        label=samples_train["downstream_class"],
        weight=samples_train["weight"],
    )
    eval_data = Pool(
        data=samples_test[bands],
        label=samples_test["downstream_class"],
        weight=samples_test["weight"],
    )

    # Train classifier
    logger.info("Training CatBoost classifier ...")
    custom_downstream_model.fit(
        calibration_data,
        eval_set=eval_data,
    )

    # Make predictions
    _ = custom_downstream_model.predict(samples_test[bands]).flatten()
