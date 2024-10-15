from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight


class NotEnoughSamplesError(Exception):
    pass


def filter_perennial(df):
    """Label 12:
    CT label
        7900 -> LANDCOVER_LABEL 10
        7910 -> LANDCOVER_LABEL 10
        7920 -> LANDCOVER_LABEL 10
        9520 -> LANDCOVER_LABEL 10
    """

    nr_perennial_before = (df["LANDCOVER_LABEL"] == 12).sum()
    logger.info(f"Perennial cropland samples before filtering: {nr_perennial_before}")

    df.loc[df["CROPTYPE_LABEL"].isin([7900, 7910, 7920, 9520]), "LANDCOVER_LABEL"] = 10

    nr_perennial_after = (df["LANDCOVER_LABEL"] == 12).sum()

    logger.info(
        (
            "Perennial cropland samples switched to annual "
            f"cropland (LANDCOVER_LABEL 10): {nr_perennial_before - nr_perennial_after}"
        )
    )

    return df


def filter_cropland_11(df):
    """Helper function to remove outliers from cropland class"""

    nr_cropland = (df["LANDCOVER_LABEL"] == 11).sum()

    logger.info(f"Annual cropland samples before filtering: {nr_cropland}")

    ignore_types = [9120, 9110, 7910, 7920, 7900, 9100]

    # Cropland should not have crop type from ignore list
    df = df[
        ~((df["LANDCOVER_LABEL"] == 11) & (df["CROPTYPE_LABEL"].isin(ignore_types)))
    ]

    nr_cropland = (df["LANDCOVER_LABEL"] == 11).sum()

    logger.info(f"Annual cropland samples after filtering: {nr_cropland}")

    return df


def filter_croptype(df):
    """Helper function to remove outliers for croptype processors"""

    nr_samples = df.shape[0]
    logger.info(f"Crop type samples before filtering: {nr_samples}")

    # Cropland should not have corner reflector signal in SAR
    df = df[~(df["SAR-VV-p90-20m"] > 0)]

    # Cropland should not have a NDVI p90 which is very low
    df = df[~(df["OPTICAL-ndvi-p90-10m"] < 0.4)]

    nr_samples = df.shape[0]

    logger.info(f"Crop type samples after filtering: {nr_samples}")

    return df


def filter_grassland(df):

    nr_grassland = (df["LANDCOVER_LABEL"] == 13).sum()

    logger.info(f"Grassland samples before filtering: {nr_grassland}")

    # Filter out grassland which is cropland according to worldcover
    # AND potapov
    df = df[
        ~(
            (df["LANDCOVER_LABEL"] == 13)
            & (df["WORLDCOVER-LABEL-10m"] == 40)
            & (df["POTAPOV-LABEL-10m"] == 1)
        )
    ]

    nr_grassland = (df["LANDCOVER_LABEL"] == 13).sum()

    logger.info(f"Grassland samples after filtering: {nr_grassland}")

    return df


def filter_ewoco(df):

    nr_crop = (df["LANDCOVER_LABEL"] == 11).sum()

    logger.info(f"Cropland samples before EWOCO filtering: {nr_crop}")

    # Filter out crop from ewoco
    # AND potapov
    df = df[~((df["LANDCOVER_LABEL"] == 11) & (df["ref_id"].str.contains("ewoco")))]

    nr_crop = (df["LANDCOVER_LABEL"] == 11).sum()

    logger.info(f"Cropland samples after EWOCO filtering: {nr_crop}")

    return df


def filter_otherlayers(df, detector):
    """Method to filter out training labels
    that do not correspond to some other values in
    other land cover layers
    """

    label = "LANDCOVER_LABEL"

    # 01 - remove pixels that are urban according to worldcover
    agri_labels = [10, 11, 12]
    ewoc_ignore = [50]
    if "annual" in detector or "cropland" in detector:
        remove_idx = (df[label].isin(agri_labels)) & (
            df["WORLDCOVER-LABEL-10m"].isin(ewoc_ignore)
        )
    else:  # crop type case
        remove_idx = df["WORLDCOVER-LABEL-10m"].isin(ewoc_ignore)

    df = df[~remove_idx].copy()
    logger.info(
        f"Removed {remove_idx.sum()} urban samples"
        f" according to worldcover information."
    )

    # 02 - remove pixels that are not crop [10, 11] according to BOTH
    # WorldCover and Potapov crop layer
    remove_idx = (
        (df[label].isin([10, 11]))
        & (df["WORLDCOVER-LABEL-10m"] != 40)
        & (df["POTAPOV-LABEL-10m"] != 1)
    )
    df = df[~remove_idx].copy()
    logger.info(
        f"Removed {remove_idx.sum()} crop samples"
        " that are not crop according "
        "to worldcover and potapov."
    )

    # 03 - remove pixels that are crop according to BOTH
    # WorldCover and Potapov crop layer but not in the label
    remove_idx = (
        (~df[label].isin([10, 11, 12]))
        & (df["WORLDCOVER-LABEL-10m"] == 40)
        & (df["POTAPOV-LABEL-10m"] == 1)
    )
    df = df[~remove_idx].copy()
    logger.info(
        f"Removed {remove_idx.sum()} non-crop samples"
        " that are crop according "
        "to both worldcover and potapov."
    )


def get_sample_weight(sample, outputlabel, options):

    # Default weight
    if sample[outputlabel] == 0:
        weight = 0
    elif sample[outputlabel] in options.get("ignorelabels", []):
        weight = 0
    else:
        weight = 1

    # Multiply focuslabel weight with provided multiplier
    if sample[outputlabel] in options.get("focuslabels", []):
        weight *= options.get("focusmultiplier", 1)

    return float(weight)


def process_training_data(
    df,
    inputfeatures,
    detector,
    options,
    minsamples=1000,
    filter_worldcover=False,
    logdir=None,
    outputlabel="LANDCOVER_LABEL",
):
    """Function that returns inputs/outputs from training DataFrame

    Args:
        df (pd.DataFrame): dataframe containing input/output features
        inputfeatures (list[str]): list of inputfeatures to be used
        detector (str): detector name
        options (dict): dictionary containing options
        minsamples (int, optional): minimum number of samples. Defaults to 500.
        logdir (str, optional): output path for logs/figures. Defaults to None.
        filter_worldcover (bool, optional): whether or not to
            remove outliers based on worldcover data. Defaults to False.

    Raises:
        NotEnoughSamplesError: obviously when not enough samples were found

    Returns:
        inputs, outputs, weights: arrays to use in training
    """

    # ----------------------------------------------------------------------
    # PART II: Check sample sizes and remove labels to be ignored

    # Check if we have enough samples at all to start with
    # that's at least 2X minsamples (binary classification)
    if df.shape[0] < 2 * minsamples:
        errormessage = (
            f"Got less than {2 * minsamples} "
            f"in total for this dataset. "
            "Cannot continue!"
        )
        logger.error(errormessage)
        raise NotEnoughSamplesError(errormessage)

    # Remove the unknown and ignore labels
    if len(options.get("ignorelabels", [])) > 0:
        remove_idx = (df[outputlabel] == 0) | (
            df[outputlabel].isin(options["ignorelabels"])
        )
    else:
        remove_idx = df[outputlabel] == 0

    df = df[~remove_idx].copy()
    logger.info(f"Removed {remove_idx.sum()} unknown/ignore samples.")

    # ----------------------------------------------------------------------
    # PART IV: Apply various thematic filters

    if "annual" not in detector and "cropland" not in detector:
        # If not looking at cropland
        # need to get rid of samples that are not part of cropland or grassland
        # We include grassland to make model aware of what grass looks like
        # in case there is grass commission inside cropland product.
        logger.info(("Removing samples that are not cropland or grassland"))
        df = df[df["LANDCOVER_LABEL"].isin([10, 11, 12, 13])].copy()

        logger.info(f'Unique LANDCOVER_LABEL values: {df["LANDCOVER_LABEL"].unique()}')

    if "annual" in detector or "cropland" in detector:
        # Rule 1: filter out dirty perennials
        # NOTE: NEEDS TO GO FIRST ALWAYS!
        df = filter_perennial(df)

        # Rule 2: filter the unkwown cropland
        # df = filter_cropland_10(df)

        # # Rule 3: filter the annual cropland
        df = filter_cropland_11(df)

        # Rule 4: filter the grassland
        df = filter_grassland(df)

        # Rule 5: remove crop from ewoco
        # df = filter_ewoco(df)

    else:
        # Filter out obvious no-crop for the crop type processors
        df = filter_croptype(df)

    # Apply some filters based on worldcover and potapov layers
    if filter_worldcover:
        filter_otherlayers(df, detector)

    # Remove corrupt rows
    remove_idx = ((df.isnull())).sum(axis=1) > int(df.shape[1] * 0.75)
    df = df[~remove_idx].copy()

    # do intermediate check of number of samples
    if df.shape[0] < 2 * minsamples:
        errormessage = (
            f"Got less than {2 * minsamples} "
            f"in total for this dataset. "
            "Cannot continue!"
        )
        logger.error(errormessage)
        raise NotEnoughSamplesError(errormessage)

    # ----------------------------------------------------------------------
    # PART VII: Select features we need
    required_columns = list(
        set(inputfeatures + [outputlabel, "ref_id", "location_id", "sample_id"])
    )
    df = df[required_columns]

    # ----------------------------------------------------------------------
    # PART VIII:
    if "annual" in detector or "cropland" in detector:
        # Remove rows with NaN
        beforenan = df.shape[0]
        df = df.dropna()
        afternan = df.shape[0]
        logger.info(f"Removed {beforenan - afternan} samples with NaN values.")

    # ----------------------------------------------------------------------
    # PART IX: Compute weights
    # This happens in two parts:
    # 1. Determine class weights in order to reach requested ratio
    # 2. Adjust sample-specific weights based on label

    # Compute class weight to get balanced samples
    binaryoutputs = df[outputlabel].copy()
    binaryoutputs[binaryoutputs.isin(options["targetlabels"])] = 1
    binaryoutputs[binaryoutputs != 1] = 0

    # check whether we still have samples from each class
    if len(np.unique(binaryoutputs)) == 1:
        singleclass = int(np.unique(binaryoutputs)[0])
        # Only one class present, no use to continue
        raise NotEnoughSamplesError(f"Only class {singleclass} found, aborting!")

    # Compute class weights that would balance the two classes
    balanced_classweight = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1]), y=binaryoutputs.values
    )

    # Adjust balanced weight according to requested ratio
    pos_neg_ratio = options.get("pos_neg_ratio", 0.5)
    logger.info(f"Using pos-neg-ratio of {pos_neg_ratio}")
    negative_classweight = (1 - pos_neg_ratio) / 0.5 * balanced_classweight[0]
    positive_classweight = pos_neg_ratio / 0.5 * balanced_classweight[1]

    # Clamp max weight to avoid excesses
    MAX_WEIGHT = 10
    positive_classweight = min(positive_classweight, MAX_WEIGHT)
    negative_classweight = min(negative_classweight, MAX_WEIGHT)

    weights = np.ones((binaryoutputs.shape[0]))
    weights[binaryoutputs.values == 0] = negative_classweight
    weights[binaryoutputs.values == 1] = positive_classweight
    df["sampleweight"] = weights

    # Get the sample-specific weights
    sample_weights = df.apply(
        lambda row: get_sample_weight(row, outputlabel, options), axis=1
    ).values

    # Adjust sample weights by the class weights and assign as final weights
    sample_weights *= df["sampleweight"].values
    df["sampleweight"] = sample_weights

    # Balancing by ref_id
    logger.info("Balancing for ref_ids ...")

    ref_id_classweights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(df["ref_id"]), y=df["ref_id"]
    )
    ref_id_classweights = {
        k: v for k, v in zip(np.unique(np.unique(df["ref_id"])), ref_id_classweights)
    }
    for ref_id in ref_id_classweights.keys():
        ref_id_classweights[ref_id] = min(ref_id_classweights[ref_id], MAX_WEIGHT)
        df.loc[df["ref_id"] == ref_id, "sampleweight"] *= ref_id_classweights[ref_id]

    # ----------------------------------------------------------------------
    # PART XI: Log various things on the situation as of now

    # Log the number of samples still present
    # per ref_id
    ref_id_counts = (
        df.groupby("ref_id")[outputlabel].value_counts().unstack().fillna(0).astype(int)
    )
    if logdir is not None:
        if not (Path(logdir) / "sample_counts.csv").is_file():
            logger.info("Saving sample counts ...")
            ref_id_counts.to_csv(Path(logdir) / "sample_counts.csv")
    ref_id_counts = ref_id_counts.sum(axis=1).to_dict()

    if logdir is not None:
        import matplotlib.pyplot as plt

        if not (Path(logdir) / "output_distribution.png").is_file():
            # Plot histogram of original outputs
            outputs = df[outputlabel].copy()
            counts = outputs.value_counts()
            labels = counts.index.astype(int)
            plt.bar(range(len(labels)), counts.values)
            plt.xticks(range(len(labels)), labels, rotation=90)
            plt.xlabel("Class")
            plt.ylabel("Amounts")
            plt.title("Output label distribution")
            outfile = Path(logdir) / "output_distribution.png"
            outfile.parent.mkdir(exist_ok=True)
            plt.savefig(outfile)
            plt.close()

    # ----------------------------------------------------------------------
    # PART XII: Extract input/output as numpy arrays, binarize the outputs
    # and do one more check if we still have enough samples

    # Get the inputs
    inputs = df[inputfeatures].values

    # Get the outputs AFTER the weights
    outputs = df[outputlabel].copy()

    # Get the final sample weights
    weights = df["sampleweight"].values

    # Get the output labels
    origoutputs = np.copy(outputs.values)  # For use in evaluation

    # Make classification binary
    outputs[outputs.isin(options["targetlabels"])] = 1
    outputs[outputs != 1] = 0
    outputs = outputs.values

    # Now check if we have enough samples to proceed
    for label in [0, 1]:
        if np.sum(outputs == label) < minsamples:
            errormessage = (
                f"Got less than {minsamples} "
                f"`{label}` samples for this dataset. "
                "Cannot continue!"
            )
            logger.error(errormessage)
            raise NotEnoughSamplesError(errormessage)

    # Log how many all-zero inputs (these are bad)
    idx = np.where(np.sum(inputs, axis=1) == 0)
    logger.info(f"#Rows with all-zero inputs: {len(idx[0])}")

    # ----------------------------------------------------------------------
    # PART XIII: Postprocessing before returning the results

    logger.info("Transforming inputs to pandas.DataFrame ...")
    data = pd.DataFrame(data=inputs, columns=inputfeatures)
    data["output"] = outputs
    data["weight"] = weights
    data["orig_output"] = origoutputs
    data["ref_id"] = df["ref_id"].values
    data["location_id"] = df["location_id"].values
    data["sample_id"] = df["sample_id"].values

    # ----------------------------------------------------------------------
    # PART XIV: Shuffle the data
    logger.info("Shuffling data ...")
    data = data.sample(frac=1)

    return data


def get_training_data(
    detector,
    options,
    inputfeatures,
    minsamples=500,
    **kwargs,
):

    dfs = (
        options["trainingfile"]
        if isinstance(options["trainingfile"], list)
        else [options["trainingfile"]]
    )

    df_data = pd.DataFrame()

    for current_df in dfs:
        df_data = pd.concat([df_data, pd.read_parquet(current_df)])

    # For cropland/croptype, we have to exclude some ref_ids
    if "cropland" in detector:
        ignore_list = [
            "2017_",
            "2018_AF",
            "2018_SSD_WFP",
            "2018_TZ_AFSIS",
            "2018_TZ_RadiantEarth",
            "2019_AF_OAF",
            "2019_KEN_WAPOR",
            "2019_TZA_CIMMYT",
            "2019_TZA_OAF",
            "2019_TZ_AFSIS",
            "2020_RW_WAPOR-Muvu",
            "2018_ES_SIGPAC-Andalucia",
            "2019_ES_SIGPAC-Andalucia",
            "2021_MOZ_WFP",
            "2021_TZA_COPERNICUS-GEOGLAM",
        ]
    else:
        ignore_list = [
            "2021_TZA_COPERNICUS-GEOGLAM",
            "2021_UKR_sunflowermap",
            "2021_EUR_EXTRACROPS",
            "2017_",
        ]

    logger.warning(f"Samples before removing refids: {df_data.shape[0]}")
    for ignore in ignore_list:
        df_data = df_data.loc[~df_data["ref_id"].str.contains(ignore)]
    logger.warning(f"Samples after removing refids: {df_data.shape[0]}")

    # Get the training data using all provided options
    data = process_training_data(
        df_data,
        inputfeatures,
        detector,
        options,
        filter_worldcover=options.get("filter_worldcover", False),
        minsamples=minsamples,
        **kwargs,
    )

    # Train/test splitting should happen on location_id as we don't want
    # a mix of (augmented) samples from the same location ending up in
    # both training and validation/test sets

    # Step 1: Split the dataset into train + validation and test sets
    samples_train, samples_test = train_test_split(
        list(data["location_id"].unique()),
        test_size=0.2,
        random_state=42,
    )

    # Step 2: Further split the train + validation set into separate train and validation sets
    samples_val, samples_test = train_test_split(
        samples_test,
        test_size=0.5,
        random_state=42,
    )

    # Get the actual data using the splitted location_ids
    data_train = data.set_index("location_id").loc[samples_train].reset_index()
    data_val = data.set_index("location_id").loc[samples_val].reset_index()
    data_test = data.set_index("location_id").loc[samples_test].reset_index()

    logger.info(f"Training on {data_train.shape[0]} samples.")
    logger.info(f"Validating on {data_val.shape[0]} samples.")
    logger.info(f"Testing on {data_test.shape[0]} samples.")

    return data_train, data_val, data_test
