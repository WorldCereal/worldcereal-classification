import glob
from pathlib import Path
from typing import Union

import pandas as pd
from loguru import logger
from presto.presto import Presto
from presto.utils import process_parquet

from worldcereal.train.data import WorldCerealTrainingDataset, get_training_df


def embeddings_from_parquet_file(
    parquet_file: Union[str, Path],
    pretrained_model: Presto,
    sample_repeats: int = 1,
    mask_ratio: float = 0.0,
    valid_date_as_token: bool = False,
    exclude_meteo: bool = False,
) -> pd.DataFrame:
    """Method to compute Presto embeddings from a parquet file of preprocessed inputs

    Parameters
    ----------
    parquet_file : Union[str, Path]
        parquet file to read data from
    pretrained_model : Presto
        Presto model to use for computing embeddings
    sample_repeats : int, optional
        number of augmented sample repeats, by default 1
    mask_ratio : float, optional
        mask ratio to apply, by default 0.0
    valid_date_as_token : bool, optional
        feed valid date as a token to Presto, by default False
    exclude_meteo : bool, optional
        if True, meteo will be masked during embedding computation, by default False

    Returns
    -------
    pd.DataFrame
        DataFrame containing Presto embeddings and other attributes
    """

    # Load parquet file
    logger.info(f"Processing {parquet_file}")
    df = pd.read_parquet(parquet_file)

    # Original file is partitioned by ref_id so we have to add
    # ref_id as a column manually
    df["ref_id"] = Path(parquet_file).parent.stem.split("=")[1]

    # Put meteo to nodata if needed
    if exclude_meteo:
        logger.warning("Excluding meteo data ...")
        df.loc[:, df.columns.str.contains("AGERA5")] = 65535

    # Convert to wide format
    logger.info("From long to wide format ...")
    df = process_parquet(df).reset_index()

    # Check if samples remain
    if df.empty:
        logger.warning("Empty dataframe: returning None")
        return None

    # Create dataset and dataloader
    logger.info("Making data loader ...")
    ds = WorldCerealTrainingDataset(
        df,
        task_type="cropland",
        augment=True,
        mask_ratio=mask_ratio,
        repeats=sample_repeats,
    )

    if len(ds) == 0:
        logger.warning("No valid samples in dataset: returning None")
        return None

    return get_training_df(
        ds,
        pretrained_model,
        batch_size=2048,
        valid_date_as_token=valid_date_as_token,
        num_workers=4,
    )


def main(
    infile,
    outfile,
    presto_model,
    sc=None,
    debug=False,
    sample_repeats: int = 1,
    mask_ratio: float = 0.0,
    valid_date_as_token: bool = False,
    exclude_meteo: bool = False,
):

    logger.info(
        f"Starting embedding computation with augmentation (sample_repeats: {sample_repeats}, mask_ratio: {mask_ratio})"
    )
    logger.info(f"Valid date as token: {valid_date_as_token}")

    # List parquet files
    parquet_files = glob.glob(str(infile) + "/**/*.parquet", recursive=True)

    if debug:
        parquet_files = parquet_files[:2]

    # Load model
    logger.info("Loading model ...")
    pretrained_model = Presto.load_pretrained(
        model_path=presto_model, strict=False, valid_month_as_token=valid_date_as_token
    )

    if sc is not None:
        logger.info(f"Parallelizing {len(parquet_files)} files ...")
        dfs = (
            sc.parallelize(parquet_files, len(parquet_files))
            .map(
                lambda x: embeddings_from_parquet_file(
                    x,
                    pretrained_model,
                    sample_repeats,
                    mask_ratio,
                    valid_date_as_token,
                    exclude_meteo,
                )
            )
            .filter(lambda x: x is not None)
            .collect()
        )
    else:
        dfs = []
        for parquet_file in parquet_files:
            dfs.append(
                embeddings_from_parquet_file(
                    parquet_file,
                    pretrained_model,
                    sample_repeats,
                    mask_ratio,
                    valid_date_as_token,
                    exclude_meteo,
                )
            )

    if isinstance(dfs, list):
        logger.info(f"Done processing: concatenating {len(dfs)} results")
        dfs = pd.concat(dfs, ignore_index=True)

    logger.info(f"Final dataframe shape: {dfs.shape}")

    # Write to parquet
    logger.info(f"Writing to parquet: {outfile}")
    dfs.to_parquet(outfile, index=False)

    logger.success("All done!")


if __name__ == "__main__":
    # Output feature basedir
    baseoutdir = Path(
        "/vitodata/worldcereal/features/preprocessedinputs-monthly-nointerp"
    )

    spark = True
    localspark = False
    debug = False
    exclude_meteo = False
    sample_repeats = 1
    valid_date_as_token = False
    presto_dir = Path("/vitodata/worldcereal/presto/finetuning")
    presto_model = (
        presto_dir
        / "presto-ss-wc-ft-ct_cropland_CROPLAND2_30D_random_time-token=none_balance=True_augment=True.pt"
    )
    identifier = ""

    if spark:
        from worldcereal.utils.spark import get_spark_context

        logger.info("Setting up spark ...")
        sc = get_spark_context(localspark=localspark)
    else:
        sc = None

    infile = baseoutdir / "worldcereal_training_data.parquet"

    if debug:
        outfile = baseoutdir / (
            f"training_df_{presto_model.stem}_presto-worldcereal{identifier}_DEBUG.parquet"
        )
    else:
        outfile = baseoutdir / (
            f"training_df_{presto_model.stem}_presto-worldcereal{identifier}.parquet"
        )

    if not infile.exists():
        raise FileNotFoundError(infile)

    main(
        infile,
        outfile,
        presto_model,
        sc=sc,
        debug=debug,
        sample_repeats=sample_repeats,
        valid_date_as_token=valid_date_as_token,
        exclude_meteo=exclude_meteo,
    )
