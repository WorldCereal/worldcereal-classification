import glob
from pathlib import Path
from typing import List, Union

import pandas as pd
import torch
from loguru import logger
from presto.dataset import WorldCerealLabelledDataset
from presto.masking import MaskParamsNoDw
from presto.presto import Presto
from presto.utils import device, process_parquet
from torch.utils.data import DataLoader
from tqdm import tqdm


class WorldCerealInputsDataset(WorldCerealLabelledDataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        task_type: str = "cropland",
        croptype_list: List = [],
        return_hierarchical_labels: bool = False,
        augment: bool = False,
        mask_ratio: float = 0.0,
        repeats: int = 1,
    ):
        dataframe = dataframe.loc[~dataframe.LANDCOVER_LABEL.isin(self.FILTER_LABELS)]

        self.task_type = task_type
        self.croptype_list = croptype_list
        self.return_hierarchical_labels = return_hierarchical_labels
        self.augment = augment
        if augment:
            logger.info(
                "Augmentation is enabled. \
    The horizontal jittering of the selected window will be performed."
            )
        self.mask_ratio = mask_ratio
        self.mask_params = MaskParamsNoDw(
            (
                "group_bands",
                "random_timesteps",
                "chunk_timesteps",
                "random_combinations",
            ),
            mask_ratio,
        )

        super(WorldCerealLabelledDataset, self).__init__(dataframe)
        logger.info(f"Original dataset size: {len(dataframe)}")

        self.indices = []
        for _ in range(repeats):
            self.indices += [i for i in range(len(self.df))]

        logger.info(f"Dataset size after {repeats} repeats: {len(self.indices)}")

    def __iter__(self):
        for idx in self.indices:
            yield self.__getitem__(idx)

    def __getitem__(self, idx):

        # Get the sample
        sample = super().__getitem__(idx)
        row = self.df.iloc[self.indices[idx], :]

        attrs = [
            "WORLDCOVER-LABEL-10m",
            "lat",
            "lon",
            "CROPTYPE_LABEL",
            "LANDCOVER_LABEL",
            "POTAPOV-LABEL-10m",
            "location_id",
            "ref_id",
            "sample_id",
        ]

        return sample + (row[attrs].to_dict(),)


def embeddings_from_parquet_file(
    parquet_file: Union[str, Path],
    pretrained_model: Presto,
    sample_repeats: int = 1,
    mask_ratio: float = 0.0,
    valid_date_as_token: bool = False,
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

    # Convert to wide format
    logger.info("From long to wide format ...")
    df = process_parquet(df).reset_index()

    # Check if samples remain
    if df.empty:
        logger.warning("Empty dataframe: returning None")
        return None

    # Create dataset and dataloader
    logger.info("Making data loader ...")
    ds = WorldCerealInputsDataset(
        df,
        task_type="cropland",
        augment=True,
        mask_ratio=mask_ratio,
        repeats=sample_repeats,
    )

    if len(ds) == 0:
        logger.warning("No valid samples in dataset: returning None")
        return None

    dl = DataLoader(
        ds,
        batch_size=2048,
        shuffle=True,
        num_workers=4,
    )

    # Initialize final dataframe
    final_df = None

    # Iterate through dataloader to consume all samples
    for x, _, dw, latlons, month, valid_month, variable_mask, attrs in tqdm(dl):
        x_f, dw_f, latlons_f, month_f, valid_month_f, variable_mask_f = [
            t.to(device) for t in (x, dw, latlons, month, valid_month, variable_mask)
        ]

        # Compute Presto embeddings; only feed valid date as token if valid_date_as_token is True
        with torch.no_grad():
            encodings = (
                pretrained_model.encoder(
                    x_f,
                    dynamic_world=dw_f.long(),
                    mask=variable_mask_f,
                    latlons=latlons_f,
                    month=month_f,
                    valid_month=valid_month_f if valid_date_as_token else None,
                )
                .cpu()
                .numpy()
            )

        # Convert to dataframe
        attrs = pd.DataFrame.from_dict(attrs)
        encodings = pd.DataFrame(
            encodings, columns=[f"presto_ft_{i}" for i in range(encodings.shape[1])]
        )
        result = pd.concat([encodings, attrs], axis=1)

        # Append to final dataframe
        final_df = result if final_df is None else pd.concat([final_df, result])

    return final_df


def main(
    infile,
    outfile,
    presto_model,
    sc=None,
    debug=False,
    sample_repeats: int = 1,
    mask_ratio: float = 0.0,
    valid_date_as_token: bool = False,
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
    pretrained_model = Presto.load_pretrained(model_path=presto_model, strict=False)

    if sc is not None:
        logger.info(f"Parallelizing {len(parquet_files)} files ...")
        dfs = (
            sc.parallelize(parquet_files, len(parquet_files))
            .map(
                lambda x: embeddings_from_parquet_file(
                    x, pretrained_model, sample_repeats, mask_ratio, valid_date_as_token
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
    sample_repeats = 3
    mask_ratio = 0.25
    valid_date_as_token = False
    presto_dir = Path("/vitodata/worldcereal/presto/finetuning")
    presto_model = (
        presto_dir
        / "presto-ss-wc-ft-ct_100epochs_30D_random_CROPLAND2_time-token=none_balance=True_augment=True_2017=True.pt"
    )

    if spark:
        from worldcereal.utils.spark import get_spark_context

        logger.info("Setting up spark ...")
        sc = get_spark_context(localspark=localspark)
    else:
        sc = None

    infile = baseoutdir / "worldcereal_training_data.parquet"

    if debug:
        outfile = baseoutdir / (
            f"training_df_{presto_model.stem}_presto-worldcereal_DEBUG.parquet"
        )
    else:
        outfile = baseoutdir / (
            f"training_df_{presto_model.stem}_presto-worldcereal.parquet"
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
        mask_ratio=mask_ratio,
        valid_date_as_token=valid_date_as_token,
    )
