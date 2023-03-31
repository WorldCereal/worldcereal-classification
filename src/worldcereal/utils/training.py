import glob
from pathlib import Path

import geopandas as gpd
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pandas as pd
from shapely.geometry import Point
from sklearn.utils import class_weight, resample
import tensorflow as tf


from worldcereal.utils.scalers import minmaxscaler, scale_df
from worldcereal.classification.weights import (get_refid_weight,
                                                load_refidweights,
                                                load_refid_lut)
from worldcereal.utils import aez

REFIDWEIGHTS = load_refidweights()


def remap_refids(ref_id_counts):

    refidlut = load_refid_lut().dropna()

    refidlut['CIB'] = ['_'.join(x.split('_')[:3]) for x in
                       refidlut['CIB'].values]
    refidlut['RDM'] = ['_'.join(x.split('_')[:3]) for x in
                       refidlut['RDM'].values]

    refidlut = refidlut.set_index('CIB')['RDM']

    return ref_id_counts.rename(index=refidlut.to_dict())


def get_tfrecord_files(indirs: list,
                       aez_ids: list = None):
    if type(aez_ids) is str or type(aez_ids) is int:
        aez_ids = [aez_ids]

    if type(indirs) is str:
        indirs = [indirs]

    all_tfrecord_files = []

    for indir in indirs:

        if aez_ids is None:
            tfrecord_files = glob.glob(str(Path(indir) / '**' / '*part*.gz'),
                                       recursive=True)
        else:
            tfrecord_files = []
            for aez_id in aez_ids:
                tfrecord_files += glob.glob(
                    str(Path(indir) / str(aez_id) / '*part*.gz'),
                    recursive=True)

        all_tfrecord_files += tfrecord_files

    if len(all_tfrecord_files) == 0:
        raise RuntimeError(('No matching TFrecord files'
                            f' found in `{indirs}`'))

    logger.info(f'TFrecord files found: {len(all_tfrecord_files)}')

    return all_tfrecord_files


def parse_record(example_proto, features, windowsize):
    """Helper function to parse TFrecords based on a recipee

    Args:
        example_proto (TFrecord): one TFrecord entry
        features (list): list of features in the TFrecord
        windowsize (int): patch size

    Returns:
        parsed TFrecord: parsed TFrecord
    """
    parse_features = {}
    for feature in features:
        parse_features[feature] = tf.io.FixedLenFeature(
            shape=[windowsize * windowsize], dtype=tf.int64
        )
    parse_features["location_id"] = tf.io.FixedLenFeature(shape=[],
                                                          dtype=tf.string)
    parse_features["split"] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)
    parse_features["aez_groupid"] = tf.io.FixedLenFeature(shape=[],
                                                          dtype=tf.int64)
    parse_features["aez_zoneid"] = tf.io.FixedLenFeature(shape=[],
                                                         dtype=tf.int64)
    parse_features["ref_id"] = tf.io.FixedLenFeature(shape=[],
                                                     dtype=tf.string)

    example = tf.io.parse_single_example(
        example_proto,
        features=parse_features
    )

    return example


def create_dataset(tfrecordfiles: list, features: list,
                   windowsize=64, batchsize=1):
    """Function to create a TFrecord dataset from TFrecord files

    Args:
        tfrecordfiles (list): list of paths to TFrecord files
        features (list): list of features present in TFrecords
        windowsize (int, optional): patch size. Defaults to 64.
        batchsize (int, optional): batch size. Defaults to 1.

    Returns:
        TFdataset: TFdataset for use in training
    """
    nrFiles = len(tfrecordfiles)

    compression_type = ""
    if tfrecordfiles[0].endswith(".gz"):
        compression_type = "GZIP"

    dataset = tf.data.Dataset.from_tensor_slices(
        tfrecordfiles).shuffle(nrFiles)

    # increase cycle length to read multiple files in parallel
    dataset = dataset.interleave(
        lambda path: (tf.data
                      .TFRecordDataset(path,
                                       compression_type=compression_type)),
        cycle_length=4,
        block_length=1,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    return (
        dataset.map(lambda x: parse_record(x, features, windowsize),
                    num_parallel_calls=nrFiles)
        .repeat()
        .batch(batchsize)
        .prefetch(5)
    )


def convert_inputs(inputbatch, inputfeatures: list, outputname: str,
                   targetlabels: list, scalefactor: int = 10000,
                   ignorelabels: list = None, focuslabels: list = None,
                   targetmultiplier: int = 1):
    """
    Function to convert raw TFrecords to training-ready numpy arrays
    Here we implement pixel-based loss weighting as well.

    Args:
        inputbatch (tfrecords): batch of TFrecords
        inputfeatures (list): list of features to extract from raw TFrecord
        outputname (str): name of the output label to extract
        targetlabels (list): list of target label values that are positives
        scalefactor (int, optional): scale factor to apply. Defaults to 10000.
        ignorelabels (list, optional): list of target label values to ignore
        focuslabels (list, optional): list of target label values
                                      to double weights
        targetmultiplier (int, optional): multiplier for weights of
                          target class (defaults to 1)

    Returns:
        tuple: numpy arrays of (inputs, outputs, weights)
    """

    inputs = []
    for feature in inputfeatures:
        inputs.append(
            minmaxscaler(
                np.expand_dims(
                    inputbatch[
                        feature].numpy().astype(
                            float), axis=2) / scalefactor,
                ft_name=feature)
        )

    inputs = np.concatenate(tuple(inputs), axis=2)
    outputs = np.expand_dims(
        inputbatch[outputname].numpy().astype(int) / scalefactor,
        axis=2)
    weights = np.ones_like(outputs)

    # First set weights to zero if we don't know the label
    weights[outputs == 0] = 0

    # Put weights of ignorelabels also to zero
    if ignorelabels is not None:
        for ignore in ignorelabels:
            weights[outputs == ignore] = 0

    # Multiply the weights by the ref_id weighting factor
    ref_ids = inputbatch['ref_id'].numpy().astype(str)
    ref_id_weights = [get_refid_weight(ref_id, outputname,
                                       refidweights=REFIDWEIGHTS)
                      for ref_id in ref_ids]
    ref_id_weights = np.expand_dims(ref_id_weights, axis=(1, 2))
    ref_id_weights = np.repeat(ref_id_weights, outputs.shape[1], axis=1)
    weights = np.multiply(weights, ref_id_weights)

    # Trpile the weights of the focus labels
    if focuslabels is not None:
        for focuslabel in focuslabels:
            weights[outputs == focuslabel] *= 3

    # Now make the classification binary
    for t in targetlabels:
        outputs[outputs == t] = 1  # cropland
    outputs[outputs != 1] = 0  # no cropland

    # Finally apply multiplier to the weights of the target
    weights[outputs == 1] *= targetmultiplier

    # Get rid of any NaN values in the inputs
    # to avoid breaking the neural net
    inputs[np.isnan(inputs)] = 0

    return inputs, outputs, weights.squeeze()


def data_generator(dataset_iterator, inputfeatures, outputname,
                   targetlabels, ignorelabels=None, focuslabels=None,
                   targetmultiplier=1, windowsize=64,
                   requestedwindowsize=None, batchsize=16,
                   clamp=(None, None), forcebinaryweights=False,
                   forcepositives=False,
                   samplepatience=1000,
                   ):
    """Function that acts as data generator for keras model training

    Args:
        dataset_iterator (iterator): a TFdataset iterator object
        inputfeatures (list): list of input features to parse
        outputname (str): name of the output label in the TFdataset
        targetlabels (list): list of target label values that are positives
        ignorelabels (list, optional): list of target label values to ignore
        focuslabels (list, optional): list of target label values
                                      to double weights
        targetmultiplier (int, optional): multiplier for weights of
                          target class (defaults to 1)
        windowsize (int, optional): size of the patch. Defaults to 64.
        requestedwindowsize (int, optional): optional random subset to cut out
        batchsize (int, optional): batch size. Defaults to 16.
        clamp (tuple, optional): min/max clamp values to apply on inputs
        forcebinaryweights (bool, optional): if True, force weights to be [0,1]
        forcepositives (bool, optional): if True, a sample must include
                  at least one positive pixel
        samplepatience (int, optional): how many samples we try if
                  forcepositives is True

    Yields:
        tuple: numpy arrays (inputs, outputs, weights))
    """

    requestedwindowsize = requestedwindowsize or windowsize

    while True:

        inputs = np.empty((batchsize, windowsize * windowsize,
                           len(inputfeatures)))
        outputs = np.empty((batchsize, windowsize * windowsize, 1))
        weights = np.empty((batchsize, windowsize * windowsize))

        for i in range(batchsize):

            if not forcepositives:
                # Get a sample
                sample = next(dataset_iterator)
            else:
                attempts = 0
                while True:

                    attempts += 1

                    if attempts > samplepatience:
                        raise RuntimeError(('Could not find an appropriate '
                                            f'sample after {samplepatience} '
                                            'attempt(s).'))

                    # Get a sample
                    sample = next(dataset_iterator)

                    # Check if we have a positive pixel
                    uniquelabels = np.unique(
                        (sample['CT'].numpy().ravel()/10000).astype(int))

                    if np.in1d(uniquelabels,
                               targetlabels,
                               assume_unique=True).any():

                        logger.debug((f'Good sample after {attempts} '
                                      'attempt(s) ...'))
                        break

                    else:
                        continue

            # Convert the data
            inputs[i, ...], outputs[i, ...], weights[i, ...] = convert_inputs(
                sample,
                inputfeatures,
                outputname,
                targetlabels,
                ignorelabels=ignorelabels,
                focuslabels=focuslabels,
                targetmultiplier=targetmultiplier)

            if forcebinaryweights:
                weights[weights > 1] = 1

            # Check if we need to clamp
            minclamp, maxclamp = clamp
            if minclamp is not None:
                inputs[inputs < minclamp] = minclamp
            if maxclamp is not None:
                inputs[inputs > maxclamp] = maxclamp

        if requestedwindowsize == windowsize:
            # Return for training
            yield inputs, outputs, weights

        elif requestedwindowsize > windowsize:
            raise ValueError(('`requestedwindowsize` cannot be larger '
                              'than `windowsize.`'))

        else:
            # Need to cut out a random part
            origindex = np.arange(windowsize ** 2).reshape(
                (windowsize, windowsize)
            )
            # Sample a random window
            randomstartX = np.random.randint(
                low=0,
                high=windowsize-requestedwindowsize)
            randomstartY = np.random.randint(
                low=0,
                high=windowsize-requestedwindowsize)
            sampledindex = origindex[
                randomstartX: randomstartX+requestedwindowsize,
                randomstartY: randomstartY+requestedwindowsize]
            sampledindex = sampledindex.ravel()

            # Return subsample for training
            yield (inputs[:, sampledindex, :],
                   outputs[:, sampledindex, :],
                   weights[:, sampledindex])


def get_sample_weight(sample, options):

    # Default weight
    if sample['OUTPUT'] == 0:
        weight = 0
    elif sample['OUTPUT'] in options.get('ignorelabels', []):
        weight = 0
    else:
        weight = 1

    # Multiply the weights by the ref_id weighting factor
    ref_id = sample['ref_id']
    ref_id_weight = get_refid_weight(ref_id, options['outputlabel'],
                                     refidweights=REFIDWEIGHTS)
    weight *= ref_id_weight / 100.

    # Multiply focuslabel weight with provided multiplier
    if sample['OUTPUT'] in options.get('focuslabels', []):
        weight *= options.get('focusmultiplier', 1)

    return float(weight)


def select_within_aez(df, aez_zone, aez_group, buffer):
    '''
    Subset the dataframes
    We use the provided buffer for this and select all samples within
    the buffered polygon(s)
    '''

    # Create points from lat/lon in DF
    points = [Point(x) for x in zip(df['lon'], df['lat'])]

    # Make gdf from df
    df = gpd.GeoDataFrame(df, geometry=points, crs='epsg:4326')

    # Load AEZ
    aez_df = aez.load()

    # Subset on group or zone
    if aez_zone is not None:
        aez_df = aez_df[aez_df['zoneID'] == aez_zone]
    if aez_group is not None:
        aez_df = aez_df[aez_df['groupID'] == aez_group]

    # Check how many samples are truly in AEZ for reporting
    logger.info('Intersecting samples with original AEZ ...')
    nr_within = df.within(aez_df.unary_union).sum()

    # Buffer the zone/group
    # buffer is in m!
    if buffer > 0:
        logger.info(('Buffering selected AEZs with '
                     f'buffer: {buffer} m'))
        aez_df['geometry'] = aez_df.to_crs(epsg=3857).buffer(
            buffer).to_crs(epsg=4326)

    # Finally intersect df with the buffered AEZ
    logger.info('Intersecting samples with buffered AEZ ...')
    df = df[df.within(aez_df.buffer(0).unary_union)]

    # Report on selected samples
    nr_surrounding = df.shape[0] - nr_within
    logger.info(f'{nr_within + nr_surrounding} samples selected '
                f'({nr_within} inside AEZ and {nr_surrounding} '
                'within the buffer)')

    return pd.DataFrame(df.drop(columns=['geometry']))


def select_within_realm(df, realm_id, buffer):
    '''
    Subset the dataframes
    We use the provided buffer for this and select all samples within
    the buffered polygon(s)
    '''

    # Create points from lat/lon in DF
    points = [Point(x) for x in zip(df['lon'], df['lat'])]

    # Make gdf from df
    df = gpd.GeoDataFrame(df, geometry=points, crs='epsg:4326')
    df = df.to_crs(epsg=3857)

    # Load Realm
    realm_df = gpd.read_file('/data/worldcereal/auxdata/realms_simplified.gpkg')
    realm_df = realm_df.to_crs(epsg=3857)

    # Subset on realm
    realm_df = realm_df[realm_df['REALM'] == int(realm_id)]

    # Check how many samples are truly in REALM for reporting
    logger.info('Intersecting samples with original REALM ...')
    nr_within = df.within(realm_df.unary_union).sum()

    # Buffer the realm
    # buffer is in meters!
    if buffer > 0:
        logger.info(('Buffering selected REALM with '
                     f'buffer: {buffer} m'))
        realm_df['geometry'] = realm_df.buffer(buffer)

    # Finally intersect df with the buffered REALM
    logger.info('Intersecting samples with buffered REALM ...')
    df = df[df.within(realm_df.buffer(0).unary_union)]

    # Report on selected samples
    nr_surrounding = df.shape[0] - nr_within
    logger.info(f'{nr_within + nr_surrounding} samples selected '
                f'({nr_within} inside REALM and {nr_surrounding} '
                'within the buffer)')

    return pd.DataFrame(df.drop(columns=['geometry']))


def rebalance_data(inputs, outputs, weights, origoutputs, irr_ratio):

    # check number of samples
    nirr = np.sum(outputs == 1)
    nnat = np.sum(outputs == 0)
    if nirr == 0:
        errormessage = ('Got no irrigation samples, '
                        'cannot continue!')
        raise NotEnoughSamplesError(errormessage)
    elif nnat == 0:
        errormessage = ('Got no rainfed samples, '
                        'cannot continue!')
        raise NotEnoughSamplesError(errormessage)

    # compute how many samples need to be added to get to desired distribution
    div = nirr / (nirr + nnat)
    if irr_ratio > div:
        n_IRR_new = int((-nnat * irr_ratio)/(irr_ratio - 1))
        n_NAT_new = nnat
    else:
        n_NAT_new = int((nirr / irr_ratio) - nirr)
        n_IRR_new = nirr

    # Calculate how much the dataset would need to be increased
    IRR_incr = n_IRR_new / nirr
    NAT_incr = n_NAT_new / nnat

    # raise error if increment would be too large
    if IRR_incr > 5:
        IRR_incr = 5
        logger.warning(('IRR_incr exceeded 5, adjusted to 5! '
                        '(less oversampling)'))

    if NAT_incr > 5:
        NAT_incr = 5
        logger.warning(('NAT_incr exceeded 5, adjusted to 5! '
                        '(less oversampling)'))

    logger.info(f'Oversampling irrigated samples with factor {IRR_incr}')
    logger.info(f'Oversampling rainfed samples with factor {NAT_incr}')

    # Get the id's of the samples that need to be added to the existing dataset
    new_ids = []
    if n_IRR_new - nirr > 0:
        irr_ids = np.where(outputs == 1)[0]
        new_ids.extend(resample(irr_ids, n_samples=n_IRR_new - nirr,
                                replace=True, random_state=0))
    if n_NAT_new - nnat > 0:
        nat_ids = np.where(outputs == 0)[0]
        new_ids.extend(resample(nat_ids, n_samples=n_NAT_new - nnat,
                                replace=True, random_state=0))

    # extend the dataset
    if len(new_ids) > 0:
        newinputs = inputs[new_ids, :]
        newoutputs = outputs[new_ids]
        newweights = weights[new_ids]
        neworigoutputs = origoutputs[new_ids]
        inputs = np.concatenate([inputs, newinputs], axis=0)
        outputs = np.concatenate([outputs, newoutputs])
        weights = np.concatenate([weights, newweights])
        origoutputs = np.concatenate([origoutputs, neworigoutputs])

    return inputs, outputs, weights, origoutputs


def sample_from_aez(df, samplenr=2500):

    logger.info(f'Sampling DF with maxsamplenr = {samplenr}')

    # Count samples per aez group
    aez_groupcounts = df['aez_groupid'].value_counts()

    # First take all samples from aez groups
    # with less than maxsamples
    subset_df = df[df['aez_groupid'].isin(
        aez_groupcounts[aez_groupcounts < samplenr].index.tolist())]

    if not len(subset_df) == len(df):
        # Then sample from the others
        sampled_df = df[df['aez_groupid'].isin(
            aez_groupcounts[aez_groupcounts >= samplenr].index.tolist())]
        sampled_df = sampled_df.groupby('aez_groupid').sample(samplenr)

        # Merge both DFs as final subsetted samples
        subset_df = subset_df.append(sampled_df)

    logger.info(f'Retained {len(subset_df)}/{len(df)} samples.')

    return subset_df


def filter_perennial(df):
    '''Label 12:
        CT label
            7900 -> LC 10
            7910 -> LC 10
            7920 -> LC 10
            9520 -> LC 10
    '''

    nr_perennial_before = (df['OUTPUT'] == 12).sum()
    logger.info(
        f'Perennial cropland samples before filtering: {nr_perennial_before}')

    df.loc[df['CT'].isin([7900, 7910, 7920, 9520]), 'OUTPUT'] = 10

    nr_perennial_after = (df['OUTPUT'] == 12).sum()

    logger.info(('Perennial cropland samples switched to annual '
                 f'cropland (LC 10): {nr_perennial_before - nr_perennial_after}'))

    return df


def filter_cropland_11(df):
    '''Helper function to remove outliers from cropland class
    '''

    nr_cropland = (df['OUTPUT'] == 11).sum()

    logger.info(f'Annual cropland samples before filtering: {nr_cropland}')

    ignore_types = [9120, 9110, 7910, 7920, 7900, 9100]

    # Cropland should not have crop type from ignore list
    df = df[~((df['OUTPUT'] == 11) & (df['CT'].isin(ignore_types)))]

    nr_cropland = (df['OUTPUT'] == 11).sum()

    logger.info(f'Annual cropland samples after filtering: {nr_cropland}')

    return df


def filter_croptype(df):
    '''Helper function to remove outliers for croptype processors
    '''

    nr_samples = df.shape[0]
    logger.info(f'Crop type samples before filtering: {nr_samples}')

    # Cropland should not have corner reflector signal in SAR
    df = df[~(df['SAR-VV-p90-20m'] > 0)]

    # Cropland should not have a NDVI p90 which is very low
    df = df[~(df['OPTICAL-ndvi-p90-10m'] < 0.4)]

    nr_samples = df.shape[0]

    logger.info(f'Crop type samples after filtering: {nr_samples}')

    return df


def filter_cropland_10(df):
    '''Helper function to remove outliers from cropland class
    '''

    nr_cropland = (df['OUTPUT'] == 10).sum()

    logger.info(f'Cropland 10 samples before filtering: {nr_cropland}')

    # Cropland should have large NDVI-IQR
    df = df[~((df['OUTPUT'] == 10) & (df['OPTICAL-ndvi-iqr-10m'] < 0.20))]

    # Cropland should have low NDVI-P10
    df = df[~((df['OUTPUT'] == 10) & ((df['OPTICAL-ndvi-p10-10m'] < 0) |
                                      (df['OPTICAL-ndvi-p10-10m'] > 0.35)))]

    # Cropland should have high NDVI-P90
    df = df[~((df['OUTPUT'] == 10) & (df['OPTICAL-ndvi-p90-10m'] < 0.6))]

    nr_cropland = (df['OUTPUT'] == 10).sum()

    logger.info(f'Cropland 10 samples after filtering: {nr_cropland}')

    return df


def filter_grassland(df):

    nr_grassland = (df['OUTPUT'] == 13).sum()

    logger.info(f'Grassland samples before filtering: {nr_grassland}')

    # Filter out grassland which is cropland according to worldcover
    # AND potapov
    df = df[~((df['OUTPUT'] == 13) & (df['WORLDCOVER-LABEL-10m'] == 40)
              & (df['POTAPOV-LABEL-10m'] == 1))]

    nr_grassland = (df['OUTPUT'] == 13).sum()

    logger.info(f'Grassland samples after filtering: {nr_grassland}')

    return df


def filter_ewoco(df):

    nr_crop = (df['OUTPUT'] == 11).sum()

    logger.info(f'Cropland samples before EWOCO filtering: {nr_crop}')

    # Filter out crop from ewoco
    # AND potapov
    df = df[~((df['OUTPUT'] == 11) & (df['ref_id'].str.contains('ewoco')))]

    nr_crop = (df['OUTPUT'] == 11).sum()

    logger.info(f'Cropland samples after EWOCO filtering: {nr_crop}')

    return df


def filter_otherlayers(df, season):
    '''Method to filter out training labels
    that do not correspond to some other values in
    other land cover layers
    '''

    if 'tc-annual' in season or 'cropland' in season:
        label = 'OUTPUT'
    else:
        label = 'LC'

    # 01 - remove pixels that are urban according to worldcover
    agri_labels = [10, 11, 12]
    ewoc_ignore = [50]
    if 'tc-annual' in season or 'cropland' in season:
        remove_idx = ((df[label].isin(agri_labels)) &
                      (df['WORLDCOVER-LABEL-10m'].isin(ewoc_ignore)))
    else:  # crop type case
        remove_idx = df['WORLDCOVER-LABEL-10m'].isin(ewoc_ignore)

    df = df[~remove_idx].copy()
    logger.info(f'Removed {remove_idx.sum()} urban samples'
                f' according to worldcover information.')

    # 02 - remove pixels that are not crop [10, 11] according to BOTH
    # WorldCover and Potapov crop layer
    remove_idx = ((df[label].isin([10, 11])) &
                  (df['WORLDCOVER-LABEL-10m'] != 40) &
                  (df['POTAPOV-LABEL-10m'] != 1))
    df = df[~remove_idx].copy()
    logger.info(f'Removed {remove_idx.sum()} crop samples'
                ' that are not crop according '
                'to worldcover and potapov.')

    # 03 - remove pixels that are crop according to BOTH
    # WorldCover and Potapov crop layer but not in the label
    remove_idx = ((~df[label].isin([10, 11, 12])) &
                  (df['WORLDCOVER-LABEL-10m'] == 40) &
                  (df['POTAPOV-LABEL-10m'] == 1))
    df = df[~remove_idx].copy()
    logger.info(f'Removed {remove_idx.sum()} non-crop samples'
                ' that are crop according '
                'to both worldcover and potapov.')


def binary_balance_within_aez(df, targetlabels, pos_neg_ratio=None,
                              use_expected_cropfraction=False):
    """This method computes sample weights at the individual AEZ
    level to reach a predefined requested balancing ratio.
    Balancing happens at the binary label level.

    Args:
        df (pd.DataFrame): dataframe containing the
            training data and labels
        targetlabels (list): list of OUTPUT labels that are considered
            positive samples. Others will be treated as negative samples
        pos_neg_ratio (float, optional): the desired ratio positive
            vs negative samples to compute the weights.
        use_expected_cropfraction (bool, optional): if True,
            match the expected crop fraction within an AEZ according
            to the crop fraction in this AEZ
            derived from Potapov et al.

    Returns:
        pd.DataFrame: Balance dataframe
    """

    logger.info('Computing weights to balance samples within each AEZ ...')

    # Reset DF index so index is unique
    df = df.reset_index(drop=True)

    if use_expected_cropfraction:
        from worldcereal.utils import aez
        aez_layer = aez.load()
        if pos_neg_ratio is not None:
            raise ValueError(('`pos_neg_ratio` cannot be set when'
                              ' `use_expected_cropfraction` is True.'))
        logger.warning(('Using expected crop fraction from '
                        'Potapov et al. for balancing samples'))

    pos_neg_ratio = pos_neg_ratio or 1  # Default value

    # Get unique AEZ zones
    unique_zones = df['aez_zoneid'].unique().tolist()

    # Make classification binary
    outputs = df['OUTPUT'].copy()
    outputs[outputs.isin(targetlabels)] = 1
    outputs[outputs != 1] = 0
    df['OUTPUT_remapped'] = outputs.values

    # Initialize new dataframe, we add a sample weight colums
    sampleweight = pd.Series(index=df.index, dtype=np.float32)

    # Set maxweight
    maxweight = 10

    # Sample classes within each AEZ
    # We use the desired balance between 1 and 0 class.
    for aez_zone in unique_zones:
        subset = df[df['aez_zoneid'] == aez_zone]
        subset_index = df[df['aez_zoneid'] == aez_zone].index

        if len(subset['OUTPUT_remapped'].unique()) == 1:
            singleclass = int(subset['OUTPUT_remapped'].unique()[0])
            # Only one class present, take default weight
            logger.info((f'Only class `{singleclass}` found in AEZ {aez_zone}: '
                         'taking default weight of 1.'))
            sampleweight.loc[subset_index] = 1
            continue

        if use_expected_cropfraction:
            expected_ratio = aez_layer.set_index('zoneID').loc[
                aez_zone]['potapov_cropfraction']
            # We set a minimum ratio of 0.01
            pos_neg_ratio = max(0.01, expected_ratio)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            # Compute class weight to get balanced samples
            balanced_classweight = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=[0, 1],
                y=subset['OUTPUT_remapped'].values)

            # Adjust balanced weight according to requested ratio
            negative_classweight = (1 - pos_neg_ratio) / 0.5 * balanced_classweight[0]
            positive_classweight = pos_neg_ratio / 0.5 * balanced_classweight[1]

            # Clamp max weight to avoid excesses
            positive_classweight = min(positive_classweight, maxweight)
            negative_classweight = min(negative_classweight, maxweight)

            logger.info(('Computed positive weight for AEZ '
                         f'{aez_zone}: {positive_classweight}'))
            logger.info(('Computed negative weight for AEZ '
                         f'{aez_zone}: {negative_classweight}'))

        weights = np.ones((subset.shape[0]))
        weights[subset['OUTPUT_remapped'] == 0] = negative_classweight
        weights[subset['OUTPUT_remapped'] == 1] = positive_classweight

        sampleweight.loc[subset_index] = weights

    # Add sampleweight to DF and limit the values
    df['sampleweight'] = sampleweight

    # Drop temporary attribute
    df = df.drop('OUTPUT_remapped', axis=1)

    # shuffle
    df = df.sample(frac=1, random_state=99)

    return df


def get_trainingdata(df, inputfeatures, season, options, aez_zone=None,
                     aez_group=None, realm_id=None, minsamples=1000,
                     logdir=None, filter_worldcover=False,
                     remove_outliers=False, outlierinputs=None, buffer=500000,
                     scale_features=True, impute_missing=True,
                     return_pandas=False, outlierfraction=0.01,
                     irr_ratio=None):
    """Function that returns inputs/outputs from training DataFrame

    Args:
        df (pd.DataFrame): dataframe containing input/output features
        inputfeatures (list[str]): list of inputfeatures to be used
        season (str): season identifier
        options (dict): dictionary containing options
        aez_zone (int, optional): ID of the AEZ to subset on. Defaults to None.
        aez_group (int, optional): AEZ group ID to subset on. Defaults to None.
        minsamples (int, optional): minimum number of samples. Defaults to 500.
        buffer (int, optional): the buffer (in m) to take around AEZs before
            selecting matching location_ids. Defaults to 500000 (500km)
        logdir (str, optional): output path for logs/figures. Defaults to None.
        filter_worldcover (bool, optional): whether or not to
            remove outliers based on worldcover data. Defaults to False.
        remove_outliers (bool, optional): whether or not to remove
            outliers from the class
            of interest based on KNN (pyod implementation). Defaults to False.
        outlierinputs (list, optional): in case outliers need to be removed,
            this list specifies which input variables need to be used here.
            Defaults to None.
        scale_features (bool, optional): If True, input features are scaled.
        impute_missing (bool, optional): If True, impute NaN by 0

    Raises:
        NotEnoughSamplesError: obviously when not enough samples were found

    Returns:
        inputs, outputs, weights: arrays to use in training
    """

    # ----------------------------------------------------------------------
    # PART I: Check sample sizes and remove labels to be ignored

    # Check if we have enough samples at all to start with
    # that's at least 2X minsamples (binary classification)
    if df.shape[0] < 2 * minsamples:
        errormessage = (f'Got less than {2 * minsamples} '
                        f'in total for this dataset. '
                        'Cannot continue!')
        logger.error(errormessage)
        raise NotEnoughSamplesError(errormessage)

    # Remove the unknown and ignore labels
    if len(options.get('ignorelabels', [])) > 0:
        remove_idx = ((df['OUTPUT'] == 0) |
                      (df['OUTPUT'].isin(options['ignorelabels'])))
    else:
        remove_idx = (df['OUTPUT'] == 0)

    df = df[~remove_idx].copy()
    logger.info(f'Removed {remove_idx.sum()} unknown/ignore samples.')

    # ----------------------------------------------------------------------
    # PART II: filter training data based on AEZ or REALM

    if aez_zone is not None and aez_group is not None:
        raise ValueError('Cannot set both `aez_zone` and `aez_group`')

    elif 'tc-annual' not in season and 'cropland' not in season:
        # No spatial sampling for croptype detectors
        pass

    # Select samples within (buffered) AEZ
    if aez_zone is not None or aez_group is not None:
        df = select_within_aez(df, aez_zone, aez_group, buffer)
    elif realm_id is not None:
        df = select_within_realm(df, realm_id, buffer)
    # else:
    #     df = sample_from_aez(df)

    # ----------------------------------------------------------------------
    # PART III: Apply various thematic filters

    if 'tc-annual' not in season and 'cropland' not in season:
        # If not looking at cropland
        # need to get rid of samples that are not part of cropland or grassland
        # We include grassland to make model aware of what grass looks like
        # in case there is grass commission inside cropland product.
        logger.info(('Removing samples that are not cropland or grassland'))
        df = df[df['LC'].isin([10, 11, 12, 13])].copy()

        logger.info(f'Unique LC labels: {df["LC"].unique()}')

    if 'tc-annual' in season or 'cropland' in season:
        # Rule 1: filter out dirty perennials
        # NOTE: NEEDS TO GO FIRST ALWAYS!
        df = filter_perennial(df)

        # Rule 2: filter the unkwown cropland
        df = filter_cropland_10(df)

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
        filter_otherlayers(df, season)

    # Remove corrupt rows
    remove_idx = ((df.isnull())).sum(axis=1) > int(df.shape[1] * 0.75)
    df = df[~remove_idx].copy()

    # do intermediate check of number of samples
    if df.shape[0] < 2 * minsamples:
        errormessage = (f'Got less than {2 * minsamples} '
                        f'in total for this dataset. '
                        'Cannot continue!')
        logger.error(errormessage)
        raise NotEnoughSamplesError(errormessage)

    # ----------------------------------------------------------------------
    # PART IV: Apply perturbations to localization features

    # Add random perturbations to lat/lon
    if 'lat' in df.columns:
        logger.info('Perturbing lat/lon ...')
        df['lat_orig'] = df['lat'].copy()
        df['lon_orig'] = df['lon'].copy()
        np.random.seed(1)

        # Perturb lat by a max of 2.5°
        df['lat'] = df['lat'] + (
            (np.random.rand(df.shape[0]) * 5 - 2.5))
        np.random.seed(2)

        # Perturb lon by a max of 10°
        df['lon'] = df['lon'] + (
            (np.random.rand(df.shape[0]) * 20 - 10))

        df.loc[df['lon'] < -180, 'lon'] += 180
        df.loc[df['lon'] > 180, 'lon'] -= 180

    # Perturb DEM by a max of 100m
    np.random.seed(3)
    logger.info('Perturbing DEM ...')
    df['DEM-alt-20m'] = df['DEM-alt-20m'] + (
        (np.random.rand(df.shape[0]) * 200 - 100))
    df.loc[df['DEM-alt-20m'] < 0, 'DEM-alt-20m'] = 0

    # Add random perturbations to biomes
    bio_cols = [col for col in df.columns if 'biome' in col]
    noise_val = 3
    for i, c in enumerate(bio_cols):
        np.random.seed(i)
        df[c] = df[c] + ((np.random.rand(df.shape[0]) * 2 - 1)
                         * noise_val * (df[c] > 0))
        df.loc[df[c] > 98, c] = 98
        df.loc[df[c] < 0, c] = 0

    # ----------------------------------------------------------------------
    # PART V: Apply simple per-class outlier detection

    # get rid of outliers
    if remove_outliers and outlierfraction > 0:
        if outlierinputs is None:
            raise ValueError('No outlier inputs provided!')
        from pyod.models.ecod import ECOD

        # split the dataset and run outlier removal on only one part
        nremoved = 0
        dfs = []
        logger.info(f'Obs before OD: {df.shape[0]}')
        if df.shape[0] > 0:
            unique_labels = df['OUTPUT'].unique()
            ref_ids = df['ref_id'].unique()
            for label in unique_labels:
                for ref_id in ref_ids:
                    # logger.info(('Outlier detection for ref_id '
                    #              f'{ref_id} and label: {label}'))
                    dftoclean = df[(df['OUTPUT'] == label) &
                                   (df['ref_id'] == ref_id)]
                    if dftoclean.shape[0] < 100:
                        # Not enough samples for this label: skip
                        # OD routine
                        dfs.append(dftoclean)
                        continue
                    # get the variables used for outlier removal
                    outlier_x = dftoclean[outlierinputs].values
                    # Get rid of any existing NaN values
                    outlier_x[np.isnan(outlier_x)] = 0
                    # fit the model
                    clf = ECOD(contamination=outlierfraction)
                    clf.fit(outlier_x)
                    # get the prediction labels
                    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)  # NOQA
                    nremoved += y_train_pred.sum()
                    retain_idx = np.where(y_train_pred == 0)
                    dfclean = dftoclean.iloc[retain_idx]
                    dfs.append(dfclean)
        # merge all cleaned dataframes
        if len(dfs) == 0:
            raise NotEnoughSamplesError('Not enough samples to perform OD!')
        df = pd.concat(dfs)
        logger.info(f'Removed {nremoved} outliers')
        logger.info(f'Obs after OD: {df.shape[0]}')

    # ----------------------------------------------------------------------
    # PART VI: Select features we need
    required_columns = list(set(inputfeatures +
                                ['OUTPUT', 'location_id', 'ref_id',
                                 'aez_groupid', 'aez_zoneid',
                                 'lat_orig', 'lon_orig']))
    df = df[required_columns]

    # ----------------------------------------------------------------------
    # PART VII: Remove samples with NaN values
    if 'tc-annual' in season or 'cropland' in season:
        # Remove rows with NaN
        beforenan = df.shape[0]
        df = df.dropna()
        afternan = df.shape[0]
        logger.info(f'Removed {beforenan - afternan} samples with NaN values.')

    # ----------------------------------------------------------------------
    # PART VIII: Compute weights
    # This happens in two parts:
    # 1. Determine class weights in order to reach requested ratio
    # 2. Adjust sample-specific weights based on conf score and label

    # Compute class weight to get balanced samples
    binaryoutputs = df['OUTPUT'].copy()
    binaryoutputs[binaryoutputs.isin(options['targetlabels'])] = 1
    binaryoutputs[binaryoutputs != 1] = 0

    # check whether we still have samples from each class
    if len(np.unique(binaryoutputs)) == 1:
        singleclass = int(np.unique(binaryoutputs)[0])
        # Only one class present, no use to continue
        raise NotEnoughSamplesError(f'Only class {singleclass} found, '
                                    'aborting!')

    # Compute class weights that would balance the two classes
    balanced_classweight = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=[0, 1],
        y=binaryoutputs.values)

    # Adjust balanced weight according to requested ratio
    pos_neg_ratio = options.get('pos_neg_ratio', 0.5)
    logger.info(f'Using pos-neg-ratio of {pos_neg_ratio}')
    negative_classweight = (1 - pos_neg_ratio) / 0.5 * balanced_classweight[0]
    positive_classweight = pos_neg_ratio / 0.5 * balanced_classweight[1]

    # Clamp max weight to avoid excesses
    MAX_WEIGHT = 10
    positive_classweight = min(positive_classweight, MAX_WEIGHT)
    negative_classweight = min(negative_classweight, MAX_WEIGHT)

    logger.info(('Computed global positive weight '
                 f'{aez_zone}: {positive_classweight}'))
    logger.info(('Computed global negative weight '
                 f'{aez_zone}: {negative_classweight}'))

    weights = np.ones((binaryoutputs.shape[0]))
    weights[binaryoutputs.values == 0] = negative_classweight
    weights[binaryoutputs.values == 1] = positive_classweight
    df['sampleweight'] = weights

    # Get the sample-specific weights
    sample_weights = df.apply(lambda row: get_sample_weight(
        row, options), axis=1).values

    # Adjust sample weights by the class weights and assign as final weights
    sample_weights *= df['sampleweight'].values
    df['sampleweight'] = sample_weights

    # ----------------------------------------------------------------------
    # PART IX: Scale inputs if needed for the model

    # Scale the input data
    if scale_features:
        # Input clamping
        minclamp = -0.1  # Do not allow inputs below this value (None to disable)
        maxclamp = 1.1  # Do not allow inputs above this value (None to disable)
        df[inputfeatures] = scale_df(df[inputfeatures],
                                     clamp=(minclamp, maxclamp),
                                     nodata=0)

    # ----------------------------------------------------------------------
    # PART X: Log various things on the situation as of now

    # Log the number of samples still present
    # per ref_id
    ref_id_counts = (df.groupby('ref_id')[
        'OUTPUT'].value_counts().unstack().fillna(0).astype(int))
    if logdir is not None:
        if not (Path(logdir) / 'sample_counts.csv').is_file():
            logger.info('Saving sample counts ...')
            ref_id_counts.to_csv(Path(logdir) / 'sample_counts.csv')
    ref_id_counts = ref_id_counts.sum(axis=1)
    ref_id_counts = remap_refids(ref_id_counts).to_dict()

    if logdir is not None:
        if not (Path(logdir) / 'output_distribution.png').is_file():
            # Plot histogram of original outputs
            outputs = df['OUTPUT'].copy()
            counts = outputs.value_counts()
            labels = counts.index.astype(int)
            plt.bar(range(len(labels)), counts.values)
            plt.xticks(range(len(labels)), labels, rotation=90)
            plt.xlabel('Class')
            plt.ylabel('Amounts')
            plt.title('Output label distribution')
            outfile = Path(logdir) / 'output_distribution.png'
            outfile.parent.mkdir(exist_ok=True)
            plt.savefig(outfile)
            plt.close()

    # ----------------------------------------------------------------------
    # PART XI: Extract input/output as numpy arrays, binarize the outputs
    # and do one more check if we still have enough samples

    # Get the inputs
    inputs = df[inputfeatures].values

    # Get the outputs AFTER the weights
    outputs = df['OUTPUT'].copy()

    # Get the final sample weights
    weights = df['sampleweight'].values

    # Get the output labels
    origoutputs = np.copy(outputs.values)  # For use in evaluation

    # Make classification binary
    outputs[outputs.isin(options['targetlabels'])] = 1
    outputs[outputs != 1] = 0
    outputs = outputs.values

    # Now check if we have enough samples to proceed
    for label in [0, 1]:
        if np.sum(outputs == label) < minsamples:
            errormessage = (f'Got less than {minsamples} '
                            f'`{label}` samples for this dataset. '
                            'Cannot continue!')
            logger.error(errormessage)
            raise NotEnoughSamplesError(errormessage)

    # Log how many all-zero inputs (these are bad)
    idx = np.where(np.sum(inputs, axis=1) == 0)
    logger.info(f'#Rows with all-zero inputs: {len(idx[0])}')

    # Get location ids
    location_ids = df['location_id'].values

    # ----------------------------------------------------------------------
    # PART XII: Postprocessing before returning the results

    # Make sure all NaNs are gone!
    if impute_missing:
        if np.sum(np.isnan(inputs)) > 0:
            logger.warning(f'Removing {np.sum(np.isnan(inputs))} NaN values!')
        inputs[np.isnan(inputs)] = 0

    if return_pandas:
        logger.info('Transforming inputs to pandas.DataFrame ...')
        inputs = pd.DataFrame(data=inputs, columns=inputfeatures)

    if logdir is not None:
        # Write proc
        if not (Path(logdir) / 'training_df.csv').is_file():
            df.to_csv(Path(logdir) / 'training_df.csv')

    return (inputs, outputs, weights, location_ids,
            origoutputs, ref_id_counts)


def filter_springcereals(df):
    '''Helper function to filter a training_df for training
    as spring wheat/cereals classifier which only takes into account
    samples from AEZs that have turned on the trigger_sw flag.
    '''

    # Load AEZ
    aez_df = aez.load()

    logger.info('Filter DataFrame from spring cereals ...')

    # Check for each sample whether Spring Wheat is possible
    df['trigger_sw'] = aez_df.set_index('zoneID').loc[
        df['aez_zoneid'].values]['trigger_sw'].values

    df_filtered = df[df['trigger_sw'] == 1]

    logger.info((f'Retaining {len(df_filtered)}/{len(df)} samples '
                 'based on `trigger_sw` flag.'))

    return df_filtered


def get_pixel_data(season, options, inputfeatures,
                   aez_zone=None, aez_group=None,
                   realm_id=None, buffer=5,
                   logdir=None, outlierinputs=None, minsamples=500,
                   detector=None, **kwargs):
    '''Wrapper function around main `get_trainingdata` method to
    gather pixel-based training data for CAL/VAL/TEST samples.
    '''

    min_req_samples = {
        'cal': minsamples if minsamples is not None else None,
        'val': int(minsamples / 5) if minsamples is not None else None,
        'test': int(minsamples / 10) if minsamples is not None else None,
    }
    pixel_data = {}

    for dataset_type in ['cal', 'val', 'test']:

        dfs = (options[f'{dataset_type}_df_files'] if
               type(options[f'{dataset_type}_df_files'])
               is list else [options[f'{dataset_type}_df_files']])

        df_data = pd.DataFrame()

        for current_df in dfs:
            df_data = pd.concat(
                [df_data, pd.read_parquet(
                    (Path(current_df) /
                     f'training_df_{options["outputlabel"]}.parquet'))])

        # For annual cropland, we need to throw out some unreliable datasets
        if 'tc-annual' in season or 'cropland' in season:
            ignore_list = [
                '2017_',    # 2017 contains not enough data
                '2018_AF',  # Noisy points
                '2018_SSD_WFP',  # Needs to be checked thoroughly
                '2018_TZ_AFSIS',  # too tricky to use?
                '2018_TZ_RadiantEarth',  # Could use some cleaning
                '2019_AF_OAF',  # Useful but noisy!
                '2019_KEN_WAPOR',  # Might contain noise
                '2019_TZA_CIMMYT',  # Geolocation inaccuracies?
                '2019_TZA_OAF',  # Noisy!
                '2019_TZ_AFSIS',  # Noisy!
                '2020_RW_WAPOR-Muvu',  # Geolocation inaccuracies?
                '2018_ES_SIGPAC-Andalucia',
                '2019_ES_SIGPAC-Andalucia',
                '2021_MOZ_WFP',  # Unfortunately contains noise
                '2021_TZA_COPERNICUS-GEOGLAM',  # There might be noise
            ]
        else:
            # For crop type
            ignore_list = [
                '2021_TZA_COPERNICUS-GEOGLAM',  # Reprocessed under different name
                '2021_UKR_sunflowermap',  # used only for CCN
                '2021_EUR_EXTRACROPS',  # Too heavy bias to Europe
                '2017_'  # Data too old, limited satellite coverage
            ]

        logger.warning(f'{dataset_type.upper()} SAMPLES BEFORE THROWING OUT REFIDs: {df_data.shape[0]}')  # NOQA
        for ignore in ignore_list:
            df_data = df_data.loc[~df_data['ref_id'].str.contains(ignore)]  # NOQA
        logger.warning(f'{dataset_type.upper()} SAMPLES AFTER THROWING OUT REFIDs: {df_data.shape[0]}')  # NOQA

        # In case of dedicated spring wheat
        if detector == 'springwheat' or detector == 'springcereals':
            logger.info((f'Serving {detector} detector: '
                         'trigger filtering mechanism'))
            df_data = filter_springcereals(df_data)

        # Get the training data using all provided options
        data = get_trainingdata(
            df_data, inputfeatures, season,
            options,
            aez_zone=aez_zone, aez_group=aez_group,
            realm_id=realm_id, logdir=logdir,
            filter_worldcover=options.get('filter_worldcover', False),
            remove_outliers=options.get('remove_outliers', False),
            outlierinputs=outlierinputs,
            buffer=buffer,
            minsamples=min_req_samples[dataset_type],
            **kwargs)

        pixel_data[dataset_type] = data

    logger.info(f'Training on {pixel_data["cal"][0].shape[0]} samples.')
    logger.info(f'Validating on {pixel_data["val"][0].shape[0]} samples.')
    logger.info(f'Testing on {pixel_data["test"][0].shape[0]} samples.')

    return pixel_data['cal'], pixel_data['val'], pixel_data['test']


class NotEnoughSamplesError(Exception):
    pass
