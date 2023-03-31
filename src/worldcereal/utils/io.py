import atexit
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import rasterio
from loguru import logger
from satio.collections import BaseCollection
from satio.features import Features
from satio.utils.geotiff import get_rasterio_profile_shape

from worldcereal.collections import AgERA5YearlyCollection

VALUE_LIMITS = {
    'uint8': {
        'min_value': 0,
        'max_value': 245,
        'nodata_value': 255
    },
    'uint16': {
        'min_value': 0,
        'max_value': 65500,
        'nodata_value': 65535
    },
    'uint13': {
        'min_value': 0,
        'max_value': 8000,
        'nodata_value': 8191
    },
    'uint14': {
        'min_value': 0,
        'max_value': 16000,
        'nodata_value': 16383
    }
}


class CollectionError(Exception):
    pass


def _clean(path):
    """Helper function to cleanup a path

    Args:
        path (str): path to file or directory to remove
    """
    if Path(path).is_file():
        Path(path).unlink()
    elif Path(path).is_dir():
        shutil.rmtree(path)


def drop_unknown(inputs, outputs, nodatavalue=0):
    idx_known = np.where(outputs != nodatavalue)[0]
    return inputs[idx_known, :], outputs[idx_known]


def drop_nan(inputs, outputs=None):
    idx = np.where(np.sum(np.isfinite(inputs), axis=1) == inputs.shape[1])[0]

    if outputs is not None:
        return inputs[idx, :], outputs[idx]
    else:
        return inputs[idx, :]


def convert_to_binary(outputs, target_labels):
    if type(target_labels) is int:
        target_labels = [target_labels]
    outputs_bin = outputs.copy()
    outputs_bin[~np.isin(outputs, target_labels)] = 0
    outputs_bin[np.isin(outputs, target_labels)] = 1
    return outputs_bin


def raise_coll_gap_failure(gap, gapkind, threshold, coll):
    msg = (f'Incomplete collection `{coll}`: '
           f'got a value of {gap} days for `{gapkind}` '
           f'which exceeds the threshold of {threshold}.')
    logger.warning(msg)
    raise CollectionError(msg)


def raise_coll_size_failure(size, threshold, coll):
    msg = (f'Incomplete collection `{coll}`: '
           f'got a collection size of {size} '
           f'which is less than the threshold of {threshold}.')
    logger.warning(msg)
    raise CollectionError(msg)


def check_collection(collection: BaseCollection, name,
                     start_date, end_date, fail_threshold=1000,
                     min_size=2):
    """Helper function to check collection completeness
    and report gap lengths

    Args:
        collection (BaseCollection): a satio collection
        name (str): name of the collection used for reporting
        start_date (str): processing start date (Y-m-d)
        end_date (str): processing end date (Y-m-d)
        fail_threshold (int): amount of days beyond which
                        an error is raised that the collection is
                        incomplete.
        min_size (int): minimum amount of products to be present
                        in collection before failing anyway.

    """

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    collstart = collection.df.date.min()

    if type(collection) == AgERA5YearlyCollection:
        # Special case: assume full year products
        collend = pd.Timestamp(year=collection.df.date.max().year,
                               month=12, day=31)
    else:
        collend = collection.df.date.max()

    if type(collection) != AgERA5YearlyCollection:
        # Check whether we have the minimum amount of
        # products to not run into a pure processing issue
        collsize = collection.df.shape[0]
        if collsize < min_size:
            raise_coll_size_failure(collsize, min_size, name)

    # Check collection start
    gapstart = collstart - start_date
    gapstart = gapstart.days
    if gapstart < 0:
        gapstart = 0

    # Check collection end
    gapend = end_date - collend
    gapend = gapend.days
    if gapend < 0:
        gapend = 0

    # Check max gap
    if type(collection) == AgERA5YearlyCollection:
        # Special case: assume uninterrupted meteo data
        maxgap = 0
    else:
        maxgap = collection.df.date.diff().max().days

    # Report on collection
    logger.info('-' * 50)
    logger.info(f'{name} first image: {collstart}')
    logger.info(f'{name} last image: {collend}')
    logger.info(f'{name} largest gap: {maxgap}')

    # Fail processing if collection is incomplete
    if gapstart > fail_threshold:
        raise_coll_gap_failure(gapstart, 'gapstart', fail_threshold, name)
    if gapend > fail_threshold:
        raise_coll_gap_failure(gapend, 'gapend', fail_threshold, name)
    if maxgap > fail_threshold:
        raise_coll_gap_failure(maxgap, 'maxgap', fail_threshold, name)

    names = [f'{name}-gap_start', f'{name}-gap_end', f'{name}-gap_max']
    return [gapstart, gapend, maxgap], names


def _get_jp2_compression_profile(compress_tag):
    """ compress_tag e.g. 'jp2-uint{nbits}-q{quality}'
    tag = f'uint{nbits}-deflate-z{z}-lsb{lsb}'
    dtype_tag = f'uint{nbits}'
    """

    mbits = re.search(r'-uint(\d*)', compress_tag)
    nbits = int(mbits.group(1)) if mbits else 16

    mqual = re.search(r'-q(\d*)', compress_tag)
    quality = int(mqual.group(1)) if mqual else 100

    dtype = np.uint16 if nbits > 8 else np.uint8

    profile = {'driver': 'JP2OpenJPEG',
               'USE_TILE_AS_BLOCK': True,
               'quality': quality,
               'reversible': False,
               'resolutions': 1,
               'nbits': nbits}

    dtype_tag = f'uint{nbits}'

    if dtype_tag not in VALUE_LIMITS.keys():
        raise ValueError(
            f"dtype tag {dtype_tag} not supported. "
            f"Available profiles: {list(VALUE_LIMITS.keys())}")

    value_limits = VALUE_LIMITS[dtype_tag]

    return profile, value_limits, dtype


def _get_deflate_compression_profile(compress_tag):
    """ compress_tag e.g. 'jp2-uint{nbits}-q{quality}'
    """

    mbits = re.search(r'-uint(\d*)', compress_tag)
    nbits = int(mbits.group(1)) if mbits else 16

    mlsb = re.search(r'-lsb(\d*)', compress_tag)
    lsb = int(mlsb.group(1)) if mlsb else None

    mz = re.search(r'-z(\d*)', compress_tag)
    z = int(mz.group(1)) if mz else 6

    dtype = np.uint16 if nbits > 8 else np.uint8

    profile = {'tiled': False,
               'compress': 'deflate',
               'interleave': 'band',
               'predictor': 2,
               'discard_lsb': lsb,
               'zlevel': z,
               'nbits': nbits}

    dtype_tag = f'uint{nbits}'

    if dtype_tag not in VALUE_LIMITS.keys():
        raise ValueError(
            f"dtype tag {dtype_tag} not supported. "
            f"Available profiles: {list(VALUE_LIMITS.keys())}")

    value_limits = VALUE_LIMITS[dtype_tag]
    return profile, value_limits, dtype


def get_compression_profile(compress_tag):
    """ compress_tag e.g. 'jp2-uint{nbits}-q{quality}'
        or 'deflate-uint{nbits}-lsb{lsb}-z{zvalue}
    """
    if compress_tag.startswith('jp2'):
        return _get_jp2_compression_profile(compress_tag)
    elif compress_tag.startswith('deflate'):
        return _get_deflate_compression_profile(compress_tag)
    else:
        raise ValueError("Compress tag not recognized")


def compress_data(arr, dtype, *, min_value, max_value, nodata_value):

    offsets = np.nanmin(arr, axis=(1, 2)) - min_value
    offsets = np.expand_dims(offsets, (1, 2))
    arr2 = arr - np.broadcast_to(offsets, arr.shape)

    scales = np.nanmax(arr2, axis=(1, 2)) / max_value
    scales = np.expand_dims(scales, (1, 2))
    with np.errstate(divide='ignore', invalid='ignore'):
        arr2 = arr2 / np.broadcast_to(scales, arr.shape)

    arr2[~np.isfinite(arr)] = nodata_value

    return arr2.round().astype(dtype), np.squeeze(scales), np.squeeze(offsets)


def restore_data(arr, scales, offsets, nodata_value):
    """
    scales == max_vals
    offsets == min_vals
    """
    arr = arr.astype(np.float32)
    arr[arr == nodata_value] = np.nan

    scales = np.expand_dims(scales, (1, 2))
    arr = arr * np.broadcast_to(scales, arr.shape)

    offsets = np.expand_dims(offsets, (1, 2))
    arr = arr + np.broadcast_to(offsets, arr.shape)

    return arr


def write_geotiff_tags(arr,
                       profile,
                       filename,
                       colormap=None,
                       nodata=None,
                       tags=None,
                       bands_tags=None,
                       scales=None,
                       offsets=None,
                       band_names=None):
    """
    tags should be a dictionary
    bands_tags should be a list of dictionarites with len == arr.shape[0]
    """
    bands_tags = bands_tags if bands_tags is not None else []

    if nodata is not None:
        profile.update(nodata=nodata)

    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)

    if os.path.isfile(filename):
        os.remove(filename)

    with rasterio.open(filename, 'w', **profile) as dst:
        dst.write(arr)

        if tags is not None:
            dst.update_tags(**tags)

        for i, bt in enumerate(bands_tags):
            dst.update_tags(i + 1, **bt)

        if band_names is not None:
            for i, b in enumerate(band_names):
                dst.update_tags(i + 1, band_name=b)

        if colormap is not None:
            dst.write_colormap(
                1, colormap)

        if scales is not None:
            dst.scales = scales

        if offsets is not None:
            dst.offsets = offsets


def save_features_geotiff(feat: Features,
                          bounds: List = [0, 1, 0, 1],
                          epsg: int = 4326,
                          filename: str = None,
                          tags: Dict = None,
                          compress_tag: str = 'deflate-uint16',
                          **profile_kwargs):

    if filename is None:
        return

    # Write to temporary file in current directory
    tempfile = f'./{Path(filename).name}'
    atexit.register(_clean, tempfile)

    compress_profile, dtype_value_limits, dtype = get_compression_profile(
        compress_tag)

    logger.debug(f"Saving {filename}...")

    data = feat.data
    data, scales, offsets = compress_data(data, dtype, **dtype_value_limits)

    profile = get_rasterio_profile_shape(data.shape, bounds,
                                         epsg, dtype)

    profile.update(nodata=dtype_value_limits['nodata_value'])
    profile.update(**compress_profile)
    profile.update(**profile_kwargs)

    scales = np.squeeze(scales).tolist()
    offsets = np.squeeze(offsets).tolist()

    default_tags = {
        'bands': feat.names,
        'epsg': epsg,
        'offsets': offsets,
        'scales': scales
    }

    tags = tags or {}
    tags = {**default_tags, **tags}

    write_geotiff_tags(data, profile, tempfile, tags=tags,
                       scales=scales, offsets=offsets,
                       band_names=feat.names)

    # Now copy the files.
    # if XML files was created, need to copy this too!
    # this is for compatibility reasons. Normally, newly created
    # features will always have the metadata in the file itself.
    shutil.move(tempfile, filename)
    xmlfile = Path(tempfile.replace('.tif', '.tif.aux.xml'))
    if xmlfile.is_file():
        shutil.move(xmlfile, Path(filename).with_suffix('.tif.aux.xml'))

    return data, scales, offsets


def load_features_geotiff(feat_fn):

    with rasterio.open(feat_fn) as src:
        arr = src.read()
        scales = src.scales
        offsets = src.offsets
        nodata = src.nodata

        if (np.all(np.array(scales) == 1) and
                np.all(np.array(offsets) == 0)):
            raise ValueError(('All scales and offsets of features '
                              'file are default, meaning that the metadata '
                              'is missing. Cannot load features!'))

        bands = eval(src.tags()['bands'])

        arr = restore_data(arr, scales, offsets, nodata)

    feats = Features(arr, bands)

    return feats
