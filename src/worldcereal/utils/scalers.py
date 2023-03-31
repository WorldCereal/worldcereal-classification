import sys
if sys.version_info.minor < 7:
    from importlib_resources import open_text
else:
    from importlib.resources import open_text

from loguru import logger
import numpy as np
import pandas as pd


with open_text('worldcereal.resources', 'scaleranges.json') as f:
    _SCALERANGES = pd.read_json(f)
    _FEATURENAMES = _SCALERANGES.columns.tolist()


def minmaxscaler(data,
                 ft_name,
                 minscaled=0,
                 maxscaled=1,
                 clamp=(None, None),
                 nodata=None):
    """Function to scale an input feature to min/max range
       Note that when clamping is applied, scaled features
       could still exceed the min/max ranges.
    Args:
        data (np.ndarray): Original feature input array
        ft_name (str): Name of the feature
        minscaled (float, optional): Desired min value to scale to.
                                     Defaults to 0.
        maxscaled (float, optional): Desired max value to scale to.
                                     Defaults to 1.
        clamp (tuple, optional): min/max values to clamp to
        nodata (float, optional): if specified, this value
                                     remains unchanged

    Raises:
        Exception: If feature is not known in the existing scale ranges

    Returns:
        np.ndarray: Scaled feature array
    """

    if 'biome' in ft_name:
        # Biome features are fractions in [0, 100] range
        return data / 100.

    if len(ft_name.split('-')) < 3:
        # Not a feature we can scale
        return data

    if 'ts' in ft_name.split('-')[2]:
        '''
        For time series features we should have one
        general TS feature
        '''

        ft_name = '-'.join([
            ft_name.split('-')[0],
            ft_name.split('-')[1],
            'ts',
            ft_name.split('-')[3]
        ])

    if ft_name not in _FEATURENAMES:
        raise ValueError(
            f'Feature `{ft_name}` not in known scalers')

    minvalue = _SCALERANGES[ft_name]['min']
    maxvalue = _SCALERANGES[ft_name]['max']

    if nodata is not None:
        # Due to numerical artefacts values can divert
        # just a tiny bit from true 0 while it's actually
        # no data.
        idxnodata = np.where(np.isclose(data, nodata))

    # Scale between minscaled and maxscaled
    datarescaled = (
        (maxscaled - minscaled) *
        (data - minvalue) /
        (maxvalue - minvalue)
        + minscaled
    )

    # Check if we need to clamp
    minclamp, maxclamp = clamp
    if minclamp is not None:
        datarescaled[datarescaled < minclamp] = minclamp
    if maxclamp is not None:
        datarescaled[datarescaled > maxclamp] = maxclamp

    # Put back nodata
    if nodata is not None:
        datarescaled[idxnodata] = nodata

    return datarescaled


def minmaxunscaler(data,
                   ft_name,
                   minscaled=0, maxscaled=1):
    """Function to unscale a scaled input feature back to
       original values.
    Args:
        data (np.ndarray): Scaled feature input array
        ft_name (str): Name of the feature
        minscaled (float, optional): Scaled min value.
                                     Defaults to 0.
        maxscaled (float, optional): Scaled max value.
                                     Defaults to 1.

    Raises:
        Exception: If feature is not known in the existing scale ranges

    Returns:
        np.ndarray: Original unscaled feature array
    """

    if 'biome' in ft_name:
        # Biome features are fractions in [0, 100] range
        return data * 100.

    if len(ft_name.split('-')) < 3:
        # Not a feature we can unscale
        return data

    if 'ts' in ft_name.split('-')[2]:
        '''
        For time series features we should have one
        general TS feature
        '''

        ft_name = '-'.join([
            ft_name.split('-')[0],
            ft_name.split('-')[1],
            'ts',
            ft_name.split('-')[3]
        ])

    if ft_name not in _FEATURENAMES:
        raise ValueError(
            f'Feature `{ft_name}` not in known scalers')

    # Unscale
    dataunscaled = (
        (data - minscaled) *
        (_SCALERANGES[ft_name]['max'] - _SCALERANGES[ft_name]['min']) /
        (maxscaled - minscaled) +
        _SCALERANGES[ft_name]['min']
    )

    return dataunscaled


def scale_df(df: pd.DataFrame,
             minscaled=0,
             maxscaled=1,
             clamp=(None, None),
             nodata=None):
    '''
    Helper function that invokes `minmaxscaler`
    directly on an entire pandas DataFrame
    '''

    nrfeatures = df.shape[1]

    logger.info((f'Scaling {nrfeatures} features to '
                 f'[{minscaled}, {maxscaled}] range'))

    dfscaled = pd.DataFrame(index=df.index)

    for ftname in df.columns.tolist():
        dfscaled[ftname + '_scaled'] = minmaxscaler(
            df[ftname].values,
            ftname,
            minscaled=minscaled,
            maxscaled=maxscaled,
            clamp=clamp,
            nodata=nodata,
        )

    return dfscaled
