import numbers

from loguru import logger
import numpy as np
from skimage.morphology import footprints
from scipy.ndimage import binary_dilation, binary_erosion
from numba import njit

SCL_MASK_VALUES = [0, 1, 3, 8, 9, 10, 11]
SWETS_WEIGHTS = np.array([1.5,  # maximum
                          0.005,  # minimum
                          0.5,  # posslope
                          0.5,  # negslope
                          1.0,  # aboutequal
                          0.0])  # default
SWETS_WEIGHTS_ONE = np.ones(6)


def dilate_mask(mask, dilate_r):

    dilate_disk = footprints.disk(dilate_r)
    for i in range(mask.shape[0]):
        mask[i] = binary_dilation(mask[i], dilate_disk)

    return mask


def erode_mask(mask, erode_r):
    erode_disk = footprints.disk(erode_r)
    for i in range(mask.shape[0]):
        mask[i] = binary_erosion(mask[i], erode_disk)

    return mask


def scl_mask(scl_data,
             *,
             mask_values,
             erode_r=None,
             dilate_r=None,
             nodata=0,
             max_invalid_ratio=None,
             **kwargs):
    """
    From a timeseries (t, y, x) returns a binary mask False for the
    given mask_values and True elsewhere.

    Parameters:
    -----------
    slc_data: 3D array
        Input array for computing the mask

    mask_values: list
        values to set to False in the mask

    erode_r : int
        Radius for eroding disk on the mask

    dilate_r : int
        Radius for dilating disk on the mask

    nodata : int
        Nodata value used to count observations

    max_invalid_ratio : float
        Will set mask values to True, when they have an
        invalid_ratio > max_invalid_ratio

    Returns:
    --------
    mask : 3D bool array
        mask True for valid pixels, False for invalid

    obs : 2D int array
        number of valid observations (different from 0 in scl_data)

    valid_before : 2D float array
        percentage of valid obs before morphological operations

    valid_after : 2D float array
        percentage of valid obs after morphological operations
    """
    scl_data = np.squeeze(scl_data)

    ts_obs = scl_data != nodata

    obs = ts_obs.sum(axis=0)

    mask = np.isin(scl_data, mask_values)
    ma_mask = (mask & ts_obs)

    invalid_before = ma_mask.sum(axis=0) / obs * 100
    invalid_before = invalid_before.astype(int)

    if erode_r is not None:
        if erode_r > 0:
            mask = erode_mask(mask, erode_r)

    if dilate_r is not None:
        if dilate_r > 0:
            mask = dilate_mask(mask, dilate_r)

    ma_mask = (mask & ts_obs)
    invalid_after = ma_mask.sum(axis=0) / obs * 100
    invalid_after = invalid_after.astype(int)

    # invert values to have True for valid pixels and False for clouds
    mask = ~mask

    if max_invalid_ratio is not None:
        max_invalid_mask = invalid_after > max_invalid_ratio * 100
        mask = mask | np.broadcast_to(max_invalid_mask, mask.shape)

    return mask, obs, 100 - invalid_before, 100 - invalid_after


def pixel_qa_mask(pixel_qa_data,
                  erode_r=None,
                  dilate_r=None,
                  max_invalid_ratio=None):
    """
    From a timeseries (t, y, x) returns a binary mask False for
    data that are not clear

    Parameters:
    -----------
    pixel_qa_data: 3D array
        Input array for computing the mask

    erode_r : int
        Radius for eroding disk on the mask

    dilate_r : int
        Radius for dilating disk on the mask

    max_invalid_ratio : float
        Will set mask values to True, when they have an
        invalid_ratio > max_invalid_ratio

    Returns:
    --------
    mask : 3D bool array
        mask True for valid pixels, False for invalid

    obs : 2D int array
        number of valid observations (different from 0 in scl_data)

    invalid_before : 2D float array
        ratio of invalid obs before morphological operations

    valid_after : 2D float array
        ratio of valid obs after morphological operations
    """
    pixel_qa_data = np.squeeze(pixel_qa_data).astype(np.uint32)

    ts_obs = pixel_qa_data != 1  # Int value 1 means Fill value

    obs = ts_obs.sum(axis=0)

    # Define the to-be-masked pixel_qa values
    cirrus = 1 << 2
    cloud = 1 << 3
    shadow = 1 << 4
    snow = 1 << 5

    clear = ((pixel_qa_data & shadow == 0) &
             (pixel_qa_data & cloud == 0) &
             (pixel_qa_data & cirrus == 0) &
             (pixel_qa_data & snow == 0))

    mask = ~clear
    ma_mask = (mask & ts_obs)

    invalid_before = ma_mask.sum(axis=0) / obs * 100
    invalid_before = invalid_before.astype(int)

    if erode_r is not None:
        if erode_r > 0:
            mask = erode_mask(mask, erode_r)

    if dilate_r is not None:
        if dilate_r > 0:
            mask = dilate_mask(mask, dilate_r)

    ma_mask = (mask & ts_obs)
    invalid_after = ma_mask.sum(axis=0) / obs * 100
    invalid_after = invalid_after.astype(int)

    # invert values to have True for valid pixels and False for clouds
    mask = ~mask

    if max_invalid_ratio is not None:
        max_invalid_mask = invalid_after > max_invalid_ratio * 100
        mask = mask | np.broadcast_to(max_invalid_mask, mask.shape)

    return mask, obs, 100 - invalid_before, 100 - invalid_after


def binary_mask(mask_data,
                *,
                erode_r=None,
                dilate_r=None,
                nodata=255,
                max_invalid_ratio=None,
                **kwargs):
    """
    Adapted masking function for binary masks
    """

    return scl_mask(mask_data, mask_values=[0],
                    erode_r=erode_r, dilate_r=dilate_r,
                    nodata=nodata,
                    max_invalid_ratio=max_invalid_ratio,
                    kwargs=kwargs)


def select_valid_obs(mask_data, mask_th,
                     min_keep_fraction,
                     within_swath=None,
                     min_acquisitions=75,
                     ):
    """Method to select observations are below a max amount of masked pixels

    Args:
        mask_data (np.ndarray): binary mask [t-x-y] where 1 means valid pixel
        mask_th (float): max fraction of masked pixels in an image
        min_keep_fraction (float): min required amount of images to keep
        within_swath (np.ndarray, optional): array matching temporal
                    dimension of mask_data describing amount of
                    within-swath pixels per image
        min_acquisitions (int): the minimum amount of acquisitions before
                    triggering the valid obs selection. If the incoming data
                    has less obs than this number, we will return everything.

    Returns:
        valid_ids: list/array of valid ids to be used to subset
                   the original acquisition DataFrame
    """

    # If within_swath array is not provided, it's supposed
    # to be the number of elements in the image
    if within_swath is None:
        within_swath = np.repeat(np.prod(mask_data.shape[1:]),
                                 mask_data.shape[0])
        valid_acquisitions = mask_data.shape[0]
    else:
        valid_acquisitions = (within_swath > 0).values.sum()

    # First check if we meet the min amount of acquisitions
    if valid_acquisitions < min_acquisitions:
        logger.info((f'Got less than {min_acquisitions} within-swath items '
                     'in collection: skipping no-data filtering.'))

        # Return all valid acquisitions
        return np.where(within_swath > 0)[0]

    # Count per obs the relative amount of unmasked pixels
    relative_valid = (np.sum(mask_data == 1,
                             axis=(1, 2)) / within_swath).values
    relative_valid[within_swath == 0] = 0

    # Iterate through progressively relaxed mask thresholds
    for current_mask_th in np.arange(mask_th, 0, -0.01):

        # Get the obs with min amount of valid pixels
        valid_ids = np.where(relative_valid > current_mask_th)[0]

        # Check if we have enough obs to continue
        if len(valid_ids) / valid_acquisitions >= min_keep_fraction:
            logger.info(('no-data filtering succeeded '
                         f'with threshold: {current_mask_th}'))
            break
        else:
            valid_ids = None

    if valid_ids is None:
        # If we end up here, non of the mask thresholds were able
        # to retain minimum amount of observations so just take the min
        # amount starting with least masked ones
        logger.warning(('Filtering no-data switched off to reach '
                        f'min_keep_fraction of {min_keep_fraction}'))

        req_obs = int(np.ceil(min_keep_fraction * mask_data.shape[0]))
        nr_vaid_pixels = np.sum(mask_data == 1, axis=(1, 2))
        valid_ids = sorted(np.argsort(nr_vaid_pixels)[::-1][:req_obs])

    return valid_ids


def makesimplelimitscube(npdatacube, limit=None):

    if limit is None:
        return None

    if not isinstance(limit, numbers.Real):
        raise ValueError("limit value '{0}' must be number".format(limit))

    def simplelimitsrasterfunction(iIdx, npdataraster):
        nplimitsraster = np.full_like(npdataraster, np.nan, dtype=np.float32)
        nplimitsraster[~np.isnan(npdataraster)] = limit
        return nplimitsraster

    return makelimitscube(npdatacube, simplelimitsrasterfunction)


def makelimitscube(npdatacube, zelimitsrasterfunction):

    if zelimitsrasterfunction is None:
        return makesimplelimitscube(npdatacube)

    numberofrasters = npdatacube.shape[0]
    nplimitsscube = np.full_like(npdatacube, np.nan, dtype=np.float32)

    for iIdx in range(numberofrasters):
        if np.isscalar(npdatacube[iIdx]):
            nplimitsscube[iIdx] = zelimitsrasterfunction(
                iIdx, npdatacube[iIdx])
        else:
            nplimitsscube[iIdx, :] = zelimitsrasterfunction(
                iIdx, npdatacube[iIdx])

    return nplimitsscube


def flaglocalminima(npdatacube, maxdip=None, maxdif=None,
                    maxgap=None, maxpasses=1, verbose=True):
    '''
    Remove dips and difs (replace by np.nan) from the input npdatacube.

    dip on position i:
        (xn - xi) < (n-l) * maxdip AND (xm - xi) < (m-i) * maxdip
        n first not-None position with value 'left' of i
        m first not-None position with value 'right' of i

    dif on position i:
        (xn - xi) < (n-l) * maxdif OR (xm - xi) < (m-i) * maxdif
        n first not-None position with value 'left' of i
        m first not-None position with value 'right' of i
    '''

    return _flaglocalextrema_ct(npdatacube, maxdip, maxdif,
                                maxgap=maxgap, maxpasses=maxpasses,
                                doflagmaxima=False, verbose=verbose)


def flaglocalmaxima(npdatacube, maxdip=None, maxdif=None,
                    maxgap=None, maxpasses=1, verbose=True):
    return _flaglocalextrema_ct(npdatacube, maxdip, maxdif,
                                maxgap=maxgap, maxpasses=maxpasses,
                                doflagmaxima=True, verbose=verbose)


def _flaglocalextrema_ct(npdatacube, maxdip, maxdif, maxgap=None,
                         maxpasses=1, doflagmaxima=False, verbose=True):
    #
    #
    #
    def slopeprev(npdatacube, maxgap):
        """
        """
        shiftedval = np.full_like(npdatacube, np.nan, dtype=float)
        shifteddis = np.full_like(npdatacube,         1, dtype=int)
        numberofrasters = npdatacube.shape[0]
        shiftedval[1:numberofrasters, ...] = npdatacube[0:numberofrasters-1, ...]

        if np.isscalar(npdatacube[0]):
            nans = np.isnan(npdatacube)
            for iIdx in range(1, numberofrasters):
                if nans[iIdx-1]:
                    shiftedval[iIdx] = shiftedval[iIdx-1]  # can still be nan in case series started with nan
                    shifteddis[iIdx] = shifteddis[iIdx-1] + 1

        else:
            for iIdx in range(1, numberofrasters):
                nans = np.isnan(npdatacube[iIdx-1])
                shiftedval[iIdx][nans] = shiftedval[iIdx-1][nans]
                shifteddis[iIdx][nans] = shifteddis[iIdx-1][nans] + 1

        slopetoprev = (shiftedval-npdatacube)/shifteddis
        comparable = ~np.isnan(slopetoprev)
        if maxgap is not None:
            comparable &= shifteddis <= maxgap

        return slopetoprev, comparable

    def slopenext(npdatacube, maxgap):
        """
        """
        shiftedval = np.full_like(npdatacube, np.nan, dtype=float)
        shifteddis = np.full_like(npdatacube,         1, dtype=int)
        numberofrasters = npdatacube.shape[0]
        shiftedval[0:numberofrasters-1, ...] = npdatacube[1:numberofrasters, ...]

        if np.isscalar(npdatacube[0]):
            nans = np.isnan(npdatacube)
            for iIdx in range(numberofrasters-2, -1, -1):
                if nans[iIdx+1]:
                    shiftedval[iIdx] = shiftedval[iIdx+1]  # can still be nan in case series started with nan
                    shifteddis[iIdx] = shifteddis[iIdx+1] + 1

        else:
            for iIdx in range(numberofrasters-2, -1, -1):
                nans = np.isnan(npdatacube[iIdx+1])
                shiftedval[iIdx][nans] = shiftedval[iIdx+1][nans]
                shifteddis[iIdx][nans] = shifteddis[iIdx+1][nans] + 1

        slopetonext = (shiftedval-npdatacube)/shifteddis
        comparable = ~np.isnan(slopetonext)
        if maxgap is not None:
            comparable &= shifteddis <= maxgap

        return slopetonext, comparable

    #
    #
    #
    def masklocalminima(slopesraster, thresholdvalue):
        return slopesraster > thresholdvalue

    def masklocalmaxima(slopesraster, thresholdvalue):
        return slopesraster < thresholdvalue
    if doflagmaxima:
        maskextrema = masklocalmaxima
    else:
        maskextrema = masklocalminima

    #
    #
    #
    if maxdip is not None and (not isinstance(maxdip, numbers.Real) or (float(maxdip) != maxdip) or (maxdip <= 0)):
        raise ValueError("maxdip must be positive number or None")
    if maxdif is not None and (not isinstance(maxdif, numbers.Real) or (float(maxdif) != maxdif) or (maxdif <= 0)):
        raise ValueError("maxdif must be positive number or None")
    if maxgap is not None and (not isinstance(maxgap, numbers.Real) or (int(maxgap) != maxgap) or (maxgap <= 0)):
        raise ValueError("maxgap must be positive integer or None")

    #
    #
    #
    initialnumberofvalues = np.sum(~np.isnan(npdatacube))
    previousnumberofvalues = initialnumberofvalues
    for iteration in range(maxpasses):
        #
        #
        #
        prevslope, prevcomparable = slopeprev(npdatacube, maxgap)
        nextslope, nextcomparable = slopenext(npdatacube, maxgap)
        #
        #
        #
        isdip = None
        if maxdip is not None:
            isdip = prevcomparable & nextcomparable
            isdip[isdip] = isdip[isdip] & maskextrema(prevslope[isdip], maxdip)
            isdip[isdip] = isdip[isdip] & maskextrema(nextslope[isdip], maxdip)

        isdif = None
        if maxdif is not None:
            isdif = np.full_like(npdatacube, False, dtype=bool)
            isdif[prevcomparable] = isdif[prevcomparable] | maskextrema(prevslope[prevcomparable], maxdif)
            isdif[nextcomparable] = isdif[nextcomparable] | maskextrema(nextslope[nextcomparable], maxdif)

        if isdip is not None:
            npdatacube[isdip] = np.nan
        if isdif is not None:
            npdatacube[isdif] = np.nan

        #
        #
        #
        remainingnumberofvalues = np.sum(~np.isnan(npdatacube))
        removednumberofvalues = previousnumberofvalues - remainingnumberofvalues
        if verbose:
            logger.debug("localextrema_ct pass(%s) - removed %s values. %s values remaining. %s values removed in total" %
                         (iteration+1, removednumberofvalues, remainingnumberofvalues, initialnumberofvalues - remainingnumberofvalues))
        previousnumberofvalues = remainingnumberofvalues
        if removednumberofvalues <= 0 and 1 < maxpasses:
            if verbose:
                logger.debug("localextrema_ct pass(%s) - exits" % (iteration+1))
            break
    #
    #
    #
    return npdatacube


def whittaker(lmbda, npdatacube, npweightscube=None,
              minimumdatavalue=None, maximumdatavalue=None,
              passes=1, dokeepmaxima=False):
    return whittaker_second_differences(
        lmbda, npdatacube, npweightscube=npweightscube,
        minimumdatavalue=minimumdatavalue, maximumdatavalue=maximumdatavalue,
        passes=passes, dokeepmaxima=dokeepmaxima)


def whittaker_second_differences(lmbda, npdatacube, npweightscube=None,
                                 minimumdatavalue=None, maximumdatavalue=None,
                                 passes=1, dokeepmaxima=False):
    """
    """
    return _dowhittaker(lmbda, 2, npdatacube,
                        npweightscube=npweightscube,
                        minimumdatavalue=minimumdatavalue,
                        maximumdatavalue=maximumdatavalue,
                        passes=passes,
                        dokeepmaxima=dokeepmaxima)


def _dowhittaker(lmbda, orderofdifferences, npdatacube, npweightscube=None,
                 minimumdatavalue=None, maximumdatavalue=None, passes=1,
                 dokeepmaxima=False):
    """
    """
    #
    #
    #
    if ((lmbda is None) or (float(lmbda) != lmbda) or (lmbda <= 0)):
        raise ValueError("lmbda must be positive value (is %s)" % (lmbda))
    if passes is None:
        passes = 1
    else:
        if ((int(passes) != passes) or (passes <= 0)):
            raise ValueError("passes must be positive integer or None (is %s)" % (passes))
    #
    #
    #
    if npweightscube is None:
        weightscube = np.full_like(npdatacube, 1.0)
    else:
        weightscube = np.copy(npweightscube)

    #
    #    allocate intermediates
    #
    smoothedcube = np.copy(npdatacube)
    notnandatacube = ~np.isnan(npdatacube)
    notnancube = np.copy(notnandatacube)
    exeedingmaximum = np.empty_like(npdatacube, dtype=bool)
    exeedingminimum = np.empty_like(npdatacube, dtype=bool)

    #
    #
    #
    if npdatacube.shape[0] < 4:
        return smoothedcube  # plain copy from input
    #
    #
    #
    for iteration in range(passes):

        #
        #    'keeping the maximum values' is only applied in case of intermediate iterations, not for the 'last' (or only - in case passes = 1)
        #    this means that the maximum values are not actually retained in the end result
        #    reason is that otherwise we get devils-horns at actual maximum values instead of a smooth transition
        #
        if iteration > 0 and dokeepmaxima:
            originalgreater = np.full_like(smoothedcube, False, dtype=bool)
            maskallnan = np.logical_and(notnandatacube, ~np.isnan(smoothedcube))
            originalgreater[maskallnan] = smoothedcube[maskallnan] < npdatacube[maskallnan]
            smoothedcube[maskallnan] = npdatacube[maskallnan]

        #
        #    we expect this to be obsolete after first pass, since all nan's would be removed
        #    but just to keep things save for future modifications we'll explicitly exclude nan's
        #
        smoothedcube[~notnancube] = 0.0
        weightscube[~notnancube] = 0.0

        #
        #    call actual Whittaker algorithm, only first and second order differences are implemented
        #
        if orderofdifferences == 1:
            smoothedcube[:] = _whittaker_first_differences(lmbda, smoothedcube, weightscube)
        elif orderofdifferences == 2:
            smoothedcube[:] = _whittaker_second_differences(lmbda, smoothedcube, weightscube)
        else:
            raise ValueError("orderofdifferences: only 1 and 2 supported (is %s)" % (orderofdifferences))

        #
        #    update the nan's raster, we expect this to be all True
        #
        notnancube[:] = ~np.isnan(smoothedcube)

        #
        #    clip results to valid data range
        #
        if maximumdatavalue is not None:
            exeedingmaximum.fill(False)
            exeedingmaximum[notnancube] = smoothedcube[notnancube] > maximumdatavalue
            smoothedcube[exeedingmaximum] = maximumdatavalue

        if minimumdatavalue is not None:
            exeedingminimum.fill(False)
            exeedingminimum[notnancube] = smoothedcube[notnancube] < minimumdatavalue
            smoothedcube[exeedingminimum] = minimumdatavalue
    #
    #
    #
    return smoothedcube


def _whittaker_first_differences(lmbda, y, w):
    """
    """
    #
    #    avoid RuntimeWarning when dividing by 0; suppose we're happy with nan and inf
    #
    old_settings = np.seterr(all='ignore')
    #
    #
    #
    numberofrasters = y.shape[0]
    #
    #
    #
    d = np.full_like(y, np.nan, dtype=float)
    c = np.full_like(y, np.nan, dtype=float)
    z = np.full_like(y, np.nan, dtype=float)
    #
    #
    #
    d[0] = w[0] + lmbda
    c[0] = - 1.0 * lmbda / d[0]
    z[0] = w[0] * y[0]

    for iIdx in range(1, numberofrasters-1):
        d[iIdx] = w[iIdx] + 2.0 * lmbda - c[iIdx-1] * c[iIdx-1] * d[iIdx-1]
        c[iIdx] = - lmbda / d[iIdx]
        z[iIdx] = w[iIdx] * y[iIdx] - c[iIdx-1] * z[iIdx-1]

    d[numberofrasters-1] = w[numberofrasters-1] + lmbda - \
        c[numberofrasters-2] * c[numberofrasters-2] * d[numberofrasters-2]
    z[numberofrasters-1] = (w[numberofrasters-1] * y[numberofrasters-1] - c[numberofrasters-2]
                            * z[numberofrasters-2]) / d[numberofrasters-1]

    for iIdx in range(numberofrasters-1)[::-1]:
        z[iIdx] = z[iIdx] / d[iIdx] - c[iIdx] * z[iIdx+1]

    #
    #    reset to default
    #
    np.seterr(**old_settings)
    #
    #
    #
    return z

#
#
#


def _whittaker_second_differences(lmbda, y, w):
    """
    """
    #
    #    avoid RuntimeWarning when dividing by 0; suppose we're happy with nan and inf
    #
    old_settings = np.seterr(all='ignore')
    #
    #
    #
    numberofrasters = y.shape[0]
    #
    #
    #
    d = np.full_like(y, np.nan, dtype=float)
    c = np.full_like(y, np.nan, dtype=float)
    e = np.full_like(y, np.nan, dtype=float)
    z = np.full_like(y, np.nan, dtype=float)
    #
    #
    #
    d[0] = w[0] + lmbda
    c[0] = -2.0 * lmbda / d[0]
    e[0] = lmbda / d[0]
    z[0] = w[0] * y[0]

    d[1] = w[1] + 5 * lmbda - d[0] * c[0] * c[0]
    c[1] = (-4 * lmbda - d[0] * c[0] * e[0]) / d[1]
    e[1] = lmbda / d[1]
    z[1] = w[1] * y[1] - c[0] * z[0]

    for iIdx in range(2, numberofrasters-2):
        i = iIdx
        i1 = iIdx - 1
        i2 = iIdx - 2
        d[i] = w[i] + 6.0 * lmbda - c[i1] * c[i1] * d[i1] - e[i2] * e[i2] * d[i2]
        c[i] = (-4.0 * lmbda - d[i1] * c[i1] * e[i1]) / d[i]
        e[i] = lmbda / d[i]
        z[i] = w[i] * y[i] - c[i1] * z[i1] - e[i2] * z[i2]

    m = numberofrasters-1
    i1 = m - 2
    i2 = m - 3
    d[m - 1] = w[m - 1] + 5.0 * lmbda - c[i1] * c[i1] * d[i1] - e[i2] * e[i2] * d[i2]
    c[m - 1] = (-2 * lmbda - d[i1] * c[i1] * e[i1]) / d[m - 1]
    z[m - 1] = w[m - 1] * y[m - 1] - c[i1] * z[i1] - e[i2] * z[i2]

    i1 = m - 1
    i2 = m - 2
    d[m] = w[m] + lmbda - c[i1] * c[i1] * d[i1] - e[i2] * e[i2] * d[i2]
    z[m] = (w[m] * y[m] - c[i1] * z[i1] - e[i2] * z[i2]) / d[m]
    z[m - 1] = z[m - 1] / d[m - 1] - c[m - 1] * z[m]

    for iIdx in range(numberofrasters-2)[::-1]:
        z[iIdx] = z[iIdx] / d[iIdx] - c[iIdx] * z[iIdx + 1] - e[iIdx] * z[iIdx + 2]

    #
    #    reset to default
    #
    np.seterr(**old_settings)
    #
    #
    #
    return z


@njit()
def nan_to_zero(x, y):
    shape = x.shape
    x = x.ravel()
    y = y.ravel()
    x[np.isnan(y)] = 0
    x = x.reshape(shape)
    return x


@njit()
def zero_to_nan(x, y):
    shape = x.shape
    x = x.ravel()
    y = y.ravel()
    x[y == 0] = np.nan
    x = x.reshape(shape)
    return x


@njit()
def replace_larger_than(x, v):
    shape = x.shape
    x = x.ravel()
    x[x > v] = v
    x = x.reshape(shape)
    return x


@njit()
def replace_smaller_than(x, v):
    shape = x.shape
    x = x.ravel()
    x[x < v] = v
    x = x.reshape(shape)
    return x


@njit()
def weightedlinearregression(npdatacube, npweightscube=None,
                             minimumdatavalue=None, maximumdatavalue=None):
    """
    weighted linear regression

    minimum data value and maximum data value: indicating
    valid range in the data rasters
    """

    #
    #    allocate weights cube
    #
    if npweightscube is None:
        #
        #    all equal weights - own allocation
        #
        npweightscube = np.full_like(npdatacube, 1.0)
    else:
        #
        #    basic check on list of weights
        #
        if npweightscube.shape != npdatacube.shape:
            raise ValueError(
                "weights cube and data cube must have identical shapes")
        #
        #    allocate own version since we're going to mess with it
        #
        npweightscube = np.copy(npweightscube)
    #
    #
    #
    numberofrasters = npdatacube.shape[0]
    #
    #    there must be a better way !
    #
    xindicescube = np.zeros_like(npdatacube, dtype=np.float64)
    for i in range(numberofrasters):
        fillvalue = float(i)
        xindicescube[i, ...] = np.full_like(npdatacube[0], fillvalue)
    xdatacube = np.copy(xindicescube)

    xdatacube = nan_to_zero(xdatacube, npdatacube)
    npweightscube = nan_to_zero(npweightscube,
                                npdatacube)
    #
    #
    #
    npregressioncube = np.zeros_like(npdatacube)
    nb, nx, ny = npregressioncube.shape
    ty = npweightscube * npdatacube
    tx = npweightscube * xdatacube
    txy = npweightscube * npdatacube * xdatacube
    txx = npweightscube * xdatacube * xdatacube

    for x in range(nx):
        for y in range(ny):
            sw = np.sum(npweightscube[:, x, y])
            sy = np.nansum(ty[:, x, y])
            sx = np.sum(tx[:, x, y])
            sxy = np.nansum(txy[:, x, y])
            sxx = np.nansum(txx[:, x, y])
            bn = (sw*sxx - sx*sx)
            if bn == 0:
                bn = np.nan
            b = (sw*sxy - sx*sy)/bn
            a = (sy - b*sx)/sw
            npregressioncube[:, x, y] = a + b * xindicescube[:, x, y]
    #
    #    clip regression (non-nan) values to valid range
    #
    if maximumdatavalue is not None:
        npregressioncube = replace_larger_than(npregressioncube,
                                               maximumdatavalue)
    if minimumdatavalue is not None:
        npregressioncube = replace_smaller_than(npregressioncube,
                                                minimumdatavalue)
    #
    #
    #
    return npregressioncube


class WeightTypeId(object):
    """
    keys to be used in dicts etc.
    """
    MAXIMUM = 1
    MINIMUM = 2
    POSSLOPE = 3
    NEGSLOPE = 4
    ABOUTEQUAL = 5
    DEFAULT = 99


class WeightValues(object):
    """
    """

    #
    #
    #
    _defaultweightvalue = 1.0

    #
    #
    #
    @staticmethod
    def defaultWeightValues():
        """"
        returns WeightValues instance
        """
        return WeightValues(
            maximum=WeightValues._defaultweightvalue,
            minimum=WeightValues._defaultweightvalue,
            posslope=WeightValues._defaultweightvalue,
            negslope=WeightValues._defaultweightvalue,
            aboutequal=WeightValues._defaultweightvalue,
            default=WeightValues._defaultweightvalue)

    #
    #
    #
    def __init__(self, maximum, minimum, posslope,
                 negslope, aboutequal, default):

        if maximum is not None and ((float(maximum) != maximum) or (maximum < 0)):
            raise ValueError(" weight for 'maximum' must be positive value or None (is %s)" % (maximum))
        if minimum is not None and ((float(minimum) != minimum) or (minimum < 0)):
            raise ValueError(" weight for 'minimum' must be positive value or None (is %s)" % (minimum))
        if posslope is not None and ((float(posslope) != posslope) or (posslope < 0)):
            raise ValueError(" weight for 'positive slope' must be positive value or None (is %s)" % (posslope))
        if negslope is not None and ((float(negslope) != negslope) or (negslope < 0)):
            raise ValueError(" weight for 'negative slope' must be positive value or None (is %s)" % (negslope))
        if aboutequal is not None and ((float(aboutequal) != aboutequal) or (aboutequal < 0)):
            raise ValueError(" weight for 'equal' must be positive value or None (is %s)" % (aboutequal))
        if default is not None and ((float(default) != default) or (default < 0)):
            raise ValueError(" weight for 'default' must be positive value, nan or None (is %s)" % (default))

        self._weightsdict = dict({
            WeightTypeId.MAXIMUM:  WeightValues._defaultweightvalue if maximum is None else maximum,
            WeightTypeId.MINIMUM:  WeightValues._defaultweightvalue if minimum is None else minimum,
            WeightTypeId.POSSLOPE:  WeightValues._defaultweightvalue if posslope is None else posslope,
            WeightTypeId.NEGSLOPE:  WeightValues._defaultweightvalue if negslope is None else negslope,
            WeightTypeId.ABOUTEQUAL:  WeightValues._defaultweightvalue if aboutequal is None else aboutequal,
            WeightTypeId.DEFAULT:  WeightValues._defaultweightvalue if default is None else default,
        })
    #
    #
    #

    def copy(self, maximum=None, minimum=None, posslope=None, negslope=None, aboutequal=None, default=None):
        """
        """
        if maximum is not None and ((float(maximum) != maximum) or (maximum < 0)):
            raise ValueError(" weight for 'maximum' must be positive value or None (is %s)" % (maximum))
        if minimum is not None and ((float(minimum) != minimum) or (minimum < 0)):
            raise ValueError(" weight for 'minimum' must be positive value or None (is %s)" % (minimum))
        if posslope is not None and ((float(posslope) != posslope) or (posslope < 0)):
            raise ValueError(" weight for 'positive slope' must be positive value or None (is %s)" % (posslope))
        if negslope is not None and ((float(negslope) != negslope) or (negslope < 0)):
            raise ValueError(" weight for 'negative slope' must be positive value or None (is %s)" % (negslope))
        if aboutequal is not None and ((float(aboutequal) != aboutequal) or (aboutequal < 0)):
            raise ValueError(" weight for 'equal' must be positive value or None (is %s)" % (aboutequal))
        if default is not None and ((float(default) != default) or (default < 0)):
            raise ValueError(" weight for 'default' must be positive value or None (is %s)" % (default))

        return WeightValues(self._weightsdict[WeightTypeId.MAXIMUM] if maximum is None else maximum,
                            self._weightsdict[WeightTypeId.MINIMUM] if minimum is None else minimum,
                            self._weightsdict[WeightTypeId.POSSLOPE] if posslope is None else posslope,
                            self._weightsdict[WeightTypeId.NEGSLOPE] if negslope is None else negslope,
                            self._weightsdict[WeightTypeId.ABOUTEQUAL] if aboutequal is None else aboutequal,
                            self._weightsdict[WeightTypeId.DEFAULT] if default is None else default)

    #
    #
    #
    def getweightsdict(self):
        return self._weightsdict.copy()

    #
    #
    #
    def getweight(self, weighttypeid):
        """
        will throw on invalid key. any compiler could prevent this.
        """
        return self._weightsdict[weighttypeid]


#
#
#


def makeweighttypescube(npdatacube, aboutequalepsilon=0):
    """
    """

    #
    #
    #
    if aboutequalepsilon is not None and ((float(aboutequalepsilon) != aboutequalepsilon) or (aboutequalepsilon < 0)):
        raise ValueError("'about equal epsilon' must be positive value or None (is %s)" % (aboutequalepsilon))
    #
    #
    #
    epsilon = aboutequalepsilon if aboutequalepsilon is not None else 0
    #
    #    ! will be used in np.select statement => sequence matters
    #
    weighttypeslist = [
        WeightTypeId.ABOUTEQUAL,
        WeightTypeId.MAXIMUM,
        WeightTypeId.MINIMUM,
        WeightTypeId.POSSLOPE,
        WeightTypeId.NEGSLOPE]
    #
    #
    #
    npweighttypescube = np.full_like(npdatacube, WeightTypeId.DEFAULT, dtype=int)

    curr_GT_prev = np.empty_like(npdatacube[0], dtype=bool)
    curr_GT_next = np.empty_like(npdatacube[0], dtype=bool)
    curr_LT_prev = np.empty_like(npdatacube[0], dtype=bool)
    curr_LT_next = np.empty_like(npdatacube[0], dtype=bool)
    curr_EQ_prev = np.empty_like(npdatacube[0], dtype=bool)
    curr_EQ_next = np.empty_like(npdatacube[0], dtype=bool)

    #
    #
    #
    numberofrasters = npdatacube.shape[0]
    #
    #
    #
    defaultweightraster = np.full_like(npdatacube[0], WeightTypeId.DEFAULT, dtype=int)
    #
    #
    #
    for iIdx in range(numberofrasters):

        #
        #    'previous' raster values can be obtained from leading rasters
        #
        prevvaluesraster = npdatacube[iIdx]  # first period stretched to left
        leadingavailableindices = list(range(iIdx))[::-1]  # reversed
        if leadingavailableindices:
            prevvaluesrasterchoicelist = [npdatacube[i] for i in leadingavailableindices]
            prevvaluesrastercondlist = [~np.isnan(npdatacube[i]) for i in leadingavailableindices]
            # defaults to current raster - stretching (could contain nan's)
            prevvaluesraster = np.select(prevvaluesrastercondlist,
                                         prevvaluesrasterchoicelist, default=npdatacube[iIdx])
        #
        #    'next' raster values can be obtained from trailing rasters
        #
        nextvaluesraster = npdatacube[iIdx]  # last period stretched to right
        trailingavailableindices = list(range(iIdx+1, numberofrasters))
        if trailingavailableindices:
            nextvaluesrasterchoicelist = [npdatacube[i] for i in trailingavailableindices]
            nextvaluesrastercondlist = [~np.isnan(npdatacube[i]) for i in trailingavailableindices]
            # defaults to current raster - stretching (could contain nan's)
            nextvaluesraster = np.select(nextvaluesrastercondlist,
                                         nextvaluesrasterchoicelist, default=npdatacube[iIdx])
        #
        #
        #
        notnanmask = ~np.isnan(npdatacube[iIdx])
        # prevvaluesraster can contain nan's only where npdatacube[iIdx] does due to default selection above
        deltacurrprev = npdatacube[iIdx][notnanmask] - prevvaluesraster[notnanmask]
        # nextvaluesraster can contain nan's only where npdatacube[iIdx] does due to default selection above
        deltacurrnext = npdatacube[iIdx][notnanmask] - nextvaluesraster[notnanmask]

        curr_GT_prev.fill(False)
        curr_GT_prev[notnanmask] = deltacurrprev > 0.
        curr_GT_next.fill(False)
        curr_GT_next[notnanmask] = deltacurrnext > 0.
        curr_LT_prev.fill(False)
        curr_LT_prev[notnanmask] = deltacurrprev < 0.
        curr_LT_next.fill(False)
        curr_LT_next[notnanmask] = deltacurrnext < 0.
        curr_EQ_prev.fill(False)
        curr_EQ_prev[notnanmask] = np.absolute(deltacurrprev) <= epsilon
        curr_EQ_next.fill(False)
        curr_EQ_next[notnanmask] = np.absolute(deltacurrnext) <= epsilon

        weightcondlist = [
            (curr_EQ_prev & curr_EQ_next),
            (curr_GT_prev & curr_GT_next),
            (curr_LT_prev & curr_LT_next),
            (curr_GT_prev | curr_LT_next),
            (curr_LT_prev | curr_GT_next)]

        npweighttypescube[iIdx] = np.select(weightcondlist, weighttypeslist, defaultweightraster)

    return npweighttypescube


def makesimpleweightscube(weighttypescube, weightvalues=WeightValues.defaultWeightValues()):
    """
    """

    #
    #
    #
    if weightvalues is None:
        weightvalues = WeightValues.defaultWeightValues()
    #
    #
    #
    npweightscube = np.full_like(weighttypescube, np.nan, dtype=float)

    npweightscube[weighttypescube == WeightTypeId.MAXIMUM] = weightvalues.getweight(WeightTypeId.MAXIMUM)
    npweightscube[weighttypescube == WeightTypeId.MINIMUM] = weightvalues.getweight(WeightTypeId.MINIMUM)
    npweightscube[weighttypescube == WeightTypeId.POSSLOPE] = weightvalues.getweight(WeightTypeId.POSSLOPE)
    npweightscube[weighttypescube == WeightTypeId.NEGSLOPE] = weightvalues.getweight(WeightTypeId.NEGSLOPE)
    npweightscube[weighttypescube == WeightTypeId.ABOUTEQUAL] = weightvalues.getweight(WeightTypeId.ABOUTEQUAL)
    npweightscube[weighttypescube == WeightTypeId.DEFAULT] = weightvalues.getweight(WeightTypeId.DEFAULT)

    return npweightscube


@njit()
def makesimpleweightscube_fromdatacube_sd(npdatacube, aboutequalepsilon=0,
                                          weightvalues=SWETS_WEIGHTS):
    """
    alternative makesimpleweightscube_fromdatacube implementation which
    runs faster on rasters with small dimensions
    """
    #
    #
    #
    def padtoright(npdatacube):
        """
        print (padtoright(np.array([np.nan, np.nan, 100, np.nan, np.nan, 200, np.nan, np.nan])))
                                      [      nan        nan  100.       100.       100. 200.       200.       200.]
        """
        nans = np.isnan(npdatacube)
        if not nans.any():
            return npdatacube  # beware - not a copy
        padded = np.copy(npdatacube)
        numberofrasters = npdatacube.shape[0]
        for iIdx in range(1, numberofrasters):
            shape = padded[iIdx].shape
            temp = padded[iIdx].ravel()
            replace = padded[iIdx-1].ravel()
            temp[np.isnan(temp)] = replace[np.isnan(temp)]
            padded[iIdx] = temp.reshape(shape)
        return padded

    def padtoleft(npdatacube):
        """
        print (padtoleft(np.array([np.nan, np.nan, 100, np.nan, np.nan, 200, np.nan, np.nan])))
                                     [      100.       100. 100.       200.       200. 200.       nan        nan]
        """
        nans = np.isnan(npdatacube)
        if not nans.any():
            return npdatacube  # beware - not a copy
        padded = np.copy(npdatacube)
        numberofrasters = npdatacube.shape[0]
        for iIdx in range(numberofrasters-2, -1, -1):
            shape = padded[iIdx].shape
            temp = padded[iIdx].ravel()
            replace = padded[iIdx+1].ravel()
            temp[np.isnan(temp)] = replace[np.isnan(temp)]
            padded[iIdx] = temp.reshape(shape)
        return padded
    #
    #
    #
    # if aboutequalepsilon is not None:
    #     if not isinstance(aboutequalepsilon, numbers.Real):
    #         raise ValueError("'about equal epsilon' must be number or None")
    #     if ((float(aboutequalepsilon) != aboutequalepsilon) or (aboutequalepsilon < 0)):
    #         raise ValueError("'about equal epsilon' must be positive value or None")
    #
    #
    #
    epsilon = aboutequalepsilon if aboutequalepsilon is not None else 0

    weightscube = np.full_like(npdatacube, weightvalues[5])  # default

    deltacurrprev = np.full_like(npdatacube, 0)
    deltacurrnext = np.full_like(npdatacube, 0)

    nans = np.isnan(npdatacube)

    #
    #    if-else might save some time and memory in absence of nan's
    #
    if nans.any():
        rightpadded = padtoleft(padtoright(npdatacube))
        leftpadded = padtoright(padtoleft(npdatacube))
        deltacurrnext[:-1, ...] = leftpadded[:-1, ...] - leftpadded[1:, ...]
        deltacurrprev[1:, ...] = rightpadded[1:, ...] - rightpadded[:-1, ...]
    else:
        deltacurrnext[:-1, ...] = npdatacube[:-1, ...] - npdatacube[1:, ...]
        deltacurrprev[1:, ...] = npdatacube[1:, ...] - npdatacube[:-1, ...]

    nb, nx, ny = weightscube.shape
    for b in range(nb):
        for x in range(nx):
            for y in range(ny):
                weight = weightscube[b, x, y]
                val = npdatacube[b, x, y]
                dcp = deltacurrprev[b, x, y]
                dcn = deltacurrnext[b, x, y]
                if ~np.isnan(val):
                    if dcp < 0 or dcn > 0:
                        weight = weightvalues[3]  # negslope
                    if dcp > 0 or dcn < 0:
                        weight = weightvalues[2]  # posslope
                    if dcp < 0 and dcn < 0:
                        weight = weightvalues[1]  # minimum
                    if dcp > 0 and dcn > 0:
                        weight = weightvalues[0]  # maximum
                    if ((np.absolute(dcp) <= epsilon) and
                            (np.absolute(dcn) <= epsilon)):
                        weight = weightvalues[4]  # aboutequal

                weightscube[b, x, y] = weight

    return weightscube


def makesimpleweightscube_fromdatacube_sd_ori(npdatacube, aboutequalepsilon=0,
                                              weightvalues=WeightValues.defaultWeightValues()):
    """
    alternative makesimpleweightscube_fromdatacube implementation which runs faster on rasters with small dimensions
    """
    #
    #
    #
    def padtoright(npdatacube):
        """
        print (padtoright(np.array([np.nan, np.nan, 100, np.nan, np.nan, 200, np.nan, np.nan])))
                                      [      nan        nan  100.       100.       100. 200.       200.       200.]
        """
        nans = np.isnan(npdatacube)
        if not nans.any():
            return npdatacube  # beware - not a copy
        padded = np.copy(npdatacube)
        numberofrasters = npdatacube.shape[0]
        for iIdx in range(1, numberofrasters):
            if np.isscalar(padded[iIdx]):
                if nans[iIdx]:
                    padded[iIdx] = padded[iIdx-1]
            else:
                padded[iIdx][nans[iIdx]] = padded[iIdx-1][nans[iIdx]]
        return padded

    def padtoleft(npdatacube):
        """
        print (padtoleft(np.array([np.nan, np.nan, 100, np.nan, np.nan, 200, np.nan, np.nan])))
                                     [      100.       100. 100.       200.       200. 200.       nan        nan]
        """
        nans = np.isnan(npdatacube)
        if not nans.any():
            return npdatacube  # beware - not a copy
        padded = np.copy(npdatacube)
        numberofrasters = npdatacube.shape[0]
        for iIdx in range(numberofrasters-2, -1, -1):
            if np.isscalar(padded[iIdx]):
                if nans[iIdx]:
                    padded[iIdx] = padded[iIdx+1]
            else:
                padded[iIdx][nans[iIdx]] = padded[iIdx+1][nans[iIdx]]
        return padded
    #
    #
    #
    if aboutequalepsilon is not None:
        if not isinstance(aboutequalepsilon, numbers.Real):
            raise ValueError("'about equal epsilon' must be number or None (is '%s')" % (aboutequalepsilon))
        if ((float(aboutequalepsilon) != aboutequalepsilon) or (aboutequalepsilon < 0)):
            raise ValueError("'about equal epsilon' must be positive value or None (is %s)" % (aboutequalepsilon))
    #
    #
    #
    epsilon = aboutequalepsilon if aboutequalepsilon is not None else 0

    weightscube = np.full_like(npdatacube, weightvalues.getweight(WeightTypeId.DEFAULT), dtype=float)

    deltacurrprev = np.full_like(npdatacube, 0, dtype=float)
    deltacurrnext = np.full_like(npdatacube, 0, dtype=float)

    nans = np.isnan(npdatacube)

    #
    #    if-else might save some time and memory in absence of nan's
    #
    if nans.any():
        rightpadded = padtoleft(padtoright(npdatacube))
        leftpadded = padtoright(padtoleft(npdatacube))
        np.subtract(leftpadded[:-1, ...], leftpadded[1:, ...],   out=deltacurrnext[:-1, ...])
        np.subtract(rightpadded[1:, ...], rightpadded[:-1, ...], out=deltacurrprev[1:, ...])
    else:
        np.subtract(npdatacube[:-1, ...], npdatacube[1:, ...],  out=deltacurrnext[:-1, ...])
        np.subtract(npdatacube[1:, ...],  npdatacube[:-1, ...], out=deltacurrprev[1:, ...])

    weightscube[((deltacurrprev < 0.) | (deltacurrnext > 0)) & ~
                nans] = weightvalues.getweight(WeightTypeId.NEGSLOPE)    # 4
    weightscube[((deltacurrprev > 0.) | (deltacurrnext < 0)) & ~
                nans] = weightvalues.getweight(WeightTypeId.POSSLOPE)   # 3
    weightscube[((deltacurrprev < 0.) & (deltacurrnext < 0)) & ~
                nans] = weightvalues.getweight(WeightTypeId.MINIMUM)    # 2
    weightscube[((deltacurrprev > 0.) & (deltacurrnext > 0)) & ~
                nans] = weightvalues.getweight(WeightTypeId.MAXIMUM)    # 1
    weightscube[(
        (np.absolute(deltacurrprev) <= epsilon) &
        (np.absolute(deltacurrnext) <= epsilon)) & ~nans] = weightvalues.getweight(WeightTypeId.ABOUTEQUAL)  # 5

    return weightscube


def weightedlinearregression_ori(npdatacube, npweightscube=None,
                                 minimumdatavalue=None, maximumdatavalue=None):
    """
    weighted linear regression

    minimum data value and maximum data value: indicating
    valid range in the data rasters
    """
    #
    #    allocate weights cube
    #
    if npweightscube is None:
        #
        #    all equal weights - own allocation
        #
        npweightscube = np.full_like(npdatacube, 1.0, dtype=float)
    else:
        #
        #    basic check on list of weights
        #
        if npweightscube.shape != npdatacube.shape:
            raise ValueError(
                "weights cube and data cube must have identical shapes")
        #
        #    allocate own version since we're going to mess with it
        #
        npweightscube = np.copy(npweightscube)
    #
    #
    #
    numberofrasters = npdatacube.shape[0]
    #
    #    there must be a better way !
    #
    xindicescube = np.array(
        [np.full_like(npdatacube[0],
                      i, dtype=float) for i in range(numberofrasters)])
    xdatacube = np.copy(xindicescube)

    isnanscube = np.isnan(npdatacube)
    xdatacube[isnanscube] = 0
    npweightscube[isnanscube] = 0
    #
    #
    #
    while True:
        sw = np.sum(npweightscube, axis=0)
        sy = np.nansum(npweightscube * npdatacube, axis=0)
        sx = np.sum(npweightscube * xdatacube, axis=0)
        sxy = np.nansum(npweightscube * npdatacube * xdatacube, axis=0)
        sxx = np.nansum(npweightscube * xdatacube * xdatacube, axis=0)
        #
        #
        #
        bn = (sw*sxx - sx*sx)
        if np.isscalar(bn):
            if bn == 0:
                bn = np.nan
        else:
            bn[(bn == 0)] = np.nan
        #
        #
        #
        b = (sw*sxy - sx*sy)/bn
        a = (sy - b*sx)/sw
        npregressioncube = a + b * xindicescube
        break
    #
    #    clip regression (non-nan) values to valid range
    #
    notnancube = ~np.isnan(npregressioncube)
    if maximumdatavalue is not None:
        exeedingmaximum = np.full_like(npregressioncube, False, dtype=bool)
        exeedingmaximum[notnancube] = npregressioncube[
            notnancube] > maximumdatavalue
        npregressioncube[exeedingmaximum] = maximumdatavalue
    if minimumdatavalue is not None:
        exeedingminimum = np.full_like(npregressioncube, False, dtype=bool)
        exeedingminimum[notnancube] = npregressioncube[
            notnancube] < minimumdatavalue
        npregressioncube[exeedingminimum] = minimumdatavalue
    #
    #
    #
    return npregressioncube
