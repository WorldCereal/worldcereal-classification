# %%
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from matplotlib import pyplot as plt
from numba import float32, float64, int16, int64, njit
from numba.types import Tuple
from rasterio.crs import CRS
from rasterio.profiles import Profile
from scipy.signal import convolve2d


class DefaultProfile(Profile):
    """Tiled, band-interleaved, LZW-compressed, 8-bit GTiff."""

    defaults = {
        "driver": "GTiff",
        "interleave": "band",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "deflate",
        "dtype": "float32",
    }


# %%
# ALL REQUIRED FUNCTIONS TO RUN THIS NOTEBOOK


def nearest_date(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


@njit(int64[:](float32[:]))
def find_peaks(x):
    peak_index = []

    for i, val in enumerate(x[1:-1], 1):
        if val >= x[i - 1] and val > x[i + 1]:
            peak_index.append(i)

    if x[-1] > x[-2]:
        peak_index.append(len(x) - 1)

    if x[0] > x[1]:
        peak_index.append(0)

    return np.array(peak_index)


def detect_seasons(
    evi,
    times,
    max_seasons=5,
    amp_thr1=0.1,
    amp_thr2=0.35,
    min_window=10,
    max_window=185,
    partial_start=False,
    partial_end=False,
):
    """
    Computes peak, SOS and EOS of all seasons based on an EVI time series

    SOS, MOS and EOS are stored as datetime.timestamp()

    :param EVI: 3D numpy array (time, x, y) from which to derive the seasons.
        The data cannot contain any NaN's!
    :param times: list of datetime objects corresponding to the time dimension
    :param max_seasons: maximum number of seasons to be detected
    :param amp_thr1: minimum threshold for amplitude of a season
    :param amp_thr2: factor with which to multiply total
        amplitude of signal to derive second minimum amplitude threshold
    :param min_window: search window for finding start and end
        before/after peak - minimum distance from peak in days
    :param max_window: search window for finding start and end
        before/after peak - maximum distance from peak in days
    :param partial_start and partial_end: whether or not to
        detect partial seasons
        (if partial_start = True, season is allowed to have
        SOS prior to start of time series;
        if partial_end = True, season is allowed to have
        EOS after end of time series)

    returns a Seasons object with for each pixel SOS, MOS, EOS and nSeasons
    """
    # some dimensionality checks...
    if evi.ndim != 3:
        raise ValueError("Input time series does not have the right dimensions")
    if evi.shape[0] != len(times):
        raise ValueError("Time dimension does not match the input data")

    times = [pd.to_datetime(t) for t in times]
    times = np.array([t.timestamp() for t in times])

    # actual definition of the function
    @njit(
        Tuple((int16[:, :], float32[:, :, :], float32[:, :, :], float32[:, :, :]))(
            float32[:, :, :], float64[:]
        )
    )
    def _detect_seasons_fast(data, times):
        def _day_to_second(days):
            return days * 24 * 3600

        def _find_nearest(array, value):
            array = np.asarray(array)
            return np.abs(array - value).argmin()

        # prepare outputs
        nx = data.shape[1]
        ny = data.shape[2]

        nseasons = np.zeros((nx, ny), dtype=np.int16)
        sos = np.zeros((max_seasons, nx, ny), dtype=np.float32)
        mos = np.zeros((max_seasons, nx, ny), dtype=np.float32)
        eos = np.zeros((max_seasons, nx, ny), dtype=np.float32)

        sos[...] = np.nan
        mos[...] = np.nan
        eos[...] = np.nan

        # loop over each individual pixel to define
        # start, peak and end of season(s)
        for i in range(nx):
            for j in range(ny):
                data_pix = data[:, i, j]
                # find all local maxima
                localmax_idx = find_peaks(data_pix)
                if localmax_idx.size == 0:
                    # no peaks found, proceed to next pixel
                    continue
                localmax = data_pix[localmax_idx]

                # sort local maxima according to VI amplitude
                sort_idx = localmax.argsort()
                localmax_idx_sorted = localmax_idx[sort_idx]

                # define outputs
                npeaks = localmax_idx_sorted.shape[0]
                valid = np.ones(npeaks, dtype=np.uint8)
                start = np.zeros(npeaks, dtype=np.int32)
                end = np.zeros(npeaks, dtype=np.int32)

                # setting some threshold values
                totalrange = np.max(data_pix) - np.min(data_pix)
                amp_thr2_fin = amp_thr2 * totalrange

                # find for each peak the associated local minima
                # and decide whether
                # the peak is valid or not
                for p in range(npeaks):
                    skip_sos = False

                    idx = localmax_idx_sorted[p]
                    # define search window for SOS
                    t_idx = times[idx]
                    t_min = t_idx - _day_to_second(max_window)
                    idx_min = _find_nearest(times, t_min)
                    t_max = t_idx - _day_to_second(min_window)
                    idx_max = _find_nearest(times, t_max)

                    # if peak is very close to start of TS...
                    if idx_max == 0:
                        if partial_start:
                            # and partial season mapping is allowed
                            # -> skip SOS detection
                            skip_sos = True
                        else:
                            # else, peak is invalid
                            valid[p] = 0
                            continue

                    # do SOS check if necessary
                    if not skip_sos:
                        # adjust search window in case there is a valid
                        # peak within the window
                        # find all intermediate VALID peaks
                        val_peaks = localmax_idx_sorted.copy()
                        val_peaks[valid == 0] = -1
                        int_peaks_idx = localmax_idx_sorted[
                            (val_peaks > idx_min) & (val_peaks < idx)
                        ]
                        # if any, find the peak nearest to original peak
                        # and set t_min to that value
                        if int_peaks_idx.shape[0] > 0:
                            idx_min = np.max(int_peaks_idx)
                            # if, by adjusting the window, idx_max <
                            # idx_min -> label peak as invalid
                            if idx_max < idx_min:
                                valid[p] = 0
                                continue

                        # identify index of local minimum in search window
                        win = data_pix[idx_min : idx_max + 1]
                        start[p] = np.where(win == np.amin(win))[0][-1] + idx_min

                        # check if amplitude conditions of the identified
                        # starting point are met
                        amp_dif = data_pix[idx] - data_pix[start[p]]
                        if not (amp_dif >= amp_thr1) & (amp_dif >= amp_thr2_fin):
                            # if partial season mapping is allowed,
                            # and search window includes start of TS,
                            # the true SOS could be before start of TS.
                            # So we skip sos check, meaning eos check
                            # should definitely be done
                            if partial_start and (idx_min == 0):
                                skip_sos = True
                            else:
                                valid[p] = 0
                                continue

                    # define search window for EOS
                    t_min = t_idx + _day_to_second(min_window)
                    idx_min = _find_nearest(times, t_min)
                    t_max = t_idx + _day_to_second(max_window)
                    idx_max = _find_nearest(times, t_max)
                    # adjust search window in case there is a valid
                    # peak within the window
                    # find all intermediate VALID peaks
                    val_peaks = localmax_idx_sorted.copy()
                    val_peaks[valid == 0] = -1
                    int_peaks_idx = localmax_idx_sorted[
                        (val_peaks > idx) & (val_peaks < idx_max)
                    ]
                    # if any, find the peak nearest to original peak
                    # and set t_max to that value
                    if int_peaks_idx.shape[0] > 0:
                        idx_max = np.min(int_peaks_idx)
                        # if, by adjusting the window, idx_max
                        # < idx_min -> label peak as invalid
                        if idx_max < idx_min:
                            valid[p] = 0
                            continue

                    # in case you've reached the end of the timeseries,
                    # adjust idx_max
                    # if idx_max == data_pix.shape[0] - 1:
                    #     idx_max -= 1
                    # identify index of local minimum in search window
                    if idx_max < idx_min:
                        end[p] = data_pix.shape[0] - 1
                    else:
                        win = data_pix[idx_min : idx_max + 1]
                        end[p] = np.where(win == np.amin(win))[0][0] + idx_min

                    # if partial season mapping is allowed
                    # AND sos check was not skipped
                    # AND search window includes end of TS
                    # THEN the end of season check can be skipped

                    if (
                        partial_end
                        and (not skip_sos)
                        and (idx_max == data_pix.shape[0] - 2)
                    ):
                        continue
                    else:
                        # check if amplitude conditions of the identified
                        # end point are met
                        amp_dif = data_pix[idx] - data_pix[end[p]]
                        if not (amp_dif >= amp_thr1) & (amp_dif >= amp_thr2_fin):
                            valid[p] = 0

                # now delete invalid peaks
                idx_valid = np.where(valid == 1)[0]
                peaks = localmax_idx_sorted[idx_valid]
                start = start[valid == 1]
                end = end[valid == 1]
                npeaks = peaks.shape[0]

                # if more than max_seasons seasons detected ->
                # select the seasons with highest amplitudes
                if npeaks > max_seasons:
                    toRemove = npeaks - max_seasons
                    maxSeason = data_pix[peaks]

                    baseSeason = np.mean(np.stack((data_pix[start], data_pix[end])))
                    amp = maxSeason - baseSeason
                    idx_remove = np.zeros_like(amp)
                    for r in range(toRemove):
                        idx_remove[np.where(amp == np.min(amp))[0][0]] = 1
                        amp[np.where(amp == np.min(amp))[0][0]] = np.max(amp)
                    # check whether enough seasons will be removed
                    check = toRemove - np.sum(idx_remove)
                    if check > 0:
                        # remove random seasons
                        for r in range(int(check)):
                            idx_remove[np.where(idx_remove == 0)[0][0]] = 1
                    # remove the identified peaks
                    peaks = peaks[idx_remove != 1]
                    start = start[idx_remove != 1]
                    end = end[idx_remove != 1]
                    npeaks = max_seasons

                # convert indices to actual corresponding dates
                peaktimes = times[peaks]
                starttimes = times[start]
                endtimes = times[end]

                # if less than max_seasons seasons detected -> add
                # dummy seasons
                if peaktimes.shape[0] < max_seasons:
                    toAdd = (
                        np.ones(max_seasons - peaktimes.shape[0], dtype=np.float32) * -1
                    )
                    starttimes = np.concatenate((starttimes, toAdd))
                    endtimes = np.concatenate((endtimes, toAdd))
                    peaktimes = np.concatenate((peaktimes, toAdd))

                # transfer to output
                mos[:, i, j] = peaktimes
                sos[:, i, j] = starttimes
                eos[:, i, j] = endtimes
                nseasons[i, j] = npeaks

        return nseasons, sos, mos, eos

    # actual call to the function
    seasons = _detect_seasons_fast(evi, times)
    return {
        "nseasons": seasons[0],
        "sos": seasons[1].astype(np.uint32),
        "mos": seasons[2].astype(np.uint32),
        "eos": seasons[3].astype(np.uint32),
    }


def visualize_seasons(seasons, x, y, evi, times, outfile=None):
    """
    Plot seasons for a given pixel x,y

    :param seasons: output of season detection function
    :param x: row index of pixel to be plotted
    :param y: column index of pixel to be plotted
    :param evi: original time series used for detection
    :param times: timestamps corresponding to the time series
    :param outfile: optional output file to save the plot
    """

    timestamps = np.array([int(pd.to_datetime(t).timestamp()) for t in times])

    # plot timeseries
    f, ax = plt.subplots()
    ts = np.squeeze(evi[:, x, y])
    ax.plot(times, ts)

    # plot all seasons for particular pixel
    npeaks = seasons["nseasons"][x, y]
    for p in range(npeaks):
        startdate = seasons["sos"][p, x, y]
        startidx = np.where(timestamps == nearest_date(timestamps, startdate))[0][0]
        peakdate = seasons["mos"][p, x, y]
        peakidx = np.where(timestamps == nearest_date(timestamps, peakdate))[0][0]
        enddate = seasons["eos"][p, x, y]
        endidx = np.where(timestamps == nearest_date(timestamps, enddate))[0][0]
        ax.plot(times[startidx], ts[startidx], "go")
        ax.plot(times[peakidx], ts[peakidx], "k+")
        ax.plot(times[endidx], ts[endidx], "ro")

    plt.show()

    # save resulting plot if requested
    if outfile is not None:
        plt.savefig(outfile)


def mask(data, mask, valid=100, maskedvalue=255):
    data[mask != valid] = maskedvalue
    return data


def _filter_with_confidence(
    prediction, confidence, conf_thr, kernel_size, no_data_value
):
    if conf_thr is None:
        raise ValueError("Confidence threshold for majority" "filter missing!")
    if conf_thr > 1:
        raise ValueError(
            "Confidence threshold for majority"
            "filtering should be between zero and one!"
        )

    filteredprediction = _filter_without_confidence(
        prediction, kernel_size, no_data_value
    )

    # determine which cells need to be updated:
    # if confidence is low
    update_mask = (confidence < conf_thr) & (prediction != no_data_value)

    # produce final result
    newprediction = np.where(update_mask, filteredprediction, prediction)

    # # Count flipped labels
    # flipped_nr = (newprediction != prediction).sum()
    # total_nr = (prediction != no_data_value).sum()
    # flipped_perc = np.round(flipped_nr / total_nr * 100, 1)

    return newprediction


def _filter_without_confidence(prediction, kernel_size, no_data_value):
    to_ignore = prediction == no_data_value

    # Convolution kernel
    k = np.ones((kernel_size, kernel_size), dtype=int)

    # count number of valid ones in each window
    pred_val_one = np.where(to_ignore, 0, prediction)
    val_ones_count = convolve2d(pred_val_one, k, "same")

    # count number of valid zeros in each window
    pred_reverse = (prediction == 0).astype(np.uint16)
    pred_val_zero = np.where(to_ignore, 0, pred_reverse)
    val_zeros_count = convolve2d(pred_val_zero, k, "same")

    # determine majority
    majority = np.where(val_ones_count > val_zeros_count, 1, 0)

    # determine which cells need to be updated:
    # if prediction is not no data and if there is a clear majority
    update_mask = (val_ones_count != val_zeros_count) & (prediction != no_data_value)

    # produce final result
    newprediction = np.where(update_mask, majority, prediction)

    return newprediction


def majority_filter(
    prediction, kernel_size, confidence=None, conf_thr=None, no_data_value=255
):
    """
    :param prediction: prediction can only be zero, one or nodata
        (see no_data_value)
    :param kernel_size: determines the size of the spatial window
        that is considered during the filtering operation. Must
        be an odd number.
    :param confidence: (optional) pixel-based confidence scores
        of the prediction. Should be values between 0 and 1.
        No data value of this input should be the same as the one
        used for prediction.
    :param conf_thr: (optional) pixels having confidence lower
        than this threshold will be updated during this process.
        Also, these pixels will not be taken into account for
        determining the majority in a window.
    :param no_data_value: (optional) No data value in both
        prediction and confidence, which will be ignored during
        the entire process.
    """

    if kernel_size % 2 == 0:
        raise ValueError(
            "Kernel size for majority filtering should be an" " an odd number!"
        )

    if confidence is not None:
        return _filter_with_confidence(
            prediction, confidence, conf_thr, kernel_size, no_data_value
        )
    else:
        return _filter_without_confidence(prediction, kernel_size, no_data_value)


def get_blocksize(val):
    """
    Blocksize needs to be a multiple of 16
    """
    if val % 16 == 0:
        return val
    else:
        return (val // 16) * 16


def get_rasterio_profile(arr, bounds, epsg, blockxsize=None, blockysize=None, **params):
    if len(arr.shape) == 2:
        arr = np.expand_dims(arr, axis=0)

    base_profile = DefaultProfile()
    shape = arr.shape

    count, height, width = shape

    if blockxsize is None:
        blockxsize = get_blocksize(width)

    if blockysize is None:
        blockysize = get_blocksize(height)

    crs = CRS.from_epsg(epsg)

    base_profile.update(
        transform=rasterio.transform.from_bounds(*bounds, width=width, height=height),
        width=width,
        height=height,
        blockxsize=blockxsize,
        blockysize=blockysize,
        dtype=arr.dtype,
        crs=crs,
        count=count,
    )

    base_profile.update(**params)

    return base_profile


def write_geotiff(arr, profile, filename, band_names=None, colormap=None, nodata=None):
    if nodata is not None:
        profile.update(nodata=nodata)

    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)

    if os.path.isfile(filename):
        os.remove(filename)

    with rasterio.open(filename, "w", **profile) as dst:
        dst.write(arr)
        if band_names is not None:
            dst.update_tags(bands=band_names)
            for i, b in enumerate(band_names):
                dst.update_tags(i + 1, band_name=b)

        if colormap is not None:
            dst.write_colormap(1, colormap)


def _to_geotiff(
    data, bounds, epsg, filename, band_names=[], colormap=None, nodata=None
):
    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)

    profile = get_rasterio_profile(data, bounds, epsg)

    write_geotiff(
        data, profile, filename, band_names=band_names, colormap=colormap, nodata=nodata
    )


# %%
# Path to the input data
infile = Path(
    "/vitodata/worldcereal/data/openeo/inputs_presto/preprocessed_merged/belgium_good_2020-12-01_2021-11-30.nc"
)
data = xr.open_dataset(infile)
data


# %%
# compute EVI and visulize it
evi = 2.5 * (data["B08"] - data["B04"]) / (data["B08"] + 2.4 * data["B04"] + 1.0)
fig, ax = plt.subplots(figsize=(10, 10))
x, y = 10, 10
ax.plot(evi.t, evi.values[:, x, y])
plt.show()


# %%
# Run the season detection function
input = np.expand_dims(evi.values[:, x, y], [1, 2])
seasons = detect_seasons(input, evi.t.values)
visualize_seasons(seasons, 0, 0, input, evi.t.values)


# %%
# now define rule that generates active cropland marker...
active_crop = seasons["nseasons"]
active_crop = np.round(active_crop).astype(np.uint8)
active_crop[active_crop > 0] = 1
active_crop[active_crop != 1] = 0

colormap = {
    0: (232, 55, 39, 255),  # inactive
    100: (77, 216, 39, 255),  # active
}
nodatavalue = 255

# save result for inspection
filename = ...
_to_geotiff(
    active_crop,
    bounds,
    epsg,
    filename,
    band_names=["Active cropland"],
    colormap=colormap,
    nodata=nodatavalue,
)

# %%
# Mask result using cropland mask
# lc_mask = None
# active_crop = mask(active_crop, lc_mask,
#                    maskedvalue=255)

# save result for inspection
filename = ...
_to_geotiff(
    active_crop,
    bounds,
    epsg,
    filename,
    band_names=["Active cropland"],
    colormap=colormap,
    nodata=nodatavalue,
)

# %%
# apply post-classification majority filtering
kernelsize = 7
active_crop = majority_filter(active_crop, kernelsize, no_data_value=nodatavalue)

# To correct dtype
active_crop = active_crop.astype(np.uint8)

# # Remap values and nodata for output product
active_crop[active_crop == 1] = 100  # Active cropland
active_crop[active_crop == 255] = nodatavalue

#
# save result for inspection
filename = ...
_to_geotiff(
    active_crop,
    bounds,
    epsg,
    filename,
    band_names=["Active cropland"],
    colormap=colormap,
    nodata=nodatavalue,
)

# ...

# %%
# now run a test comparing dekadal timeseries vs monthly timeseries in different parts around the world!
