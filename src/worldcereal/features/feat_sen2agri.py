from datetime import datetime

from numba import njit, float32, int16, float64
import numpy as np

'''
USE IN SATIO:

S2_features_meta = {
     "sen2agri_temp_feat":{
        "function": sen2agri_temp_feat,
        "parameters":{
            'time_start': start_date,
            'time_freq': 10,
            'w': 2,
            'delta': 0.05,
            'tsoil': 0.2
        },
        "bands": ['ndvi'],
        "names": ['maxdif', 'mindif', 'difminmax', 'peak',
                    'lengthpeak', 'areapeak',
                      'ascarea', 'asclength', 'ascratio',
                      'descarea', 'desclength',
                     'descratio', 'soil1', 'soil2']
    }
}

'''


@njit(float64[:](float32[:], float64[:], int16, float64, float64))
def sen2agri_calc_feat(data, times, w, delta, tsoil):
    '''
    Actual function to calculate the features on individual timeseries

    '''
    features = np.zeros(14)
    ts_size = data.shape[0]

    # compute the index transitions only if the size of the input pixel
    # is greater or equal to
    # twice the size of the temporal window
    if np.greater_equal(ts_size, 2*w):
        for i in range(ts_size - (2 * w) + 1):
            # average of first window
            first = 0
            for j in range(i, i+w, 1):
                first += data[j]
            first = first / w
            # average of second window
            second = 0
            for j in range(i+w, i+(2*w), 1):
                second += data[j]
            second = second / w

            dif = first - second

            # max and min
            if i == 0 or features[0] < dif:
                features[0] = dif
            if i == 0 or features[1] > dif:
                features[1] = dif

        # compute the difference between max and min
        features[2] = features[0] - features[1]

    # compute the mbiFeatures associated to the maximum index value:
    if np.greater_equal(ts_size, w):
        minIndex = 0
        maxIndex = 0
        for i in range(ts_size - w + 1):
            # compute slice average
            avg = 0
            for j in range(i, i+w, 1):
                avg += data[j]
            avg = avg / w

            if np.greater(avg, features[3]):
                features[3] = avg
                minIndex = i
                maxIndex = i + w - 1
        # compute interval
        maxAvg = features[3]
        while (((features[3] > 0) and
                (data[minIndex] >= maxAvg - delta)) and
               (data[minIndex] <= maxAvg + delta)):
            minIndex += -1
            if minIndex < 0:
                minIndex = 0
                break
        while (((features[3] > 0) and
                (data[maxIndex] >= maxAvg - delta)) and
               (data[maxIndex] <= maxAvg + delta)):
            maxIndex += 1
            if maxIndex >= ts_size:
                maxIndex = ts_size - 1
                break
        # compute length of interval in days and the area
        intLength = times[maxIndex] - times[minIndex]  # in seconds
        features[4] = intLength / (3600 * 24)   # in days
        features[5] = features[3] * features[4]

    # compute the largest increasing and the largest decreasing features
    minIndexAsc = 0
    maxIndexAsc = 0
    minIndexDesc = 0
    maxIndexDesc = 0
    # previous variation: 0 - undefined, 1 - ascending, 2 - descending
    prevVar = 0
    for i in range(1, ts_size, 1):
        isDesc = data[i-1] > data[i]
        if prevVar == 0:
            # this is the first step, just save the current indices
            if isDesc:
                # we have descending slope
                minIndexDesc = i-1
                maxIndexDesc = i
                prevVar = 2
            else:
                # we have ascending slope
                minIndexAsc = i-1
                maxIndexAsc = i
                prevVar = 1
        elif (prevVar == 1) and not isDesc:
            # slope still positive -> update maxIndex ascending
            maxIndexAsc = i
        elif (prevVar == 2) and isDesc:
            # slope still negative -> update maxindex descending
            maxIndexDesc = i
        elif prevVar == 1:
            # ended an increasing interval -> compute area and
            # decide to keep or not
            difValue = data[maxIndexAsc] - data[minIndexAsc]
            difTime = times[maxIndexAsc] - times[minIndexAsc]
            difTime = difTime / (3600 * 24)
            area = (difValue * difTime) / 2
            if features[6] < area:
                features[6] = area
                features[7] = difTime
                features[8] = difValue / difTime
            # set new slope to descending
            minIndexDesc = i-1
            maxIndexDesc = i
            prevVar = 2
        else:
            # ended a decreasing interval -> compute area and decide
            # to keep or not
            difValue = data[minIndexDesc] - data[maxIndexDesc]
            difTime = times[maxIndexDesc] - times[minIndexDesc]
            difTime = difTime / (3600 * 24)
            area = (difValue * difTime) / 2
            if features[9] < area:
                features[9] = area
                features[10] = difTime
                features[11] = difValue / difTime
            # set new slope to ascending
            minIndexAsc = i-1
            maxIndexAsc = i
            prevVar = 1
    # process the last area
    if prevVar == 1:
        # ended an increasing interval -> compute area and decide
        # to keep or not
        difValue = data[maxIndexAsc] - data[minIndexAsc]
        difTime = times[maxIndexAsc] - times[minIndexAsc]
        difTime = difTime / (3600 * 24)
        area = (difValue * difTime) / 2
        if features[6] < area:
            features[6] = area
            features[7] = difTime
            features[8] = difValue / difTime
    elif prevVar == 2:
        # ended a decreasing interval -> compute area and decide
        # to keep or not
        difValue = data[minIndexDesc] - data[maxIndexDesc]
        difTime = times[maxIndexDesc] - times[minIndexDesc]
        difTime = difTime / (3600 * 24)
        area = (difValue * difTime) / 2
        if features[9] < area:
            features[9] = area
            features[10] = difTime
            features[11] = difValue / difTime

    # compute bare soil transitions
    # look for transitions
    for i in range(1, ts_size, 1):
        if (data[i-1] <= tsoil) and (data[i] >= tsoil):
            features[12] = 1
        elif (data[i-1] >= tsoil) and (data[i] <= tsoil):
            features[13] = 1

    return features


@njit(float64[:, :, :](float32[:, :, :],
                       float64[:], int16, float64, float64))
def sen2agri_pixelwrapper(x, times, w, delta, tsoil):

    _, nx, ny = x.shape

    features = np.zeros((14, nx, ny))
    for i in range(nx):
        for j in range(ny):
            features[:, i, j] = sen2agri_calc_feat(x[:, i, j],
                                                   times,
                                                   w,
                                                   delta,
                                                   tsoil)

    return features


def sen2agri_temp_feat(x, time_start='2018-01-01',
                       time_freq=10,  w=2, delta=0.05,
                       tsoil=0.2):
    '''
    Temporal features as defined in Sen2Agri project
    Source: Valero et al. (2016). Production of a dynamic cropland
    mask by processing
    remote sensing image series at high temporal and spatial resolutions.

    :param x: timeseries array with shape (time, x, y)
    :param times: 1D array with timestamps (datetime objects)
    :param w: size of window used for aggregating data while
            calculating features
    :param delta: difference threshold to determine whether a value belongs
        to the plateau of the curve
    :param tsoil: threshold of vi value below which a pixel is considered
        to be soil
    '''

    # create array of datetime.timestamp objects based on start and freq
    base = datetime.strptime(time_start, '%Y-%m-%d').timestamp()
    times = np.array([base + (i * time_freq*24*3600)
                      for i in range(x.shape[0])])

    features = sen2agri_pixelwrapper(x, times, w, delta, tsoil)

    return features
