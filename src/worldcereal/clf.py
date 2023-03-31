from typing import Dict

from loguru import logger
import numpy as np
from satio.features import Features
from scipy.signal import convolve2d

from worldcereal.classification.models import WorldCerealModel
from worldcereal.utils.scalers import minmaxscaler


def get_default_filtersettings():
    return {'kernelsize': 0,
            'conf_threshold': 0.9}


def mask(data, mask, valid=100, maskedvalue=255):
    data[mask != valid] = maskedvalue
    return data


def _filter_with_confidence(prediction, confidence, conf_thr,
                            kernel_size, no_data_value):

    if conf_thr is None:
        raise ValueError('Confidence threshold for majority'
                         'filter missing!')
    if conf_thr > 1:
        raise ValueError('Confidence threshold for majority'
                         'filtering should be between zero and one!')

    filteredprediction = _filter_without_confidence(
        prediction, kernel_size, no_data_value)

    # determine which cells need to be updated:
    # if confidence is low
    update_mask = ((confidence < conf_thr) &
                   (prediction != no_data_value))

    # produce final result
    newprediction = np.where(update_mask, filteredprediction, prediction)

    # Count flipped labels
    flipped_nr = (newprediction != prediction).sum()
    total_nr = (prediction != no_data_value).sum()
    flipped_perc = np.round(flipped_nr / total_nr * 100, 1)

    logger.info(f'Flipped {flipped_perc}% of pixels during spatial cleaning.')

    return newprediction


def _filter_without_confidence(prediction, kernel_size,
                               no_data_value):

    to_ignore = prediction == no_data_value

    # Convolution kernel
    k = np.ones((kernel_size, kernel_size), dtype=int)

    # count number of valid ones in each window
    pred_val_one = np.where(to_ignore, 0, prediction)
    val_ones_count = convolve2d(pred_val_one, k, 'same')

    # count number of valid zeros in each window
    pred_reverse = (prediction == 0).astype(np.uint16)
    pred_val_zero = np.where(to_ignore, 0, pred_reverse)
    val_zeros_count = convolve2d(pred_val_zero, k, 'same')

    # determine majority
    majority = np.where(val_ones_count > val_zeros_count, 1, 0)

    # determine which cells need to be updated:
    # if prediction is not no data and if there is a clear majority
    update_mask = ((val_ones_count != val_zeros_count) &
                   (prediction != no_data_value))

    # produce final result
    newprediction = np.where(update_mask, majority, prediction)

    return newprediction


def majority_filter(prediction, kernel_size,
                    confidence=None, conf_thr=None,
                    no_data_value=255):
    '''
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
    '''

    if kernel_size % 2 == 0:
        raise ValueError('Kernel size for majority filtering should be an'
                         ' an odd number!')

    if confidence is not None:
        return _filter_with_confidence(prediction, confidence, conf_thr,
                                       kernel_size, no_data_value)
    else:
        return _filter_without_confidence(prediction, kernel_size,
                                          no_data_value)


class WorldCerealClassifier(object):

    def __init__(self, worldcerealmodel: WorldCerealModel,
                 filtersettings: Dict = None,
                 maskdata=None):

        self.model = worldcerealmodel
        self.modeltype = worldcerealmodel.modeltype
        self.modelclass = worldcerealmodel.modelclass
        self.feature_names = worldcerealmodel.feature_names
        self.requires_scaling = worldcerealmodel.requires_scaling
        self.filtersettings = filtersettings or get_default_filtersettings()
        self.maskdata = maskdata

        self._check_model()

    def _check_model(self):
        if not isinstance(self.model, WorldCerealModel):
            raise ValueError(('Associated model should be '
                              'instance of WorldCerealModel '
                              f'but got: `{type(self.model)}`'))
        if self.modeltype is None:
            raise ValueError(('Associated model is of unknown '
                              'type. Should be `pixel` or `patch`.'))

    def predict(self, features: Features, threshold=0.5,
                fillnodata=0, nodatavalue=255):

        if threshold != 0.5:
            logger.warning(('Using custom decision '
                            f'threshold of: {threshold}'))

        # Select the features on which the model was trained
        inputfeatures = features.select(self.feature_names)

        # Scale the input features if the model requires it
        if self.requires_scaling:
            inputfeatures = self._scale(inputfeatures)

        # Get rid of any remaining NaN values
        if fillnodata is not None and self.model.impute_missing:
            inputfeatures.data[np.isnan(inputfeatures.data)] = fillnodata

        # do prediction
        if self.modeltype == 'pixel':
            prediction, confidence = self._predict_pixel_based(
                inputfeatures, threshold=threshold)
        elif self.modeltype == 'patch':
            prediction, confidence = self._predict_patch_based(
                inputfeatures, threshold=threshold)
        else:
            raise ValueError(('Unknown modeltype: '
                              f'{self.modeltype}'))

        # Set confidence of 0 to 0.01 to not interfere with nodata
        confidence[confidence < 0.01] = 0.01

        # mask prediction and confidence if necessary
        if self.maskdata is not None:
            prediction = mask(prediction, self.maskdata,
                              maskedvalue=nodatavalue)
            confidence = mask(confidence, self.maskdata,
                              maskedvalue=nodatavalue)

        # perform majority filtering
        if self.filtersettings['kernelsize'] > 0:
            # apply majority filter on prediction
            prediction = majority_filter(
                prediction,
                self.filtersettings['kernelsize'],
                confidence=confidence,
                conf_thr=self.filtersettings['conf_threshold'],
                no_data_value=nodatavalue
            )

        # Convert prediction/confidence to uint8
        prediction[prediction != nodatavalue] *= 100
        confidence[confidence != nodatavalue] *= 100

        return prediction.astype(np.uint8), confidence.astype(np.uint8)

    def _predict_pixel_based(self, features, threshold):
        logger.debug('Start pixel-based prediction ...')
        orig_shape = features.data.shape[1:3]
        inputs = features.data.transpose(
            (1, 2, 0)).reshape((-1, len(self.feature_names)))
        prediction, confidence = self.model.predict(inputs,
                                                    threshold=threshold,
                                                    orig_shape=orig_shape)
        prediction = prediction.reshape(orig_shape)
        confidence = confidence.reshape(orig_shape)

        return prediction, confidence

    def _predict_patch_based(self, features, threshold):
        '''
        First implementation of patch-based classifier.
        Should be improved
        '''
        logger.debug('Start patch-based prediction ...')
        logger.info("Running classification ...")
        windowsize = self.model.parameters['windowsize']
        xdim = features.data.shape[1]
        ydim = features.data.shape[2]

        if xdim == ydim == windowsize:
            # Features are already in correct spatial shape
            # we can directly make prediction
            prediction, confidence = self.model.predict(
                features.data.transpose((1, 2, 0)).reshape(
                    (1,
                     windowsize * windowsize,
                     -1)),
                threshold=threshold)
            prediction = prediction.squeeze().reshape((windowsize, windowsize))
            confidence = confidence.squeeze().reshape((windowsize, windowsize))
        else:
            # Slide through the block with overlap and make predictions

            prediction = np.empty((xdim, ydim))
            confidence = np.empty((xdim, ydim))

            for xStart in range(0, xdim, windowsize):
                for yStart in range(0, ydim, windowsize):
                    # We need to check if we're at the end of the master image
                    # We have to make sure we have a full subtile
                    # so we need to expand such tile and the resulting overlap
                    # with previous subtile is not an issue
                    if xStart + windowsize > xdim:
                        xStart = xdim - windowsize
                        xEnd = xdim
                    else:
                        xEnd = xStart + windowsize
                    if yStart + windowsize > ydim:
                        yStart = ydim - windowsize
                        yEnd = ydim
                    else:
                        yEnd = yStart + windowsize

                    features_patch = features.data[:,
                                                   xStart:xEnd,
                                                   yStart:yEnd]
                    patchprediction, patchconfidence = self.model.predict(
                        features_patch.transpose((1, 2, 0)).reshape(
                            (1, windowsize * windowsize, -1)),
                        threshold=threshold)

                    patchprediction = patchprediction.squeeze().reshape(
                        (windowsize, windowsize))
                    patchconfidence = patchconfidence.squeeze().reshape(
                        (windowsize, windowsize))

                    prediction[xStart:xEnd, yStart:yEnd] = patchprediction
                    confidence[xStart:xEnd, yStart:yEnd] = patchconfidence

        return prediction, confidence

    def _scale(self, features):

        scaled_features = []
        logger.info('Scaling input features ...')

        # Scale the data
        for ft in features.names:
            ftdata = features.select([ft]).data
            ftscaled = minmaxscaler(ftdata,
                                    ft_name=ft,
                                    clamp=(-0.1, 1.1),
                                    nodata=0)
            scaled_features.append(Features(data=ftscaled,
                                            names=[ft]))

        logger.info('Scaling done.')
        return Features.from_features(*scaled_features)
