import abc
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import List

from catboost import CatBoostClassifier
import joblib
from loguru import logger
import numpy as np
import pandas as pd
from satio.utils.retry import retry
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (StratifiedShuffleSplit,
                                     GridSearchCV,
                                     cross_val_score)
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score,
                             recall_score)

import tensorflow as tf
from tensorflow.keras.layers import (UpSampling2D, Dropout, Conv2D,
                                     BatchNormalization, Concatenate,
                                     Reshape, MaxPooling2D,
                                     Conv2DTranspose, Dense,
                                     LeakyReLU, concatenate, LSTM,
                                     Activation, ConvLSTM2D, Conv3D,
                                     )
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import Input, Model

from worldcereal.utils import (get_sensor_config,
                               get_sensor_config_timeseries)
from worldcereal.utils import (probability_to_binary,
                               probability_to_confidence)


SUPPORTED_MODELS = [
    'WorldCerealRFModel', 'WorldCerealUNET',
    'WorldCerealCNN', 'WorldCerealPatchLSTM',
    'WorldCerealFFNN', 'WorldCerealPixelLSTM',
    'WorldCerealCatBoostModel'
]

TRIES = 5
BACKOFF = 10
DELAY = 5


class WorldCerealModel(object, metaclass=abc.ABCMeta):
    '''
    Abstract base class for WorldCereal model implementations
    '''

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def predict(self):
        pass

    @abc.abstractmethod
    def save(self):
        pass

    @abc.abstractclassmethod
    def load(self):
        pass

    @abc.abstractmethod
    def summary(self):
        pass

    @abc.abstractmethod
    def evaluate(self):
        pass

    @staticmethod
    @retry(exceptions=TimeoutError, tries=TRIES, delay=DELAY,
           backoff=BACKOFF, logger=logger)
    def from_config(configfile):
        logger.info(('Loading WorldCereal model '
                     f'from config: {configfile} ...'))
        configfile = str(configfile)
        if not configfile.endswith('json'):
            raise ValueError('Configfile should be json.')
        if configfile.startswith("https"):
            from urllib.request import urlopen
            response = urlopen(configfile)
            config = json.loads(response.read())
        else:
            with open(configfile) as f:
                config = json.load(f)

        modelclass = config['settings']['modelclass']
        if modelclass not in SUPPORTED_MODELS:
            raise ValueError((f'Model class `{modelclass}` not known. '
                              f'Should be one of: {SUPPORTED_MODELS}'))

        basedir = config['paths'].get(
            'basedir', tempfile.mkdtemp())

        # Load the model
        modelfile = config['paths']['modelfile']
        if modelfile is None:
            raise ValueError('Config file has no path to a model')
        model = eval(modelclass).load(modelfile)

        # Get other parameters
        feature_names = config['feature_names']
        parameters = config['parameters']

        return eval(modelclass)(
            model=model,
            basedir=basedir,
            feature_names=feature_names,
            parameters=parameters,
            exist_ok=True,
            config=config
        )


class WorldCerealBaseModel(WorldCerealModel):

    def __init__(self,
                 model=None,
                 modeltype=None,
                 feature_names=None,
                 requires_scaling=False,
                 parameters=None,
                 basedir=None,
                 overwrite=False,
                 exist_ok=False,
                 parentmodel=None,
                 config=None):
        self.model = model
        self.modeltype = modeltype
        self.modelclass = type(self).__name__
        self.basedir = basedir
        self.feature_names = feature_names
        self.requires_scaling = requires_scaling
        self.parameters = parameters or {}
        self.parentmodel = parentmodel
        self.config = config
        self.impute_missing = True

        if modeltype is None:
            raise ValueError('`modeltype` cannot be None')

        if feature_names is None:
            raise ValueError('`feature_names` cannot be None')

        if basedir is None:
            raise ValueError('`basedir` cannot be None')

        if Path(basedir).is_dir():
            if exist_ok:
                pass
            elif not overwrite:
                raise ValueError((f'Basedir `{basedir}` is '
                                  'not empty. Please delete '
                                  'or use `overwrite=True`'))
            else:
                shutil.rmtree(basedir)

        Path(basedir).mkdir(parents=True, exist_ok=True)

        if self.config is None:
            self.create_config()

    @ classmethod
    def load(cls, file):
        raise NotImplementedError('No model loader available.')

    def save(self, file):
        raise NotImplementedError(('Cannot save model directly '
                                   'from base class.'))

    def predict(self, inputs, threshold=0.5, **kwargs):
        '''Main prediction method for binary WorldCerealModel
        Actual model-specific prediction happens in the subclass.
        Argument threshold allows to tune the prediction threshold.
        '''

        if self.model is None:
            raise Exception('No model loaded yet')

        probabilities = self._predict(inputs, **kwargs)

        if not np.allclose(probabilities.sum(axis=1), 1):
            raise ValueError('Probabilities do not sum to 1!')

        # Transform to binary prediction
        prediction = probability_to_binary(probabilities, threshold)

        # Get the sample-specific confidence
        confidence = probability_to_confidence(probabilities)

        return prediction, confidence

    def evaluate(self, inputs, outputs, original_labels=None,
                 outdir=None, pattern='', encoder=None):
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        import matplotlib.pyplot as plt

        if self.model is None:
            raise ValueError('No model initialized yet.')
        predictions, confidence = self.predict(inputs)

        outdir = outdir or self.basedir

        if encoder is not None:
            predictions = encoder.inverse_transform(predictions)

        # Make sure predictions are now 1D
        predictions = predictions.squeeze()

        # Make absolute confusion matrix
        cm = confusion_matrix(outputs, predictions,
                              labels=np.unique(outputs))
        disp = ConfusionMatrixDisplay(cm, display_labels=np.unique(outputs))
        _, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
        plt.tight_layout()
        plt.savefig(str(Path(outdir) / f'{pattern}CM_abs.png'))
        plt.close()

        # Make relative confusion matrix
        cm = confusion_matrix(outputs, predictions, normalize='true',
                              labels=np.unique(outputs))
        disp = ConfusionMatrixDisplay(cm, display_labels=np.unique(outputs))
        _, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='.1f',
                  colorbar=False)
        plt.tight_layout()
        plt.savefig(str(Path(outdir) / f'{pattern}CM_norm.png'))
        plt.close()

        # Compute evaluation metrics
        metrics = {}
        if len(np.unique(outputs)) == 2:
            metrics['OA'] = np.round(accuracy_score(
                outputs, predictions), 3)
            metrics['F1'] = np.round(f1_score(
                outputs, predictions), 3)
            metrics['Precision'] = np.round(precision_score(
                outputs, predictions), 3)
            metrics['Recall'] = np.round(recall_score(outputs, predictions), 3)
        else:
            metrics['OA'] = np.round(accuracy_score(
                outputs, predictions), 3)
            metrics['F1'] = np.round(f1_score(
                outputs, predictions, average='macro'), 3)
            metrics['Precision'] = np.round(precision_score(
                outputs, predictions, average='macro'), 3)
            metrics['Recall'] = np.round(recall_score(
                outputs, predictions, average='macro'), 3)

        # Write metrics to disk
        with open(str(Path(outdir) / f'{pattern}metrics.txt'), 'w') as f:
            f.write('Test results:\n')
            for key in metrics.keys():
                f.write(f'{key}: {metrics[key]}\n')
                logger.info(f'{key} = {metrics[key]}')

        cm = confusion_matrix(outputs, predictions)
        outputlabels = list(np.unique(outputs).astype(int))
        predictlabels = list(np.unique(predictions).astype(int))
        outputlabels.extend(predictlabels)
        outputlabels = list(dict.fromkeys(outputlabels))
        outputlabels.sort()
        cm_df = pd.DataFrame(data=cm, index=outputlabels, columns=outputlabels)
        outfile = Path(outdir) / f'{pattern}confusion_matrix.txt'
        cm_df.to_csv(outfile)
        if original_labels is not None:
            datadict = {'ori': original_labels.astype(int),
                        'pred': predictions.astype(int)}
            data = pd.DataFrame.from_dict(datadict)
            count = data.groupby(['ori', 'pred']).size()
            result = count.to_frame(name='count').reset_index()
            outfile = (Path(outdir) /
                       f'{pattern}confusion_matrix_original_labels.txt')
            result.to_csv(outfile, index=False)

        return metrics

    def create_config(self):

        config = {}
        config['parameters'] = self.parameters
        config['settings'] = dict(
            modeltype=self.modeltype,
            modelclass=self.modelclass,
            requires_scaling=self.requires_scaling
        )
        config['feature_names'] = self.feature_names
        config['paths'] = dict(
            basedir=str(self.basedir),
            modelfile=None,
            modelweights=None,
            parentmodel=self.parentmodel
        )
        self.config = config
        self.save_config()

    def save_config(self):
        configpath = Path(self.basedir) / 'config.json'
        with open(configpath, 'w') as f:
            json.dump(self.config, f, indent=4)

    def transfer(self, basedir):
        '''
        Method to transfer the model to a new directory
        where it can be used to retrain the model on other data
        '''
        if self.model is None:
            raise ValueError('No model loaded to transfer.')
        logger.info(f'Transferring model to: {basedir}')
        return self.__class__(basedir=basedir,
                              feature_names=self.feature_names,
                              parameters=self.parameters,
                              model=self.model,
                              parentmodel=self.config['paths']['modelfile'])

    def _predict(inputs, **kwargs):
        raise NotImplementedError


class WorldCerealSklearnModel(WorldCerealBaseModel):

    def __init__(self, modeltype='pixel', **kwargs):
        super().__init__(modeltype=modeltype, **kwargs)

    def tune(self, tuning, inputs, outputs, outfile):

        # check whether there are classes which have less than 3 samples
        # these should be removed!
        unique, counts = np.unique(outputs, return_counts=True)
        idx = np.nonzero(counts < 3)
        toremove = list(unique[idx])
        if len(toremove) > 0:
            logger.warning(f'{len(toremove)} classes will be discared'
                           'for hyperparameter tuning!')
            for tr in toremove:
                idx = np.nonzero(outputs != tr)
                inputs = inputs[idx]
                outputs = outputs[idx]

        # if you only have one class -> don't do anything!
        if np.all(outputs == outputs[0]):
            logger.info('Only one target class for hyperparameter tuning...'
                        'skipped!')
        else:
            logger.info('Running hyperparameter tuning...')
            # setup the cross-validation via shuffle split
            # (3 folded validation enough for rough search)
            cv = StratifiedShuffleSplit(n_splits=3, test_size=0.3)
            # setup grid search
            grid = GridSearchCV(self.model,
                                param_grid=tuning,
                                cv=cv, verbose=2, n_jobs=None)
            # run grid search
            grid.fit(inputs, outputs)
            # save result to the model
            for param in tuning.keys():
                setattr(self.model, param, grid.best_params_[param])

            # run a separate 5folded cross-validation to get the RF_score
            logger.info('Estimating train_score...')
            train_score = np.mean(cross_val_score(self.model, inputs,
                                                  outputs, cv=5))

            if outfile is not None:
                logger.info(f'Writing results to {outfile}')
                result = {'n_estimators': self.model.n_estimators,
                          'min_samples_split': self.model.min_samples_split,
                          'min_samples_leaf': self.model.min_samples_leaf,
                          'max_features': self.model.max_features,
                          'criterion': self.model.criterion,
                          'train_score': train_score}
                with open(outfile, 'w') as out:
                    json.dump(result, out)

    def train(self, inputs, outputs,
              resamplers=None,
              tuning=None,
              outfile=None):

        if inputs.shape[1] != len(self.feature_names):
            raise ValueError(('Model was initialized for '
                              f'{len(self.feature_names)} '
                              'features but got '
                              f'{inputs.shape[1]} for '
                              'fitting.'))

        if self.model is None:
            raise ValueError('No model initialized yet.')
        if resamplers is not None:
            for res in resamplers:
                inputs, outputs = res.fit_resample(inputs,
                                                   outputs)
        if tuning is not None:
            self.tune(tuning, inputs, outputs, outfile)
        self.model.fit(inputs, outputs)

    def _predict(self, inputs, **kwargs):
        return self.model.predict_proba(inputs)

    def save(self, modelfile):
        modelfile = str(modelfile)
        if self.model is None:
            raise ValueError('No model initialized yet.')
        if not modelfile.endswith('pkl'):
            modelfile += '.pkl'
        logger.info(f'Saving model to: {modelfile}')
        joblib.dump(self.model, modelfile)

        # Update config
        self.config['paths']['modelfile'] = modelfile
        self.save_config()

    @ classmethod
    @retry(exceptions=TimeoutError, tries=TRIES, delay=DELAY,
           backoff=BACKOFF, logger=logger)
    def load(cls, modelfile):
        if not modelfile.endswith('pkl'):
            modelfile += '.pkl'
        if modelfile.startswith("https"):
            import urllib
            modelfile, _ = urllib.request.urlretrieve(modelfile)

        logger.info(f'Restoring model from: {modelfile}')
        return joblib.load(modelfile)

    def summary(self):
        if self.model is None:
            raise ValueError('No model initialized yet.')
        logger.info(self.model)


class WorldCerealRFModel(WorldCerealSklearnModel):

    def __init__(self, parameters: dict = {},
                 model=None, **kwargs):
        '''
        Create a WorldCereal Random Forest model
        pass additional parameters for
        the sklearn RandomForestClassifier
        through `parameters` dict
        '''

        if model is None:
            model = RandomForestClassifier(
                **parameters
            )
        super().__init__(model=model, modeltype='pixel',
                         parameters=parameters, **kwargs
                         )

    def featImp(self, outfile):
        '''
        outfile: path without extension!
        '''
        df = pd.DataFrame(self.model.feature_importances_,
                          columns=['fimp'])
        df['labels'] = self.feature_names

        df = df.sort_values('fimp', ascending=False)
        # from matplotlib import pyplot as plt
        # y_pos = np.arange(df.shape[0])
        # fig, ax = plt.subplots()
        # ax.barh(y_pos, df['fimp'].values)
        # ax.set_yticks(y_pos)
        # ax.set_yticklabels(df['labels'].values)
        # ax.invert_yaxis()
        # ax.set_xlabel('Feature importance')
        # plt.savefig(outfile + '.png', dpi=600)
        # plt.close()
        df.to_csv(outfile + '.csv')


class WorldCerealCatBoostModel(WorldCerealBaseModel):
    def __init__(self, gpu=False, model=None,
                 iterations=8000, depth=8,
                 random_seed=1234, classes_count=None,
                 learning_rate=0.05, early_stopping_rounds=20,
                 **kwargs):

        if gpu:
            task_type = "GPU"
            devices = '0'
        else:
            task_type = "CPU"
            devices = None

        if model is None:
            model = CatBoostClassifier(
                iterations=iterations, depth=depth,
                random_seed=random_seed,
                learning_rate=learning_rate,
                early_stopping_rounds=early_stopping_rounds,
                task_type=task_type,
                classes_count=classes_count,
                devices=devices,
                l2_leaf_reg=3
            )
        super().__init__(model=model, modeltype='pixel',
                         **kwargs)

        self.impute_missing = False  # CatBoost can handle NaN

    def train(self, inputs, outputs=None, cat_features=None, **kwargs):

        if inputs.shape[1] != len(self.feature_names):
            raise ValueError(('Model was initialized for '
                              f'{len(self.feature_names)} '
                              'features but got '
                              f'{inputs.shape[1]} for '
                              'fitting.'))

        if self.model is None:
            raise ValueError('No model initialized yet.')

        if 'init_model' in kwargs:
            logger.info('Continuing training from previous model!')

        self.model.fit(inputs, outputs,
                       cat_features=cat_features, **kwargs)

    def _predict(self, inputs, orig_shape=None, **kwargs):
        if type(inputs) == np.ndarray:
            inputs = pd.DataFrame(data=inputs,
                                  columns=self.feature_names)

        # Make sure categorical features or categorical
        for ft in self.model.get_cat_feature_indices():
            inputs.iloc[:, ft] = inputs.iloc[:, ft].astype(int)

        if orig_shape is not None:
            '''Pathway where we can smooth the probabilities
            '''
            raw_probs = self.model.predict_proba(inputs).reshape(
                list(orig_shape) + [2])

            kernel = self.get_gaussian_kernel()
            filtered_probs = self.convolve_probs(raw_probs, kernel)
            mask = raw_probs.max(axis=-1) >= 0.85
            filtered_probs[mask] = raw_probs[mask]
            filtered_probs = filtered_probs.reshape((-1, 2))
        else:
            filtered_probs = self.model.predict_proba(inputs)

        return filtered_probs

    def grid_search(self, grid, X):
        model = CatBoostClassifier(early_stopping_rounds=20,
                                   eval_metric='F1')
        logger.info(f'Starting grid search for parameter grid: {grid}')
        results = model.grid_search(grid, X, verbose=False)
        return results

    def save(self, modelfile):
        modelfile = str(modelfile)
        if self.model is None:
            raise ValueError('No model initialized yet.')
        if not modelfile.endswith('cbm'):
            modelfile += '.cbm'
        logger.info(f'Saving model to: {modelfile}')
        self.model.save_model(modelfile)

        # Update config
        self.config['paths']['modelfile'] = modelfile
        self.save_config()

    @ classmethod
    @retry(exceptions=TimeoutError, tries=TRIES, delay=DELAY,
           backoff=BACKOFF, logger=logger)
    def load(cls, modelfile):
        logger.info(f'Restoring model from: {modelfile}')
        if modelfile.startswith("https"):
            import urllib
            modelfile, _ = urllib.request.urlretrieve(modelfile)

        model = CatBoostClassifier()

        return model.load_model(modelfile)

    def summary(self):
        if self.model is None:
            raise ValueError('No model initialized yet.')
        logger.info(self.model.get_params())

    def transfer(self, basedir, **kwargs):
        '''Override parent method because the model
        will need to be re-initialized instead of
        transferred.
        '''
        if self.model is None:
            raise ValueError('No model loaded to transfer.')
        logger.info(f'Transferring and resetting model to: {basedir}')
        return self.__class__(basedir=basedir,
                              feature_names=self.feature_names,
                              parameters=self.parameters,
                              parentmodel=self.config['paths']['modelfile'],
                              **kwargs)

    def retrain(self, init_model, inputs, outputs=None,
                cat_features=None, **kwargs):
        if self.model is None:
            raise ValueError('No model loaded yet.')

        self.train(inputs, outputs=outputs, cat_features=cat_features,
                   init_model=init_model, **kwargs)

    @staticmethod
    def convolve_probs(probs, kernel):
        """
        Perform 2d convolution of kernel to array of probabilities
        along the labels axis
        """
        import scipy

        filtered_probs = np.zeros(probs.shape)
        for i in range(probs.shape[-1]):
            filtered_probs[..., i] = scipy.signal.convolve2d(
                probs[..., i],
                kernel,
                mode='same',
                boundary='symm')

        return filtered_probs

    @staticmethod
    def get_gaussian_kernel(kernlen=7, std=1):
        """Returns a 2D Gaussian kernel array."""
        import scipy

        gkern1d = scipy.signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
        gkern2d = np.outer(gkern1d, gkern1d)
        gkern2d = gkern2d/gkern2d.sum()
        return gkern2d


class WorldCerealKerasModel(WorldCerealBaseModel):

    def train(self,
              calibrationx=None,
              calibrationy=None,
              validationdata=None,
              steps_per_epoch=None,
              validation_steps=None,
              epochs=100,
              modelfile=None,
              weightsfile=None,
              learning_rate=None,
              earlystopping=True,
              reducelronplateau=True,
              tensorboard=False,
              csvlogger=False,
              customcallbacks: List = None,
              **kwargs
              ):

        if self.model is None:
            raise Exception('No model loaded yet')

        if validationdata is None:
            earlystopping = False

        callbacks = []

        if modelfile is not None:
            checkpointermodel = tf.keras.callbacks.ModelCheckpoint(
                filepath=modelfile, save_best_only=True
            )
            callbacks.append(checkpointermodel)

            # Also save the network architecture to a text file
            with open(Path(modelfile).parent /
                      (str(Path(modelfile).stem) +
                       '_architecture.log'), 'w') as f:
                f.write("--- " + str(Path(modelfile).stem) + " ---" + "\n\n\n")
                self.model.summary(
                    print_fn=lambda x: f.write(x + "\n"))

            # Plot the model
            try:
                plotfile = (Path(modelfile).parent /
                            (str(Path(modelfile).stem) +
                             '_architecture.png'))
                plot_model(
                    self.model,
                    to_file=plotfile,
                    show_shapes=True,
                    dpi=96,
                )
            except ImportError:
                logger.warning('Could not plot model!')

        if weightsfile is not None:
            checkpointerweights = tf.keras.callbacks.ModelCheckpoint(
                filepath=weightsfile,
                save_best_only=True,
                save_weights_only=True
            )
            callbacks.append(checkpointerweights)

        if earlystopping:
            earlystoppingcallback = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', min_delta=0.001, patience=5,
                restore_best_weights=True, verbose=1
            )
            callbacks.append(earlystoppingcallback)

        if reducelronplateau:
            reducelrcallback = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.1, patience=3,
                verbose=1,  mode="auto", min_delta=0.0001)
            callbacks.append(reducelrcallback)

        if tensorboard:
            log_dir = Path(modelfile).parent / 'tensorboardlogs'
            tensorboardcallback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir
            )
            callbacks.append(tensorboardcallback)

        if csvlogger:
            log_file = Path(modelfile).parent / 'kerastraininglog.csv'

            # Delete log file if it exists
            if os.path.exists(log_file):
                os.remove(log_file)
            csvloggercallback = tf.keras.callbacks.CSVLogger(log_file,
                                                             separator=",",
                                                             append=True)
            callbacks.append(csvloggercallback)

        if customcallbacks is not None:
            callbacks += customcallbacks

        if learning_rate is not None:
            logger.info(f'Adjusting learning rate to: {learning_rate}')
            K.set_value(self.model.optimizer.learning_rate, learning_rate)

        logger.info('-'*30)
        logger.info('Starting model training ...')
        logger.info('-'*30)

        self.model.fit(
            x=calibrationx,
            y=calibrationy,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=validationdata,
            validation_steps=validation_steps,
            **kwargs
        )

    def _predict(self, inputs, **kwargs):
        probabilities = self.model.predict(inputs,
                                           verbose=1)

        # We need to return probs for each class
        probabilities = np.array([1-probabilities,
                                  probabilities]).transpose()

        return probabilities

    @ classmethod
    @retry(exceptions=TimeoutError, tries=TRIES, delay=DELAY,
           backoff=BACKOFF, logger=logger)
    def load(cls, file: str):
        if file.startswith("https"):
            import h5py
            from urllib.request import urlopen
            from io import BytesIO
            file = h5py.File(BytesIO(urlopen(file).read()), 'r')

        return tf.keras.models.load_model(
            file,
            custom_objects={'DiceBCELoss': DiceBCELoss}
        )

    def save(self, file):
        if self.model is None:
            raise Exception('No model loaded yet')

        bn = Path(file).stem.replace('_weights', '').replace('.h5', '')
        bn_weights = str(Path(file).parent / (bn + '_weights.h5'))
        logger.info(f'Saving model weights to: {bn_weights}')
        self.model.save_weights(bn_weights)

        bn_model = Path(file).parent / (bn + '.h5')
        logger.info(f'Saving model to: {bn_model}')
        self.model.save(bn_model)

        # Update config
        self.config['paths']['modelfile'] = str(bn_model)
        self.config['paths']['modelweights'] = str(bn_weights)
        self.save_config()

    def summary(self, **kwargs):
        if self.model is None:
            raise ValueError('No model associated with this object')
        else:
            self.model.summary(**kwargs)

    def retrain(self, **kwargs):
        raise NotImplementedError(
            'Method should be implemented for specific model!')


class WorldCerealUNET(WorldCerealKerasModel):
    def __init__(self,
                 model=None,
                 feature_names=None,
                 parameters: dict = {},
                 **kwargs
                 ):

        if feature_names is None:
            raise ValueError('`feature_names` cannot be None')

        _req_params = ['windowsize']

        for param in _req_params:
            if param not in parameters:
                raise ValueError((f'Parameter `{param}` is '
                                  'compulsory for this model '
                                  'but was not found.'))

        self.windowsize = parameters['windowsize']
        self.dropoutfraction = 0.5
        self.startchannels = parameters.get('startchannels', 64)
        self.unetdepth = parameters.get('unetdepth', 5)
        self.sensorconfig = get_sensor_config(feature_names)

        parameters['sensorconfig'] = self.sensorconfig

        super().__init__(modeltype='patch',
                         parameters=parameters,
                         requires_scaling=True,
                         feature_names=feature_names,
                         **kwargs)

        if model is None:
            self.model = self.unetmodel()
        else:
            self.model = model

        optimizer = Adam(0.0002, 0.5)
        self.model.compile(optimizer=optimizer, loss=DiceBCELoss,
                           weighted_metrics=['accuracy'],
                           sample_weight_mode="temporal")

    def DiceBCELoss(y_true, y_pred, smooth=1):

        BCE = binary_crossentropy(y_true, y_pred)
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        dice_loss = (1 - (2. * intersection + smooth) /
                     (K.sum(K.square(y_true), -1) +
                      K.sum(K.square(y_pred), -1) + smooth))
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

    def unetmodel(self):
        '''
        U-NET implementation from https://github.com/pietz/unet-keras
        Adapted here to take pixel-based loss weighting into account.
        Original U-NET paper: https://arxiv.org/abs/1505.04597
        '''
        def conv_block(m, dim, acti, bn, res, do=0):
            n = Conv2D(dim, 3, activation=acti, padding='same')(m)
            n = BatchNormalization()(n) if bn else n
            n = Dropout(do)(n) if do else n
            n = Conv2D(dim, 3, activation=acti, padding='same')(n)
            n = BatchNormalization()(n) if bn else n
            return Concatenate()([m, n]) if res else n

        def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
            if depth > 0:
                n = conv_block(m, dim, acti, bn, res)
                m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2,
                                                        padding='same')(n)
                m = level_block(m, int(inc * dim), depth - 1, inc, acti,
                                do, bn, mp, up, res)
                if up:
                    m = UpSampling2D()(m)
                    m = Conv2D(dim, 2, activation=acti, padding='same')(m)
                else:
                    m = Conv2DTranspose(dim, 3, strides=2, activation=acti,
                                        padding='same')(m)
                n = Concatenate()([n, m])
                m = conv_block(n, dim, acti, bn, res)
            else:
                m = conv_block(m, dim, acti, bn, res, do)
            return m

        def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2.,
                 activation='relu',
                 dropout=0.5, batchnorm=True, maxpool=True, upconv=True,
                 residual=False):
            i = Input(shape=(None, img_shape[2]))
            x = Reshape(img_shape)(i)
            o = level_block(x, start_ch, depth, inc_rate, activation,
                            dropout, batchnorm, maxpool, upconv, residual)
            o = Conv2D(out_ch, 1, activation='sigmoid')(o)
            o = Reshape((-1, 1))(o)
            return Model(inputs=i, outputs=o)

        return UNet((self.windowsize, self.windowsize,
                     len(self.feature_names)),
                    start_ch=self.startchannels, depth=self.unetdepth)


class WorldCerealCNN(WorldCerealKerasModel):
    def __init__(self,
                 model=None,
                 feature_names=None,
                 parameters: dict = {},
                 **kwargs
                 ):

        if feature_names is None:
            raise ValueError('`feature_names` cannot be None')

        _req_params = ['windowsize']

        for param in _req_params:
            if param not in parameters:
                raise ValueError((f'Parameter `{param}` is '
                                  'compulsory for this model '
                                  'but was not found.'))

        self.windowsize = parameters['windowsize']
        self.dropoutfraction = 0.5
        self.sensorconfig = get_sensor_config(feature_names)

        parameters['sensorconfig'] = self.sensorconfig

        super().__init__(modeltype='patch',
                         parameters=parameters,
                         requires_scaling=True,
                         feature_names=feature_names,
                         **kwargs)

        if model is None:
            self.model = self.encoderdecodermodel()
        else:
            self.model = model

        self.optimizer = Adam(0.0002, 0.5)
        self.model.compile(optimizer=self.optimizer, loss=DiceBCELoss,
                           weighted_metrics=['accuracy'],
                           sample_weight_mode="temporal")

    def encoderdecodermodel(self):
        '''
        This method outlines the main CNN model
        architecture consisting of sensor-specific
        encoding stacks, a concatenated bottleneck layer
        and a merged decoding step to which skip connections
        with the encoder are foreseen.
        '''

        # Get a local copy of this because we'll modify it
        sensorconfig = get_sensor_config(self.feature_names)

        defaultkernelsize = 4
        defaultstride = 2
        init = RandomNormal(mean=0.0, stddev=0.02)

        def conv2d(layer_input, filters, kernelsize, stride, init,
                   batchnormalization=True, l2_weight_regularizer=1e-5):
            c = Conv2D(filters, kernel_size=(kernelsize, kernelsize),
                       strides=stride, padding='same',
                       activation=None,
                       kernel_initializer=init,
                       kernel_regularizer=l2(
                l2_weight_regularizer))(layer_input)
            if batchnormalization:
                c = BatchNormalization()(c)
            c = LeakyReLU(alpha=0.2)(c)
            return c

        def deconv2d(layer_input, skip_input, filters, kernelsize, stride,
                     init, dropout, batchnormalization=True,
                     l2_weight_regularizer=1e-5):
            d = Conv2DTranspose(filters, kernel_size=kernelsize,
                                strides=stride, padding='same',
                                activation=None,
                                kernel_initializer=init,
                                kernel_regularizer=l2(
                                    l2_weight_regularizer))(layer_input)
            if batchnormalization:
                d = BatchNormalization()(d)
            if dropout:
                d = Dropout(dropout)(d)
            d = Activation('relu')(d)
            if skip_input:
                if type(skip_input) is not list:
                    skip_input = [skip_input]
                d = concatenate([d] + skip_input)
            return d

        def _encoder(enc_input, filters=64):

            input_reshaped = Reshape((self.windowsize,
                                      self.windowsize,
                                      enc_input.shape[2]))(enc_input)

            enc1 = conv2d(input_reshaped,
                          filters=filters,
                          kernelsize=defaultkernelsize,
                          stride=defaultstride,
                          init=init)
            enc2 = conv2d(enc1,
                          filters=filters * 2,
                          kernelsize=defaultkernelsize,
                          stride=defaultstride,
                          init=init)
            enc3 = conv2d(enc2,
                          filters=filters * 4,
                          kernelsize=defaultkernelsize,
                          stride=defaultstride,
                          init=init)
            enc4 = conv2d(enc3,
                          filters=filters * 4,
                          kernelsize=defaultkernelsize,
                          stride=defaultstride,
                          init=init)

            # Bottleneck
            enc_output = conv2d(enc4,
                                filters=filters * 4,
                                kernelsize=defaultkernelsize,
                                stride=defaultstride,
                                init=init)

            return {'enc1': enc1, 'enc2': enc2, 'enc3': enc3,
                    'enc4': enc4, 'enc_output': enc_output}

        # --------------------------------------
        # Inputs
        # --------------------------------------

        # Inputs
        modelinput = Input(
            shape=(None, len(self.feature_names)),
            name='full_input'
        )

        # The tf.gather operator selects the
        # sensor-specific channels from the full input
        for sensor in sensorconfig.keys():
            sensorconfig[sensor]['input'] = tf.gather(
                params=modelinput,
                indices=sensorconfig[sensor]['positions'],
                axis=2
            )

        # --------------------------------------
        # Spatial encoding step
        # --------------------------------------

        # Encoders
        for sensor in sensorconfig.keys():
            sensorconfig[sensor]['encoded'] = _encoder(
                sensorconfig[sensor]['input']
            )

        # --------------------------------------
        # Concatenation of the encoded features
        # --------------------------------------
        encoded = [
            sensorconfig[sensor]['encoded']['enc_output']
            for sensor in sensorconfig.keys()
        ]

        if len(encoded) > 1:
            concatenated = concatenate(encoded)
        else:
            concatenated = encoded[0]

        # --------------------------------------
        # Spatial decoding step
        # --------------------------------------

        dec4 = deconv2d(concatenated,
                        skip_input=[sensorconfig[sensor]['encoded']['enc4']
                                    for sensor in sensorconfig.keys()],
                        filters=256,
                        kernelsize=defaultkernelsize,
                        stride=defaultstride,
                        init=init, dropout=self.dropoutfraction)
        dec3 = deconv2d(dec4,
                        skip_input=[sensorconfig[sensor]['encoded']['enc3']
                                    for sensor in sensorconfig.keys()],
                        filters=256,
                        kernelsize=defaultkernelsize,
                        stride=defaultstride,
                        init=init, dropout=self.dropoutfraction)
        dec2 = deconv2d(dec3,
                        skip_input=[sensorconfig[sensor]['encoded']['enc2']
                                    for sensor in sensorconfig.keys()],
                        filters=128,
                        kernelsize=defaultkernelsize,
                        stride=defaultstride,
                        init=init, dropout=0)
        dec1 = deconv2d(dec2,
                        skip_input=[sensorconfig[sensor]['encoded']['enc1']
                                    for sensor in sensorconfig.keys()],
                        filters=64,
                        kernelsize=defaultkernelsize,
                        stride=defaultstride,
                        init=init, dropout=0)

        # OUTPUT LAYER
        classification = Conv2DTranspose(
            filters=1,
            kernel_size=defaultkernelsize,
            strides=defaultstride, padding='same',
            activation='sigmoid',
            kernel_initializer=init)(dec1)

        # Define final inputs and outputs of the model
        # and return the model
        output = Reshape(
            (self.windowsize * self.windowsize, 1))(classification)

        return Model(inputs=modelinput, outputs=output)

    def retrain(self, trainablelayers=20, **kwargs):
        if self.model is None:
            raise ValueError('No model loaded yet.')

        '''
        Retraining happens in two steps. First we freeze
        the entire base model and train a new classification
        head from scratch.

        Second - optionally - we unfreeze the base model
        and finetune a certain amount of pretrained layers
        of the base model as well.
        '''

        # First freeze the entire model
        logger.info('Freezing base model ...')
        for i in range(len(self.model.layers)):
            self.model.layers[i].trainable = False

        # Now reset the model weights of the classification head
        # and make them trainable
        logger.info('Resetting classification head ...')
        if 'conv2d' not in self.model.layers[-2].name:
            raise ValueError(('Classification head does not '
                              'seem to be classification '
                              f'layer (`{self.model.layers[-2].name}`)'))
        weight_initializer = self.model.layers[-2].kernel_initializer
        bias_initializer = self.model.layers[-2].bias_initializer
        old_weights, old_biases = self.model.layers[-2].get_weights()

        self.model.layers[-2].set_weights([
            weight_initializer(shape=old_weights.shape),
            bias_initializer(shape=old_biases.shape)])
        self.model.layers[-2].trainable = True

        # Recompile the model
        self.model.compile(optimizer=self.optimizer, loss=DiceBCELoss,
                           weighted_metrics=['accuracy'],
                           sample_weight_mode="temporal")

        # Train the classification head
        self.train(**kwargs)

        # Now let's finetune the entire model
        # start by unfreezing desired nr of layers,
        # except for batchnormalization layers (!!!)
        for i in range(
                len(self.model.layers)-trainablelayers, len(
                    self.model.layers)):
            if 'batch_normalization' not in self.model.layers[i].name:
                self.model.layers[i].trainable = True

        # Recompile the model
        self.model.compile(optimizer=self.optimizer, loss=DiceBCELoss,
                           weighted_metrics=['accuracy'],
                           sample_weight_mode="temporal")

        # Start finetuning
        logger.info(('Starting retraining on the last '
                     f'{trainablelayers} layers ...'))
        self.train(**kwargs)


class WorldCerealPatchLSTM(WorldCerealKerasModel):
    def __init__(self,
                 model=None,
                 feature_names=None,
                 parameters: dict = {},
                 **kwargs
                 ):

        if feature_names is None:
            raise ValueError('`feature_names` cannot be None')

        _req_params = ['windowsize']

        for param in _req_params:
            if param not in parameters:
                raise ValueError((f'Parameter `{param}` is '
                                  'compulsory for this model '
                                  'but was not found.'))

        self.windowsize = parameters['windowsize']
        self.dropoutfraction = 0.5
        self.sensorconfigts = get_sensor_config_timeseries(feature_names)
        self.sensorconfigaux = get_sensor_config(feature_names)

        parameters['sensorconfigts'] = self.sensorconfigts
        parameters['sensorconfigaux'] = self.sensorconfigaux

        super().__init__(modeltype='patch',
                         parameters=parameters,
                         requires_scaling=True,
                         feature_names=feature_names,
                         **kwargs)

        if model is None:
            self.model = self.convlstmmodel()
        else:
            self.model = model

        self.model.compile(optimizer=Adam(0.0002, 0.5), loss=DiceBCELoss,
                           weighted_metrics=['accuracy'],
                           sample_weight_mode="temporal")

    def convlstmmodel(self):
        '''
        This method outlines the main ConvLSTM model
        architecture
        '''

        # Get a local copy of this because we'll modify it
        sensorconfigts = get_sensor_config_timeseries(self.feature_names)
        sensorconfigaux = get_sensor_config(self.feature_names)

        defaultkernelsize = 4
        defaultstride = 2
        init = RandomNormal(mean=0.0, stddev=0.02)

        def conv2d(layer_input, filters, kernelsize, stride, init,
                   batchnormalization=True, l2_weight_regularizer=1e-5):
            c = Conv2D(filters, kernel_size=(kernelsize, kernelsize),
                       strides=stride, padding='same',
                       activation=None,
                       kernel_initializer=init,
                       kernel_regularizer=l2(
                l2_weight_regularizer))(layer_input)
            if batchnormalization:
                c = BatchNormalization()(c)
            c = LeakyReLU(alpha=0.2)(c)
            return c

        def deconv2d(layer_input, skip_input, filters, kernelsize, stride,
                     init, dropout, batchnormalization=True,
                     l2_weight_regularizer=1e-5):
            # d = Conv2DTranspose(filters, kernel_size=kernelsize,
            #                     strides=stride, padding='same',
            #                     activation=None,
            #                     kernel_initializer=init,
            #                     kernel_regularizer=l2(
            #                         l2_weight_regularizer))(layer_input)

            upsampled = tf.image.resize(layer_input,
                                        [2*layer_input.shape[1],
                                         2*layer_input.shape[2]])

            d = Conv2D(filters, kernel_size=(kernelsize, kernelsize),
                       strides=1, padding='same',
                       activation=None,
                       kernel_initializer=init,
                       kernel_regularizer=l2(
                l2_weight_regularizer))(upsampled)

            if batchnormalization:
                d = BatchNormalization()(d)
            if dropout:
                d = Dropout(dropout)(d)
            d = Activation('relu')(d)
            if skip_input:
                if type(skip_input) is not list:
                    skip_input = [skip_input]
                d = concatenate([d] + skip_input)
            return d

        def _time_encoder(enc_input):

            filters = 8**(int(enc_input.shape[3]/5) + 1)

            input_reshaped = Reshape((enc_input.shape[1],
                                      self.windowsize,
                                      self.windowsize,
                                      enc_input.shape[3]))(enc_input)

            enc1 = BatchNormalization()(ConvLSTM2D(
                filters=filters,
                kernel_size=1,
                padding='same',
                strides=(1, 1),
                return_sequences=True
            )(input_reshaped))
            enc_output = BatchNormalization()(ConvLSTM2D(
                filters=filters,
                kernel_size=1,
                padding='same',
                strides=(1, 1),
                return_sequences=False
            )(enc1))

            return enc_output

        def _time_encoder_conv3D(enc_input,
                                 l2_weight_regularizer=1e-5):

            filters = 64

            input_reshaped = Reshape((enc_input.shape[1],
                                      self.windowsize,
                                      self.windowsize,
                                      enc_input.shape[3]))(enc_input)

            # First layer no batch normalisation
            enc1 = LeakyReLU(alpha=0.2)(
                Conv3D(filters=filters,
                       kernel_size=(7, 1, 1),
                       strides=(5, 1, 1),
                       padding="same",
                       kernel_regularizer=l2(
                           l2_weight_regularizer))(input_reshaped))
            enc2 = LeakyReLU(alpha=0.2)(BatchNormalization()(
                Conv3D(filters=filters*2,
                       kernel_size=(5, 1, 1),
                       strides=(5, 1, 1),
                       padding="same",
                       kernel_regularizer=l2(
                           l2_weight_regularizer))(enc1)))
            enc3 = tf.keras.backend.squeeze(
                LeakyReLU(alpha=0.2)(BatchNormalization()(
                    Conv3D(filters=filters*2,
                           kernel_size=(5, 1, 1),
                           strides=(5, 1, 1),
                           padding="same",
                           kernel_regularizer=l2(
                               l2_weight_regularizer))(enc2))),
                axis=1)

            return enc3

        def _encoder(enc_input):

            filters = enc_input.shape[-1]

            enc1 = conv2d(enc_input,
                          filters=filters,
                          kernelsize=defaultkernelsize,
                          stride=defaultstride,
                          init=init)
            enc2 = conv2d(enc1,
                          filters=filters * 2,
                          kernelsize=defaultkernelsize,
                          stride=defaultstride,
                          init=init)
            enc3 = conv2d(enc2,
                          filters=filters * 2,
                          kernelsize=defaultkernelsize,
                          stride=defaultstride,
                          init=init)
            enc4 = conv2d(enc3,
                          filters=filters * 4,
                          kernelsize=defaultkernelsize,
                          stride=defaultstride,
                          init=init)

            # Bottleneck
            enc_output = conv2d(enc4,
                                filters=filters * 4,
                                kernelsize=defaultkernelsize,
                                stride=defaultstride,
                                init=init)

            return {'enc1': enc1, 'enc2': enc2, 'enc3': enc3,
                    'enc4': enc4, 'enc_output': enc_output}

        # --------------------------------------
        # Inputs
        # --------------------------------------

        # Inputs
        modelinput = Input(
            shape=(None, len(self.feature_names)),
            name='full_input'
        )

        # The tf.gather operator selects the
        # sensor-specific channels from the full input
        for sensor in sensorconfigts.keys():
            sensorinputs = []
            for sensorchannel in sensorconfigts[sensor]['channels']:
                sensorinputs.append(
                    tf.transpose(
                        tf.gather(
                            params=modelinput,
                            indices=sensorconfigts[sensor][
                                sensorchannel]['positions'],
                            axis=2
                        ),
                        perm=[0, 2, 1]  # Switch space and time dims
                    )
                )
            # Stack the different channels in a new channel dimension
            sensorconfigts[sensor]['input'] = tf.stack(
                sensorinputs, axis=3)

        for sensor in sensorconfigaux.keys():
            sensorconfigaux[sensor]['input'] = tf.gather(
                params=modelinput,
                indices=sensorconfigaux[sensor]['positions'],
                axis=2
            )

        # --------------------------------------
        # Temporal encoding step
        # --------------------------------------

        # Encoders
        for sensor in sensorconfigts.keys():
            sensorconfigts[sensor]['time_encoded'] = _time_encoder(
                sensorconfigts[sensor]['input']
            )

        # --------------------------------------
        # Spatial encoding step
        # --------------------------------------

        # Encoders
        for sensor in sensorconfigts.keys():
            sensorconfigts[sensor]['encoded'] = _encoder(
                sensorconfigts[sensor]['time_encoded']
            )

        for sensor in sensorconfigaux.keys():
            sensorconfigaux[sensor]['encoded'] = _encoder(
                Reshape(
                    (self.windowsize,
                     self.windowsize,
                     sensorconfigaux[sensor]['input'].shape[2]))(
                    sensorconfigaux[sensor]['input'])
            )

        # --------------------------------------
        # Concatenation of all encoded features
        # --------------------------------------
        encoded = [
            sensorconfigts[sensor]['encoded']['enc_output']
            for sensor in sensorconfigts.keys()
        ] + [
            sensorconfigaux[sensor]['encoded']['enc_output']
            for sensor in sensorconfigaux.keys()
        ]

        if len(encoded) > 1:
            concatenated = concatenate(encoded)
        else:
            concatenated = encoded[0]

        # --------------------------------------
        # Spatial decoding step
        # --------------------------------------

        dec4 = deconv2d(concatenated,
                        skip_input=[sensorconfigts[sensor]['encoded']['enc4']
                                    for sensor in sensorconfigts.keys()],
                        filters=256,
                        kernelsize=defaultkernelsize,
                        stride=defaultstride,
                        init=init, dropout=self.dropoutfraction)
        dec3 = deconv2d(dec4,
                        skip_input=[sensorconfigts[sensor]['encoded']['enc3']
                                    for sensor in sensorconfigts.keys()],
                        filters=256,
                        kernelsize=defaultkernelsize,
                        stride=defaultstride,
                        init=init, dropout=self.dropoutfraction)
        dec2 = deconv2d(dec3,
                        skip_input=[sensorconfigts[sensor]['encoded']['enc2']
                                    for sensor in sensorconfigts.keys()],
                        filters=128,
                        kernelsize=defaultkernelsize,
                        stride=defaultstride,
                        init=init, dropout=0)
        dec1 = deconv2d(dec2,
                        skip_input=[sensorconfigts[sensor]['encoded']['enc1']
                                    for sensor in sensorconfigts.keys()],
                        filters=64,
                        kernelsize=defaultkernelsize,
                        stride=defaultstride,
                        init=init, dropout=0)

        # OUTPUT LAYER
        upsampled = tf.image.resize(dec1,
                                    [2*dec1.shape[1],
                                        2*dec1.shape[2]])
        classification = Conv2D(
            filters=1,
            kernel_size=defaultkernelsize,
            activation='sigmoid',
            strides=1, padding='same',
            kernel_initializer=init)(upsampled)

        # Define final inputs and outputs of the model
        # and return the model
        output = Reshape(
            (self.windowsize * self.windowsize, 1))(classification)

        return Model(inputs=modelinput, outputs=output)

    def retrain(self, trainablelayers=20, **kwargs):
        if self.model is None:
            raise ValueError('No model loaded yet.')

        '''
        Retraining happens in two steps. First we freeze
        the entire base model and train a new classification
        head from scratch.

        Second - optionally - we unfreeze the base model
        and finetune a certain amount of pretrained layers
        of the base model as well.
        '''

        # First freeze the entire model
        logger.info('Freezing base model ...')
        for i in range(len(self.model.layers)):
            self.model.layers[i].trainable = False

        # Now reset the model weights of the classification head
        # and make them trainable
        logger.info('Resetting classification head ...')
        if 'conv2d' not in self.model.layers[-2].name:
            raise ValueError(('Classification head does not '
                              'seem to be classification '
                              f'layer (`{self.model.layers[-2].name}`)'))
        weight_initializer = self.model.layers[-2].kernel_initializer
        bias_initializer = self.model.layers[-2].bias_initializer
        old_weights, old_biases = self.model.layers[-2].get_weights()

        self.model.layers[-2].set_weights([
            weight_initializer(shape=old_weights.shape),
            bias_initializer(shape=old_biases.shape)])
        self.model.layers[-2].trainable = True

        # Recompile the model
        self.model.compile(optimizer=Adam(0.0002, 0.5), loss=DiceBCELoss,
                           weighted_metrics=['accuracy'],
                           sample_weight_mode="temporal")

        # Train the classification head
        self.train(**kwargs)

        # Now let's finetune the model
        # start by unfreezing desired nr of layers,
        # except for batchnormalization layers (!!!)
        for i in range(
                len(self.model.layers)-trainablelayers, len(
                    self.model.layers)):
            if 'batch_normalization' not in self.model.layers[i].name:
                self.model.layers[i].trainable = True

        # Recompile the model
        self.model.compile(optimizer=Adam(0.0002, 0.5), loss=DiceBCELoss,
                           weighted_metrics=['accuracy'],
                           sample_weight_mode="temporal")

        # Start finetuning
        logger.info(('Starting retraining on the last '
                     f'{trainablelayers} layers ...'))
        self.train(**kwargs)


class WorldCerealFFNN(WorldCerealKerasModel):
    def __init__(self,
                 model=None,
                 parameters: dict = {},
                 **kwargs
                 ):

        _req_params = ['depth', 'nodes']

        for param in _req_params:
            if param not in parameters:
                raise ValueError((f'Parameter `{param}` is '
                                  'compulsory for this model '
                                  'but was not found.'))

        super().__init__(modeltype='pixel',
                         parameters=parameters,
                         requires_scaling=True,
                         **kwargs)

        self.depth = parameters['depth']
        self.nodes = parameters['nodes']
        self.nrfeatures = len(self.feature_names)

        if model is None:
            self.model = self.FFNN()
        else:
            self.model = model

        self.model.compile(optimizer='adam', loss='binary_crossentropy',
                           weighted_metrics=['accuracy'])

    def FFNN(self):

        def _dense(nodes, layerinput, activation='relu'):
            return Dense(nodes,
                         activation=activation)(layerinput)

        inputlayer = Input(shape=(self.nrfeatures))
        x = inputlayer

        for i in range(self.depth):
            x = _dense(self.nodes, x)

        outputlayer = _dense(2, x, activation='sigmoid')

        return Model(inputs=inputlayer, outputs=outputlayer)


class WorldCerealPixelLSTM(WorldCerealKerasModel):
    def __init__(self,
                 model=None,
                 feature_names=None,
                 parameters: dict = {},
                 **kwargs
                 ):

        if feature_names is None:
            raise ValueError('`feature_names` cannot be None')

        _req_params = []

        for param in _req_params:
            if param not in parameters:
                raise ValueError((f'Parameter `{param}` is '
                                  'compulsory for this model '
                                  'but was not found.'))

        self.dropoutfraction = 0.2
        self.sensorconfigts = get_sensor_config_timeseries(feature_names)
        self.sensorconfigaux = get_sensor_config(feature_names)

        parameters['sensorconfigts'] = self.sensorconfigts
        parameters['sensorconfigaux'] = self.sensorconfigaux

        super().__init__(modeltype='pixel',
                         parameters=parameters,
                         requires_scaling=True,
                         feature_names=feature_names,
                         **kwargs)

        if model is None:
            self.model = self.lstmmodel()
        else:
            self.model = model

        self.model.compile(optimizer=Adam(0.0002, 0.5),
                           loss=['binary_crossentropy',
                                 'binary_crossentropy',
                                 'binary_crossentropy'],
                           weighted_metrics=['accuracy'])

    def DiceBCELoss(y_true, y_pred, smooth=1):

        BCE = binary_crossentropy(y_true, y_pred)
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        dice_loss = (1 - (2. * intersection + smooth) /
                     (K.sum(K.square(y_true), -1) +
                      K.sum(K.square(y_pred), -1) + smooth))
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

    def estimate_units(self, inputchannels):
        if inputchannels < 5:
            return 32
        if inputchannels < 12:
            return 64
        elif inputchannels < 25:
            return 128
        elif inputchannels < 50:
            return 256
        elif inputchannels < 100:
            return 512
        else:
            return 512

    def lstmmodel(self):
        '''
        This method outlines the main lstmmodel model
        architecture
        '''

        # Get a local copy of this because we'll modify it
        sensorconfigts = get_sensor_config_timeseries(self.feature_names)
        sensorconfigaux = get_sensor_config(self.feature_names)

        def _time_encoder(enc_input, sensor, units=None):

            units = units or self.estimate_units(enc_input.shape[2])

            # Masking slows down training terribly
            # and does not seem to impact the results
            # masked_input = Masking()(enc_input)

            regularizer = 0.005

            enc_1 = Dropout(self.dropoutfraction)(
                LSTM(
                    units,
                    return_sequences=True,
                    kernel_regularizer=l2(regularizer),
                    recurrent_regularizer=l2(regularizer),
                    bias_regularizer=l2(regularizer),
                    name=f'{sensor}_lstm_initial'
                )(enc_input))
            enc_output = Dropout(self.dropoutfraction)(
                LSTM(
                    units * 2,
                    return_sequences=False,
                    kernel_regularizer=l2(regularizer),
                    recurrent_regularizer=l2(regularizer),
                    bias_regularizer=l2(regularizer),
                    name=f'{sensor}_lstm_out'
                )(enc_1))

            return enc_output

        def _encoder(enc_input, sensor):

            # units = min(32, 8**(int(enc_input.shape[1]/5) + 1))

            # enc_output = Dropout(self.dropoutfraction)(
            #     Activation('relu')(
            #         BatchNormalization()(
            #             Dense(
            #                 units,
            #                 name=f'{sensor}_dense_initial'
            #             )(enc_input))))
            # enc_output = Dropout(self.dropoutfraction)(
            #     Activation('relu')(
            #         BatchNormalization()(
            #             Dense(
            #                 units*2,
            #                 name=f'{sensor}_dense_out'
            #             )(enc1))))

            enc_output = BatchNormalization()(enc_input)

            return enc_output

        # --------------------------------------
        # Inputs
        # --------------------------------------

        # Inputs
        modelinput = Input(
            shape=(len(self.feature_names),),
            name='full_input'
        )

        # The tf.gather operator selects the
        # sensor-specific channels from the full input
        for sensor in sensorconfigts.keys():
            sensorinputs = []
            for sensorchannel in sensorconfigts[sensor]['channels']:
                sensorinputs.append(
                    tf.gather(
                        params=modelinput,
                        indices=sensorconfigts[sensor][
                            sensorchannel]['positions'],
                        axis=1,
                    ),
                )
            # Stack the different channels in a new channel dimension
            sensorconfigts[sensor]['input'] = tf.stack(
                sensorinputs, axis=2, name=f'{sensor}_ts_input')

        for sensor in sensorconfigaux.keys():
            sensorconfigaux[sensor]['input'] = tf.gather(
                params=modelinput,
                indices=sensorconfigaux[sensor]['positions'],
                axis=1,
                name=f'{sensor}_aux_input'
            )

        # --------------------------------------
        # Temporal encoding step
        # --------------------------------------

        # Encoders
        for sensor in sensorconfigts.keys():
            sensorconfigts[sensor]['encoded'] = _time_encoder(
                sensorconfigts[sensor]['input'],
                sensor=sensor
            )

        # --------------------------------------
        # Non temporal encoding step
        # --------------------------------------
        for sensor in sensorconfigaux.keys():
            sensorconfigaux[sensor]['encoded'] = _encoder(
                sensorconfigaux[sensor]['input'],
                sensor=sensor
            )

        # --------------------------------------
        # Concatenation of all encoded features
        # --------------------------------------
        encoded = ([
            sensorconfigts[sensor]['encoded']
            for sensor in sensorconfigts.keys()
        ]
            + [
            sensorconfigaux[sensor]['encoded']
            for sensor in sensorconfigaux.keys()
        ])

        if len(encoded) > 1:
            concatenated = concatenate(encoded)
        else:
            concatenated = encoded[0]

        # --------------------------------------
        # Classification heads
        # --------------------------------------

        clf_outputs = []

        for sensor in sensorconfigts.keys():
            # Sensor-specific classification head
            sensor_specific = ([sensorconfigts[sensor]['encoded']]
                               +
                               [sensorconfigaux[sensor]['encoded']
                               for sensor in sensorconfigaux.keys()]
                               )

            if len(sensor_specific) > 1:
                sensor_concatenated = concatenate(sensor_specific)
            else:
                sensor_concatenated = sensor_specific[0]

            l2regularization = 1e-3

            # Final dense layer for classification
            clf1 = Dropout(self.dropoutfraction,
                           name=f'{sensor}_clf_dropout1_finetunable')(
                BatchNormalization()(
                    Dense(64, activation='relu',
                          name=f'{sensor}_clf_dense1_finetunable',
                          kernel_regularizer=l2(l2regularization),
                          bias_regularizer=l2(l2regularization))(
                        sensor_concatenated)))
            clf2 = Dropout(self.dropoutfraction,
                           name=f'{sensor}_clf_dropout2_finetunable')(
                BatchNormalization()(
                    Dense(32, activation='relu',
                          name=f'{sensor}_clf_dense2_finetunable',
                          kernel_regularizer=l2(l2regularization),
                          bias_regularizer=l2(l2regularization))(clf1)))
            output = Dense(1, activation='sigmoid',
                           name=f'{sensor}_out_finetunable',
                           kernel_regularizer=l2(l2regularization),
                           bias_regularizer=l2(l2regularization))(clf2)

            clf_outputs.append(output)

        # Combined classification head
        clf1 = Dropout(self.dropoutfraction,
                       name='combined_clf_dropout1_finetunable')(
            BatchNormalization()(
                Dense(64, activation='relu',
                      name='combined_clf_dense1_finetunable',
                      kernel_regularizer=l2(l2regularization),
                      bias_regularizer=l2(l2regularization))(concatenated)))
        clf2 = Dropout(self.dropoutfraction,
                       name='combined_clf_dropout2_finetunable')(
            BatchNormalization()(
                Dense(32, activation='relu',
                      name='combined_clf_dense2_finetunable',
                      kernel_regularizer=l2(l2regularization),
                      bias_regularizer=l2(l2regularization))(clf1)))
        output = Dense(1, activation='sigmoid',
                       name='comb_out_finetunable',
                       kernel_regularizer=l2(l2regularization),
                       bias_regularizer=l2(l2regularization))(clf2)
        clf_outputs.append(output)

        return Model(inputs=modelinput, outputs=clf_outputs)

    def retrain(self, learning_rate=None, **kwargs):
        if self.model is None:
            raise ValueError('No model loaded yet.')

        '''
        Retraining happens in two steps. First we freeze
        the entire base model and train a new classification
        head from scratch.

        Second - optionally - we unfreeze the base model
        and finetune the pretrained layers
        of the base model as well.
        '''

        # First freeze the entire model
        logger.info('Freezing base model ...')

        for i in range(len(self.model.layers)):
            self.model.layers[i].trainable = False

        # # Now reset the model weights of the classification head
        # # and make them trainable
        # logger.info('Resetting classification head ...')
        # for i in range(len(self.model.layers)-5, len(self.model.layers)):
        #     if 'dense' in self.model.layers[i].name:

        #         logger.info(f'Resetting: {self.model.layers[i].name}')
        #         weight_initializer = self.model.layers[i].kernel_initializer
        #         bias_initializer = self.model.layers[i].bias_initializer
        #         old_weights, old_biases = self.model.layers[i].get_weights()

        #         self.model.layers[i].set_weights([
        #             weight_initializer(shape=old_weights.shape),
        #             bias_initializer(shape=old_biases.shape)])
        #         self.model.layers[i].trainable = True

        # # Recompile the model
        # lr = learning_rate or 0.0002
        # self.model.compile(optimizer=Adam(lr, 0.5),
        #                    loss=DiceBCELoss,
        #                    weighted_metrics=['accuracy'])

        # # Train the classification head
        # self.train(**kwargs)

        # Now let's finetune the entire model
        # start by unfreezing all layers,
        # except for batchnormalization (!!!) and initial layers
        lr = learning_rate or 0.0002
        for i in range(len(self.model.layers)):
            if 'finetunable' in self.model.layers[i].name:
                logger.info((f'Making layer `{self.model.layers[i].name}`'
                             ' trainable'))
                self.model.layers[i].trainable = True

        # Recompile the model
        self.model.compile(optimizer=Adam(lr, 0.5),
                           loss=['binary_crossentropy',
                                 'binary_crossentropy',
                                 'binary_crossentropy'],
                           weighted_metrics=['accuracy'])

        # Start finetuning
        logger.info('Starting finetuning of all layers ...')
        self.train(**kwargs)

    def _predict(self, inputs, **kwargs):
        '''
        Overwrite default method for batchsize
        '''

        nrsamples = inputs.shape[0]
        if nrsamples > 1024:
            batch_size = 1024
        else:
            batch_size = nrsamples

        probabilities = self.model.predict(
            inputs, batch_size=batch_size, verbose=1)

        # We need to return probs for each class
        probabilities = np.array([1-probabilities,
                                  probabilities]).transpose()

        return probabilities


def DiceBCELoss(y_true, y_pred, smooth=1):

    BCE = binary_crossentropy(y_true, y_pred)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    dice_loss = (1 - (2. * intersection + smooth) /
                 (K.sum(K.square(y_true), -1) +
                 K.sum(K.square(y_pred), -1) + smooth))
    Dice_BCE = BCE + dice_loss

    return Dice_BCE
