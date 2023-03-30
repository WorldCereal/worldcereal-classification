import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
from catboost import Pool
from satio.utils.logs import proclogs

from worldcereal.utils.spark import get_spark_context
from worldcereal.utils.training import get_pixel_data
from worldcereal.classification.weights import get_refid_weight
from worldcereal.classification.models import WorldCerealCatBoostModel
from worldcereal.train import get_training_settings


MODELVERSION = '750'
BUFFER = 0  # Buffer (m) around Realms


class Trainer:

    def __init__(self, settings, basedir, detector, modeltag):

        self.settings = settings
        self.bands = settings['bands']
        self.outputlabel = settings['outputlabel']
        self.detector = detector
        self.minsamples = settings.get('minsamples', 250)
        self.basedir = Path(basedir)
        self.proclogs_dir = Path(basedir) / 'proclogs'
        self.modeltag = modeltag

        self.gpu = False

        Path(basedir).mkdir(parents=True, exist_ok=True)

        # Input parameters
        cal_df_files = settings.get('cal_df_files')
        val_df_files = settings.get('val_df_files')
        test_df_files = settings.get('test_df_files')

        logger.info('-' * 50)
        logger.info('Initializing CatBoost trainer ...')
        logger.info('-' * 50)
        logger.info(f'CAL files: {cal_df_files}')
        logger.info(f'VAL files: {val_df_files}')
        logger.info(f'TEST files: {test_df_files}')

        # Load a test dataframe to derive some information
        test_df = self._load_df(test_df_files[0])

        # Get the list of features present in the training data
        self.present_features = Trainer._get_training_df_features(test_df)

        # Check weights completeness
        self._check_weights(test_df)

    def _train(self, outputdir, modelname, realm_id=None, **kwargs):
        '''Function to train a model
        '''

        # Set the output directory
        if Path(outputdir).is_dir():
            logger.info((f'Model directory `{outputdir}` '
                         'already exists -> skipping.'))
            return

        # Check if all required features are present
        self._check_features()

        # Get the categorical features
        cat_features = self._get_categorical_features()

        # Setup the model
        model = self._setup_model(outputdir)

        # Add some specific parameters and settings to config file
        model.config['parameters']['realm_id'] = realm_id
        model.config['trainingsettings'] = dict()
        for key, value in self.settings.items():
            if key not in ['bands']:
                model.config['trainingsettings'][key] = value
        model.config['trainingsettings'].update(kwargs)
        model.save_config()

        # in case of IRR: set global fraction of irr
        if self.outputlabel == 'IRR':
            irr_ratio = 0.2
        else:
            irr_ratio = None

        # Get and check trainingdata
        cal_data, val_data, test_data = self._get_trainingdata(
            outputdir, detector=self.detector, irr_ratio=irr_ratio,
            realm_id=realm_id, buffer=BUFFER, **kwargs)
        Trainer._check_trainingdata(cal_data, val_data, outputdir)

        # Save processed data to disk for debugging
        logger.info('Saving processed data ...')
        cal_data[0].to_csv(Path(outputdir) / 'cal_inputs.csv')
        np.save(Path(outputdir) / 'cal_outputs', cal_data[1])
        np.save(Path(outputdir) / 'cal_weights', cal_data[2])
        np.save(Path(outputdir) / 'cal_locationids', cal_data[3])

        # Setup datapools for training
        calibration_data, eval_data = Trainer._setup_datapools(
            cal_data, val_data, cat_features)

        # Store the ref_id counts in the model config
        model.config['training_refids'] = cal_data[5]
        model.save_config()

        logger.info('Starting training ...')
        model.train(
            inputs=calibration_data,
            eval_set=eval_data,
            verbose=50
        )

        # Save the model
        Trainer.save_model(model, outputdir, modelname)

        # Test the model
        Trainer.evaluate(model, test_data, outputdir)

        # Plot feature importances
        Trainer._plot_feature_importance(model, outputdir)

        logger.success(f'Model `{modelname}` trained!')

    def train_realms(self, realms,
                     outlierfraction=0,
                     sc=None):
        '''Realm-specific training
        '''

        logger.info(f'Starting training on {len(realms)} realms ...')

        minsamples = self.minsamples

        @proclogs(self.proclogs_dir, level='DEBUG')
        def _log_train_realm(processing_tuple):
            _train_realm(processing_tuple[1])

        def _train_realm(realm_id):
            logger.info(f'---WORKING ON REALM {realm_id}---')

            # Set the output directory
            outputdir = (Path(self.basedir) /
                         (f'Realm_{realm_id}' + self.modeltag))
            modelname = (f'WorldCerealPixelCatBoost_{self.detector}_'
                         f'Realm_{realm_id}')

            # Train the model for this realm
            return self._train(outputdir,
                               modelname,
                               realm_id=realm_id,
                               minsamples=minsamples,
                               outlierfraction=outlierfraction)

        if sc is None:
            # Working in serial
            for realm_id in realms:
                _log_train_realm((f'Realm_{realm_id}', realm_id))
        else:
            # Working in parallel
            logger.info('Training on executors ...')
            processing_tuples = [(f'Realm_{realm_id}', realm_id) for realm_id in realms]
            sc.parallelize(processing_tuples,
                           len(processing_tuples)).foreach(_log_train_realm)

        logger.success('Realm models trained!')

    def _load_df(self, file):
        df = pd.read_parquet(
            Path(file) / f'training_df_{self.outputlabel}.parquet')

        return df

    @staticmethod
    def evaluate(model, testdata, outdir, pattern=''):
        testinputs = testdata[0]
        testoutputs = testdata[1]
        testweights = testdata[2]
        testorigoutputs = testdata[4]

        logger.info('Getting test results ...')

        # In test mode, all valid samples are equal
        idxvalid = np.where(testweights > 0)[0]
        testinputs = testinputs.iloc[idxvalid, :]
        testoutputs = testoutputs[idxvalid]
        testorigoutputs = testorigoutputs[idxvalid]

        # Run evaluation
        metrics = model.evaluate(testinputs, testoutputs,
                                 original_labels=testorigoutputs,
                                 outdir=outdir, pattern=pattern)

        return metrics

    @staticmethod
    def save_model(model, outputdir, modelname):
        model.save(Path(outputdir) / (modelname + '.cbm'))

    @staticmethod
    def _get_training_df_features(df):

        present_features = df.columns.tolist()

        return present_features

    def _setup_model(self, outputdir):
        # Setup the model
        logger.info('Setting up model ...')
        worldcerealcatboost = WorldCerealCatBoostModel(
            feature_names=self.bands,
            gpu=self.gpu,
            basedir=outputdir,
            overwrite=False)

        # Print a summary of the model
        worldcerealcatboost.summary()

        return worldcerealcatboost

    def _get_trainingdata(self, outputdir, minsamples=500,
                          **kwargs):

        # set outlierinputs settings
        if self.settings['remove_outliers']:
            outlierinputs = self.settings['outlierinputs']
        else:
            outlierinputs = None

        # Get the data
        cal_data, val_data, test_data = get_pixel_data(
            self.detector, self.settings, self.bands, logdir=outputdir,
            outlierinputs=outlierinputs, scale_features=False,
            impute_missing=False, minsamples=minsamples,
            return_pandas=True,
            **kwargs)

        return cal_data, val_data, test_data

    @staticmethod
    def _setup_datapools(cal_data, val_data, cat_features):

        # Setup dataset Pool
        calibration_data = Pool(
            data=cal_data[0],
            label=cal_data[1],
            weight=cal_data[2],
            cat_features=cat_features
        )
        eval_data = Pool(
            data=val_data[0],
            label=val_data[1],
            weight=val_data[2],
            cat_features=cat_features
        )

        return calibration_data, eval_data

    @staticmethod
    def _check_trainingdata(cal_data, val_data, outputdir):

        # Run some checks
        plt.hist(val_data[0].values.ravel(), 100)
        plt.savefig(Path(outputdir) / ('inputdist_val.png'))
        plt.close()
        plt.hist(cal_data[0].values.ravel(), 100)
        plt.savefig(Path(outputdir) / ('inputdist_cal.png'))
        plt.close()
        logger.info(f'Unique CAL outputs: {np.unique(cal_data[1])}')
        logger.info(f'Unique VAL outputs: {np.unique(val_data[1])}')
        logger.info(f'Unique CAL weights: {np.unique(cal_data[2])}')
        logger.info(f'Unique VAL weights: {np.unique(val_data[2])}')
        logger.info(f'Mean Pos. weight: '
                    f'{np.mean(cal_data[2][cal_data[1] == 1])}')
        logger.info(f'Mean Neg. weight: '
                    f'{np.mean(cal_data[2][cal_data[1] == 0])}')
        ratio_pos = np.sum(cal_data[1] == 1) / cal_data[1].size
        logger.info(f'Ratio pos/neg outputs: {ratio_pos}')

    def _check_weights(self, df):

        # Check ref_id weights completeness
        ref_ids = df['ref_id'].unique().astype(str)
        ref_id_weights = [get_refid_weight(ref_id, self.outputlabel)
                          for ref_id in ref_ids]
        nr_unknown = (np.array(ref_id_weights) == 90).sum()
        logger.info((f'{nr_unknown}/{len(ref_id_weights)} ref_ids '
                     'got a default weight assigned!'))

    def _check_features(self):

        present_features = [ft for ft in self.present_features]

        for band in self.bands:
            if band not in present_features:
                raise RuntimeError(
                    f'Feature `{band}` not found in features.')

    def _get_categorical_features(self):

        cat_features = []
        # for band in self.bands:
        #     if not is_real_feature(band):
        #         if band not in ['lat', 'lon']:
        #             cat_features.append(band)

        logger.info((f'Will train on {len(self.bands) - len(cat_features)} '
                     'normal features ...'))
        logger.info(
            f'Will train on {len(cat_features)} categorical features ...')

        return cat_features

    @staticmethod
    def _plot_feature_importance(model, outputdir):

        # Save feature importance plot
        logger.info('Plotting feature importance ...')
        ft_imp = model.model.get_feature_importance()
        sorting = np.argsort(np.array(ft_imp))[::-1]

        f, ax = plt.subplots(1, 1, figsize=(20, 8))
        ax.bar(np.array(model.feature_names)[
            sorting], np.array(ft_imp)[sorting])
        ax.set_xticklabels(np.array(
            model.feature_names)[sorting], rotation=90)
        plt.tight_layout()
        plt.savefig(str(Path(outputdir) / 'feature_importance.png'))

    @staticmethod
    def _grid_search(model, calibration_data):
        # Perform a grid search for best parameters
        grid = {'depth': [4, 8],
                'l2_leaf_reg': [3, 5]
                }
        results = model.grid_search(grid, calibration_data)
        logger.info(f'Grid search best parameters: {results["params"]}')

        # Adjust parameters of main model with the results
        model.model.set_params(**results['params'])

        return model


def main(detector, trainingsettings, outdir_base, sc=None):

    # Plot without display
    plt.switch_backend('Agg')

    # Realms for which to train models
    realms = [2, 3, 4, 5, 6, 7]

    for modeltype in ['hybrid', 'OPTICAL']:

        logger.info(f'Working on model type: {modeltype}')

        modeltag = ''

        if modeltype == 'OPTICAL':
            outdir_base = outdir_base + '-OPTICAL'
            modeltag = '-OPTICAL'

            trainingsettings['bands'] = [
                b for b in trainingsettings['bands']
                if 'SAR' not in b]

        logger.info(f'Training on bands: {trainingsettings["bands"]}')

        # Initialize trainer
        trainer = Trainer(trainingsettings, outdir_base, detector, modeltag)

        # Now train realm models; remove 1% outliers
        trainer.train_realms(realms, outlierfraction=0.01,
                             sc=sc)

        logger.success('Models trained!')


if __name__ == '__main__':

    spark = False

    if spark:
        logger.info('Setting up spark ...')
        sc = get_spark_context()
    else:
        sc = None

    # Suppress debug messages
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Specify the detector(s) to train
    detectors = [
        'cropland'
    ]

    for detector in detectors:

        # Get the trainingsettings
        trainingsettings = get_training_settings(detector)

        outdir = ('/data/worldcereal/models/'
                  f'WorldCerealPixelCatBoost/{detector}_detector_'
                  f'WorldCerealPixelCatBoost'
                  f'_v{MODELVERSION}')

        main(detector, trainingsettings, outdir, sc=sc)
