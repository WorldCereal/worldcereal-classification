import shutil
import sys
from pathlib import Path
import copy
import threading

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import Pool
from loguru import logger
from satio.utils.logs import proclogs

from worldcereal.utils.spark import get_spark_context
from worldcereal.utils.training import (get_pixel_data,
                                        NotEnoughSamplesError)
from worldcereal.classification.weights import get_refid_weight
from worldcereal.utils import aez
from worldcereal.classification.models import WorldCerealCatBoostModel
from worldcereal.train import get_training_settings


MODELVERSION = '751'
BUFFER = 0  # Buffer (m) around AEZs


def thread_id():
    return threading.current_thread().ident


class Trainer:

    def __init__(self, settings, basemodeldir, detector):

        self.settings = settings
        self.bands = settings['bands']
        self.outputlabel = settings['outputlabel']
        self.basemodeldir = basemodeldir
        self.proclogs_dir = Path(basemodeldir) / 'proclogs'
        self.basemodel = None
        self.detector = detector
        self.minsamples = settings.get('minsamples',
                                       [1000, 1000, 1000])

        if type(self.minsamples) == int:
            self.minsamples = [self.minsamples, self.minsamples,
                               self.minsamples]

        # gpu = True if spark else False
        self.gpu = False

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

    def _train(self, **kwargs):
        '''Function to train the base model
        '''

        # Set the output directory
        outputdir = self.basemodeldir
        if Path(outputdir).is_dir():
            logger.info((f'Model directory `{outputdir}` '
                         'already exists -> skipping.'))
            return

        # Check if all required features are present
        self._check_features()

        # Get the categorical features
        cat_features = self._get_categorical_features()

        # Setup the basemodel
        self.basemodel = self._setup_model(outputdir)

        # Add some specific parameters and settings to config file
        self.basemodel.config['trainingsettings'] = dict()
        for key, value in self.settings.items():
            if key not in ['bands', 'pos_neg_ratio_override']:
                self.basemodel.config['trainingsettings'][key] = value
        self.basemodel.config['trainingsettings'].update(kwargs)
        self.basemodel.save_config()

        # in case of IRR: set global fraction of irr
        if self.outputlabel == 'IRR':
            irr_ratio = 0.2
        else:
            irr_ratio = None

        # Get and check trainingdata
        cal_data, val_data, test_data = self._get_trainingdata(
            outputdir, detector=self.detector, irr_ratio=irr_ratio,
            settings=self.settings, **kwargs)
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
        self.basemodel.config['training_refids'] = cal_data[5]
        self.basemodel.save_config()

        logger.info('Starting training ...')
        self.basemodel.train(
            inputs=calibration_data,
            eval_set=eval_data,
            verbose=50
        )

        # Save the model
        modelname = f'WorldCerealPixelCatBoost_{self.detector}'
        Trainer.save_model(self.basemodel, outputdir, modelname)

        # Test the model
        Trainer.evaluate(self.basemodel, test_data, outputdir)

        # Plot feature importances
        Trainer._plot_feature_importance(self.basemodel, outputdir)

        logger.success('Base model trained!')

    def _retrain(self, parentmodeldir, outputdir, modelname,
                 aez_zone=None, aez_group=None, minsamples=500,
                 learning_rate=0.05, iterations=8000,
                 train_from_scratch=False, **kwargs):
        '''Method to finetune a model based on a
        parentmodeldir to continue from.
        '''

        settings = copy.deepcopy(self.settings)

        if aez_zone is None and aez_group is None:
            raise ValueError(
                'Both `aez_zone` and `aez_group` cannot be None.')
        key = aez_zone if aez_zone is not None else aez_group

        if Path(outputdir).is_dir():
            # Check if model is functional
            try:
                _ = WorldCerealCatBoostModel.from_config(
                    Path(outputdir) / 'config.json')
                logger.info(f'Model `{modelname}` already trained!')
                return Trainer.load_log_df(outputdir)
            except Exception:
                logger.warning(f'Model `{modelname}` does not work: removing')
                shutil.rmtree(outputdir)

        # Get parent model
        parentmodelconfig = Path(parentmodeldir) / 'config.json'
        model = WorldCerealCatBoostModel.from_config(parentmodelconfig)

        # Transfer the base model
        transferredmodel = model.transfer(outputdir,
                                          learning_rate=learning_rate,
                                          iterations=iterations)

        # Write settings to config
        if aez_zone is not None:
            transferredmodel.config['parameters']['aez_zone'] = int(aez_zone)
        if aez_group is not None:
            transferredmodel.config['parameters']['aez_group'] = int(aez_group)

        # Get pos_neg_ratio and apply override if required
        if 'pos_neg_ratio' in settings.keys():
            localization_id = aez_zone if aez_zone is not None else aez_group
            pos_neg_ratio_override = settings.get('pos_neg_ratio_override', None)
            if (pos_neg_ratio_override is not None) and (localization_id is not None):
                for newratio, groups in pos_neg_ratio_override.items():
                    if localization_id in groups:
                        settings['pos_neg_ratio'] = newratio

        transferredmodel.config['trainingsettings'] = dict()
        for key, value in settings.items():
            if key not in ['bands', 'pos_neg_ratio_override']:
                transferredmodel.config['trainingsettings'][key] = value
        transferredmodel.config['trainingsettings'].update(kwargs)
        transferredmodel.config['trainingsettings']['buffer'] = BUFFER
        transferredmodel.config['trainingsettings']['train_from_scratch'] = train_from_scratch  # NOQA
        transferredmodel.save_config()

        # in case of IRR, get irr ratio for specific AEZ
        irr_ratio = None
        if self.outputlabel == 'IRR':
            if aez_zone is not None:
                irr_ratio = aez.load().set_index('zoneID').loc[
                    aez_zone]['irr_stats'] / 100
                if irr_ratio < 0.01:
                    irr_ratio = 0.01

        try:
            cal_data, val_data, test_data = self._get_trainingdata(
                outputdir, buffer=BUFFER, aez_zone=aez_zone,
                aez_group=aez_group, minsamples=minsamples,
                detector=self.detector, settings=settings,
                **kwargs)
        except NotEnoughSamplesError:
            logger.warning('Not enough samples to train model!')
            shutil.rmtree(outputdir)
            return None

        # Run some checks
        Trainer._check_trainingdata(cal_data, val_data, outputdir)

        # Test the parent model
        parentmetrics = Trainer.evaluate(model, test_data,
                                         outputdir, 'Parentmodel_')

        # Get categorical features
        cat_features = self._get_categorical_features()

        # Setup datapools for training
        calibration_data, eval_data = Trainer._setup_datapools(
            cal_data, val_data, cat_features)

        # Print a summary of the model
        transferredmodel.summary()

        # Store the ref_id counts in the model config
        transferredmodel.config['training_refids'] = cal_data[5]
        transferredmodel.save_config()

        # Do the training
        logger.info('Starting training ...')
        if not train_from_scratch:
            transferredmodel.retrain(
                init_model=model.model,
                inputs=calibration_data,
                eval_set=eval_data,
                verbose=50
            )
        else:
            logger.warning('Training from scratch!')
            transferredmodel.train(
                inputs=calibration_data,
                eval_set=eval_data,
                verbose=50
            )

        # Save the model
        Trainer.save_model(transferredmodel, outputdir, modelname)

        # Test the retrained model
        metrics = Trainer.evaluate(transferredmodel, test_data, outputdir,
                                   'Finetunedmodel_')

        # Plot feature importances
        Trainer._plot_feature_importance(transferredmodel, outputdir)

        # Write results to a DF logfile
        Trainer.write_log_df(metrics, key, modelname, cal_data,
                             outputdir, parentmetrics=parentmetrics)

        logger.success(f'Model `{modelname}` retrained!')

        return Trainer.load_log_df(outputdir)

    def train_base(self, outlierfraction=0, **kwargs):

        # get min samples for base model from settings
        minsamples = self.minsamples[0]
        # train model
        self._train(minsamples=minsamples,
                    outlierfraction=outlierfraction,
                    **kwargs)

    def train(self, **kwargs):
        self.train_base(**kwargs)

    def train_groups(self, groups,
                     outlierfraction=0,
                     train_from_scratch=False,
                     sc=None):
        '''Group-specific finetuning
        '''

        logger.info(f'Starting finetuning on {len(groups)} groups ...')

        # get min samples for group models from settings
        minsamples = self.minsamples[1]

        def _train_group(aez_groupid):
            logger.info(f'---WORKING ON AEZ GROUP {aez_groupid}---')

            logfolder = Path(self.proclogs_dir)
            logfolder.mkdir(parents=True, exist_ok=True)

            log = logfolder / f'Group_{aez_groupid}.log'
            sink = logger.add(
                log, level='DEBUG',
                filter=lambda record: record["thread"].id == thread_id())

            # Set the output directory
            outputdir = (Path(self.basemodeldir) /
                         (f'Group_{aez_groupid}'))
            modelname = (f'WorldCerealPixelCatBoost_{self.detector}_retrained_'
                         f'Group_{aez_groupid}')

            # Retrain the model for this group
            result = self._retrain(self.basemodeldir, outputdir,
                                   modelname, aez_group=aez_groupid,
                                   minsamples=minsamples,
                                   outlierfraction=outlierfraction,
                                   learning_rate=0.05, iterations=4000,
                                   train_from_scratch=train_from_scratch)

            logger.remove(sink)

            return result

        if sc is None:
            # Working in serial
            logs = []
            for aez_groupid in groups:
                log = _train_group(aez_groupid)
                if log is not None:
                    logs.append(log)
        else:
            # Working in parallel
            logger.info('Training on executors ...')
            logs = sc.parallelize(groups, len(groups)).map(
                _train_group).filter(lambda x: x is not None).collect()

        # Combine the results in one DF and write to folder
        outfile = Path(self.basemodeldir) / 'aez_group_models.csv'
        all_logs = pd.concat(logs, axis=0)
        all_logs.to_csv(outfile)
        logger.success(
            f'Group models trained, log written to: {outfile}')

    def train_zones(self, zones,
                    outlierfraction=0.05,
                    train_from_scratch=False, sc=None):
        '''Zone-specific finetuning
        '''

        logger.info(f'Starting finetuning on {len(zones)} zones ...')

        # get min samples for zone models from settings
        minsamples = self.minsamples[2]

        @proclogs(self.proclogs_dir, level='DEBUG')
        def _log_train_zone(processing_tuple):
            return _train_zone(processing_tuple[1])

        def _train_zone(aez_zoneid):
            logger.info(f'---WORKING ON AEZ ZONE {aez_zoneid}---')

            logfolder = Path(self.proclogs_dir)
            logfolder.mkdir(parents=True, exist_ok=True)

            log = logfolder / f'Zone_{aez_zoneid}.log'
            sink = logger.add(
                log, level='DEBUG',
                filter=lambda record: record["thread"].id == thread_id())

            # Find corresponding group and its model dir
            group_id = aez.group_from_id(aez_zoneid)
            parentdir = (Path(self.basemodeldir) /
                         (f'Group_{group_id}'))

            if not Path(parentdir).is_dir():
                logger.warning((f'AEZ zone {aez_zoneid} has no trained '
                                'group model -> starting from global model.'))
                parentdir = self.basemodeldir

            # Set the output directory
            outputdir = (Path(self.basemodeldir) /
                         (f'Zone_{aez_zoneid}'))
            modelname = (f'WorldCerealPixelCatBoost_{self.detector}_retrained_'
                         f'Zone_{aez_zoneid}')

            # Retrain the model for this zone
            result = self._retrain(parentdir, outputdir,
                                   modelname, aez_zone=aez_zoneid,
                                   minsamples=minsamples, learning_rate=0.05,
                                   outlierfraction=outlierfraction,
                                   iterations=4000,
                                   train_from_scratch=train_from_scratch)

            logger.remove(sink)

            return result

        if sc is None:
            # Working in serial
            logs = []
            for aez_zoneid in zones:
                log = _log_train_zone((f'Zone_{aez_zoneid}', aez_zoneid))
                if log is not None:
                    logs.append(log)
        else:
            # Working in parallel
            logger.info('Training on executors ...')
            processing_tuples = [(f'Zone_{aez_zoneid}', aez_zoneid) for aez_zoneid in zones]
            logs = sc.parallelize(processing_tuples,
                                  len(processing_tuples)).foreach(
                _log_train_zone).filter(lambda x: x is not None).collect()

        # Combine the results in one DF and write to folder
        outfile = Path(self.basemodeldir) / 'aez_zone_models.csv'
        all_logs = pd.concat(logs, axis=0)
        all_logs.to_csv(outfile)
        logger.success(
            f'Zone models trained, log written to: {outfile}')

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
                          settings=None, **kwargs):

        settings = self.settings if settings is None else settings

        # set outlierinputs settings
        if self.settings.get('remove_outliers', False):
            outlierinputs = self.settings['outlierinputs']
        else:
            outlierinputs = None

        # Get the data
        cal_data, val_data, test_data = get_pixel_data(
            self.detector, settings, self.bands, logdir=outputdir,
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

    @staticmethod
    def write_log_df(metrics, aez, modelname, cal_data,
                     outputdir, parentmetrics=None):

        outfile = Path(outputdir) / 'log_df.csv'

        nr_cal_samples = cal_data[0].shape[0]
        logdata = {
            'model': [modelname],
            'aez': [aez],
            'cal_samples': [nr_cal_samples],
            'OA': [metrics['OA']],
            'OA_parent': [np.nan],
            'F1': [metrics['F1']],
            'F1_parent': [np.nan],
            'Precision': [metrics['Precision']],
            'Precision_parent': [np.nan],
            'Recall': [metrics['Recall']],
            'Recall_parent': [np.nan]
        }

        if parentmetrics is not None:
            logdata['OA_parent'] = [parentmetrics['OA']]
            logdata['F1_parent'] = [parentmetrics['F1']]
            logdata['Precision_parent'] = [parentmetrics['Precision']]
            logdata['Recall_parent'] = [parentmetrics['Recall']]

        log_df = pd.DataFrame.from_dict(logdata).set_index('model')
        log_df.to_csv(outfile)

    @staticmethod
    def load_log_df(outputdir):

        outfile = Path(outputdir) / 'log_df.csv'
        if not outfile.is_file():
            raise FileNotFoundError(f'Logfile `{outfile}` not found.')

        log_df = pd.read_csv(outfile, index_col=0)

        return log_df


def main(detector, trainingsettings, outdir_base, sc=None):

    # Plot without display
    plt.switch_backend('Agg')

    for modeltype in ['hybrid', 'OPTICAL']:

        logger.info(f'Working on model type: {modeltype}')

        if modeltype == 'OPTICAL':
            outdir_base = outdir_base + '-OPTICAL'

            trainingsettings['bands'] = [
                b for b in trainingsettings['bands']
                if 'SAR' not in b]

        logger.info(f'Training on bands: {trainingsettings["bands"]}')

        # Get path to parent model
        basemodeldir = Path(outdir_base)

        # Initialize trainer
        trainer = Trainer(trainingsettings, basemodeldir, detector)

        # First train the base model; remove 1% outliers
        # trainer.train_base(outlierfraction=0.01, realm_id=3)
        trainer.train_base(outlierfraction=0.01)

        if trainingsettings.get('train_group', False):
            # Now train AEZ groups; remove 1% outliers
            aez_groups = list(aez.load().groupID.unique().astype(int))
            trainer.train_groups(aez_groups, outlierfraction=0.01,
                                 sc=sc, train_from_scratch=False)

        if trainingsettings.get('train_zone', False):
            # Finally train AEZ zones; remove 1% outliers per class
            aez_zones = list(aez.load().zoneID.unique().astype(int))
            trainer.train_zones(aez_zones, outlierfraction=0.01,
                                sc=sc, train_from_scratch=False)

        logger.success('Models trained!')


if __name__ == '__main__':

    spark = True

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
        'maize',
        'wintercereals',
        'springcereals',
        'irrigation'
    ]

    for detector in detectors:

        # Get the trainingsettings
        trainingsettings = get_training_settings(detector)

        # Output parameters
        outdir = ('/data/worldcereal/models/'
                  f'WorldCerealPixelCatBoost/{detector}_detector_'
                  f'WorldCerealPixelCatBoost'
                  f'_v{MODELVERSION}')

        main(detector, trainingsettings, outdir, sc=sc)
