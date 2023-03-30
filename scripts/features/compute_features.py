import geopandas as gpd
from pathlib import Path
from loguru import logger
import pandas as pd

from satio.collections import (L2ATrainingCollection,
                               PatchLabelsTrainingCollection,
                               SIGMA0TrainingCollection,
                               AgERA5TrainingCollection,
                               DEMCollection,
                               WorldCoverCollection)
from worldcereal.collections import L8ThermalTrainingCollection

from worldcereal.processors import PixelTrainingChain
from worldcereal.utils.spark import get_spark_context
from worldcereal.utils import aez
from worldcereal import (TBASE, GDDTLIMIT)
from satio.features import percentile_iqr

from worldcereal.features.settings import (get_cropland_catboost_parameters,
                                           get_cropland_tsteps_parameters,
                                           get_croptype_tsteps_parameters,
                                           get_croptype_catboost_parameters,
                                           get_irr_parameters)


class FeatureComputer:

    def __init__(self,
                 cib_rootdir, experiment_id,
                 outdir, season, parameters,
                 outputlabel='LC', augment=False,
                 append=False):

        self.cib_rootdir = cib_rootdir
        self.experiment_id = experiment_id
        self.outdir = outdir
        self.season = season
        self.augment = augment
        self.parameters = parameters
        self.aez_layer = aez.load()
        self.outputlabel = outputlabel
        self.append = append
        self.database = None

    def load_database(self, **kwargs):
        logger.info('Opening database ...')
        database = gpd.read_file(
            Path(self.cib_rootdir) / self.experiment_id / 'database.json')
        logger.info(f'Database opened with {len(database)} samples!')

        # Filter database
        database = self._filter_database(database, **kwargs)

        # Convert into SATIO format
        df = database[['location_id', 'tile', 'epsg', 'path', 'split',
                       'ref_id', 'year', 'start_date', 'end_date', 'bounds']]

        logger.info(f'Retained {len(df)} samples after filtering')

        self.database = database
        self.df = df

    def create_collections(self):

        collections = {}

        # Collections we always need
        collections['LABELS'] = PatchLabelsTrainingCollection(
            self.df, dataformat='worldcereal')
        collections['OPTICAL'] = L2ATrainingCollection(
            self.df, dataformat='worldcereal')
        collections['SAR'] = SIGMA0TrainingCollection(
            self.df, dataformat='worldcereal')
        collections['METEO'] = AgERA5TrainingCollection(
            self.df, dataformat='worldcereal')
        collections['DEM'] = DEMCollection(
            folder='/data/MEP/DEM/COP-DEM_GLO-30_DTED/S2grid_20m')
        collections['WorldCover'] = WorldCoverCollection(
            folder='/data/worldcereal/auxdata/WORLDCOVER/2020/')

        # Variable collections: check settings
        if 'TIR' in self.parameters['settings'].keys():
            collections['TIR'] = L8ThermalTrainingCollection(
                self.df, dataformat='worldcereal')

        self.collections = collections

    def build_trainingchain(self, gddnormalization=False):

        settings = self.parameters['settings']
        rsi_meta = self.parameters['rsi_meta']
        features_meta = self.parameters['features_meta']
        ignore_def_feat = self.parameters['ignore_def_feat']
        season_prior_buffer = self.parameters.get('season_prior_buffer', None)
        season_post_buffer = self.parameters.get('season_post_buffer', None)

        # adding some features for outlier detection
        # (not used for inference)
        settings['OPTICAL']['rsis'].extend(['ndvi', 'rgbBR'])
        settings['OPTICAL']['rsis'] = list(
            dict.fromkeys(settings['OPTICAL']['rsis']))
        if 'percentile_iqr' not in features_meta['OPTICAL'].keys():
            features_meta['OPTICAL']['percentile_iqr'] = {
                "function": percentile_iqr,
                "parameters": {
                    'q': [10, 50, 90],
                    'iqr': [25, 75]
                },
                "names": ['p10', 'p50', 'p90', 'iqr']
            }

        # -----------------------------------------------------------
        # Now the important part
        # specify for all sensors that we want GDD
        # normalization with season-specific tbase if it's required
        if gddnormalization:
            for sensor in settings.keys():
                settings[sensor]['normalize_gdd'] = dict(
                    tbase=TBASE[season],
                    tlimit=GDDTLIMIT[season],
                    season=season
                )
        # -----------------------------------------------------------

        # Setup training chain
        logger.info('Setting up trainingchain ...')
        self.trainingchain = PixelTrainingChain(
            basedir=outdir,
            collections_dict=self.collections,
            trainingdb=self.database,
            settings=settings,
            rsi_meta=rsi_meta,
            features_meta=features_meta,
            ignore_def_feat=ignore_def_feat,
            location_ids=self.database['location_id'].tolist(),
            season=self.season,
            prior_buffer=season_prior_buffer,
            post_buffer=season_post_buffer,
            aez=self.aez_layer)

    def compute_features(self, sparkcontext=None,
                         max_pixels=1, seed=1234,
                         filter_function=None, debug=False):

        logger.info('Getting training dataframe ...')
        training_df = self.trainingchain.get_training_df(
            sparkcontext=sparkcontext,
            label=self.outputlabel,
            save=False,
            max_pixels=max_pixels,
            seed=seed,
            augment=self.augment,
            format='parquet',
            filter_function=filter_function,
            debug=debug)

        if self.season == 'winter':
            if self.outputlabel == 'CT':
                # Correct winter cereal samples
                training_df = self._correct_wintercereals(training_df)

        if self.append and self.former_train_df is not None:
            logger.info('Appending new data to existing dataframe ...')
            training_df = pd.concat([training_df, self.former_train_df])

        # Save to disk
        train_df_file = (Path(self.outdir)
                         / f'training_df_{self.outputlabel}.parquet')
        logger.info(f'Saving training df to: {train_df_file}')
        training_df.to_parquet(train_df_file)

    def _filter_database(self, database, filt_loc=None,
                         filt_type=None, filt_cat=None,
                         filt_content=None, limitsamples=None):

        # Filter for samples of the requested type
        if filt_type is not None:
            logger.info(f'Subsetting on {filt_type} labels ...')
            database = database[database['labeltype'] == filt_type]
            logger.info(f'{len(database)} samples remaining')

        # Filter for samples of requested location(s)/time(s)
        if filt_loc is not None:
            logger.info('Subsetting on location ...')
            database = database[database['ref_id'].isin(filt_loc)]
            logger.info(f'{len(database)} samples remaining')

        # Filter for samples of requested content type
        if filt_content is not None:
            logger.info('Subsetting on content ...')
            database = database[database['contenttype'].isin(filt_content)]
            logger.info(f'{len(database)} samples remaining')

        # And filter on samples of requested category
        if filt_cat is not None:
            logger.info(f'Subsetting on sample type {filt_cat}...')
            database = database[database['split'] == filt_cat]
            logger.info(f'{len(database)} samples remaining')

        # If L8 data is requested, get rid of samples with L8 issues
        if 'TIR' in self.parameters['settings'].keys():
            logger.info('Filtering on Landsat-8 data availability...')
            database = database.loc[database['L8_issue'] != 1]
            logger.info(f'{len(database)} samples remaining')

        # Take a subset of samples if needed
        if limitsamples is not None:
            database = database.iloc[0:limitsamples, :]
            logger.info(f'Limiting samples to {limitsamples}')

        # If required, load previous trainingDF
        # and only retain new samples
        if self.append:
            former_train_df_file = (
                Path(self.outdir)
                / f'training_df_{self.outputlabel}.parquet')
            if former_train_df_file.is_file():
                logger.info('Loading existing dataframe ...')
                former_train_df = pd.read_parquet(former_train_df_file)
                self.former_train_df = former_train_df
                existing_location_ids = list(
                    former_train_df.location_id.unique())
                logger.info((f'Removing {len(existing_location_ids)} '
                             'samples from database ...'))
                database = database[~database['location_id'].isin(
                    existing_location_ids)]
            else:
                self.former_train_df = None

        return database

    def _correct_wintercereals(self, feature_df):

        logger.info('Correcting winter cereals ...')

        # Check for each sample whether Spring Wheat is possible
        feature_df['trigger_sw'] = self.aez_layer.set_index('zoneID').loc[
            feature_df['aez_zoneid'].values]['trigger_sw'].values

        # Find samples that cannot be spring-planted according to
        # our definition
        idx_1110 = feature_df[
            (feature_df['trigger_sw'] == 0) &
            (
                (feature_df['OUTPUT'] == 1100) |
                (feature_df['OUTPUT'] == 1120)
            )
        ].index
        idx_1510 = feature_df[
            (feature_df['trigger_sw'] == 0) &
            (
                (feature_df['OUTPUT'] == 1500) |
                (feature_df['OUTPUT'] == 1520)
            )
        ].index
        idx_1610 = feature_df[
            (feature_df['trigger_sw'] == 0) &
            (
                (feature_df['OUTPUT'] == 1600) |
                (feature_df['OUTPUT'] == 1620)
            )
        ].index
        idx_1910 = feature_df[
            (feature_df['trigger_sw'] == 0) &
            (
                (feature_df['OUTPUT'] == 1900) |
                (feature_df['OUTPUT'] == 1920)
            )
        ].index

        total_changes = (len(idx_1110) + len(idx_1510) + len(idx_1610)
                         + len(idx_1910))

        logger.info((f'Switching {total_changes} winter cereal samples '
                     'to their winter version.'))

        feature_df.loc[idx_1110, 'OUTPUT'] = 1110
        feature_df.loc[idx_1510, 'OUTPUT'] = 1510
        feature_df.loc[idx_1610, 'OUTPUT'] = 1610
        feature_df.loc[idx_1910, 'OUTPUT'] = 1910

        return feature_df


def main(cib_rootdir, experiment_id, outdir,
         parameters, augment=False,
         filt_loc=None, filt_type=None, filt_cat=None,
         filt_content=None, maxnrpixels=1,
         limitsamples=None, season='annual', outputlabel='LC',
         filter_function=None,
         append=False, sc=None, debug=False):

    # Initialize FeatureComputer
    featurecomputer = FeatureComputer(cib_rootdir, experiment_id,
                                      outdir, season, parameters,
                                      augment=augment,
                                      outputlabel=outputlabel,
                                      append=append)

    # Load and filter database
    featurecomputer.load_database(
        filt_loc=filt_loc,
        filt_type=filt_type, filt_cat=filt_cat,
        filt_content=filt_content, limitsamples=limitsamples
    )

    # Create the collections
    featurecomputer.create_collections()

    # Setup the TrainingChain
    featurecomputer.build_trainingchain(gddnormalization)

    # Compute and save features
    featurecomputer.compute_features(sparkcontext=sc,
                                     max_pixels=maxnrpixels,
                                     filter_function=filter_function,
                                     debug=debug)

    logger.success('All done!')


if __name__ == '__main__':

    # CIB specs
    cib_rootdir = '/data/worldcereal/cib'
    experiment_id = 'CIB_V1'

    # Output feature basedir
    baseoutdir = '/data/worldcereal/features-nowhitakker-final'

    append = False
    overwrite = True
    limitsamples = None
    debug = False
    spark = True
    localspark = False
    filt_loc = None

    if spark:
        logger.info('Setting up spark ...')
        sc = get_spark_context(localspark=localspark)
    else:
        sc = None

    scenario_settings = {
        # 'annual': {
        #     'season': 'annual',
        #     'maxnrpixels': 3,  # THIS IS NOW THE NR. OF PIXELS PER CLASS IN PATCH!  # NOQA
        #     'outputlabel': 'LC',
        #     'filt_content': None,
        #     'gddnormalization': False,
        #     'augment': True,
        #     'parameters': get_cropland_catboost_parameters()
        #     },
        'winter':
            {
            'season': 'winter',
            'maxnrpixels': 5,  # THIS IS NOW THE NR. OF PIXELS PER CLASS IN PATCH!  # NOQA
            'outputlabel': 'CT',
            'filt_content': ['110', '111'],
            'gddnormalization': True,
            'augment': True,
            'parameters': get_croptype_catboost_parameters()
            },
        # 'summer1-75dayspriorEOS':
        #     {
        #     'season': 'summer1',
        #     'maxnrpixels': 5,  # THIS IS NOW THE NR. OF PIXELS PER CLASS IN PATCH!  # NOQA
        #     'outputlabel': 'CT',
        #     'filt_content': ['110', '111'],
        #     'gddnormalization': True,
        #     'augment': True,
        #     'season_post_buffer': -75,  # End the season 75 days earlier
        #     'parameters': get_croptype_catboost_parameters()
        #     # },
        'summer1':
            {
            'season': 'summer1',
            'maxnrpixels': 5,  # THIS IS NOW THE NR. OF PIXELS PER CLASS IN PATCH!  # NOQA
            'outputlabel': 'CT',
            'filt_content': ['110', '111'],
            'gddnormalization': True,
            'augment': True,
            'parameters': get_croptype_catboost_parameters()
            },
        'summer2':
            {
            'season': 'summer2',
            'maxnrpixels': 5,  # THIS IS NOW THE NR. OF PIXELS PER CLASS IN PATCH!  # NOQA
            'outputlabel': 'CT',
            'filt_content': ['110', '111'],
            'gddnormalization': True,
            'augment': True,
            'parameters': get_croptype_catboost_parameters()
            },
        # 'irr_summer1':
        #     {
        #     'season': 'summer1',
        #     # 'maxnrpixels': 5,  # THIS IS NOW THE NR. OF PIXELS PER CLASS IN PATCH!  # NOQA
        #     'maxnrpixels': 1,  # for spain datasets
        #     'outputlabel': 'IRR',
        #     'filt_content': ['111', '101'],
        #     'gddnormalization': False,
        #     'parameters': get_irr_parameters(),
        #     'filter_function': 'irr_max_ndvi_filter'
        #     },
        # 'irr_winter':
        #     {
        #         'season': 'winter',
        #         # 'maxnrpixels': 5,  # THIS IS NOW THE NR. OF PIXELS PER CLASS IN PATCH!  # NOQA
        #         'maxnrpixels': 1,  # for spain datasets
        #         'outputlabel': 'IRR',
        #         'filt_content': ['101', '111'],
        #         'gddnormalization': False,
        #         'parameters': get_irr_parameters(),
        #         'filter_function': 'irr_max_ndvi_filter'
        #     }
        # 'irr_summer2':
        #     {
        #         'season': 'summer2',
        #         'maxnrpixels': 5,  # THIS IS NOW THE NR. OF PIXELS PER CLASS IN PATCH!  # NOQA
        #         'outputlabel': 'IRR',
        #         'filt_content': ['111', '101'],
        #         'gddnormalization': False,
        #         'parameters': get_irr_parameters(),
        #         'filter_function': 'irr_max_ndvi_filter'
        # }
    }

    for scenario in scenario_settings.keys():
        outname = scenario + '_CIB'
        season = scenario_settings[scenario]['season']
        outputlabel = scenario_settings[scenario]['outputlabel']
        maxnrpixels = scenario_settings[scenario]['maxnrpixels']
        gddnormalization = scenario_settings[scenario]['gddnormalization']
        augment = scenario_settings[scenario].get('augment', False)
        filt_content = scenario_settings[scenario]['filt_content']
        filter_function = scenario_settings[scenario].get('filter_function',
                                                          None)

        # Get parameters for this scenario
        parameters = scenario_settings[scenario]['parameters']

        for filt_cat in ['CAL', 'VAL', 'TEST']:
            outdir = (Path(
                baseoutdir)
                / outname / filt_cat)
            outdir.mkdir(exist_ok=True, parents=True)
            if (not (outdir / (f'training_df_{outputlabel}.parquet')).is_file()
                    or overwrite):
                main(cib_rootdir, experiment_id, outdir,
                     parameters,
                     augment=augment,
                     filt_cat=filt_cat,
                     filt_loc=filt_loc,
                     filt_content=filt_content,
                     maxnrpixels=maxnrpixels,
                     limitsamples=limitsamples,
                     season=season, outputlabel=outputlabel,
                     filter_function=filter_function,
                     append=append,
                     sc=sc,
                     debug=debug)
