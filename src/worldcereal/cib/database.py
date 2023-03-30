import glob
import multiprocessing as mp
import os
from pathlib import Path

from loguru import logger
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr


# Define the products that should be in the CIB
_PRODUCTS = ['S2_L2A_10m',
             'S2_L2A_20m',
             'AgERA5_DAILY_1000m',
             'S1_GRD-ASCENDING_20m',
             'S1_GRD-DESCENDING_20m',
             'OUTPUT',
             'L8_L2_10m',
             'L8_L2-ST_10m']


def parametrized(dec):
    """This decorator can be used to create other
    decorators that accept arguments"""

    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)

        return repl

    return layer


@parametrized
def sigsev_guard(fcn, default_value=None, timeout=None):
    """Used as a decorator with arguments.
    The decorated function will be called with its
    input arguments in another process.

    If the execution lasts longer than *timeout* seconds,
    it will be considered failed.

    If the execution fails, *default_value* will be returned.
    """

    def _fcn_wrapper(*args, **kwargs):
        q = mp.Queue()
        p = mp.Process(target=lambda q: q.put(fcn(*args, **kwargs)), args=(q,))
        p.start()
        p.join(timeout=timeout)
        exit_code = p.exitcode

        if exit_code == 0:
            return q.get()

        logger.warning(('Process did not exit correctly. '
                        f'Exit code: {exit_code}'))
        return default_value

    return _fcn_wrapper


@sigsev_guard(default_value=False, timeout=60)
def open_file_safely(file):
    _ = xr.open_dataset(file,
                        engine='h5netcdf',
                        mask_and_scale=False
                        )
    return True


class DataBase(object):

    def __init__(self,
                 experiment_id,
                 rootdir,
                 sample_files: list,
                 spark=False):
        logger.info(('Setting up CIB database for '
                     f'{experiment_id} ...'))

        self.experiment_id = experiment_id
        self.rootdir = rootdir
        self.source_files = sample_files
        self._logdir = os.path.join(rootdir, 'logs')
        self._proc_logdir = os.path.join(rootdir, 'logs', 'proc_logs')
        self._gdf = None
        self._spark = spark

        os.makedirs(self._logdir, exist_ok=True)
        os.makedirs(self._proc_logdir, exist_ok=True)

        if spark:
            import sys
            logger.info('Initializing spark ...')
            logger.info(f'Python version: {sys.version_info}')
            import pyspark
            import cloudpickle
            pyspark.serializers.cloudpickle = cloudpickle
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
            self._sc = spark.sparkContext

    @ classmethod
    def from_folder(cls,
                    infolder,
                    jsonfiles=None,
                    spark=False):

        logger.info(('Looking for json files with training samples '
                     f'in {infolder} ...'))

        # Infer experiment name
        experiment_id = Path(infolder).stem

        # Find all sample jsons for the experiment
        # (if no specific json files are specified)
        if jsonfiles is None:
            jsonfiles = glob.glob(os.path.join(infolder,
                                               '*', '*',
                                               '*_samples.json'))
        # Print the results
        for file in jsonfiles:
            logger.info(f'Found {file} ...')

        # Make an instance of the database class
        return cls(experiment_id, infolder, jsonfiles,
                   spark=spark)

    def _read_file(self, file: str):
        if not Path(file).is_file():
            raise ValueError(f'File {file} does not exist!')

        logger.debug(f'Opening {file} ...')
        gdf = gpd.read_file(file)

        return gdf

    def to_json(self, outfile):
        if self.database is None:
            raise ValueError('First populate the database!')
        logger.info((f'Writing {self.database.shape[0]} samples '
                     f'to {outfile} ...'))
        self.database.to_file(outfile, driver='GeoJSON')

    def populate(self):

        logger.info('Populating database ...')
        gdf = gpd.GeoDataFrame()
        for file in self.source_files:
            gdf = gdf.append(self._read_file(file))

        logger.info(f'Added a total of {len(gdf)} samples to database!')

        # Set the location_id attribute as the index
        gdf = gdf.set_index('location_id')

        # Make sure there's no duplicates
        nr_duplicates = gdf.index.duplicated().sum()
        if nr_duplicates > 0:
            logger.warning((f'Will drop {nr_duplicates} samples '
                            'with duplicate location_id!'))
            gdf = gdf[~gdf.index.duplicated()]

        # Only keep the required attributes
        gdf = gdf[_BASE_ATTRIBUTES]

        self._gdf = gdf

    def add_root(self):
        """
        Function to add "path" column to df
        referring to data location on disk
        """
        if self.database is None:
            raise ValueError('First populate the database!')

        def _add_root(row):
            rootpath = (Path(self.rootdir) / str(row['labeltype']) /
                        row['ref_id'] / 'data' /
                        str(row['epsg']) / row['tile'] / str(row.name))
            return str(rootpath)

        logger.info('Adding root paths ...')

        self.database['path'] = self.database.apply(_add_root,
                                                    axis=1)

    def add_aez(self, aez_file):
        """
        Function to add AEZ information
        """
        if self.database is None:
            raise ValueError('First populate the database!')

        aez = gpd.read_file(aez_file).to_crs(epsg=4326)
        aez['geometry'] = aez['geometry'].buffer(0)

        def _get_aez_info(row, aez):

            if row['labeltype'] == 'POLY':
                row['geometry'] = row['geometry'].buffer(0)

            matching_aez = aez[aez.intersects(row.geometry)]
            aez_id = matching_aez['zoneID'].values[0]
            aez_group = matching_aez['groupID'].values[0]

            aez_info = {
                'location_id': row.name,
                'aez_id': aez_id,
                'aez_group': aez_group
            }

            return aez_info

        logger.info('Adding AEZ information ...')

        if not self._spark:
            # Run locally
            aez_info = list(self.database.apply(
                lambda row: _get_aez_info(
                    row,
                    aez=aez),
                axis=1).values)
        else:
            aez_info = self._sc.parallelize(
                self.database.iterrows(),
                len(self.database.index.tolist())).map(
                lambda row: _get_aez_info(
                    row[1],
                    aez=aez)).collect()

        aez_info = pd.DataFrame.from_dict(
            aez_info).set_index('location_id')
        self._gdf = pd.concat([self._gdf,
                               aez_info], axis=1)

    def add_l8(self, l8issues):

        df = pd.DataFrame(index=self._gdf.index,
                          columns=['L8_issue'],
                          data=np.zeros(len(self._gdf), dtype=np.uint16))
        df.loc[l8issues.index, 'L8_issue'] = 1

        self._gdf = pd.concat([self._gdf,
                               df], axis=1)

    def add_products(self, products=_PRODUCTS, remove_errors=False):
        if self.database is None:
            raise ValueError('First populate the database!')

        def _check_dates(ds, startdate, enddate, file):
            logger.debug(f'Checking start and end of: {file}')
            start = pd.to_datetime(ds.timestamp.values[0])
            end = pd.to_datetime(ds.timestamp.values[-1])
            if ((start - pd.to_datetime(startdate))
                    > pd.Timedelta(days=60)):
                logger.error((f'"{file}" only starts at '
                              f'{start} while stack should start '
                              f' at {startdate}!'))
                return False
            elif ((pd.to_datetime(enddate) - end)
                  > pd.Timedelta(days=60)):
                logger.error((f'"{file}" ends at '
                              f'{end} while stack should end '
                              f' at {enddate}!'))
                return False
            else:
                logger.info('Date check OK!')
                return True

        # @exitlogs(self._logdir)
        # @proclogs(self._proc_logdir, level='INFO')
        def _log_find_products(task_args, rootdir=None,
                               products=_PRODUCTS, local=True,
                               remove_errors=False):
            task_id, row = task_args
            return _find_products(row, rootdir, products, local,
                                  remove_errors)

        def _find_products(row, rootdir, products, local=True,
                           remove_errors=False):

            def _check_file(file, remove_errors=False):
                # We check if file can be opened
                # and if we have expected time range
                # in the product
                logger.info(f'Attempt opening file: {file}')

                if open_file_safely(file):
                    ds = xr.open_dataset(file,
                                         engine='h5netcdf',
                                         mask_and_scale=False)
                    logger.info('Successfully openened ...')

                    if ('OUTPUT' not in pattern
                            and 'CropCalendar' not in pattern):
                        is_ok = _check_dates(
                            ds,
                            row['start_date'],
                            row['end_date'],
                            file)
                    else:
                        is_ok = True
                else:
                    is_ok = False
                if not is_ok and remove_errors:
                    logger.warning(f'Removing: {file}')
                    os.remove(file)
                return is_ok

            datapath = (Path(rootdir) / row['labeltype'] /
                        row['ref_id'] / 'data' /
                        str(row['epsg']) / row['tile'] / str(row.name))

            productpaths = {'location_id': row.name}
            for pattern in products:
                if pattern != 'OUTPUT':
                    productpath = (datapath / '_'.join([
                        pattern, str(row.name), str(row.epsg),
                        row.start_date, row.end_date + '.nc'
                    ]))
                else:
                    productpath = (datapath / '_'.join([
                        pattern,
                        row.labeltype + '-' + row.contenttype,
                        '10m', str(row.name), str(row.epsg),
                        row.start_date, row.end_date + '.nc'
                    ]))
                try:
                    is_ok = _check_file(productpath,
                                        remove_errors=remove_errors)
                    productpaths[pattern] = str(productpath) if is_ok else None

                except Exception as e:
                    # Fails when after retries product
                    # is not found or cannot be opened
                    logger.error(f'Got an error for file : {productpath}')
                    logger.error(e)
                    productpaths[pattern] = None

            return productpaths

        if not self._spark:
            # Run locally
            logger.info(f'Looking for products ...')
            products = list(self.database.apply(
                lambda row: _log_find_products(
                    (row.name,
                     row),
                    rootdir=self.rootdir,
                    products=products,
                    remove_errors=remove_errors),
                axis=1).values)
        else:
            logger.info(f'Looking for products using spark ...')
            rootdir = self.rootdir
            products = self._sc.parallelize(
                self.database.iterrows(),
                len(self.database.index.tolist())).map(
                lambda row: _log_find_products(
                    (row[0],
                     row[1]),
                    rootdir=rootdir,
                    products=products,
                    local=False,
                    remove_errors=remove_errors)).collect()

        productpaths = pd.DataFrame.from_dict(
            products).set_index('location_id')
        self._gdf = pd.concat([self._gdf,
                               productpaths], axis=1)

    def check_s2_resolutionmatch(self):
        if self.database is None:
            raise ValueError('First populate the database!')
        if 'S2_L2A_10m' not in self.database.columns:
            raise Exception('"S2_L2A_10m" not in paths')
        if 'S2_L2A_20m' not in self.database.columns:
            raise Exception('"S2_L2A_20m" not in paths')

        def _is_matching(row):
            logger.debug(f'Checking sample {row.name} ...')

        logger.info('Checking S2 resolution match ...')
        result = self.database.apply(_is_matching, axis=1)
        result = result[result]
        return result

    def check_issues(self, products=_PRODUCTS):

        # @exitlogs(self._logdir)
        # @proclogs(self._proc_logdir, level='DEBUG')
        def _log_check_sample(task_args):
            task_id, row = task_args
            return _check_sample(row)

        def _check_sample(row):

            checks = dict()
            checks['location_id'] = row.name
            logger.debug(f'Checking sample: {row.name}')

            # Check if the files can be opened
            for product in products:
                if product not in row:
                    logger.warning(f'"{product}" not in paths')
                    checks[product] = False
                elif row[product] is None:
                    logger.warning(f'"{product}" does not exist or is corrupt')
                    checks[product] = False
                else:
                    # File is OK
                    checks[product] = True

            # For S2: check if timestamps of both
            # resolutions are identical
            if (checks['S2_L2A_10m']
                    and checks['S2_L2A_20m']):

                ds_10m = xr.open_dataset(row['S2_L2A_10m'],
                                         engine='h5netcdf',
                                         mask_and_scale=False)
                ds_20m = xr.open_dataset(row['S2_L2A_20m'],
                                         engine='h5netcdf',
                                         mask_and_scale=False)
                if ((ds_10m.timestamp.size ==
                     ds_20m.timestamp.size) and all(
                         ds_10m.timestamp == ds_20m.timestamp)):
                    checks['s2_identicaltimestamps'] = True
                else:
                    logger.warning((f'Sample {row.name} has '
                                    ' a different number of '
                                    'timestamps!'))
                    checks['s2_identicaltimestamps'] = False

            else:
                checks['s2_identicaltimestamps'] = False

            return checks

        if not self._spark:
            # Run locally
            checks = self.database.apply(
                lambda row: _check_sample(row), axis=1).apply(
                    pd.Series
            ).drop(columns=['location_id'])
        else:
            logger.info('Running checks on spark ...')

            results = self._sc.parallelize(
                self.database.iterrows(),
                len(self.database.index.tolist())).map(
                lambda row: _log_check_sample(row)).collect()

            checks = pd.DataFrame.from_dict(results)
            checks = checks.set_index('location_id')
            # checks = self.database.apply(
            #     lambda row: _check_sample(row), axis=1)

        # combine S1 ASCENDING and DESCENDING in one check
        checks['S1_GRD_20m'] = checks[['S1_GRD-ASCENDING_20m', 'S1_GRD-DESCENDING_20m']].any(axis=1)
        checks = checks.drop(columns=['S1_GRD-ASCENDING_20m', 'S1_GRD-DESCENDING_20m'])

        # combine L8_L2 and L8_L2-ST in one check
        checks['L8_L2*_10m'] = checks[['L8_L2_10m', 'L8_L2-ST_10m']].any(axis=1)
        checks = checks.drop(columns=['L8_L2_10m', 'L8_L2-ST_10m'])

        # add source information on samples
        checks['source'] = self._gdf.loc[checks.index]['source']
        # if there is no IRR information, we don't need L8
        checks.loc[checks['source'].str.endswith(('100', '110')),
                   'L8_L2*_10m'] = True

        # drop source column again for next check
        checks = checks.drop(columns=['source'])

        # only retain samples with issues
        issuesinv = ~checks
        issues = checks[issuesinv.sum(axis=1) > 0]
        logger.info(f'Found issues with {len(issues)} files!')

        # if the only issue is L8,
        # then we can still keep the sample in the CIB
        l8issues = checks[(issuesinv.sum(axis=1) == 1) &
                          (checks['L8_L2*_10m'] == False)]
        logger.info(f'Found {len(l8issues)} samples with only L8 issues!')

        # Remove these from issues
        issues = issues.drop(index=l8issues.index)
        logger.info(f'Found {len(issues)} samples with remaining issues!')

        return issues, l8issues

    def drop(self, index: list):
        '''
        Method to drop certain samples from database
        based on list of indexes
        '''
        logger.info('Dropping indexes from database.')
        self._gdf = self._gdf.drop(index=index)

    @ property
    def database(self):
        if self._gdf is None:
            logger.warning('Database not yet populated!')
            return None
        else:
            return self._gdf


_BASE_ATTRIBUTES = [
    'userConf',
    'validityTi',
    'split',
    'LC',
    'CT',
    'IRR',
    'tile',
    'easting',
    'northing',
    'epsg',
    'zonenumber',
    'zoneletter',
    'round_lat',
    'round_lon',
    'bounds',
    'year',
    'start_date',
    'end_date',
    'ref_id',
    'labeltype',
    'contenttype',
    'source',
    'source_path',
    'geometry'
]
