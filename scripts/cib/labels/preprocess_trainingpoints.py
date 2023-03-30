import os
import satio.layers
import pandas as pd
import numpy as np
from loguru import logger
import json
from pathlib import Path
import subprocess

from worldcereal.cib.labels import TrainingInput
from worldcereal.utils.spark import get_spark_context

TIMERANGE_LUT = {
    2016: {'start': '2015-11-01',
           'end': '2016-11-30'},
    2017: {'start': '2016-11-01',
           'end': '2017-11-30'},
    2018: {'start': '2017-08-01',
           'end': '2018-11-30'},
    2019: {'start': '2018-08-01',
           'end': '2019-11-30'},
    2020: {'start': '2019-08-01',
           'end': '2020-11-30'},
    2021: {'start': '2020-08-01',
           'end': '2021-11-30'}
}

HARD_START = '2017-01-01'


def refdata_to_tile(infile, s2grid, s2tile, outputbasename):
    logger.info(f'Writing full samples in s2 tile: {s2tile} ...')
    wgs84bounds = s2grid[s2grid['tile'] == s2tile]
    epsg = s2grid[s2grid['tile'] == s2tile].epsg.values[0]
    projbounds = wgs84bounds.to_crs(epsg=epsg).buffer(1000)
    xmin, ymin, xmax, ymax = projbounds.to_crs(
        epsg=4326).geometry.values[0].bounds
    outfile = outputbasename + f'_{s2tile}.json'

    if Path(outfile).is_file():
        os.remove(outfile)

    logger.info(f'Cutting features from {infile} into {outfile} ...')

    # We need to use ogr2ogr for speed and memory issues
    cmd = (f'ogr2ogr -of GeoJSON -overwrite -spat {xmin} {ymin} '
           f'{xmax} {ymax} {outfile} {infile}')

    logger.debug(cmd)

    subprocess.call(cmd, shell=True)

    logger.success('ogr2ogr process finished!')


def infer_start_date(df, prior_days=365):
    '''
    By default we take 12 months prior to validity time
    '''
    logger.info('Inferring start date for samples ...')
    if 'validityTi' not in df.columns:
        raise Exception('Column `validityTi` not found in DataFrame.')

    validityTimes = [pd.to_datetime(date) for date in df['validityTi'].values]
    start_dates = [(date - pd.Timedelta(
        days=prior_days)).to_period('M').to_timestamp().strftime('%Y-%m-%d')
        for date in validityTimes]

    return start_dates


def infer_end_date(df, post_days=365):
    '''
    By default we take 6 months past validity time
    '''
    logger.info('Inferring end date for samples ...')
    if 'validityTi' not in df.columns:
        raise Exception('Column `validityTi` not found in DataFrame.')

    validityTimes = [pd.to_datetime(date) for date in df['validityTi'].values]
    end_dates = [(date + pd.Timedelta(
        days=post_days)).to_period('M').to_timestamp().strftime('%Y-%m-%d')
        for date in validityTimes]

    return end_dates


def process_dataset(settings, sc,
                    skip_processed=True):

    # get settings
    idattr = settings.get('idattr', 'sampleID')
    labeltype = settings['labeltype']
    contenttype = settings['contenttype']
    ref_id = settings['ref_id']
    year = settings['year']
    cib_experiment_id = settings['cib_experiment_id']
    start_date = settings.get('start_date', None)
    end_date = settings.get('end_date', None)
    remove_autocorrelation = settings.get('remove_autocorrelation', 0)
    remove_autocorrelation = bool(remove_autocorrelation)

    # Infer the input file
    infile = (f'/data/worldcereal/data/ref/VITO_processed/{labeltype}/'
              f'{contenttype}/{year}/{ref_id}_'
              f'{labeltype}_{contenttype}/'
              f'{ref_id}_{labeltype}_{contenttype}.shp')
    if not Path(infile).is_file():
        infile = infile.replace('shp', 'gpkg')
        if not Path(infile).is_file():
            infile = infile.replace('gpkg', 'shp')
            raise ValueError(f'File `{infile}` not found!')

    # Infer output directory
    outdir = (f'/data/worldcereal/cib/{cib_experiment_id}'
              f'/{labeltype}/{ref_id}')
    os.makedirs(outdir, exist_ok=True)

    # Set the output file for the actual training samples
    outfile_samples = str(Path(outdir) / (Path(infile).stem + '_samples.json'))

    if os.path.exists(outfile_samples) and skip_processed:
        return 'ok'
    else:
        logger.info(f'START PROCESSING DATASET: {ref_id}')

        # Infer start/end dates for imagery
        if start_date is None:
            logger.warning('No start date provided -> taking default')
            start_date = TIMERANGE_LUT[year]['start']
        if end_date is None:
            end_date = TIMERANGE_LUT[year]['end']
            logger.warning('No end date provided -> taking default')

        # Infer the samples file
        samplesfile = (f'/data/worldcereal/data/ref/VITO_processed/{labeltype}/'
                       f'{contenttype}/{year}/{ref_id}_'
                       f'{labeltype}_{contenttype}/'
                       f'{ref_id}_{labeltype}_{contenttype}_samples.csv')
        # Check if a samples file exist
        if not os.path.isfile(samplesfile):
            logger.warning(('No subset samples file found -> '
                            'using full input shapefile'))
            samplesfile = None
        else:
            logger.info(f'Found a subset samples file: {samplesfile}')

        # NOTE: if samplesfile is set to "None", ALL samples in shapefile
        # will be processed

        kernelsize = 32 if labeltype != 'POINT' else 2
        rounding = 20

        # Set the output directory for all samples per s2 tile
        outdir_tiles = Path(outdir) / 'ref_tiles'
        os.makedirs(outdir_tiles, exist_ok=True)

        # Read in the training data
        trainingdata = TrainingInput.from_file(infile, idattr=idattr,
                                               overwriteattrs=False,
                                               kernelsize=kernelsize,
                                               rounding=rounding)

        # remove any invalid geometries
        trainingdata.data = trainingdata.data.dropna(subset=['geometry'])

        # Let's do some checks to see if the required attributes exist
        logger.info('Checking required attributes ...')
        req_attrs = ['LC', 'CT', 'IRR']

        missing = []
        for attr in req_attrs:
            if attr not in trainingdata.data.columns:
                missing.append(attr)
        if len(missing) > 0:
            raise ValueError(('Following required attributes not found '
                              f'in file: ({missing} found in file!'))

        # checking userconf...
        # Hack to avoid breaking with slightly different name
        if 'userconf' in trainingdata.data.columns:
            trainingdata.data = trainingdata.data.rename(
                columns={'userconf': 'userConf'}
            )
        # adding userconf if it is not there
        if 'userConf' not in trainingdata.data.columns:
            logger.info('No userConf! Manually adding it (default = 0)')
            nsamples = trainingdata.data.shape[0]
            trainingdata = trainingdata.add(userConf=np.zeros(nsamples))

        # We need to make sure the output labels are Integers
        trainingdata.data['LC'] = trainingdata.data['LC'].astype(int)
        trainingdata.data['CT'] = trainingdata.data['CT'].astype(int)
        trainingdata.data['IRR'] = trainingdata.data['IRR'].astype(int)

        # Also make sure that validityTi is present and is a string
        if 'validityTi' not in trainingdata.data.columns:
            if 'validityTime' in trainingdata.data.columns:
                trainingdata.data = trainingdata.data.rename(
                    columns={'validityTime': 'validityTi'})
            elif 'ValidityTi' in trainingdata.data.columns:
                trainingdata.data = trainingdata.data.rename(
                    columns={'ValidityTi': 'validityTi'})
            elif 'valtime' in trainingdata.data.columns:
                trainingdata.data = trainingdata.data.rename(
                    columns={'valtime': 'validityTi'})
            else:
                raise Exception('Column `validityTi` not found in DataFrame.')
        trainingdata.data['validityTi'] = trainingdata.data[
            'validityTi'].astype(str)

        # Subset the training data if needed
        if samplesfile is not None:
            # TODO: this is specifically for current formatting
            # which might not be the ideal way
            subset = pd.read_csv(samplesfile)
            samples = subset['sampleID'].values.tolist()
            trainingdatasubset = trainingdata.subset(samples)
        else:
            trainingdatasubset = trainingdata.copy()

        # Remove autocorrelated samples
        if remove_autocorrelation:
            trainingdatasubset = trainingdatasubset.remove_autocorrelation()

        # Add a cal/val/test split if it's missing
        if ('split' not in trainingdatasubset.data.columns or
                any(trainingdatasubset.data['split'] == '') or
                any(trainingdatasubset.data['split'].isnull())):
            trainingdatasubset = trainingdatasubset.add(
                split=trainingdatasubset.split_cal_val_test())

        # As the source attribute, we take the filename
        source = os.path.splitext(os.path.basename(infile))[0]

        # Add all required attributes, a.k.a. metadata
        # to be used downstream in the CIB
        trainingdatasubset = trainingdatasubset.add(
            bounds=trainingdatasubset.bounds,
            year=year,
            ref_id=ref_id,
            labeltype=labeltype,
            contenttype=contenttype,
            source=source,
            source_path=infile)

        # Add start and end dates
        docheck = False
        if start_date == 'infer':
            docheck = True
            start_date = infer_start_date(trainingdatasubset.data)
        if end_date == 'infer':
            docheck = True
            end_date = infer_end_date(trainingdatasubset.data)

        trainingdatasubset = trainingdatasubset.add(
            start_date=start_date,
            end_date=end_date
        )

        if docheck:
            # Make sure the requested time series does not start
            # before 2017-01-01 because there is no L2A availability
            trainingdatasubset.data.loc[pd.to_datetime(
                trainingdatasubset.data['start_date'])
                < pd.to_datetime(HARD_START), 'start_date'] = HARD_START

        if 'index_right' in trainingdatasubset.data:
            trainingdatasubset = trainingdatasubset.drop('index_right')

        # Save final training data to json
        trainingdatasubset.save(outfile_samples, driver='json')
        # trainingdatasubset.save(outfile_samples, driver='gpkg')

        # Load the s2 tile grid
        s2grid = satio.layers.load('s2grid')

        # # List all unique S2 tiles for which we have training samples selected
        uniquetiles = list(set(trainingdatasubset.data['tile'].to_list()))

        # Save FULL training files per buffered S2tile
        outputbasename = str(outdir_tiles / Path(infile).stem)

        # Attempt to fix geometry issues
        if sc is None:
            # Process in serial
            for s2tile in uniquetiles:
                logger.info(f'Processing tile {s2tile} ...')
                refdata_to_tile(infile, s2grid, s2tile, outputbasename)
        else:
            logger.info('Sending tile jobs to executors ...')
            sc.parallelize(uniquetiles, len(uniquetiles)).foreach(
                lambda s2tile: refdata_to_tile(
                    infile, s2grid, s2tile, outputbasename)
            )

        logger.success(f'Dataset {ref_id} processed successfully!')
        return 'ok'

        # UNCOMMENT TO EXPORT THE BOUNDS AS SHP FOR DEBUGGING

        # gdf = trainingdatasubset.bounds_as_geometry()
        # gdf.to_file(os.path.splitext(
        #     outfile_samples)[0] + f'_bounds{kernelsize}.shp')

        # UNCOMMENT TO EXPORT THE SHIFTED COORDS AS SHP FOR DEBUGGING

        # from shapely.geometry import Point
        # import geopandas as gpd

        # test = gpd.GeoDataFrame(trainingdatasubset.data.apply(
        #     lambda row: Point(row['round_lon'], row['round_lat']),
        #     axis=1)).rename(columns={0: 'geometry'})
        # test.to_file(os.path.splitext(outfile_samples)[0] + '_shiftedcoords.shp')


def rename_old_sample_files(datasets):

    for dataset, settings in datasets.items():

        cib_experiment_id = settings['cib_experiment_id']
        labeltype = settings['labeltype']
        ref_id = settings['ref_id']
        year = settings['year']
        contenttype = settings['contenttype']

        # Infer output directory
        outdir = (f'/data/worldcereal/cib/{cib_experiment_id}'
                  f'/{labeltype}/{ref_id}')
        os.makedirs(outdir, exist_ok=True)

        # Infer the input file
        infile = (f'/data/worldcereal/data/ref/VITO_processed/{labeltype}/'
                  f'{contenttype}/{year}/{ref_id}_'
                  f'{labeltype}_{contenttype}/'
                  f'{ref_id}_{labeltype}_{contenttype}.shp')
        if not Path(infile).is_file():
            infile = infile.replace('shp', 'gpkg')
            if not Path(infile).is_file():
                infile = infile.replace('gpkg', 'shp')
                raise ValueError(f'File `{infile}` not found!')

        # Get the output file for the actual training samples
        outfile_samples = str(Path(outdir) / (Path(infile).stem + '_samples.json'))

        if os.path.exists(outfile_samples):
            newfilename = outfile_samples.replace('_samples.json',
                                                  '_samples_OLD.json')
            logger.info(f'Renaming: {outfile_samples} to {newfilename}')
            os.rename(outfile_samples, newfilename)


def main(datasets, skip_processed=True, spark=False):

    if spark:
        logger.info('Setting up spark ...')
        sc = get_spark_context()
    else:
        sc = None

    # rename previous versions of datasets
    # rename_old_sample_files(datasets)

    processed = []
    for dataset, settings in datasets.items():
        result = process_dataset(settings, sc,
                                 skip_processed=skip_processed)
        if result == 'ok':
            processed.append(dataset)

    logger.info(f'{len(processed)} out of {len(datasets.keys())}'
                f' datasets processed successfully: {processed}')


if __name__ == '__main__':

    spark = True
    skip_processed = False

    # Load the list of datasets
    dataset_file = '/data/worldcereal/data/ref/VITO_processed/datasets_preprocess_trainingpoints_new.json'  # NOQA
    if not Path(dataset_file).is_file():
        raise FileNotFoundError(
            'Required datasets file '
            f'`{dataset_file}` not found. Cannot continue.')
    datasets = json.load(open(dataset_file, 'r'))

    # Run the main program
    main(datasets, spark=spark, skip_processed=skip_processed)
