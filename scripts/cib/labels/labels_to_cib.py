import geopandas as gpd
import os
from pathlib import Path
import json
from loguru import logger

from worldcereal.cib.labels import TrainingPoint, TrainingPolygon, TrainingMap
from worldcereal.utils.spark import get_spark_context


def read_ref_tile(ref_tile_file, s2tile):
    logger.info(f'Loading: {ref_tile_file}')
    ref_tile_gdf = gpd.read_file(ref_tile_file)
    logger.success('Done!')

    return ref_tile_gdf


def check_exists(gdf_samples, location_id, labeltype, contenttype,
                 resolution, outdir):
    sample = gdf_samples[gdf_samples['location_id'] == location_id]
    tile = sample['tile'].values[0]
    epsg = sample['epsg'].values[0]
    start_date = sample['start_date'].values[0]
    end_date = sample['end_date'].values[0]

    prefix = f'OUTPUT_{labeltype}-{contenttype}'

    name = (f"{prefix}_{resolution}"
            f"m_{location_id}")

    ds_basename = (name
                   + f'_{epsg}'
                   + f'_{start_date}'
                   + f'_{end_date}'
                   + '.nc')

    filename = os.path.join(outdir,
                            str(epsg),
                            tile,
                            location_id,
                            ds_basename)

    if os.path.exists(filename):
        return True
    else:
        return False


def _process_sample(TrainingLabel,
                    gdf_samples,
                    ref_tile_files,
                    location_id,
                    patchsize,
                    contenttype,
                    labeltype,
                    outdir,
                    lc_fillvalue=None,
                    tifffile=None,
                    overwrite=False):
    logger.info(f'Processing sample {location_id} ...')

    if check_exists(gdf_samples, location_id, labeltype, contenttype,
                    10, outdir) and not overwrite:
        logger.warning('Output file exists -> skipping')

        return

    else:
        if labeltype in ['POINT', 'POLY']:

            ref_tile_file = ref_tile_files[
                gdf_samples.set_index(
                    'location_id').loc[location_id].tile]

            logger.info(f'Loading: {ref_tile_file}')
            gdf_full = gpd.read_file(ref_tile_file)
            logger.info('Done!')

            result = TrainingLabel.from_gdf(
                gdf_samples,
                gdf_full,
                location_id,
                patchsize,
                contenttype,
                lc_fillvalue=lc_fillvalue,
                overwrite=overwrite)
            if result is not None:
                result.to_netcdf(outdir)
        elif labeltype == 'MAP':
            TrainingLabel.from_tiff(
                gdf_samples,
                tifffile,
                location_id,
                patchsize,
                contenttype,
                overwrite=overwrite).to_netcdf(outdir)
        return


def process_samples(TrainingLabel, gdf_samples, patchsize, contenttype,
                    labeltype, outdir, location_ids,
                    spark, lc_fillvalue=None, tifffile=None,
                    ref_tile_files=None, sc=None, overwrite=False):

    # Loop over all samples
    if spark:
        logger.info('Processing samples in parallel on spark ...')
        sc.parallelize(location_ids,
                       len(location_ids)).foreach(
            lambda location_id: _process_sample(TrainingLabel,
                                                gdf_samples,
                                                ref_tile_files,
                                                location_id,
                                                patchsize,
                                                contenttype,
                                                labeltype,
                                                outdir,
                                                lc_fillvalue=lc_fillvalue,
                                                tifffile=tifffile,
                                                overwrite=overwrite))

    else:
        logger.info('Processing samples in serial ...')
        for location_id in location_ids:
            logger.info(f'Processing sample {location_id}...')
            _process_sample(TrainingLabel,
                            gdf_samples,
                            ref_tile_files,
                            location_id,
                            patchsize,
                            contenttype,
                            labeltype,
                            outdir,
                            lc_fillvalue=lc_fillvalue,
                            tifffile=tifffile,
                            overwrite=overwrite)


def process_dataset(settings, sc, overwrite=False):

    # get settings
    labeltype = settings['labeltype']
    contenttype = settings['contenttype']
    ref_id = settings['ref_id']
    year = settings['year']
    cib_experiment_id = settings['cib_experiment_id']
    lc_fillvalue = settings.get('lcfill', 0)
    tifffile = settings.get('tifffile', None)

    # Infer the input file
    reffile = (f'/data/worldcereal/data/ref/VITO_processed/{labeltype}/'
               f'{contenttype}/{year}/{ref_id}_'
               f'{labeltype}_{contenttype}/'
               f'{ref_id}_{labeltype}_{contenttype}.shp')
    if not Path(reffile).is_file():
        reffile = reffile.replace('shp', 'gpkg')
        if not Path(reffile).is_file():
            reffile = reffile.replace('gpkg', 'shp')
            raise ValueError(f'File `{reffile}` not found!')
    outdir = (f'/data/worldcereal/cib/{cib_experiment_id}'
              f'/{labeltype}/{ref_id}')
    file = str(Path(outdir) / (Path(reffile).stem + '_samples.json'))

    logger.info(f'Reading file: {file}...')
    gdf_samples = gpd.read_file(file)

    # Choose the correct class to start from
    if labeltype == 'POINT':
        TrainingLabel = TrainingPoint
    elif labeltype == 'POLY':
        TrainingLabel = TrainingPolygon
    elif labeltype == 'MAP':
        TrainingLabel = TrainingMap
    else:
        raise Exception(f'Unsupported labeltype: {labeltype}')

    patchsize = 64 if labeltype != 'POINT' else 4
    location_ids = gdf_samples.location_id.to_list()

    # Set output dir
    outdir = str(Path(file).parent / 'data')

    ###################################################
    # THE MAP TRACK
    if labeltype == 'MAP':
        if tifffile is None:
            raise ValueError(('`tifffile` cannot be None'
                              ' when creating MAP samples.'))

        process_samples(TrainingLabel, gdf_samples, patchsize, contenttype,
                        labeltype, outdir, location_ids,
                        spark, tifffile=tifffile,
                        sc=sc, overwrite=overwrite)

    else:

        ###################################################
        # THE VECTOR TRACK

        if labeltype in ['POINT', 'POLY']:
            # Check if the required corresponding ref tiles
            # exist and load them as well
            uniquetiles = list(set(gdf_samples['tile'].to_list()))
            ref_tile_files = {}
            for tile in uniquetiles:
                ref_tile_file = (Path(file).parent / 'ref_tiles' /
                                 (Path(file).stem.replace(
                                     'samples', tile) + '.json'))
                if ref_tile_file.is_file():
                    logger.info(f'Found: {ref_tile_file}')
                    ref_tile_files[tile] = str(ref_tile_file)
                else:
                    raise Exception(('The required file '
                                     f'{ref_tile_file} was not found!'))

            logger.info('-'*50)
            process_samples(TrainingLabel, gdf_samples, patchsize, contenttype,
                            labeltype, outdir, location_ids,
                            spark, lc_fillvalue=lc_fillvalue,
                            ref_tile_files=ref_tile_files, sc=sc,
                            overwrite=overwrite)


def main(datasets, spark=False, overwrite=False):

    if spark:
        logger.info('Setting up spark ...')
        sc = get_spark_context()
    else:
        sc = None

    i = 1
    for dataset, settings in datasets.items():
        logger.info(f'Processing {i} out of {len(datasets)}')
        process_dataset(settings, sc,
                        overwrite=overwrite)
        i += 1


if __name__ == '__main__':

    spark = True
    overwrite = True

    # Load the list of datasets
    dataset_file = '/data/worldcereal/data/ref/VITO_processed/datasets_preprocess_trainingpoints_new.json'  # NOQA
    if not Path(dataset_file).is_file():
        raise FileNotFoundError(
            'Required datasets file '
            f'`{dataset_file}` not found. Cannot continue.')
    datasets = json.load(open(dataset_file, 'r'))

    # Run the main program
    main(datasets, spark=spark, overwrite=overwrite)
