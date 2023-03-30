import geopandas as gpd
import os
import json
from pathlib import Path
from loguru import logger

from worldcereal.cib.inputs import AgERA5InputPatch
from worldcereal.utils.spark import get_spark_context


def check_exists(gdf_samples, location_id,
                 resolution, outdir):
    sample = gdf_samples[gdf_samples['location_id'] == location_id]
    ref_id = sample['ref_id'].values[0]
    labeltype = sample['labeltype'].values[0]
    tile = sample['tile'].values[0]
    epsg = sample['epsg'].values[0]
    start_date = sample['start_date'].values[0]
    end_date = sample['end_date'].values[0]

    prefix = 'AgERA5_DAILY'

    name = (f"{prefix}_{resolution}"
            f"m_{location_id}")

    ds_basename = (name
                   + f'_{epsg}'
                   + f'_{start_date}'
                   + f'_{end_date}'
                   + '.nc')

    filename = os.path.join(outdir,
                            labeltype,
                            ref_id,
                            'data',
                            str(epsg),
                            tile,
                            location_id,
                            ds_basename)

    if os.path.exists(filename):
        return True
    else:
        return False


def process_sample(gdf_samples,
                   location_id,
                   agera5colldf,
                   outdir,
                   overwrite=False):

    logger.info(f'Processing sample {location_id} ...')

    if check_exists(gdf_samples, location_id,
                    1000, outdir) and not overwrite:
        logger.warning('Output file exists -> skipping')

        return

    else:
        logger.info('Extracting AgERA5 patches ...')

        try:
            inputpatch = AgERA5InputPatch.from_folder(
                agera5colldf,
                gdf_samples,
                location_id,
                overwrite=overwrite)

            logger.info('Writing to output file ...')
            outdir = outdir / inputpatch.labeltype / inputpatch.ref_id / 'data'
            inputpatch.to_netcdf(outdir)
        except Exception:
            logger.error('Got an error: file was not created.')
            raise

        return


def proces_dataset(file, outdir, sc=None,
                   agera5colldf=None, overwrite=False):

    agera5colldf = agera5colldf or '/data/MTDA/AgERA5/'

    logger.info(f'Reading file: {file}...')
    gdf_samples = gpd.read_file(file)

    # location_id = '10898463'
    location_ids = gdf_samples.location_id.to_list()

    # Loop over all samples
    if sc is not None:
        logger.info('Processing samples in parallel on spark ...')
        sc.parallelize(location_ids,
                       len(location_ids)).foreach(
            lambda location_id: process_sample(gdf_samples,
                                               location_id,
                                               agera5colldf=agera5colldf,
                                               outdir=outdir,
                                               overwrite=overwrite))

    else:
        logger.info('Processing samples in serial ...')
        for location_id in location_ids:
            process_sample(gdf_samples,
                           location_id,
                           agera5colldf=agera5colldf,
                           outdir=outdir,
                           overwrite=overwrite)


def main(task_file, cib_experiment_id, agera5colldf,
         spark=False, overwrite=False):

    if spark:
        logger.info('Setting up spark ...')
        sc = get_spark_context()
    else:
        sc = None

    # get list of tasks
    if not Path(task_file).is_file():
        raise FileNotFoundError(
            'Required tasklist file '
            f'`{task_file}` not found. Cannot continue.')
    task_list = json.load(open(task_file, 'r'))

    logger.info(f'Starting {len(task_list)} tasks!')

    outdir = Path('/data/worldcereal/cib') / cib_experiment_id

    for file, settings in task_list.items():

        # check if it already has been done
        status = settings.get('status', None)
        if status is None:

            logger.info(f'Now processing {file}')

            labeltype = file.split('_')[-2]
            contenttype = file.split('_')[-1]
            ref_id = '_'.join(file.split('_')[:-2])
            logger.info(f'Labeltype = {labeltype}')
            logger.info(f'Contenttype = {contenttype}')
            logger.info(f'Ref_id = {ref_id}')

            infile = (f'/data/worldcereal/cib/{cib_experiment_id}/'
                      f'{labeltype}/{ref_id}/{file}_samples.json')

            proces_dataset(infile, outdir, sc=sc,
                           agera5colldf=agera5colldf,
                           overwrite=overwrite)

            # writing to task list
            task_list[file]['status'] = 'done'
            with open(task_file, 'w') as fp:
                json.dump(task_list, fp, indent=4)

            logger.success(f'Task {file} done!')

        else:
            logger.info('-'*50)
            logger.info(f'Skipping task: {file}')

    logger.success('All tasks done!')


if __name__ == '__main__':

    # some settings
    spark = True
    overwrite = False
    cib_experiment_id = "CIB_V1"
    # Set the input yearly AgERA5 collection DF
    agera5colldf = '/data/worldcereal/tmp/kristof/collections/satio_agera5_yearly.csv'

    # list of tasks
    task_file = '/data/worldcereal/data/ref/VITO_processed/agera5_tasklist_new.json'  # NOQA

    # Run the main program
    main(task_file, cib_experiment_id, agera5colldf,
         spark=spark, overwrite=overwrite)
