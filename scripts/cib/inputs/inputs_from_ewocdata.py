import os
import argparse

from loguru import logger
import pandas as pd
import xarray as xr
import numpy as np
from rasterio.crs import CRS
import geopandas as gpd
from pathlib import Path

from worldcereal.worldcereal_products import get_worldcereal_collections


_BASE_DIR = '/data/worldcereal/cib'


def data_to_xarr(data, prefix, source='worldcereal'):

    arr = data['data']
    location_id = data['location_id']
    bounds = eval(data['bounds'])
    resolution = data['resolution']

    xmin, ymin, xmax, ymax = bounds

    attrs = {a: data[a] for a in ['location_id', 'epsg',
                                  'resolution',
                                  'bounds',
                                  'start_date', 'end_date',
                                  'tile']}
    attrs['source'] = source

    attrs['crs'] = CRS.from_epsg(data['epsg']).to_proj4()

    dims = ['band', 'timestamp', 'y', 'x']
    dims = {k: arr.shape[i] for i, k in enumerate(dims)}

    resolution = data['resolution']

    xmin, ymin, xmax, ymax = bounds

    if arr.shape[2] > 1:

        x = np.linspace(xmin + resolution / 2,
                        xmax - resolution / 2,
                        dims['x'])

        y = np.linspace(ymax - resolution / 2,
                        ymin + resolution / 2,
                        dims['y'])

    else:
        # Special case where we have only one pixel
        # Set the center coordinate
        x = [xmin + (xmax - xmin) / 2]
        y = [ymin + (ymax - ymin) / 2]

    timestamps = data['timestamps']
    bands = data['bands']
    coords = {'band': bands,
              'timestamp': timestamps,
              'x': x,
              'y': y}

    da = xr.DataArray(arr,
                      coords=coords,
                      dims=dims,
                      attrs=attrs)

    # Some things we need to do to make GDAL
    # and other software recognize the CRS
    # cfr: https://github.com/pydata/xarray/issues/2288
    da.coords['spatial_ref'] = 0
    da.coords['spatial_ref'].attrs['spatial_ref'] = CRS.from_epsg(
        data['epsg']).wkt
    da.coords['spatial_ref'].attrs['crs_wkt'] = CRS.from_epsg(
        data['epsg']).wkt
    da.attrs['grid_mapping'] = 'spatial_ref'

    # Now we convert DataArray to Dataset to set
    # band-specific metadata
    ds = da.to_dataset(dim='band')
    ds.attrs['name'] = f"{prefix}_{resolution}m_{location_id}"
    ds.attrs['grid_mapping'] = 'spatial_ref'

    # Attributes of DataArray were copied to Dataset at global level
    # we need them in the individual DataArrays of the new Dataset as well
    for band in ds.data_vars:
        ds[band].attrs = ds.attrs
        ds[band].attrs['grid_mapping'] = 'spatial_ref'

    # Apply scale factor
    # ATTENTION: at this point, the data is still in Float32, which
    # is needed for proper scaling. By constructing the proper
    # NetCDF4 encoding, the data will be scaled again and stored
    # in the desired dtype. Upon loading, Xarray will recognize the
    # encoding and unscale again to float32.
    # Note that by passing 'mask_and_scale=False' or
    # 'decode_cf=False', you can instruct Xarray to leave the
    # compressed values untouched, if you like to work on int16 e.g.

    for band in ds.data_vars:
        ds[band].values = ds[band]*data['bands_scaling'][band]

    # Construct the NetCDF4 encoding
    # TODO: check if we need to define _FillValue
    encoding = dict()
    for band in data['bands']:
        if type(data['bands_dtype']) is str:
            dtype = data['bands_dtype']
        else:
            dtype = data['bands_dtype'][band]
        encoding[band] = dict(dtype=dtype,
                              scale_factor=data['bands_scaling'][band])

    # Set the encoding for the dataset
    ds.encoding = encoding

    return ds


def check_exists(sample, cib_experiment_id, sensor_id, resolution):
    '''
    Function to check if a decoded file was already
    created
    '''
    tile = sample['tile']
    epsg = sample['epsg']
    start_date = sample['start_date']
    end_date = sample['end_date']
    location_id = sample['location_id']
    labeltype = sample['labeltype']
    ref_id = sample['ref_id']

    decoded_folder = (f'{_BASE_DIR}/{cib_experiment_id}'
                      f'/{labeltype}/{ref_id}/data')

    prefix = f'{sensor_id}_{resolution}m_{location_id}'

    ds_basename = (prefix
                   + f'_{epsg}'
                   + f'_{start_date}'
                   + f'_{end_date}'
                   + '.nc')

    filename = os.path.join(decoded_folder,
                            str(epsg),
                            tile,
                            location_id,
                            ds_basename)

    if os.path.exists(filename):
        return True
    else:
        return False


def to_file(dataset, filename, encoding):
    if os.path.exists(filename):
        os.remove(filename)
    # Write to file
    # NOTE: when using default netcdf4 engine, it appears
    # the resulting .nc files are sometimes corrupt
    # this has not been observed with h5netcdf engine (yet)
    dataset.to_netcdf(filename, encoding=encoding,
                      engine='h5netcdf')


def decode_to_file(data, prefix, outfolder):
    ds = data_to_xarr(data, prefix)

    # Check if we actually have any timestamps
    # which is very rarely not the case
    if len(ds['timestamp']) == 0:
        logger.warning(('No timestamps in dataset!'
                        ' Skipping file.'))
        return

    # Check if the dates cover the requested start and end
    start = pd.to_datetime(ds.timestamp.values[0])
    end = pd.to_datetime(ds.timestamp.values[-1])
    if ((start - pd.to_datetime(ds.start_date))
            > pd.Timedelta(days=60)):
        logger.error((f'Sample only starts at '
                      f'{start} while stack should start '
                      f' at {ds.start_date}!'))
        return
    elif ((pd.to_datetime(ds.end_date) - end)
            > pd.Timedelta(days=60)):
        logger.error((f'Sample ends at '
                      f'{end} while stack should end'
                      f' at {ds.end_date}!'))
        return

    # Cut out the requested start and end
    # for this particular sample
    ds = ds.sel(timestamp=slice(ds.start_date, ds.end_date))

    ds_basename = (ds.attrs['name']
                   + f'_{ds.epsg}'
                   + f'_{ds.start_date}'
                   + f'_{ds.end_date}'
                   + '.nc')

    filename = (Path(outfolder) / str(ds.epsg) /
                ds.tile / ds.location_id / ds_basename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    # Set _FillValue
    for band in ds.encoding.keys():
        ds.encoding[band]['_FillValue'] = 0

    # Get the encoding with renamed band values
    encoding = ds.encoding

    # Write to encoded NetCDF
    try:
        logger.info(f'Writing to: {filename}')
        to_file(ds, filename, encoding)
    except Exception:
        # If we end up here, it means we couldn't
        # write successfully after a few attempts
        # Remove the bad file if it exists
        if os.path.exists(filename):
            os.remove(filename)
        logger.error(f'Could not write to file: {filename}')
    finally:
        ds.close()


def _process_optical(row, collection, output_folder, cib_experiment_id):

    bounds = eval(row.bounds)
    epsg = row.epsg
    tile = row.tile

    collection = collection.filter_bounds(bounds, epsg).filter_tiles(tile)

    if collection.df.shape[0] == 0:
        logger.warning('Emppty collection -> skipping point!')
        return

    for resolution in [10, 20]:

        if check_exists(row, cib_experiment_id, 'S2_L2A', resolution):
            logger.info('File exists -> skipping')
            continue

        logger.info(f'Working on OPTICAL resolution: {resolution}m')

        if resolution == 10:
            bands = ['B02', 'B03', 'B04', 'B08']
            bands_scaling = dict.fromkeys(bands, 0.0001)
            bands_dtype = dict.fromkeys(bands, 'uint16')
        elif resolution == 20:
            bands = ['B05', 'B06', 'B07', 'B11', 'B12', 'MASK']
            bands_scaling = dict.fromkeys(bands, 0.0001)
            bands_dtype = dict.fromkeys(bands, 'uint16')
            bands_scaling['SCL'] = 1  # This band is not scaled!
            bands_dtype['SCL'] = 'uint8'  # This band is different!

        logger.info('Loading timeseries ...')
        data = collection.load_timeseries(*bands).to_xarray()

        validacquisitions = list(data.time[(data.sum(
            axis=(2, 3))[0, :] != 0)].time.values)
        data = data.sel(time=validacquisitions)

        if resolution == 20:
            # Need to create a dummy SCL mask
            data_ds = data.to_dataset('bands')
            scl = data_ds['MASK'].values
            scl[scl == 1] = 4  # Valid data
            scl[scl == 0] = 1  # Invalid data
            scl[scl == 255] = 0  # No data
            data_ds['MASK'].values = scl
            data_ds = data_ds.rename({'MASK': 'SCL'})
            data = data_ds.to_array(dim='bands')

        timestamps = [pd.Timestamp(x).to_pydatetime()
                      for x in data.time.values]

        # Wrap in dictionary
        data = dict(
            data=data.values,
            bands=list(data.bands.values),
            location_id=row.location_id,
            bounds=row.bounds,
            resolution=resolution,
            epsg=row.epsg,
            start_date=row.start_date,
            end_date=row.end_date,
            tile=row.tile,
            timestamps=timestamps,
            bands_scaling=bands_scaling,
            bands_dtype=bands_dtype
        )

        prefix = 'S2_L2A'

        decode_to_file(data, prefix, output_folder)


def _process_sar(row, collection, output_folder, cib_experiment_id):

    bounds = eval(row.bounds)
    epsg = row.epsg
    tile = row.tile
    resolution = 20

    collection = collection.filter_bounds(bounds, epsg).filter_tiles(tile)
    if collection.df.shape[0] == 0:
        logger.warning('Emppty collection -> skipping point!')
        return

    ex_file = collection.df.path.iloc[0]
    orbit = Path(ex_file).name.split('_')[2]
    if orbit == 'ASC':
        prefix = 'S1_GRD-ASCENDING'
    elif orbit == 'DES':
        prefix = 'S1_GRD-DESCENDING'
    else:
        raise ValueError(f'Could not derive S1 orbit for: {ex_file}')

    if check_exists(row, cib_experiment_id, prefix, resolution):
        logger.info('File exists -> skipping')
        return

    logger.info(f'Working on SAR resolution: {resolution}m')

    bands = ['VV', 'VH']
    bands_scaling = dict.fromkeys(bands, 0.0001)
    bands_dtype = dict.fromkeys(bands, 'int32')

    logger.info('Loading timeseries ...')
    data = collection.load_timeseries(*bands).to_xarray()

    data = data.astype(np.float32)

    # Remove nodata
    nodata = 65535
    validacquisitions = list(data.time[((data == nodata).sum(
        axis=(2, 3))[0, :] == 0)].time.values)
    data = data.sel(time=validacquisitions)

    '''
    OTB-processed backscatter puts all pixels that have
    backscatter below the noise level to "0". This is different
    from SNAP which is what the system is trained on.
    We need to be remove those 0 values with a HACK, by assigning
    low backscatter values to these 0 pixels.
    '''

    # Identify below-noise pixels
    idx_belownoise = data.values == 0

    # Get random low number between DN 50-300 (~[-50dB, -33dB])
    logger.warning('Introducing random noise [hack]')
    random_low = np.random.randint(50, 300, size=idx_belownoise.sum())

    # Replace below noise with low value
    data.values[idx_belownoise] = random_low

    # Remove product-specific scaling towards true dB
    data.values = 20 * np.log10(data.values) - 83

    # And then reintroduce with GEE-scaling in Int32
    data.values = data.values * 10000
    data = data.astype(np.int32)

    # Get the timestamps
    timestamps = [pd.Timestamp(x).to_pydatetime()
                  for x in data.time.values]

    # Wrap in dictionary
    data = dict(
        data=data.values,
        bands=list(data.bands.values),
        location_id=row.location_id,
        bounds=row.bounds,
        resolution=resolution,
        epsg=row.epsg,
        start_date=row.start_date,
        end_date=row.end_date,
        tile=row.tile,
        timestamps=timestamps,
        bands_scaling=bands_scaling,
        bands_dtype=bands_dtype
    )

    decode_to_file(data, prefix, output_folder)


def process_sample(gdf_samples,
                   location_id,
                   collections,
                   output_folder,
                   cib_experiment_id,
                   ):

    sample = gdf_samples[gdf_samples['location_id'] == location_id].iloc[0, :]

    _process_optical(sample, collections['OPTICAL'], output_folder,
                     cib_experiment_id)
    _process_sar(sample, collections['SAR'], output_folder,
                 cib_experiment_id)


def main(samples_file, cib_experiment_id,
         sc=None):

    inputpaths = {
        "OPTICAL": "/data/worldcereal_data/collections/OPTICAL.csv",  # NOQA
        "SAR": "/data/worldcereal_data/collections/SAR.csv",  # NOQA
        "TIR": "/data/worldcereal_data/collections/TIR.csv",  # NOQA
        "DEM": "/data/MEP/DEM/COP-DEM_GLO-30_DTED/S2grid_20m",
        "METEO": "/data/MTDA/AgERA5"
    }

    # Create the collections
    collections, _ = get_worldcereal_collections(inputpaths, False, False)

    ref_id = '_'.join(Path(samples_file).name.split('_')[:3])
    labeltype = Path(samples_file).name.split('_')[3]

    # Output folder
    output_folder = (f'{_BASE_DIR}/{cib_experiment_id}'
                     f'/{labeltype}/{ref_id}/data')
    os.makedirs(output_folder, exist_ok=True)

    logger.info(f"Initializing training data extraction of {samples_file}")

    # Load the samples
    gdf = gpd.read_file(samples_file)
    logger.info(f'File has {gdf.shape[0]} samples!')

    # Get the list of samples
    location_ids = gdf.location_id.to_list()

    # location_ids = ['2019_TZA_EXTRAPOINTS_222359']

    if sc is not None:
        logger.info('Processing samples in parallel on spark ...')
        sc.parallelize(location_ids,
                       len(location_ids)).foreach(
            lambda location_id: process_sample(
                gdf,
                location_id,
                collections=collections,
                output_folder=output_folder,
                cib_experiment_id=cib_experiment_id)
        )

    else:
        logger.info('Processing samples in serial ...')
        for location_id in location_ids:
            process_sample(
                gdf,
                location_id,
                collections=collections,
                output_folder=output_folder,
                cib_experiment_id=cib_experiment_id)

    logger.success('Done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True,
                        help='Input GeoJSON file to process')
    parser.add_argument('--cib', type=str, required=True,
                        help='CIB experiment ID')
    parser.add_argument('-s', '--spark',
                        action="store_true",  help='Run on spark')

    args = parser.parse_args()

    # Get the parameters
    file = args.file
    cib_experiment_id = args.cib
    spark = args.spark

    # # Get the parameters
    # file = '/data/worldcereal/cib/CIB_V1/POINT/2021_UKR_sunflowermap/2021_UKR_sunflowermap_POINT_110_samples.json'  # NOQA
    # cib_experiment_id = 'CIB_V1'
    # spark = False

    # Log some information for this run
    logger.info('-'*80)
    logger.info('USING FOLLOWING SETTINGS:')
    logger.info(f'file: {file}')
    logger.info(f'cib_experiment_id: {cib_experiment_id}')
    logger.info('-'*80)

    if spark:
        import pyspark
        import cloudpickle
        pyspark.serializers.cloudpickle = cloudpickle
        sc = pyspark.SparkContext()
    else:
        sc = None

    main(file, cib_experiment_id, sc)
