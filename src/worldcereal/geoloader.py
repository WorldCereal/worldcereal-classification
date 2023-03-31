import configparser
import os
from pathlib import Path

from loguru import logger
import numpy as np
import rasterio
from satio.geoloader import (S3ParallelLoader, S3LatLonReprojectLoader,
                             S3LandsatLoader, S3WarpLoader,
                             ParallelLoader, LatLonReprojectLoader,
                             LandsatLoader, WarpLoader)
from satio.utils.retry import retry

NR_THREADS = int(os.environ.get('SATIO_MAXTHREADS', 5))
RETRIES = int(os.environ.get('SATIO_RETRIES', 50))
DELAY = int(os.environ.get('SATIO_DELAY', 5))
BACKOFF = int(os.environ.get('SATIO_BACKOFF', 1))
TIMEOUT = int(os.environ.get('SATIO_TIMEOUT', 30))

EWOC_RIO_GDAL_OPTIONS = {
    'AWS_VIRTUAL_HOSTING': False,
    'AWS_REQUEST_PAYER': 'requester',
    'GDAL_DISABLE_READDIR_ON_OPEN': 'FALSE',
    'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
    'VSI_CACHE': False
}

EWOC_REGION_NAME = os.environ.get("EWOC_REGION_NAME",
                                  'RegionOne')
EWOC_ENDPOINT_URL = os.environ.get("EWOC_ENDPOINT_URL",
                                   "cf2.cloudferro.com:8080")


class EWOCS3ParallelLoader(S3ParallelLoader):
    def __init__(self,
                 region_name=EWOC_REGION_NAME,
                 endpoint_url=EWOC_ENDPOINT_URL,
                 max_workers=NR_THREADS,
                 **kwargs):

        super().__init__(region_name=region_name,
                         endpoint_url=endpoint_url,
                         rio_gdal_options=EWOC_RIO_GDAL_OPTIONS,
                         max_workers=max_workers,
                         **kwargs)


class EWOCS3WarpLoader(S3WarpLoader):
    def __init__(self,
                 region_name=EWOC_REGION_NAME,
                 endpoint_url=EWOC_ENDPOINT_URL,
                 max_workers=NR_THREADS,
                 progressbar=True,
                 **kwargs):

        super().__init__(region_name=region_name,
                         endpoint_url=endpoint_url,
                         rio_gdal_options=EWOC_RIO_GDAL_OPTIONS,
                         max_workers=max_workers,
                         progressbar=progressbar,
                         **kwargs)


class EWOCS3LatLonReprojectLoader(S3LatLonReprojectLoader):
    def __init__(self,
                 region_name=EWOC_REGION_NAME,
                 endpoint_url=EWOC_ENDPOINT_URL,
                 max_workers=NR_THREADS,
                 **kwargs):

        super().__init__(region_name=region_name,
                         endpoint_url=endpoint_url,
                         rio_gdal_options=EWOC_RIO_GDAL_OPTIONS,
                         max_workers=max_workers,
                         **kwargs)


class EWOCS3LandsatLoader(S3LandsatLoader):
    def __init__(self,
                 region_name=EWOC_REGION_NAME,
                 endpoint_url=EWOC_ENDPOINT_URL,
                 max_workers=NR_THREADS,
                 **kwargs):

        super().__init__(region_name=region_name,
                         endpoint_url=endpoint_url,
                         rio_gdal_options=EWOC_RIO_GDAL_OPTIONS,
                         max_workers=max_workers,
                         **kwargs)


class EWOCGDALWarpLoader:

    def __init__(self, **gdal_options):
        self._gdal_options = gdal_options or {}

    @retry(exceptions=Exception, tries=RETRIES, delay=DELAY,
           backoff=BACKOFF, logger=logger)
    def _warp(self, fn, bounds, epsg, resolution,
              resampling='cubic'):
        import tempfile

        logger.debug(f'Start loading of: {fn}')

        load_resolution = min([resolution,
                               bounds[2] - bounds[0],
                               bounds[3] - bounds[1]])

        if load_resolution != resolution:
            logger.debug((f'Loading at {load_resolution}m resolution '
                          'due to small bounds.'))

        with tempfile.NamedTemporaryFile() as tmp:
            self.gdal_warp([fn],
                           tmp.name,
                           bounds,
                           dst_epsg=epsg,
                           resolution=load_resolution,
                           resampling=resampling)
            with rasterio.open(tmp.name) as src:
                arr = src.read()

        logger.debug(f'Loading of {fn} completed.')

        return arr

    def gdal_warp(self,
                  src_fnames,
                  dst_fname,
                  bounds,
                  dst_epsg,
                  resolution=100,
                  center_long=0,
                  gdal_cachemax=2000,
                  resampling='cubic'):

        import sys
        import subprocess

        # First try system gdal
        bin = '/bin/gdalwarp'
        if not Path(bin).is_file():
            # Try environment gdal
            py = sys.executable.split('/')[-1]
            bin = sys.executable.replace(py, 'gdalwarp')
            if not Path(bin).is_file():
                raise FileNotFoundError(
                    'Could not find a GDAL installation.')

        if isinstance(src_fnames, str):
            src_fnames = [src_fnames]

        fns = " ".join(list(map(str, src_fnames)))
        str_bounds = " ".join(list(map(str, bounds)))

        env_vars_str = " ".join([f'{k}={v}' for
                                 k, v in self._gdal_options.items()])

        cmd = (
            f"{env_vars_str} "
            f"{bin} -of GTiff "
            f"-t_srs EPSG:{dst_epsg} "
            f"-te {str_bounds} "
            f"-tr {resolution} {resolution} -multi "
            f"-r {resampling} "
            f"--config CENTER_LONG {center_long} "
            f"--config GDAL_CACHEMAX {gdal_cachemax} "
            f"-co COMPRESS=DEFLATE "
            f"{fns} "
            f"{dst_fname}"
        )

        p = subprocess.run(cmd, shell=True, timeout=TIMEOUT)
        if p.returncode != 0:
            raise IOError("GDAL warping failed")
        else:
            logger.debug(f'Warping of {fns} completed.')

    def _dates_interval(self, start_date, end_date):
        import datetime
        days = [start_date + datetime.timedelta(days=x)
                for x in range((end_date - start_date).days)]
        return days

    def _yearly_dates(self, year):
        import datetime

        d1 = datetime.datetime(year, 1, 1)
        d2 = datetime.datetime(year + 1, 1, 1)

        return self._dates_interval(d1, d2)

    def load(self, collection, bands, resolution,
             src_nodata=None, dst_nodata=None,
             resampling='cubic'):
        from satio.timeseries import Timeseries
        from dateutil.parser import parse

        if len(bands) > 1:
            # can load only 1 band per time
            raise NotImplementedError

        if not isinstance(bands, (list, tuple)):
            raise TypeError("'bands' should be a list/tuple of bands. "
                            f"Its type is: {type(bands)}")

        band = bands[0]
        dst_bounds = list(collection.bounds)
        dst_epsg = collection.epsg
        # these are the bounds and epsg requested for the final data
        # we need to check the epsgs of source data and get the filenames

        filenames = collection.get_band_filenames(band)
        start_date = parse(collection.start_date)
        end_date = parse(collection.end_date)

        arr = None
        timestamps = None

        for fn in filenames:
            arr_tmp = self._warp(fn,
                                 dst_bounds,
                                 dst_epsg,
                                 resolution,
                                 resampling=resampling)

            year = int(Path(fn).name.split('.')[0].split('_')[-1])
            ts_tmp = self._yearly_dates(year)

            if arr is None:
                arr = arr_tmp
                timestamps = ts_tmp
            else:
                arr = np.concatenate([arr, arr_tmp], axis=0)
                timestamps += ts_tmp

        # filter dates
        filtered_timestamps = self._dates_interval(start_date,
                                                   end_date)
        ts_arr = np.array(timestamps)
        time_flag = (ts_arr >= start_date) & (ts_arr < end_date)
        arr = arr[time_flag, ...]
        arr = np.expand_dims(arr, axis=0)

        ts = Timeseries(arr, filtered_timestamps, bands)
        ts.attrs['sensor'] = collection.sensor

        return ts


class EWOCS3GDALWarpLoader(EWOCGDALWarpLoader):
    def __init__(self,
                 region_name=EWOC_REGION_NAME,
                 endpoint_url=EWOC_ENDPOINT_URL,
                 access_key_id=None,
                 secret_access_key=None,
                 **kwargs):

        if access_key_id is None:
            access_key_id, secret_access_key = _get_s3_keys()

        gdal_options = {
            'AWS_ACCESS_KEY_ID': access_key_id,
            'AWS_SECRET_ACCESS_KEY': secret_access_key,
            'AWS_REGION': region_name,
            'AWS_S3_ENDPOINT': endpoint_url,
            'AWS_VIRTUAL_HOSTING': 'FALSE',
            'AWS_REQUEST_PAYER': 'requester',
            'GDAL_DISABLE_READDIR_ON_OPEN': 'FALSE',
            'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
            'VSI_CACHE': 'FALSE'
        }

        super().__init__(**gdal_options)


def _get_s3_keys():
    # Try to get keys from environment
    try:
        CLF_SECRET_ACCESS_KEY = os.environ['EWOC_S3_SECRET_ACCESS_KEY']
        CLF_ACCESS_KEY_ID = os.environ['EWOC_S3_ACCESS_KEY_ID']
    except KeyError:
        raise RuntimeError('Could not retrieve s3 access/secret keys!')

    return CLF_ACCESS_KEY_ID, CLF_SECRET_ACCESS_KEY


def make_s3_collection(coll,
                       access_key_id=None,
                       secret_access_key=None):
    """Transform an ordinary input collection to a
    S3 collection that will load directly from bucket

    Args:
        coll (satio Collection): input collection to make s3 compatible
        aws_access_key_id (str, optional): optional access key to use.
        aws_secret_access_key (str, optional): optional secret key to use.

    """
    if access_key_id is None or secret_access_key is None:
        CLF_ACCESS_KEY_ID, CLF_SECRET_ACCESS_KEY = _get_s3_keys()

    if type(coll.loader) == ParallelLoader:
        coll.loader = EWOCS3ParallelLoader(
            access_key_id=CLF_ACCESS_KEY_ID,
            secret_access_key=CLF_SECRET_ACCESS_KEY,
            fill_value=coll.loader._fill_value)
    elif type(coll.loader) == LatLonReprojectLoader:
        coll.loader = EWOCS3LatLonReprojectLoader(
            access_key_id=CLF_ACCESS_KEY_ID,
            secret_access_key=CLF_SECRET_ACCESS_KEY,
            fill_value=coll.loader._fill_value)
    elif type(coll.loader) == LandsatLoader:
        coll.loader = EWOCS3LandsatLoader(
            access_key_id=CLF_ACCESS_KEY_ID,
            secret_access_key=CLF_SECRET_ACCESS_KEY,
            fill_value=coll.loader._fill_value)
    elif type(coll.loader) == WarpLoader:
        coll.loader = EWOCS3WarpLoader(
            access_key_id=CLF_ACCESS_KEY_ID,
            secret_access_key=CLF_SECRET_ACCESS_KEY,
            fill_value=coll.loader._fill_value,
            buffer_bounds=15000)
    elif type(coll.loader) == EWOCGDALWarpLoader:
        coll.loader = EWOCS3GDALWarpLoader(
            access_key_id=CLF_ACCESS_KEY_ID,
            secret_access_key=CLF_SECRET_ACCESS_KEY
        )
    else:
        raise ValueError(('Cannot transform loader of type '
                          f'`{coll.loader}` into S3 alternative.'))

    return coll
