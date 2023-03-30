import os
from pathlib import Path
from subprocess import Popen, PIPE

import geopandas as gpd
from loguru import logger
import numpy as np
import rasterio
from rasterio.crs import CRS
import xarray as xr

from worldcereal.collections import AgERA5YearlyCollection

_AGERA5VARS = [
    'dewpoint_temperature',
    'precipitation_flux',
    'solar_radiation_flux',
    'temperature_min',
    'temperature_max',
    'temperature_mean',
    'vapour_pressure',
    'wind_speed'
]

# How currently AgERA5 is stored on disk
_AGERA5SRCATTRS = {
    'dewpoint_temperature': {
        'scale': 0.01,
        'nodata': 65535
    },
    'precipitation_flux': {
        'scale': 0.01,
        'nodata': 65535
    },
    'solar_radiation_flux': {
        'scale': 1,
        'nodata': 0
    },
    'temperature_min': {
        'scale': 0.01,
        'nodata': 65535
    },
    'temperature_max': {
        'scale': 0.01,
        'nodata': 65535
    },
    'temperature_mean': {
        'scale': 0.01,
        'nodata': 65535
    },
    'vapour_pressure': {
        'scale': 0.001,
        'nodata': 0
    },
    'wind_speed': {
        'scale': 0.01,
        'nodata': 65535
    }
}


_AGERA5ATTRS = {
    'dewpoint_temperature': {
        'units': 'K',
        'nodata': -9999
    },
    'precipitation_flux': {
        'units': 'mm d-1',
        'nodata': -9999
    },
    'solar_radiation_flux': {
        'units': 'J m-2 d-1',
        'nodata': -9999
    },
    'temperature_min': {
        'units': 'K',
        'nodata': -9999
    },
    'temperature_max': {
        'units': 'K',
        'nodata': -9999
    },
    'temperature_mean': {
        'units': 'K',
        'nodata': -9999
    },
    'vapour_pressure': {
        'units': 'hPa',
        'nodata': -9999
    },
    'wind_speed': {
        'units': 'm s-1',
        'nodata': -9999
    }
}

_CROPCALENDARVARS = [
    'WW_SOS',
    'WW_EOS',
    'M1_SOS',
    'M1_EOS',
    'M2_SOS',
    'M2_EOS'
]


def _read_pixel_values(tifffile, lat, lon):
    """A function to read one pixel
        based on a bounds tuple

    Args:
        tifffile (str): Path to tiff file that holds the data
        bounds (tuple): tuple describing the bounds
        patchsize (int): Size of the patch to generate
        epsg (str): EPSG code to work in
    """

    cmd = ['gdallocationinfo', '-wgs84', '-valonly',
           str(tifffile), str(lon), str(lat)]
    p = Popen(cmd, stdout=PIPE)
    p.wait()
    stdout = p.stdout.read().decode('utf-8').split('\n')[:-1]
    values = np.array(stdout).astype(float)

    data = values.reshape((1, 1, -1))

    return data


def _read_raster_patch(tifffile, bounds, patchsize, epsg):
    """A function to read raster patches
        based on a bounds tuple

    Args:
        tifffile (str): Path to tiff file that holds the data
        bounds (tuple): tuple describing the bounds
        patchsize (int): Size of the patch to generate
        epsg (str): EPSG code to work in
    """

    from rasterio.vrt import WarpedVRT

    # Open the source raster file
    with rasterio.open(tifffile) as src:
        # Set a virtual output raster
        with WarpedVRT(src,
                       crs=rasterio.crs.CRS.from_epsg(epsg)
                       ) as vrt:

            # Determine the window to use in reading from the dataset.
            dst_window = vrt.window(bounds[0], bounds[1], bounds[2], bounds[3])

            # Read from input raster to VRT coordinates
            data = vrt.read(window=dst_window, out_shape=(
                src.count, patchsize, patchsize))

    # Need to transpose to [x, y, bands]
    data = data.transpose((1, 2, 0))

    return data


class InputPatch:

    def __init__(self,
                 data,
                 sensor,
                 labeltype=None,
                 resolution=10,
                 patchsize=None,
                 tile=None,
                 epsg=None,
                 bounds=None,
                 location_id=None,
                 start_date=None,
                 end_date=None,
                 source=None,
                 ref_id=None,
                 nodata=0,
                 dtype=np.uint16,
                 attributes=None,
                 overwrite=False):

        if labeltype not in ['POINT', 'POLY', 'MAP']:
            raise ValueError(f'Label type {labeltype} not supported!')

        for datavar in data.keys():
            if 'date_range' in data[datavar]:
                if len(data[datavar]['date_range']) != data[
                        datavar]['patches'].shape[2]:
                    raise ValueError(
                        ('Mismatch between shape of '
                         f'`date_range` ({data[datavar]["date_range"]}) '
                         f' and `data` ({data[datavar]["patches"].shape})'))

        self.data = data
        self.sensor = sensor
        self.resolution = resolution
        self.labeltype = labeltype
        self.NoData = nodata
        self.patchsize = patchsize
        self.tile = tile
        self.epsg = epsg
        self.bounds = bounds
        self.location_id = location_id
        self.start_date = start_date
        self.end_date = end_date
        self.source = source
        self.ref_id = ref_id
        self.dtype = dtype

        self._overwrite = overwrite
        self._prefix = f'{sensor}'
        self.attributes = attributes

    def to_netcdf(self, outdir):

        bands = list(self.data.keys())
        if 'date_range' in self.data[bands[0]]:
            temporal = True
            timestamps = self.data[bands[0]]['date_range']
        else:
            temporal = False
        xdim = self.data[bands[0]]['patches'].shape[0]
        ydim = self.data[bands[0]]['patches'].shape[1]

        if temporal:
            data = np.empty((len(bands), len(timestamps), xdim, ydim))
        else:
            data = np.empty((len(bands), xdim, ydim))

        # Fetch the data
        for i, datavar in enumerate(self.data.keys()):
            currentdata = self.data[datavar]['patches']

            if temporal:
                currentdata = np.expand_dims(
                    currentdata.transpose(2, 0, 1), axis=0)
                data[i, ...] = currentdata
            else:
                data[i, ...] = currentdata

        xmin, ymin, xmax, ymax = self.bounds

        attrs = {a: getattr(self, a) for a in ['location_id', 'epsg',
                                               'resolution',
                                               'bounds',
                                               'start_date', 'end_date',
                                               'tile', 'source']}

        attrs['crs'] = CRS.from_epsg(self.epsg).to_proj4()

        if temporal:
            dims = ['band', 'timestamp', 'y', 'x']
        else:
            dims = ['band', 'y', 'x']
        dims = {k: data.shape[i] for i, k in enumerate(dims)}

        if data.shape[1] > 1:

            x = np.linspace(xmin + self.resolution / 2,
                            xmax - self.resolution / 2,
                            dims['x'])

            # NOTE: y needs to go from ymax to ymin !!
            y = np.linspace(ymax - self.resolution / 2,
                            ymin + self.resolution / 2,
                            dims['y'])

        else:
            # Special case where we have only one pixel
            # Set the center coordinate
            x = [xmin + (xmax - xmin) / 2]
            y = [ymin + (ymax - ymin) / 2]

        if temporal:
            coords = {'band': bands,
                      'timestamp': timestamps,
                      'x': x,
                      'y': y}
        else:
            coords = {'band': bands,
                      'x': x,
                      'y': y}

        da = xr.DataArray(data.astype(self.dtype),
                          coords=coords,
                          dims=dims,
                          attrs=attrs)

        # Some things we need to do to make GDAL
        # and other software recognize the CRS
        # cfr: https://github.com/pydata/xarray/issues/2288
        da.coords['spatial_ref'] = 0
        da.coords['spatial_ref'].attrs['spatial_ref'] = CRS.from_epsg(
            self.epsg).wkt
        da.coords['spatial_ref'].attrs['crs_wkt'] = CRS.from_epsg(
            self.epsg).wkt
        da.attrs['grid_mapping'] = 'spatial_ref'

        # Now we convert DataArray to Dataset to set
        # band-specific metadata
        ds = da.to_dataset(dim='band')
        ds.attrs['name'] = (f"{self._prefix}_{self.resolution}"
                            f"m_{self.location_id}")
        ds.attrs['grid_mapping'] = 'spatial_ref'

        # Attributes of DataArray were copied to Dataset at global level
        # we need them in the individual DataArrays of the new Dataset as well
        # We add band-specific attributes too if needed
        # TODO: apparently the units of one band are still
        # in the attrs of the dataset
        for band in ds.data_vars:
            attrs = ds.attrs
            if self.attributes is not None:
                attrs.update(self.attributes[band])
            ds[band].attrs = attrs
            ds[band].attrs['grid_mapping'] = 'spatial_ref'

        # We need to encode the no data value "0" into the netcdf
        encoding = dict()
        for band in bands:
            if (self.attributes is not None
                    and 'nodata' in self.attributes[band]):
                nodata = self.attributes[band]['nodata']
            else:
                nodata = self.NoData
            encoding[band] = dict(_FillValue=nodata,
                                  dtype=self.dtype)

        ds_basename = (ds.attrs['name']
                       + f'_{self.epsg}'
                       + f'_{self.start_date}'
                       + f'_{self.end_date}'
                       + '.nc')

        filename = os.path.join(outdir,
                                str(ds.epsg),
                                ds.tile,
                                ds.location_id,
                                ds_basename)
        oldmask = os.umask(0o022)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if not os.path.exists(filename) or self._overwrite:
            logger.info(f'Writing to {filename} ...')
            ds.to_netcdf(filename, encoding=encoding, engine='h5netcdf')
        os.umask(oldmask)


class AgERA5InputPatch(InputPatch):

    def __init__(self,
                 *args,
                 **kwargs):

        sensor = 'AgERA5_DAILY'
        dtype = np.float32

        super().__init__(*args,
                         sensor=sensor,
                         dtype=dtype,
                         **kwargs)

    @ classmethod
    def from_folder(cls,
                    folder: str,
                    gdf_samples: gpd.GeoDataFrame,
                    location_id: str,
                    attr_name='location_id',
                    patchsize: int = 1,
                    resolution=1000,
                    variables=_AGERA5VARS,
                    attributes=_AGERA5ATTRS,
                    overwrite=False):

        # Select the sample
        if location_id not in gdf_samples[attr_name].values:
            raise ValueError(f'Sample "{location_id} not found in GDF!')
        sample = gdf_samples[gdf_samples[attr_name] == location_id]

        tile = sample['tile'].values[0]
        epsg = sample['epsg'].values[0]
        if type(sample['bounds'].values[0]) is tuple:
            bounds = sample['bounds'].values[0]
        else:
            bounds = eval(sample['bounds'].values[0])
        # lat = sample['round_lat'].values[0]
        # lon = sample['round_lon'].values[0]
        source = sample['source'].values[0]
        labeltype = sample['labeltype'].values[0]
        ref_id = sample['ref_id'].values[0]

        # if (bounds[2] - bounds[0])/resolution != patchsize:
        #     raise Exception((f'Bounds {bounds} do not correspond '
        #                      f'to patchsize {patchsize} and '
        #                      f'resolution {resolution}m!'))

        start_date = sample['start_date'].values[0]
        end_date = sample['end_date'].values[0]

        # Get patches
        data = dict()
        for variable in variables:
            patches, date_range = cls.get_patches(
                folder, variable, bounds,
                epsg, start_date, end_date
            )
            data[variable] = dict(
                patches=patches,
                date_range=date_range
            )

        return cls(data,
                   labeltype=labeltype,
                   patchsize=patchsize,
                   tile=tile,
                   epsg=epsg,
                   bounds=bounds,
                   location_id=location_id,
                   start_date=start_date,
                   end_date=end_date,
                   source=source,
                   ref_id=ref_id,
                   attributes=attributes,
                   resolution=resolution,
                   overwrite=overwrite)

    @ staticmethod
    def get_patches(agera5colldf, variable, bounds, epsg,
                    start_date, end_date):

        # Create an AgERA5 yearly collection and filter dates and bounds
        agera5coll = AgERA5YearlyCollection.from_path(
            agera5colldf).filter_dates(
            start_date,
            end_date
        ).filter_bounds(bounds, epsg=epsg)

        # Load the timeseries for this variable
        bands_ts = agera5coll.load_timeseries(variable,
                                              resolution=1000)

        # Get out numpy array and reshape
        data = np.array(bands_ts.data).reshape((1, 1, -1)).astype(np.float32)

        return (data, bands_ts.timestamps)


class CropCalendarInputPatch(InputPatch):

    def __init__(self,
                 *args,
                 **kwargs):

        sensor = 'CropCalendar_AEZ'
        dtype = np.int16

        super().__init__(*args,
                         sensor=sensor,
                         dtype=dtype,
                         **kwargs)

    @ classmethod
    def from_folder(cls,
                    folder: str,
                    gdf_samples: gpd.GeoDataFrame,
                    location_id: str,
                    attr_name='location_id',
                    patchsize: int = 1,
                    resolution=50000,
                    variables=_CROPCALENDARVARS,
                    overwrite=False):

        # Select the sample
        if location_id not in gdf_samples[attr_name].values:
            raise ValueError(f'Sample "{location_id} not found in GDF!')
        sample = gdf_samples[gdf_samples[attr_name] == location_id]

        tile = sample['tile'].values[0]
        epsg = sample['epsg'].values[0]
        bounds = eval(sample['bounds'].values[0])
        lat = sample['round_lat'].values[0]
        lon = sample['round_lon'].values[0]
        source = sample['source'].values[0]
        labeltype = sample['labeltype'].values[0]
        ref_id = sample['ref_id'].values[0]
        start_date = sample['start_date'].values[0]
        end_date = sample['end_date'].values[0]

        # Get patches
        data = dict()
        for variable in variables:
            patches = cls.get_patches(
                folder, variable, lat, lon,
                patchsize, epsg
            )
            patches = patches.reshape((patchsize, patchsize))
            data[variable] = dict(
                patches=patches
            )

        return cls(data,
                   labeltype=labeltype,
                   patchsize=patchsize,
                   tile=tile,
                   epsg=epsg,
                   bounds=bounds,
                   location_id=location_id,
                   start_date=start_date,
                   end_date=end_date,
                   source=source,
                   ref_id=ref_id,
                   resolution=resolution,
                   overwrite=overwrite)

    @ staticmethod
    def get_patches(folder, variable, lat, lon, patchsize, epsg):

        infile = Path(folder) / (variable + '_WGS84.tif')

        if infile.is_file():
            logger.info(f'Extracting patch from: {infile}')
            patch = _read_pixel_values(infile, lat, lon)

        else:
            logger.warning((f'File {infile} not found -> '
                            'using no data value `0`'))
            patch = np.zeros((patchsize, patchsize, 1))

        return patch


if __name__ == '__main__':

    agera5folder = '/data/MTDA/AgERA5/'

    database = gpd.read_file('/data/worldcereal/cib/CIB_V1/database.json')
    location_id = database.iloc[0, :].location_id
    ref_id = database.iloc[0, :].ref_id
    labeltype = database.iloc[0, :].labeltype

    outdir = (Path('/data/worldcereal/cib/CIB_V1/') /
              labeltype / ref_id / 'data')

    AgERA5InputPatch.from_folder(agera5folder,
                                 database,
                                 location_id).to_netcdf(outdir)

    # cropcalendarfolder = '/data/worldcereal/data/cropcalendars/'

    # database = gpd.read_file('/data/worldcereal/cib/CIB_V1/database.json')
    # location_id = database.iloc[0, :].location_id
    # ref_id = database.iloc[0, :].ref_id
    # labeltype = database.iloc[0, :].labeltype

    # outdir = (Path('/data/worldcereal/cib/CIB_V1/') /
    #           labeltype / ref_id / 'data')

    # CropCalendarInputPatch.from_folder(cropcalendarfolder,
    #                                    database,
    #                                    location_id).to_netcdf(outdir)
