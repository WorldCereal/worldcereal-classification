import os

import geopandas as gpd
from loguru import logger
import numpy as np
try:
    import osr
except ImportError:
    from osgeo import osr
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
import satio
from shapely.geometry import Polygon
import utm
import xarray as xr


DRIVER_LUT = {
    'geojson': 'GeoJSON',
    'json': 'GeoJSON',
    'gpkg': 'GPKG',
    'shp': 'ESRI Shapefile'
}


def _rasterize_labels(gdf, bounds, patchsize, epsg, lc_fillvalue=None):
    """A function to rasterize points/polygons
       LC/CT/IRR labels based on a bounds tuple

    Args:
        gdf (GeoDataFrame): The GeoDataFrame holding the features
        bounds (tuple): tuple describing the bounds
        patchsize (int): Size of the patch to generate
        epsg (str): EPSG code to work in
        lc_fillvalue: optional fill value to use
    """
    # Convert the gdf
    gdf = gdf.to_crs(epsg=int(epsg))

    # Create a polygon from the bounds
    boundspoly = Polygon.from_bounds(
        bounds[0], bounds[1], bounds[2], bounds[3])

    # Select polygons in the bounds
    gdf = gdf[gdf.intersects(boundspoly)]
    if gdf.shape[0] == 0:
        return None
    else:

        # Setup a rasterio transform from the bounds
        transform = rasterio.transform.from_bounds(
            bounds[0], bounds[1], bounds[2], bounds[3], patchsize, patchsize)

        # Need to convert to 3-band label (LC, croptype, irrigation)
        data = np.zeros((patchsize, patchsize, 3),
                        dtype=np.uint16)

        # Set the fill_value
        lc_fillvalue = lc_fillvalue or 0

        # replace missing labels by 0
        gdf['LC'] = gdf['LC'].fillna(0)
        gdf['CT'] = gdf['CT'].fillna(0)
        gdf['IRR'] = gdf['IRR'].fillna(0)

        # We need to make sure the output labels are Integers
        gdf['LC'] = gdf['LC'].astype(int)
        gdf['CT'] = gdf['CT'].astype(int)
        gdf['IRR'] = gdf['IRR'].astype(int)

        # Get LC information
        lc = ((geom, value)
              for geom, value in zip(gdf.geometry, gdf.LC))
        data[:, :, 0] = rasterize(
            lc,
            out_shape=(patchsize, patchsize),
            transform=transform,
            fill=lc_fillvalue,
            all_touched=False,
            dtype=rasterio.uint16)

        # Compile croptype
        ct = ((geom, value)
              for geom, value in zip(gdf.geometry, gdf.CT))
        ct = rasterize(
            ct,
            out_shape=(patchsize, patchsize),
            transform=transform,
            fill=0,
            all_touched=False,
            dtype=rasterio.uint16)

        data[:, :, 1] = ct

        # Compile irrigation
        irr = ((geom, value)
               for geom, value in zip(gdf.geometry, gdf.IRR))
        irr = rasterize(
            irr,
            out_shape=(patchsize, patchsize),
            transform=transform,
            fill=0,
            all_touched=False,
            dtype=rasterio.uint16) * 100

        data[:, :, 2] = irr

        return data


def _read_raster_labels(tifffile, bounds, patchsize, epsg):
    """A function to read MAP-based
       LC/CT/IRR labels based on a bounds tuple

    Args:
        tifffile (str): Path to tiff file that holds the labels
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
                3, patchsize, patchsize))

    # Need to transpose to [x, y, bands]
    data = data.transpose((1, 2, 0))

    return data


def _get_validity_time(sample: pd.Series):
    try:
        valtime = sample.validityTi
    except AttributeError:
        try:
            valtime = sample.ValidityTi
        except AttributeError:
            try:
                valtime = sample.valtime
            except AttributeError:
                raise Exception('No validity time found.')

    return valtime


def select_samples_by_date(gdf, date):
    '''
    filter gdf to exclude samples not in the
    same period of time compared to validity time of the sample
    check validity time in gdf_full
    '''
    if 'validityTi' not in gdf.columns:
        if 'validityTime' in gdf.columns:
            gdf = gdf.rename(
                columns={'validityTime': 'validityTi'})
        elif 'ValidityTi' in gdf.columns:
            gdf = gdf.rename(
                columns={'ValidityTi': 'validityTi'})
        elif 'valtime' in gdf.columns:
            gdf = gdf.rename(
                columns={'valtime': 'validityTi'})
        else:
            raise Exception('Column `validityTi` not found in DataFrame.')
    # make sure it is string first
    gdf['validityTi'] = gdf['validityTi'].astype(str)
    # to datetime
    gdf['valtime'] = pd.to_datetime(gdf['validityTi'])
    # determine start and end of valid range (1 month buffer)
    start_range = date - pd.DateOffset(months=1)
    end_range = date + pd.DateOffset(months=1)
    gdf = gdf.loc[(gdf.valtime <= end_range) & (gdf.valtime >= start_range)]

    return gdf


class TrainingLabel:

    def __init__(self,
                 data,
                 labeltype=None,
                 contenttype=None,
                 resolution=10,
                 patchsize=None,
                 tile=None,
                 epsg=None,
                 bounds=None,
                 location_id=None,
                 start_date=None,
                 end_date=None,
                 source=None,
                 nodata=0,
                 overwrite=False):

        if labeltype not in ['POINT', 'POLY', 'MAP']:
            raise ValueError(f'Lable type {labeltype} not supported!')

        if data.shape[2] != 3:
            raise ValueError('Training data should have exactly 3 bands!')

        self.data = data
        self.labeltype = labeltype
        self.NoData = nodata
        self.contenttype = contenttype
        self.resolution = resolution
        self.patchsize = patchsize
        self.tile = tile
        self.epsg = epsg
        self.bounds = bounds
        self.location_id = location_id
        self.start_date = start_date
        self.end_date = end_date
        self.source = source

        self._overwrite = overwrite
        self._prefix = f'OUTPUT_{labeltype}-{contenttype}'

    def to_netcdf(self, outdir):

        if self.data is None:
            return

        data = self.data.transpose(2, 0, 1)

        xmin, ymin, xmax, ymax = self.bounds

        attrs = {a: getattr(self, a) for a in ['location_id', 'epsg',
                                               'resolution',
                                               'bounds', 'NoData',
                                               'start_date', 'end_date',
                                               'tile', 'source']}

        attrs['crs'] = CRS.from_epsg(self.epsg).to_proj4()

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

        bands = ['LC', 'CT', 'IRR']
        coords = {'band': bands,
                  'x': x,
                  'y': y}

        da = xr.DataArray(data.astype(np.uint16),
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
        for band in ds.data_vars:
            ds[band].attrs = ds.attrs
            ds[band].attrs['grid_mapping'] = 'spatial_ref'

        # We need to encode the no data value "0" into the netcdf
        encoding = dict()
        for band in bands:
            encoding[band] = dict(_FillValue=self.NoData,
                                  dtype='uint16')

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


class TrainingPoint(TrainingLabel):
    def __init__(self,
                 *args,
                 **kwargs):

        super().__init__(*args,
                         labeltype='POINT',
                         **kwargs)

    @classmethod
    def from_gdf(cls,
                 gdf_samples: gpd.GeoDataFrame,
                 gdf_full: gpd.GeoDataFrame,
                 location_id: str,
                 patchsize: int,
                 contenttype: str,
                 attr_name='location_id',
                 resolution=10,
                 lc_fillvalue=None,
                 overwrite=False):

        # Select the sample
        if location_id not in gdf_samples[attr_name].values:
            raise ValueError(f'Sample "{location_id} not found in GDF!')
        sample = gdf_samples[gdf_samples[attr_name] == location_id]

        if sample.geom_type.values[0] != 'Point':
            raise Exception(('Sample is not a Point! '
                             f'It is a: {sample.geom_type.values[0]}'))

        tile = sample['tile'].values[0]
        epsg = sample['epsg'].values[0]
        bounds = eval(sample['bounds'].values[0])
        source = sample['source'].values[0]

        if (bounds[2] - bounds[0])/resolution != patchsize:
            raise Exception((f'Bounds {bounds} do not correspond '
                             f'to patchsize {patchsize} and '
                             f'resolution {resolution}m!'))

        start_date = sample['start_date'].values[0]
        end_date = sample['end_date'].values[0]

        # Get validity time of sample
        samplevaltime = pd.to_datetime(
            sample.validityTi.values[0]).to_pydatetime()

        # Subset surrounding samples based on validity time
        gdf_full = select_samples_by_date(gdf_full, samplevaltime)

        # Get rasterized labels
        data = _rasterize_labels(gdf_full, bounds, patchsize, epsg)
        if data is None:
            return None
        else:
            return cls(data,
                       contenttype=contenttype,
                       patchsize=patchsize,
                       tile=tile,
                       epsg=epsg,
                       bounds=bounds,
                       location_id=location_id,
                       start_date=start_date,
                       end_date=end_date,
                       source=source,
                       resolution=resolution,
                       overwrite=overwrite)


class TrainingPolygon(TrainingLabel):
    def __init__(self,
                 *args,
                 **kwargs):

        super().__init__(*args,
                         labeltype='POLY',
                         **kwargs)

    @classmethod
    def from_gdf(cls,
                 gdf_samples: gpd.GeoDataFrame,
                 gdf_full: gpd.GeoDataFrame,
                 location_id: str,
                 patchsize: int,
                 contenttype: str,
                 attr_name='location_id',
                 resolution=10,
                 lc_fillvalue=None,
                 overwrite=False):

        # Select the sample
        if location_id not in gdf_samples[attr_name].values:
            raise ValueError(f'Sample "{location_id} not found in GDF!')
        sample = gdf_samples[gdf_samples[attr_name] == location_id]

        if sample.geom_type.values[0] not in ['Polygon',
                                              'MultiPolygon']:
            raise Exception(('Sample is not a Polygon! '
                             f'It is a: {sample.geom_type.values[0]}'))

        tile = sample['tile'].values[0]
        epsg = sample['epsg'].values[0]
        bounds = eval(sample['bounds'].values[0])
        source = sample['source'].values[0]

        if (bounds[2] - bounds[0])/resolution != patchsize:
            raise Exception((f'Bounds {bounds} do not correspond '
                             f'to patchsize {patchsize} and '
                             f'resolution {resolution}m!'))

        start_date = sample['start_date'].values[0]
        end_date = sample['end_date'].values[0]

        if lc_fillvalue is not None:
            logger.warning(f'Using a LC fill value of {lc_fillvalue}!')

        # Get validity time of sample
        samplevaltime = pd.to_datetime(
            sample.validityTi.values[0]).to_pydatetime()

        # Subset surrounding samples based on validity time
        gdf_full = select_samples_by_date(gdf_full, samplevaltime)

        # Get rasterized labels
        data = _rasterize_labels(gdf_full, bounds, patchsize, epsg,
                                 lc_fillvalue=lc_fillvalue)

        return cls(data,
                   contenttype=contenttype,
                   patchsize=patchsize,
                   tile=tile,
                   epsg=epsg,
                   bounds=bounds,
                   location_id=location_id,
                   start_date=start_date,
                   end_date=end_date,
                   source=source,
                   resolution=resolution,
                   overwrite=overwrite)


class TrainingMap(TrainingLabel):
    def __init__(self,
                 *args,
                 **kwargs):

        super().__init__(*args,
                         labeltype='MAP',
                         **kwargs)

    @ classmethod
    def from_tiff(cls,
                  gdf_samples: gpd.GeoDataFrame,
                  tifffile: str,
                  location_id: str,
                  patchsize: int,
                  contenttype: str,
                  attr_name='location_id',
                  resolution=10,
                  overwrite=False):

        # Select the sample
        if location_id not in gdf_samples[attr_name].values:
            raise ValueError(f'Sample "{location_id} not found in GDF!')
        sample = gdf_samples[gdf_samples[attr_name] == location_id]

        tile = sample['tile'].values[0]
        epsg = sample['epsg'].values[0]
        bounds = eval(sample['bounds'].values[0])
        source = sample['source'].values[0]

        if (bounds[2] - bounds[0])/resolution != patchsize:
            raise Exception((f'Bounds {bounds} do not correspond '
                             f'to patchsize {patchsize} and '
                             f'resolution {resolution}m!'))

        start_date = sample['start_date'].values[0]
        end_date = sample['end_date'].values[0]

        # Get rasterized labels
        data = _read_raster_labels(tifffile, bounds, patchsize, epsg)

        return cls(data,
                   contenttype=contenttype,
                   patchsize=patchsize,
                   tile=tile,
                   epsg=epsg,
                   bounds=bounds,
                   location_id=location_id,
                   start_date=start_date,
                   end_date=end_date,
                   source=source,
                   resolution=resolution,
                   overwrite=overwrite)


class TrainingInput:
    def __init__(self, data, kernelsize, index=None,
                 source=None, driver=None, overwriteattrs=False,
                 rounding=0):

        if type(data) != gpd.GeoDataFrame or data is None:
            raise ValueError('Should provide a valid '
                             'GeoPandas GDF as "data" argument to '
                             'create an instance of this class!')

        self.data = data
        self.source = source
        self.rounding = rounding
        self.overwriteattrs = overwriteattrs
        if driver is None:
            self.driver = 'shapefile'
        elif driver not in DRIVER_LUT.values():
            raise ValueError(f'Driver {driver} not supported.')
        else:
            self.driver = driver
        if index is None:
            self.index = data.index
        else:
            self.index = index
        self.kernelsize = kernelsize

        if 'location_id' not in self.data.columns:
            self.data['location_id'] = self.data.index.astype(
                str).tolist()

    @classmethod
    def from_file(cls, file, idattr=None,
                  overwriteattrs=False, kernelsize=64,
                  rounding=0):
        """Entry point to instantiate class from a file

        Args:
            file (str): Path to compatible file (e.g. SHP)
            idattr (str, optional): Attribute name for sample identifier.
            Defaults to None.
            overwriteattrs (bool, optional): Whether or not to overwrite
            existing attributes. Defaults to False.
            kernelsize (int, optional): Full height/width of patch.
            Defaults to 64.
            rounding (int, optional): Nearest whole number to round to.
            Defaults to 0.

        Returns:
            TrainingInput class: instance of class
        """

        # Our input should always be in wgs84
        logger.info(f'Loading data from: {file}')
        df = gpd.read_file(file)
        logger.info('File loaded!')

        if idattr is not None:
            if idattr not in df.columns:
                raise ValueError(f'Specified idattr {idattr} '
                                 f'not found in {file}!')
            df = df.set_index(idattr)
        else:
            logger.info('No idattr specified, taking feature index!')

        if df.crs.to_string() != "EPSG:4326":
            logger.warning('Input file needs '
                           f'reprojection from {df.crs.to_string()} '
                           'to EPSG:4326!')
            df = df.to_crs(epsg=4326)

        index = df.index.tolist()

        # Check if index is unique
        logger.info('Checking index uniqueness ...')
        if len(index) != len(list(set(index))):
            raise ValueError('Index not unique!')

        driver = DRIVER_LUT[os.path.splitext(file)[-1][1:]]

        return cls(df, kernelsize, index=index,
                   source=file, driver=driver,
                   overwriteattrs=overwriteattrs,
                   rounding=rounding)

    @classmethod
    def from_tiffs(cls, path):

        raise NotImplementedError('Not implemented yet :(')

    @property
    def centroids(self):
        return self.data.centroid

    def _clone(self,
               data: gpd.GeoDataFrame = None,
               kernelsize: int = None,
               index=None,
               source=None,
               driver=None,
               overwriteattrs=None,
               rounding=None
               ):

        data = self.data if data is None else data
        kernelsize = self.kernelsize if kernelsize is None else kernelsize
        index = self.index if index is None else index
        source = self.source if source is None else source
        driver = self.driver if driver is None else driver
        overwriteattrs = (self.overwriteattrs if
                          overwriteattrs is None else overwriteattrs)
        rounding = self.rounding if rounding is None else rounding

        return self.__class__(data, kernelsize, index,
                              source, driver, overwriteattrs, rounding)

    def add_s2tile(self, s2grid='/data/worldcereal/auxdata/'
                   's2tiles/s2tiles_voronoi.shp'):
        # TODO: current shapefile is in higher latitudes not correct!

        logger.info('Reading s2tiles grid ...')
        s2tiles = gpd.read_file(s2grid)

        # Get the data
        data = self.data.copy()

        # Make sure s2tiles is in wgs84
        if s2tiles.crs.to_string() != 'EPSG:4326':
            s2tiles = s2tiles.to_crs(epsg=4326)

        # Temporarily set polygon centroids as the geometry
        # since these are needed to assign S2 tiles
        realgeom = data.geometry
        data.geometry = data.centroid

        logger.info('Assigning matching S2 tiles ...')
        data = gpd.sjoin(data,
                         s2tiles,
                         how='left',
                         op='intersects').rename(
            columns={'s2tile': 'tile'}
        )

        # Restore geometry
        data.geometry = realgeom

        # add epsg code based on s2grid from satio
        s2grid = satio.layers.load('s2grid')
        data = pd.merge(data, s2grid[['epsg', 'tile']], on='tile', how='left')

        return self._clone(data=data)

    def add_latlon(self):
        df = self.data.copy()
        df['centroids'] = df.centroid
        data = self.data.copy()
        latlon = df.apply(
            lambda row: self.latlon2utm(row.centroids.y,
                                        row.centroids.x,
                                        rounding=self.rounding,
                                        epsg=row['epsg']),
            axis=1, result_type='expand'
        ).rename(columns={0: 'easting', 1: 'northing', 2: 'epsg',
                          3: 'zonenumber', 4: 'zoneletter'})

        data['easting'] = latlon['easting']
        data['northing'] = latlon['northing']
        data['zonenumber'] = latlon['zonenumber']
        data['zoneletter'] = latlon['zoneletter']

        rounded_centroid = latlon.apply(
            lambda row: self.utm2latlon(row['easting'],
                                        row['northing'],
                                        row['zonenumber'],
                                        row['zoneletter']),
            axis=1, result_type='expand'
        ).rename(columns={0: 'round_lat', 1: 'round_lon'})

        data['round_lat'] = rounded_centroid['round_lat']
        data['round_lon'] = rounded_centroid['round_lon']

        return self._clone(data=data)

    def add(self, **newattributes):

        data = self.data.copy()

        if len(newattributes) == 0:
            raise ValueError('Should at least add one attribute!')
        for k, v in newattributes.items():
            logger.info(f'Adding "{k}" to dataframe ...')
            if type(v) is list:
                if type(v[0]) not in [str, int, float, np.float32,
                                      np.float]:
                    # Likely unsupported format: cast to string
                    v = [str(x) for x in v]
            elif type(v) is str or type(v) is int or type(v) is float:
                pass
            elif v.dtype not in [str, int, float, np.float32,
                                 np.float]:
                # Likely unsupported format: cast to string
                v = v.astype(str)
            data[k] = v
        return self._clone(data=data)

    def drop(self, attr):

        data = self.data.copy()

        if attr not in self.data.columns.tolist():
            raise ValueError(f'Attribute "{attr}" not in dataframe!"')
        logger.info(f'Dropping "{attr}" from dataframe ...')
        data = self.data.drop(columns=attr)
        return self._clone(data=data)

    def subset(self, samples: list):
        """Function to subset the data on a list of sample IDs

        Args:
            samples (list): list of sample IDs to subset on

        Raises:
            ValueError: Error raised when subsetting fails
        """

        logger.info('Subsetting on sample list ...')
        try:
            data = self.data.loc[samples]
            return self._clone(data=data)
        except Exception:
            raise ValueError(('Could not subset on samples. '
                              'Most likely the sample ids could '
                              ' could not be found.'))

    def save(self, outfile=None, driver=None):
        if outfile is None:
            outfile = self.source
        if outfile is None:
            raise ValueError('No output file specified')
        if driver is None:
            driver = self.driver
        else:
            driver = DRIVER_LUT[driver]
        logger.info(f'Saving to {outfile} ...')
        if os.path.exists(outfile):
            logger.warning(f'File {outfile} exists -> deleting ...')
            os.remove(outfile)
        oldmask = os.umask(0o022)
        self.data.to_file(outfile, driver=driver)
        os.umask(oldmask)

    def copy(self):
        return self._clone()

    def split_cal_val_test(self, cal=0.7, val=0.2, test=0.1):
        """Method to randomly split the training data in CAL/VAL/TEST

        Args:
            cal (float, optional): Portion to use for CAL. Defaults to 0.7.
            val (float, optional): Portion to use for VAL. Defaults to 0.2.
            test (float, optional): Portion to use for TEST. Defaults to 0.1.

        Returns:
            pd.Series: series describing for each sample to which
            category it belongs
        """

        if not np.isclose(cal + val + test, 1):
            raise ValueError(('Cal/Val/Test should sum to 1 '
                              f'but got: {cal + val + test}'))
        logger.info(('Splitting in CAL/VAL/TEST according '
                     f'to: [{cal}, {val}, {test}]'))

        from sklearn.model_selection import train_test_split

        x_train, x_val = train_test_split(self.index, test_size=val+test)
        x_val, x_test = train_test_split(x_val, test_size=test/(test+val))

        split = pd.Series(index=self.index)
        split.loc[x_train] = 'CAL'
        split.loc[x_val] = 'VAL'
        split.loc[x_test] = 'TEST'

        return split

    def remove_autocorrelation(self):
        """Method to remove autocorrelated samples

        Returns:
            TrainingInput: Subset on non-autocorrelated samples
        """
        logger.info('Removing auto-correlated samples ...')

        samples = []
        nonoverlapping = gpd.GeoDataFrame(columns=['id', 'geometry'])
        df = self.data
        df = df.to_crs(epsg=3857)
        dfbuffered = df.buffer(int(self.kernelsize / 2) * 10)

        for ix, geom in dfbuffered.iteritems():
            if not nonoverlapping.intersects(geom).any():
                nonoverlapping = nonoverlapping.append({'id': ix,
                                                        'geometry': geom},
                                                       ignore_index=True)
                samples.append(ix)
            else:
                # Found overlap, check if dates are also similar
                sampledate = pd.to_datetime(
                    _get_validity_time(df.loc[ix])).to_pydatetime()
                overlaps = df.loc[nonoverlapping.loc[
                    nonoverlapping.intersects(geom)].id]
                matchingoverlaps = select_samples_by_date(overlaps, sampledate)

                if matchingoverlaps.empty:
                    # None of the overlapping features is close in time
                    # so we should include it as an independent sample
                    nonoverlapping = nonoverlapping.append({'id': ix,
                                                            'geometry': geom},
                                                           ignore_index=True)
                    samples.append(ix)

        return self.subset(samples)

    @ property
    def bounds(self):
        # If the closest S2 tile has not been determined yes,
        # we need to do this first
        # this function also adds the epsg
        if 'tile' not in self.data.columns:
            logger.info('Need to find S2 tiles ...')
            withs2 = self.add_s2tile()
            self.data = withs2.data
        else:
            logger.info('S2 tiles already determined!')

        # add rounded lat and lon
        logger.info('Adding rounded lat, lon coordinates...')
        withlatlon = self.add_latlon()
        self.data = withlatlon.data

        logger.info(f'Getting bounds for kernelsize: {self.kernelsize}')
        bounds = self.data.apply(
            lambda row: self.get_bounds(
                row['easting'],
                row['northing'],
                self.kernelsize
            ), axis=1
        )
        return bounds

    def bounds_as_geometry(self):
        if 'bounds' not in self.data:
            self.add(bounds=self.bounds)

        coords = self.data.apply(
            lambda row: self.bounds_to_latlon(row['bounds'],
                                              row['zonenumber'],
                                              row['zoneletter']),
            axis=1, result_type='expand'
        ).rename(columns={0: 'llx', 1: 'lly', 2: 'ulx',
                          3: 'uly', 4: 'lrx', 5: 'lry',
                          6: 'urx', 7: 'ury'})
        logger.info('Acquiring bounds as lat/lon polygons ...')
        gdf = gpd.GeoDataFrame(
            coords.apply(
                lambda row: self.to_geometry([row['llx'],
                                              row['lly'],
                                              row['ulx'],
                                              row['uly'],
                                              row['urx'],
                                              row['ury'],
                                              row['lrx'],
                                              row['lry']]),
                axis=1
            ).rename(
                'geometry')
        )
        gdf.crs = "EPSG:4326"

        return gdf

    @ staticmethod
    def bounds_to_latlon(bounds, zonenumber, zoneletter):
        bounds = eval(bounds)

        llx, lly = utm2latlon(bounds[0], bounds[1],
                              zonenumber, zoneletter)
        ulx, uly = utm2latlon(bounds[2], bounds[1],
                              zonenumber, zoneletter)
        lrx, lry = utm2latlon(bounds[0], bounds[3],
                              zonenumber, zoneletter)
        urx, ury = utm2latlon(bounds[2], bounds[3],
                              zonenumber, zoneletter)

        return [llx, lly, ulx, uly, lrx, lry, urx, ury]

    @ staticmethod
    def to_geometry(coords):
        return Polygon([(coords[1], coords[0]),
                        (coords[3], coords[2]),
                        (coords[5], coords[4]),
                        (coords[7], coords[6])])

    @ staticmethod
    def latlon2utm(lat, lon, rounding=0,
                   epsg=None):
        if epsg is not None:
            crs = CRS.from_epsg(epsg)
            srs = osr.SpatialReference(wkt=crs.to_wkt())
            # zoneletter = srs.GetAttrValue('projcs')[-1]
            zoneletter = None
            zonenumber = int(srs.GetAttrValue('projcs')[-3:-1])
        else:
            zonenumber = None
            zoneletter = None
        easting, northing, zonenumber, zoneletter = utm.from_latlon(
            lat, lon, force_zone_number=zonenumber,
            force_zone_letter=zoneletter)
        if rounding > 0:
            easting = TrainingInput.customround(easting, rounding)
            northing = TrainingInput.customround(northing, rounding)
            logger.debug('Rounded EPSG coordinates: '
                         f'({easting}, {northing})')
        if epsg is None:
            if zoneletter >= 'N':
                # Northern hemisphere
                epsg = f'{32600 + zonenumber}'
            else:
                # Southern hemisphere
                epsg = f'{32700 + zonenumber}'
        return [easting, northing, epsg,
                zonenumber, zoneletter]

    @ staticmethod
    def utm2latlon(easting, northing, zonenumber, zoneletter):
        lat, lon = utm.to_latlon(easting, northing, zonenumber, zoneletter)
        return [lat, lon]

    @ staticmethod
    def get_bounds(easting, northing, windowsize):
        """[summary]

        Args:
            easting ([type]): [description]
            northing ([type]): [description]
            windowsize ([type]): [description]

        Returns:
            [tuple]: [(x1, y1, x2, y2)]
        """

        x1 = easting - windowsize * 10
        y1 = northing - windowsize * 10
        x2 = easting + windowsize * 10
        y2 = northing + windowsize * 10

        return (x1, y1, x2, y2)

    @ staticmethod
    def customround(x, base=5):
        return int(np.floor(x/base)*base)


def utm2latlon(easting, northing, zonenumber: int, zoneletter: str):
    """Function to get lat/lon coordinates from UTM

    Args:
        easting (float): UTM easting coordinate
        northing (float): UTM northing coordinate
        zonenumber (int): UTM zone number
        zoneletter (str): UTM zone letter

    Returns:
        [float, float]: list with lat, lon coordinate
    """
    lat, lon = utm.to_latlon(easting, northing, zonenumber,
                             zoneletter, strict=False)
    return [lat, lon]
