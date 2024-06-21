from ipyleaflet import (Map, basemaps, DrawControl,
                        SearchControl)
import geopandas as gpd
from shapely import geometry
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import leafmap


def get_probability_cmap():
    colormap = plt.get_cmap('RdYlGn')
    cmap = {}
    for i in range(101):
        cmap[i] = tuple((np.array(colormap(int(2.55 * i))) * 255).astype(int))
    return cmap


COLORMAP = {
    'active-cropland': {
        0: (232, 55, 39, 255),  # inactive
        1: (77, 216, 39, 255),  # active
    },
    'temporary-crops': {
        0: (186, 186, 186, 0),  # no cropland
        1: (224, 24, 28, 200),  # cropland
    },
    'maize': {
        0: (186, 186, 186, 255),  # other
        1: (252, 207, 5, 255),  # maize
    },
    'probabilities': get_probability_cmap()
}

NODATAVALUE = 255


def _get_colormap(product):

    if product in COLORMAP.keys():
        colormap = COLORMAP[product]
    else:
        logger.warning((f'Unknown product `{product}`: '
                        'cannot assign colormap'))
        logger.info(f'Supported products: {COLORMAP.keys()}')
        colormap = None

    return colormap


def _get_nodata(product):
    return NODATAVALUE


def postprocess_map(infile, product=None, colormap=None, nodata=None):
    '''
    Function to set colormap and nodata value of a WorldCereal product.
    The function assumes the input data consists of two layers:
    1) the classification layer
    2) the probabilities layer, indicating class probability of winning class

    Args:
        infile (str): path to the input file
        product (str): product identifier
        colormap (dict): colormap to assign to the classification layer
        nodata (int): nodata value to assign to the classification layer

    Returns:
        outfiles (list): list of output files
        (first file is the classification layer,
        second file is the probabilities layer)
    '''
    # infer name of output file
    outclas = infile.replace('.tif', '-classification.tif')
    outprob = infile.replace('.tif', '-probabilities.tif')

    # get colormap and nodata value
    if product is not None:
        nodata = _get_nodata(product)
        colormap = _get_colormap(product)

    if nodata is None:
        nodata = NODATAVALUE

    # get properties and data from input file
    with rasterio.open(infile, 'r') as src:
        labels = src.read(1)
        probs = src.read(2)
        meta = src.meta

    meta.update(count=1)

    # write output files
    with rasterio.open(outclas, 'w', **meta) as dst:
        dst.write(labels, indexes=1)
        dst.nodata = nodata
        if colormap is not None:
            dst.write_colormap(1, colormap)

    with rasterio.open(outprob, 'w', **meta) as dst:
        dst.write(probs, indexes=1)
        dst.nodata = nodata
        colormap_prob = _get_colormap('probabilities')
        dst.write_colormap(1, colormap_prob)

    return [outclas, outprob]


def visualize_product(infile, product=None, colormap=None,
                      nodata=None, port=8888):
    '''
    Function to visualize a WorldCereal product.
    The function assumes the input data consists of two layers:
    1) the classification layer
    2) the probabilities layer, indicating class probability of winning class

    Args:
        infiles (list): list of input files
        port (int): port to use for visualization

    Returns:
        None
    '''
    outfiles = postprocess_map(infile, product=product,
                               colormap=colormap, nodata=nodata)

    m = leafmap.Map()
    m.add_basemap('Esri.WorldImagery')
    m.add_raster(outfiles[0], indexes=[1],
                 layer_name=f'{product}-classification',
                 port=port)
    m.add_raster(outfiles[1], indexes=[1],
                 layer_name=f'{product}-probabilities',
                 port=port)

    return m


def get_ui_map():

    m = Map(basemap=basemaps.Esri.WorldStreetMap,
            center=(51.1872, 5.1154), zoom=10)

    draw_control = DrawControl()

    draw_control.rectangle = {
        "shapeOptions": {
            "fillColor": "#6be5c3",
            "color": '#00F',
            "fillOpacity": 0.3,
        },
        "drawError": {
            "color": "#dd253b",
            "message": "Oups!"
        },
        "allowIntersection": False
    }
    draw_control.circle = {}
    draw_control.polyline = {}
    draw_control.circlemarker = {}
    draw_control.polygon = {}

    m.add_control(draw_control)

    search = SearchControl(
        position="topleft",
        url='https://nominatim.openstreetmap.org/search?format=json&q={s}',
        zoom=20
    )
    m.add_control(search)

    return m, draw_control


def _latlon_to_utm(bbox):
    '''This function converts a bounding box defined in lat/lon
    to local UTM coordinates.
    It returns the bounding box in UTM and the epsg code
    of the resulting UTM projection.'''

    # convert bounding box to geodataframe
    bbox_poly = geometry.box(*bbox)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_poly],
                                crs='EPSG:4326')

    # estimate best UTM zone
    crs = bbox_gdf.estimate_utm_crs()
    epsg = int(crs.to_epsg())

    # convert to UTM
    bbox_utm = bbox_gdf.to_crs(crs).total_bounds

    return bbox_utm, epsg
