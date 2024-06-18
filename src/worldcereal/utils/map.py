from ipyleaflet import (Map, basemaps, DrawControl,
                        SearchControl)
import geopandas as gpd
from shapely import geometry
import rasterio


def get_ui_map():

    m = Map(basemap=basemaps.Esri.WorldImagery,
            center=(51.1872, 5.1154), zoom=5)

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
