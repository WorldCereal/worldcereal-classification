import geopandas as gpd
from ipyleaflet import DrawControl, LayersControl, Map, SearchControl, basemaps
from loguru import logger
from openeo_gfmap import BoundingBoxExtent
from shapely import geometry
from shapely.geometry import Polygon, shape


def handle_draw(self, action, geo_json, area_limit=250):
    if action == "created":
        poly = Polygon(shape(geo_json.get("geometry")))
        bbox = poly.bounds
        logger.info(f"Your processing extent: {bbox}")

        # We convert our bounding box to local UTM projection
        # for further processing
        bbox_utm, epsg = _latlon_to_utm(bbox)
        area = (bbox_utm[2] - bbox_utm[0]) * (bbox_utm[3] - bbox_utm[1]) / 1000000
        logger.info(f"Area of processing extent: {area:.2f} km²")

        if area_limit is not None and area > area_limit:
            logger.error(
                f"Area of processing extent is too large. "
                f"Please select an area smaller than {area_limit} km²."
            )
            self.last_draw = {"type": "Feature", "geometry": None}

    elif action == "deleted":
        self.clear()
        self.last_draw = {"type": "Feature", "geometry": None}

    else:
        raise ValueError(f"Unknown action: {action}")


class ui_map:
    def __init__(self):
        from ipyleaflet import basemap_to_tiles

        osm = basemap_to_tiles(basemaps.OpenStreetMap.Mapnik)
        osm.base = True
        osm.name = "Open street map"

        img = basemap_to_tiles(basemaps.Esri.WorldImagery)
        img.base = True
        img.name = "Satellite imagery"

        self.map = Map(
            center=(51.1872, 5.1154), zoom=2, layers=[img, osm], scroll_wheel_zoom=True
        )
        self.map.add_control(LayersControl())

        self.draw_control = DrawControl(edit=False)

        self.draw_control.rectangle = {
            "shapeOptions": {
                "fillColor": "#6be5c3",
                "color": "#00F",
                "fillOpacity": 0.3,
            },
            "drawError": {"color": "#dd253b", "message": "Oups!"},
            "allowIntersection": False,
            "metric": ["km"],
        }
        self.draw_control.circle = {}
        self.draw_control.polyline = {}
        self.draw_control.circlemarker = {}
        self.draw_control.polygon = {}

        # Attach the event listener to the draw control
        self.draw_control.on_draw(handle_draw)

        self.map.add_control(self.draw_control)

        search = SearchControl(
            position="topleft",
            url="https://nominatim.openstreetmap.org/search?format=json&q={s}",
            zoom=20,
        )
        self.map.add_control(search)

        self.spatial_extent = None
        self.bbox = None
        self.poly = None

    def show_map(self):
        return self.map

    def get_processing_extent(self):

        obj = self.draw_control.last_draw

        if obj.get("geometry") is None:
            raise ValueError(
                "Please first draw a rectangle on the map before proceeding."
            )

        self.poly = Polygon(shape(obj.get("geometry")))
        bbox = self.poly.bounds

        # We convert our bounding box to local UTM projection
        # for further processing
        bbox_utm, epsg = _latlon_to_utm(bbox)

        self.spatial_extent = BoundingBoxExtent(*bbox_utm, epsg)

        logger.info(f"Your processing extent: {bbox}")

        return self.spatial_extent

    def get_polygon_latlon(self):
        self.get_processing_extent()
        return self.poly


def _latlon_to_utm(bbox):
    """This function converts a bounding box defined in lat/lon
    to local UTM coordinates.
    It returns the bounding box in UTM and the epsg code
    of the resulting UTM projection."""

    # convert bounding box to geodataframe
    bbox_poly = geometry.box(*bbox)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_poly], crs="EPSG:4326")

    # estimate best UTM zone
    crs = bbox_gdf.estimate_utm_crs()
    epsg = int(crs.to_epsg())

    # convert to UTM
    bbox_utm = bbox_gdf.to_crs(crs).total_bounds

    return bbox_utm, epsg
