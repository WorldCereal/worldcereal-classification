import warnings
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
from ipyleaflet import (
    DrawControl,
    GeoJSON,
    LayersControl,
    Map,
    SearchControl,
    WidgetControl,
    basemaps,
)
from IPython.display import display
from ipywidgets import HTML, Layout, VBox, widgets
from matplotlib.colors import to_hex
from openeo_gfmap import BoundingBoxExtent
from shapely import geometry
from shapely.geometry import Polygon, shape

from worldcereal.utils.legend import translate_ewoc_codes


def handle_draw(instance, action, geo_json, output, area_limit):
    with output:
        if action == "created":
            poly = Polygon(shape(geo_json.get("geometry")))
            bbox = poly.bounds
            display(HTML(f"<b>Your extent:</b> {bbox}"))

            # We convert our bounding box to local UTM projection
            # for further processing
            bbox_utm, epsg = _latlon_to_utm(bbox)
            area = (bbox_utm[2] - bbox_utm[0]) * (bbox_utm[3] - bbox_utm[1]) / 1000000
            display(HTML(f"<b>Area of extent:</b> {area:.2f} km²"))

            if area_limit is not None:
                if area > area_limit:
                    message = f"Area of extent is too large. Please select an area smaller than {area_limit} km²."
                    display(HTML(f'<span style="color: red;"><b>{message}</b></span>'))
                    instance.last_draw = {"type": "Feature", "geometry": None}

        elif action == "deleted":
            instance.clear()
            instance.last_draw = {"type": "Feature", "geometry": None}

        else:
            raise ValueError(f"Unknown action: {action}")


class ui_map:
    def __init__(self, area_limit: Optional[int] = None):
        """
        Initializes an ipyleaflet map with a draw control to select an extent.

        Parameters
        ----------
        area_limit : int, optional
            The maximum area in km² that can be selected on the map.
            By default no restrictions are imposed.
        """
        from ipyleaflet import basemap_to_tiles

        self.output = widgets.Output()
        self.area_limit = area_limit
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

        # Wrapper to pass additional arguments
        def draw_handler(instance, action, geo_json):
            handle_draw(
                instance, action, geo_json, self.output, area_limit=self.area_limit
            )

        # Attach the event listener to the draw control
        self.draw_control.on_draw(draw_handler)

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

        self.show_map()

    def show_map(self):
        vbox = widgets.VBox(
            [self.map, self.output],
            layout={"height": "600px"},
        )
        return display(vbox)

    def get_extent(self, projection="utm") -> BoundingBoxExtent:
        """Get extent from last drawn rectangle on the map.

        Parameters
        ----------
        projection : str, optional
            The projection to use for the extent.
            You can either request "latlon" or "utm". In case of the latter, the
            local utm projection is automatically derived.

        Returns
        -------
        BoundingBoxExtent
            The extent as a bounding box in the requested projection.

        Raises
        ------
        ValueError
            If no rectangle has been drawn on the map.
        """

        obj = self.draw_control.last_draw

        if obj.get("geometry") is None:
            raise ValueError(
                "Please first draw a rectangle on the map before proceeding."
            )

        self.poly = Polygon(shape(obj.get("geometry")))
        if self.poly is None:
            return None

        bbox = self.poly.bounds

        if projection == "utm":
            bbox_utm, epsg = _latlon_to_utm(bbox)
            self.spatial_extent = BoundingBoxExtent(*bbox_utm, epsg)
        else:
            self.spatial_extent = BoundingBoxExtent(*bbox)

        return self.spatial_extent

    def get_polygon_latlon(self):
        self.get_extent()
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


def visualize_rdm_geoparquet(src_path: str):
    """Visualize an RDM collection geoparquet file on a map.
    Parameters
    ----------
    src_path : str
        Path to the geoparquet file.
    """

    gdf = gpd.read_parquet(src_path)

    # Compute centroid, ignoring warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        center = (gdf.centroid.y.mean(), gdf.centroid.x.mean())

    # Extract unique ewoc_code values and assign colors
    unique_codes = sorted(gdf["ewoc_code"].unique())
    cmap = plt.get_cmap("tab20")  # Use a colormap with distinct colors
    colors = {
        code: to_hex(cmap(i / len(unique_codes))) for i, code in enumerate(unique_codes)
    }

    # Add a new column for the color associated with each ewoc_code
    gdf["color"] = gdf["ewoc_code"].map(colors)

    m = Map(
        basemap=basemaps.Esri.WorldImagery,
        center=center,
        zoom=7,
        scroll_wheel_zoom=True,
    )

    # convert dataframe to geojson
    data = gdf.__geo_interface__

    def style_callback(feature):
        """Apply color based on the ewoc_code attribute."""
        properties = feature["properties"]
        return {
            "color": "black",
            "fillColor": properties["color"],
            "opacity": 1,
            "fillOpacity": 0.7,
            "weight": 2,
        }

    # construct layer compatible with ipyleaflet
    layer = GeoJSON(
        data=data,
        style_callback=style_callback,
        point_style={
            "radius": 5,
        },
        name="Reference data",
    )

    # Add to the map
    m.add_layer(layer)

    # Translate ewoc_codes
    crop_types = translate_ewoc_codes(unique_codes)

    # Create a legend
    legend_items = []
    for code, color in colors.items():
        if code not in crop_types.index:
            legend_items.append(
                HTML(f"<span style='color:{color};'>⬤</span> {code} (unknown)")
            )
        else:
            legend_items.append(
                HTML(
                    f"<span style='color:{color};'>⬤</span> {crop_types.loc[code]['label_full']}"
                )
            )
    # Adjust legend size dynamically with a scrollable container
    legend_box = VBox(
        legend_items,
        layout=Layout(
            max_height="300px",  # Set a maximum height
            overflow="auto",  # Add scrolling if content exceeds max height
        ),
    )
    legend_control = WidgetControl(widget=legend_box, position="topright")

    # Add the legend to the map
    m.add_control(legend_control)

    layer_control = LayersControl(position="topleft")
    m.add_control(layer_control)

    return m
