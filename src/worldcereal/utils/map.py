import json
import tempfile
import warnings
import zipfile
from pathlib import Path
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
from shapely.geometry import Polygon, box, shape

from worldcereal.utils.legend import translate_ewoc_codes


def validate_area(bbox, area_limit: int, output: widgets.Output) -> bool:
    """
    Validates the area of a bbox against a limit.
    Displays feedback in the given output widget.

    Returns True if valid, False otherwise.
    """
    with output:
        output.clear_output()
        display(HTML(f"<b>Your extent:</b> {bbox}"))

        bbox_utm, epsg = _latlon_to_utm(bbox)
        area = (bbox_utm[2] - bbox_utm[0]) * (bbox_utm[3] - bbox_utm[1]) / 1_000_000
        display(HTML(f"<b>Area of extent:</b> {area:.2f} km²"))

        if area_limit is not None and area > area_limit:
            display(
                HTML(
                    f"<span style='color:red'><b>Area too large "
                    f"(>{area_limit} km²). Please select a smaller area.</b></span>"
                )
            )
            return False

    return True


class ui_map:
    def __init__(self, area_limit: Optional[int] = None, display_ui: bool = True):
        """
        Initializes an ipyleaflet map with a draw control and file upload functionality
        to select an extent.

        Parameters
        ----------
        area_limit : int, optional
            The maximum area in km² that can be selected on the map.
            By default no restrictions are imposed.
        display_ui : bool, optional
            Whether to display the map UI immediately upon initialization.
        """

        self.area_limit = area_limit
        self.spatial_extent = None
        self.poly = None

        self.output = widgets.Output(
            layout={
                "border": "1px solid #ccc",
                "max_height": "120px",
                "overflow": "auto",
            }
        )

        self._build_map()
        self._add_controls()
        self._wire_events()

        if display_ui:
            display(widgets.VBox([self.map, self.output]))

    def _build_map(self):

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
        self.map.add_control(
            SearchControl(
                position="topleft",
                url="https://nominatim.openstreetmap.org/search?format=json&q={s}",
                zoom=20,
            )
        )

        self.draw_control = DrawControl(
            rectangle={
                "shapeOptions": {
                    "fillColor": "#6be5c3",
                    "color": "#00F",
                    "fillOpacity": 0.3,
                },
                "allowIntersection": False,
                "metric": ["km"],
            },
            circle={},
            circlemarker={},
            polyline={},
            polygon={},
            edit=False,
        )
        self.map.add_control(self.draw_control)

    def _add_controls(self):
        self.uploader = widgets.FileUpload(
            description="Upload (.zip/.gpkg)",
            accept=".zip,.gpkg",
            multiple=False,
            layout=widgets.Layout(width="200px"),
        )
        self.map.add_control(WidgetControl(widget=self.uploader, position="topright"))

    def _wire_events(self):
        self.draw_control.on_draw(self._handle_draw)
        self.uploader.observe(self._handle_upload, "value")

    def _handle_draw(self, _, action, geo_json):
        if action == "created":
            self._clear_extent()  # delete previous rectangle & layer
            poly = Polygon(shape(geo_json["geometry"]))
            bbox = poly.bounds
            if validate_area(bbox, self.area_limit, self.output):
                self._set_extent_from_bbox(bbox)
            else:
                self.draw_control.clear()
                self.draw_control.last_draw = {"type": "Feature", "geometry": None}
        elif action == "deleted":
            self._clear_extent()
            with self.output:
                self.output.clear_output()
                display(HTML("<i>Extent removed.</i>"))
        else:
            raise ValueError(f"Unknown action: {action}")

    def _handle_upload(self, change):
        upload_value = self.uploader.value
        if not upload_value:
            return

        # Handle both new (tuple) and old (dict) structures
        if isinstance(upload_value, dict):
            # old structure
            name, fileinfo = next(iter(upload_value.items()))
            content = fileinfo["content"]
        elif isinstance(upload_value, (tuple, list)):
            # new structure
            fileinfo = upload_value[0]
            name = fileinfo["name"]
            content = fileinfo["content"]
        else:
            raise ValueError("Unexpected upload format.")

        try:
            # we are uploading a new file, so we clear the previous extent
            self._clear_extent()
            gdf = self._load_vector_file(name, content)
            bbox = gdf.total_bounds  # (minx, miny, maxx, maxy)
            if validate_area(bbox, self.area_limit, self.output):
                self._set_extent_from_bbox(bbox)
        except Exception as e:
            with self.output:
                self.output.clear_output()
                display(
                    HTML(f"<span style='color:red'><b>Upload failed:</b> {e}</span>")
                )
            return

        # Reset uploader widget in a version-compatible way
        if isinstance(self.uploader.value, dict):
            self.uploader.value.clear()
        else:
            self.uploader.value = ()

    def _set_extent_from_bbox(self, bbox):
        # Remove any previously drawn/uploaded geometry
        self._clear_extent()

        # Create a new polygon from the bounding box
        self.poly = box(*bbox)  # shapely polygon
        self.get_extent("latlon")  # sets self.spatial_extent
        self._remove_geojson_layer()

        geojson_layer = GeoJSON(data=json.loads(gpd.GeoSeries([self.poly]).to_json()))
        self.map.add_layer(geojson_layer)
        self._last_geojson_layer = geojson_layer

        self.map.fit_bounds([[bbox[1], bbox[0]], [bbox[3], bbox[2]]])

    def _remove_geojson_layer(self):
        # remove previously added polygon layer if it exists
        if hasattr(self, "_last_geojson_layer"):
            try:
                self.map.remove_layer(self._last_geojson_layer)
            except Exception:
                pass

    def _clear_extent(self):
        self._remove_geojson_layer()
        self.draw_control.clear()
        self.poly = None
        self.spatial_extent = None

    def _load_vector_file(self, name, raw_bytes):
        tmpdir = tempfile.mkdtemp()
        path = Path(tmpdir, name)
        path.write_bytes(raw_bytes)

        if name.endswith(".zip"):
            with zipfile.ZipFile(path) as zf:
                zf.extractall(tmpdir)
            shp = next(p for p in Path(tmpdir).glob("*.shp"))
            gdf = gpd.read_file(shp)
        elif name.endswith(".gpkg"):
            gdf = gpd.read_file(str(path))
        else:
            raise ValueError("Unsupported file format.")

        gdf = gdf.to_crs(4326)  # convert to WGS84

        # Run some basic checks
        if gdf.empty:
            raise ValueError("The provided vector file is empty or invalid.")
        if gdf.geom_type.isin(["Point"]).all():
            coords = gdf.geometry.apply(lambda pt: (pt.x, pt.y)).drop_duplicates()
            if len(coords) < 2:
                raise ValueError(
                    "Upload must contain at least 2 distinct points to define an extent."
                )

        return gdf

    def get_extent(self, projection="utm") -> "BoundingBoxExtent":
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
            If no rectangle has been drawn on the map or no file has been uploaded.
        """

        if self.poly is None:
            raise ValueError("No extent selected; draw or upload first.")
        bbox = self.poly.bounds
        if projection == "utm":
            bbox, epsg = _latlon_to_utm(bbox)
            self.spatial_extent = BoundingBoxExtent(*bbox, epsg)
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
