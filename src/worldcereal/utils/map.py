import datetime
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Literal, Optional

import geopandas as gpd
from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
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


def validate_area(
    bbox,
    area_limit: int,
    output: widgets.Output,
    label: Optional[str] = None,
    display_output: bool = True,
    return_area: bool = False,
):
    """
    Validate bbox area against a limit.

    Parameters
    ----------
    bbox : tuple
        Bounding box as (minx, miny, maxx, maxy) in lat/lon.
    area_limit : int
        Maximum area in km^2. None disables the limit.
    output : widgets.Output
        Output widget for status messages.
    label : str, optional
        Prefix label used in the output message.
    display_output : bool, optional
        When True, writes the extent and area to the output widget.
    return_area : bool, optional
        When True, returns a tuple (is_valid, area_km2).

    Returns
    -------
    bool or tuple
        True/False when return_area is False, otherwise (is_valid, area_km2).
    """
    bbox_utm, _ = _latlon_to_utm(bbox)
    area = (bbox_utm[2] - bbox_utm[0]) * (bbox_utm[3] - bbox_utm[1]) / 1_000_000
    is_valid = area_limit is None or area <= area_limit

    if display_output:
        with output:
            output.clear_output()
            label_prefix = f"{label} " if label else ""
            display(HTML(f"<b>{label_prefix}extent:</b> {bbox}"))
            display(HTML(f"<b>Area of extent:</b> {area:.2f} km²"))

            if not is_valid:
                display(
                    HTML(
                        f"<span style='color:red'><b>Area too large "
                        f"(>{area_limit} km²). Please select a smaller area.</b></span>"
                    )
                )

    if return_area:
        return is_valid, area
    return is_valid


class ui_map:
    def __init__(
        self,
        area_limit: Optional[int] = None,
        display_ui: bool = True,
        mode: Literal["single", "multi"] = "single",
    ):
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
        self.mode = mode
        self.gdf: Optional[gpd.GeoDataFrame] = None
        self._last_drawn_geometry: Optional[Polygon] = None
        self._pending_source: Optional[str] = None
        self._pending_gdf: Optional[gpd.GeoDataFrame] = None

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
            controls = [self.map]
            if hasattr(self, "input"):
                controls.append(self.input)
            controls.append(self.output)
            display(widgets.VBox(controls))

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

        rectangle_options = {
            "shapeOptions": {
                "fillColor": "#6be5c3",
                "color": "#00F",
                "fillOpacity": 0.3,
            },
            "allowIntersection": False,
            "metric": ["km"],
        }

        polygon_options = {
            "shapeOptions": {
                "fillColor": "#6be5c3",
                "color": "#00F",
                "fillOpacity": 0.3,
            },
            "allowIntersection": False,
            "metric": ["km"],
        }

        self.draw_control = DrawControl(
            rectangle=rectangle_options,
            polygon=polygon_options if self.mode == "multi" else {},
            circle={},
            circlemarker={},
            polyline={},
            remove=True,
            edit=False,
        )
        self.map.add_control(self.draw_control)

    def _add_controls(self):
        self.uploader = widgets.FileUpload(
            description="Upload (.zip/.gpkg/.parquet)",
            accept=".zip,.gpkg,.parquet",
            multiple=False,
            layout=widgets.Layout(width="200px"),
        )
        self.map.add_control(WidgetControl(widget=self.uploader, position="topright"))

        self.id_input = widgets.Text(
            placeholder="Enter an ID",
            description="AOI ID",
            layout=widgets.Layout(width="260px"),
        )
        self.id_field_input = widgets.Dropdown(
            options=[],
            description="Upload ID",
            layout=widgets.Layout(width="260px"),
        )
        self.submit_button = widgets.Button(
            description="Submit", button_style="success"
        )
        self.submit_button.on_click(self._handle_submit)
        self.input = widgets.VBox(
            [self.id_input, self.id_field_input, self.submit_button]
        )
        self._hide_inputs()

    def _wire_events(self):
        self.draw_control.on_draw(self._handle_draw)
        self.uploader.observe(self._handle_upload, "value")

    def _handle_draw(self, _, action, geo_json):
        if action == "created":
            poly = Polygon(shape(geo_json["geometry"]))
            bbox = poly.bounds
            if validate_area(bbox, self.area_limit, self.output):
                self._pending_source = "draw"
                self._pending_gdf = None
                self._last_drawn_geometry = poly
                self._show_draw_inputs()
                with self.output:
                    display(
                        HTML(
                            "<i>Geometry captured. Provide an AOI ID and click Submit.</i>"
                        )
                    )
            else:
                self.draw_control.clear()
                self.draw_control.last_draw = {"type": "Feature", "geometry": None}
                if self.mode == "multi" and self.gdf is not None and not self.gdf.empty:
                    self._set_geojson_layer_from_gdf(self.gdf)
        elif action == "deleted":
            if self.mode == "single":
                self._clear_extent()
                with self.output:
                    self.output.clear_output()
                    display(HTML("<i>Extent removed.</i>"))
                return

            deleted_geom = None
            if geo_json and geo_json.get("geometry"):
                deleted_geom = Polygon(shape(geo_json["geometry"]))

            if deleted_geom is None:
                with self.output:
                    self.output.clear_output()
                    display(HTML("<i>Geometry removed.</i>"))
                return

            if (
                self._last_drawn_geometry is not None
                and self._last_drawn_geometry.equals_exact(deleted_geom, tolerance=1e-6)
            ):
                self._last_drawn_geometry = None
                self._pending_source = None
                self._pending_gdf = None
                self._hide_inputs()

            if self.gdf is None or self.gdf.empty:
                with self.output:
                    self.output.clear_output()
                    display(HTML("<i>Geometry removed.</i>"))
                return

            mask = self.gdf.geometry.apply(
                lambda geom: geom.equals_exact(deleted_geom, tolerance=1e-6)
            )
            if mask.any():
                self.gdf = self.gdf.loc[~mask].reset_index(drop=True)
                if self.gdf.empty:
                    self._remove_geojson_layer()
                else:
                    self._set_geojson_layer_from_gdf(self.gdf)
                with self.output:
                    self.output.clear_output()
                    display(HTML("<i>AOI removed.</i>"))
            else:
                with self.output:
                    self.output.clear_output()
                    display(
                        HTML(
                            "<i>Geometry removed. No saved AOI matched that shape.</i>"
                        )
                    )
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
            if self.mode == "single":
                self._clear_extent()
            else:
                self._clear_pending_state()
            gdf = self._load_vector_file(name, content)
            if self.mode == "single":
                bbox = gdf.total_bounds  # (minx, miny, maxx, maxy)
                if not validate_area(bbox, self.area_limit, self.output):
                    return

            else:
                if self.area_limit is not None:
                    for idx, geom in enumerate(gdf.geometry, start=1):
                        is_valid, area_km2 = validate_area(
                            geom.bounds,
                            self.area_limit,
                            self.output,
                            display_output=False,
                            return_area=True,
                        )
                        if not is_valid:
                            raise ValueError(
                                f"Geometry {idx} exceeds {self.area_limit} km^2 "
                                f"({area_km2:.2f} km^2)."
                            )
            # display the resulting geometries on the map
            self._set_geojson_layer_from_gdf(gdf)
            self._pending_source = "upload"
            self._pending_gdf = gdf
            self._update_upload_id_options(gdf)
            self._show_upload_inputs()
            with self.output:
                display(
                    HTML(
                        f"<i>Upload captured ({len(gdf)} geometries). Select the ID column and click Submit.</i>"
                    )
                )
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

    def _handle_submit(self, _):
        with self.output:
            self.output.clear_output()

        if self._pending_source is None:
            with self.output:
                display(HTML("<i>Draw or upload an AOI first.</i>"))
            return

        if self._pending_source == "draw":
            if self._last_drawn_geometry is None:
                with self.output:
                    display(HTML("<i>Draw an AOI first.</i>"))
                return

            object_id = self.id_input.value.strip()
            if not object_id:
                with self.output:
                    display(HTML("<i>Please enter an AOI ID before submitting.</i>"))
                return
            if (
                self.mode == "multi"
                and self.gdf is not None
                and not self.gdf.empty
                and object_id in set(self.gdf["id"])
            ):
                with self.output:
                    display(HTML("<i>AOI ID already exists. Choose a unique ID.</i>"))
                return

            self._set_gdf_from_polygon(
                self._last_drawn_geometry,
                object_id,
                replace=self.mode == "single",
            )
            self._last_drawn_geometry = None
            self.id_input.value = ""

        elif self._pending_source == "upload":
            if self._pending_gdf is None:
                with self.output:
                    display(HTML("<i>Upload an AOI first.</i>"))
                return

            id_field = self.id_field_input.value
            if not id_field:
                with self.output:
                    display(
                        HTML("<i>Please select the ID column before submitting.</i>")
                    )
                return

            try:
                self._set_gdf_from_upload(self._pending_gdf, id_field)
            except Exception as exc:
                with self.output:
                    display(
                        HTML(
                            f"<span style='color:red'><b>Upload failed:</b> {exc}</span>"
                        )
                    )
                return

            self.id_field_input.value = None
            self._pending_gdf = None
        else:
            with self.output:
                display(HTML("<i>Unknown submit action.</i>"))
            return

        self._pending_source = None
        self._hide_inputs()
        if self.mode == "single":
            bbox = self.gdf.total_bounds
            validate_area(bbox, self.area_limit, self.output)
        self._set_geojson_layer_from_gdf(self.gdf)
        with self.output:
            if self.mode == "multi":
                display(
                    HTML("<i>AOI saved. You can add more polygons if you want.</i>")
                )
            else:
                display(HTML("<i>AOI saved.</i>"))

    def _set_geojson_layer_from_gdf(self, gdf: gpd.GeoDataFrame) -> None:
        if gdf is None or gdf.empty:
            return
        self._remove_geojson_layer()
        geojson = json.loads(gdf.to_json())
        self.draw_control.data = geojson.get("features", [])

        bounds = gdf.total_bounds
        self.map.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    def _remove_geojson_layer(self):
        # remove previously added polygon layer if it exists
        if hasattr(self, "_last_geojson_layer"):
            try:
                self.map.remove_layer(self._last_geojson_layer)
            except Exception:
                pass
        self.draw_control.data = []

    def _clear_extent(self):
        self._remove_geojson_layer()
        self.draw_control.clear()
        self._last_drawn_geometry = None
        self.gdf = None
        self._pending_source = None
        self._pending_gdf = None
        self._hide_inputs()

    def _clear_pending_state(self):
        self._last_drawn_geometry = None
        self._pending_source = None
        self._pending_gdf = None
        self._hide_inputs()

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
        elif name.endswith(".parquet"):
            gdf = gpd.read_parquet(str(path))
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

    def get_gdf(self) -> gpd.GeoDataFrame:
        if self.gdf is None or self.gdf.empty:
            raise ValueError("No geometries available; draw or upload first.")
        return self.gdf.copy()

    def get_poly(self) -> Polygon:
        if self.mode != "single":
            raise ValueError("get_poly is only available in single mode.")
        gdf = self.get_gdf()
        if len(gdf) != 1:
            raise ValueError("Expected exactly one geometry in single mode.")
        return gdf.geometry.iloc[0]

    def get_bbox(self) -> BoundingBoxExtent:
        if self.mode != "single":
            raise ValueError("get_bbox is only available in single mode.")
        gdf = self.get_gdf()
        return gdf_to_bbox_extent(gdf)

    def save_gdf(self, output_dir: Path, outputname: Optional[str] = None) -> Path:
        """Save the current GeoDataFrame to a GeoPackage file."""
        gdf = self.get_gdf()
        if outputname is None:
            if self.mode == "single":
                outputname = gdf.iloc[0]["id"]
            else:
                raise ValueError("You must provide an output name for your file.")
        output_path = output_dir / f"{outputname}.gpkg"
        output_dir.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_path, driver="GPKG")
        logger.info(f"Saved AOI GeoDataFrame to {output_path}")
        return output_path

    def _set_gdf_from_polygon(
        self, geometry_obj: Polygon, object_id: str, replace: bool
    ) -> None:
        new_entry = gpd.GeoDataFrame(
            [{"id": object_id, "geometry": geometry_obj}],
            geometry="geometry",
            crs="EPSG:4326",
        )
        if replace or self.gdf is None:
            self.gdf = new_entry
        else:
            self.gdf = gpd.pd.concat([self.gdf, new_entry], ignore_index=True)

    def _set_gdf_from_upload(self, gdf: gpd.GeoDataFrame, id_field: str) -> None:
        gdf = gdf.copy()
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        if "geometry" not in gdf.columns:
            raise ValueError("Uploaded file must contain a geometry column.")

        if id_field not in gdf.columns:
            raise ValueError(f"ID column '{id_field}' not found in upload.")

        if not gdf[id_field].is_unique:
            raise ValueError("ID column must contain unique values.")

        gdf = gdf[[id_field, "geometry"]].copy()
        gdf = gdf.rename(columns={id_field: "id"})

        if self.mode == "single" and len(gdf) != 1:
            raise ValueError("Single mode requires exactly one geometry.")

        if self.mode == "multi" and self.gdf is not None and not self.gdf.empty:
            # check for ID conflicts with existing geometries
            existing_ids = set(self.gdf["id"])
            new_ids = set(gdf["id"])
            overlap = existing_ids.intersection(new_ids)
            if overlap:
                raise ValueError(
                    "Uploaded IDs already exist: "
                    + ", ".join(sorted(str(v) for v in overlap))
                )
            # Add uploaded geometries to existing ones
            self.gdf = gpd.pd.concat([self.gdf, gdf], ignore_index=True)
        else:
            self.gdf = gdf

    def _show_draw_inputs(self) -> None:
        self.id_input.layout.display = ""
        self.id_field_input.layout.display = "none"
        self.submit_button.layout.display = ""

    def _show_upload_inputs(self) -> None:
        self.id_input.layout.display = "none"
        self.id_field_input.layout.display = ""
        self.submit_button.layout.display = ""

    def _hide_inputs(self) -> None:
        self.id_input.layout.display = "none"
        self.id_field_input.layout.display = "none"
        self.submit_button.layout.display = "none"

    def _update_upload_id_options(self, gdf: gpd.GeoDataFrame) -> None:
        cols = [c for c in gdf.columns if c != "geometry"]
        options = ["", *cols]
        self.id_field_input.options = options
        self.id_field_input.value = None


def gdf_to_bbox_extent(gdf: gpd.GeoDataFrame) -> BoundingBoxExtent:
    """Convert a single-row GeoDataFrame to a BoundingBoxExtent.

    The extent is returned in the GeoDataFrame CRS.
    """
    if gdf is None or gdf.empty:
        raise ValueError("GeoDataFrame is empty.")
    if len(gdf) != 1:
        raise ValueError("GeoDataFrame must contain exactly one geometry.")
    if gdf.crs is None:
        raise ValueError("GeoDataFrame must have a defined CRS.")

    bounds = gdf.total_bounds
    epsg = gdf.crs.to_epsg()
    if epsg is None:
        raise ValueError("GeoDataFrame CRS does not define an EPSG code.")
    return BoundingBoxExtent(
        west=bounds[0],
        south=bounds[1],
        east=bounds[2],
        north=bounds[3],
        epsg=epsg,
    )


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


def visualize_rdm_geoparquet(
    src_path: str,
    selected_sample_ids: Optional[list] = None,
):
    """Visualize an RDM collection geoparquet file on a map.

    Parameters
    ----------
    src_path : str
        Path to the geoparquet file.
    selected_sample_ids : Optional[list], optional
        List of sample_ids that are selected. If provided, will display
        selected samples (large, colored by crop type) and non-selected
        samples (small, red) separately.
    """

    gdf = gpd.read_parquet(src_path)

    # Preprocess datetime columns to ISO format strings
    gdf = gdf.copy()
    geometry_col = gdf.geometry.name if hasattr(gdf, "geometry") else None
    for col in gdf.columns:
        if col == geometry_col:
            continue
        if pd.api.types.is_datetime64_any_dtype(gdf[col]) or gdf[col].dtype == "object":
            gdf[col] = gdf[col].apply(
                lambda v: (
                    v.isoformat()
                    if isinstance(v, (datetime.date, datetime.datetime))
                    else v
                )
            )

    # Ensure WGS84 for map display
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # Compute total bounds for map centering and auto-zoom
    minx, miny, maxx, maxy = gdf.total_bounds

    # Extract unique ewoc_code values and assign colors
    unique_codes = sorted(gdf["ewoc_code"].unique())

    # If a selection is provided, restrict legend/colors to selected crop types
    selected_codes = None
    if selected_sample_ids is not None and "sample_id" in gdf.columns:
        selected_mask = gdf["sample_id"].isin(selected_sample_ids)
        selected_codes = sorted(
            gdf.loc[selected_mask, "ewoc_code"].dropna().unique().tolist()
        )

    codes_for_colors = (
        selected_codes
        if selected_codes is not None and len(selected_codes) > 0
        else unique_codes
    )

    # Use multiple colormaps to maximize color diversity
    if len(codes_for_colors) <= 10:
        cmap = plt.get_cmap("tab10")
    elif len(codes_for_colors) <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        # For many crop types, use a combination of colormaps
        # Sample colors from different colormaps to maximize distinctiveness
        cmap_names = ["tab20", "tab20b", "tab20c", "Set3", "Paired"]
        all_colors = []
        for cmap_name in cmap_names:
            cm = plt.get_cmap(cmap_name)
            n_colors = cm.N if hasattr(cm, "N") else 20
            for i in range(n_colors):
                all_colors.append(to_hex(cm(i)))

        # Select colors by maximizing distance (every Nth color)
        step = max(1, len(all_colors) // len(codes_for_colors))
        selected_colors = [
            all_colors[i * step % len(all_colors)] for i in range(len(codes_for_colors))
        ]
        colors = {code: selected_colors[i] for i, code in enumerate(codes_for_colors)}

    if len(codes_for_colors) <= 20:
        # For smaller sets, use equal spacing
        colors = {
            code: to_hex(cmap(i / max(1, len(codes_for_colors) - 1)))
            for i, code in enumerate(codes_for_colors)
        }

    # Add a new column for the color associated with each ewoc_code
    gdf["color"] = gdf["ewoc_code"].map(colors)

    m = Map(
        basemap=basemaps.Esri.WorldImagery,
        scroll_wheel_zoom=True,
    )
    # Fit map to data extent
    m.fit_bounds([[miny, minx], [maxy, maxx]])

    # Handle selected vs non-selected visualization
    if selected_sample_ids is not None and "sample_id" in gdf.columns:
        # Mark which samples are selected
        gdf["is_selected"] = gdf["sample_id"].isin(selected_sample_ids)

        # Split into selected and non-selected
        selected_gdf = gdf[gdf["is_selected"]].copy()
        non_selected_gdf = gdf[~gdf["is_selected"]].copy()

        # Add non-selected samples layer (smaller, red)
        if len(non_selected_gdf) > 0:
            non_selected_data = non_selected_gdf.__geo_interface__

            def non_selected_style(feature):
                return {
                    "color": "darkred",
                    "fillColor": "red",
                    "opacity": 0.6,
                    "fillOpacity": 0.4,
                    "weight": 1,
                }

            non_selected_layer = GeoJSON(
                data=non_selected_data,
                style_callback=non_selected_style,
                point_style={"radius": 3},
                name="Non-selected samples",
            )
            m.add_layer(non_selected_layer)

        # Add selected samples layer (larger, colored by crop type)
        if len(selected_gdf) > 0:
            selected_data = selected_gdf.__geo_interface__

            def selected_style(feature):
                properties = feature["properties"]
                return {
                    "color": "darkgreen",
                    "fillColor": properties["color"],
                    "opacity": 1,
                    "fillOpacity": 0.8,
                    "weight": 2,
                }

            selected_layer = GeoJSON(
                data=selected_data,
                style_callback=selected_style,
                point_style={"radius": 6},
                name="Selected samples",
            )
            m.add_layer(selected_layer)
    else:
        # Original behavior: display all samples uniformly
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
    legend_codes = (
        selected_codes
        if selected_codes is not None and len(selected_codes) > 0
        else unique_codes
    )
    crop_types = translate_ewoc_codes(legend_codes)

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
