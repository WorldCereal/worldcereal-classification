from ipyleaflet import (Map, basemaps, DrawControl,
                        SearchControl)


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
