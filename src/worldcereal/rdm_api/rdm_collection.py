from typing import List


class RdmCollection:
    """Data class to host collections queried from the RDM API."""

    def __init__(self, **metadata):
        """Initializes the RdmCollection object with metadata from the RDM API.
        Collection metadata is passed as keyword arguments and stored as attributes.
        All default collection metadata items are expected.

        RdmCollection can be initialized directly from the RDM API response resulting from
        a query to the collections endpoint (see get_collections function in rdmi_interaction.py).
        """
        # check presence of mandatory metadata items
        if not metadata.get("collectionId"):
            raise ValueError(
                "Collection ID is missing, cannot create RdmCollection object."
            )
        if not metadata.get("accessType"):
            raise ValueError(
                "Access type is missing, cannot create RdmCollection object."
            )

        # Get all metadata items
        self.id = metadata.get("collectionId")
        self.title = metadata.get("title")
        self.feature_count = metadata.get("featureCount")
        self.data_type = metadata.get("type")
        self.access_type = metadata.get("accessType")
        self.observation_method = metadata.get("typeOfObservationMethod")
        self.confidence_lc = metadata.get("confidenceLandCover")
        self.confidence_ct = metadata.get("confidenceCropType")
        self.confidence_irr = metadata.get("confidenceIrrigationType")
        self.ewoc_codes = metadata.get("ewocCodes")
        self.irr_codes = metadata.get("irrTypes")
        self.extent = metadata.get("extent")
        if self.extent:
            self.spatial_extent = self.extent["spatial"]
            self.temporal_extent = self.extent["temporal"]["interval"][0]
        else:
            self.spatial_extent = None
            self.temporal_extent = None
        self.additional_data = metadata.get("additionalData")
        self.crs = metadata.get("crs")
        self.last_modified = metadata.get("lastModificationTime")
        self.last_modified_by = metadata.get("lastModifierId")
        self.creation_time = metadata.get("creationTime")
        self.created_by = metadata.get("creatorId")
        self.fid = metadata.get("id")

    def print_metadata(self):

        print("#######################")
        print("Collection Metadata:")
        print(f"ID: {self.id}")
        print(f"Title: {self.title}")
        print(f"Number of samples: {self.feature_count}")
        print(f"Data type: {self.data_type}")
        print(f"Access type: {self.access_type}")
        print(f"Observation method: {self.observation_method}")
        print(f"Confidence score for land cover: {self.confidence_lc}")
        print(f"Confidence score for crop type: {self.confidence_ct}")
        print(f"Confidence score for irrigation label: {self.confidence_irr}")
        print(f"List of available crop types: {self.ewoc_codes}")
        print(f"List of available irrigation labels: {self.irr_codes}")
        print(f"Spatial extent: {self.spatial_extent}")
        print(f"Coordinate reference system (CRS): {self.crs}")
        print(f"Temporal extent: {self.temporal_extent}")
        print(f"Additional data: {self.additional_data}")
        print(f"Last modified: {self.last_modified}")
        print(f"Last modified by: {self.last_modified_by}")
        print(f"Creation time: {self.creation_time}")
        print(f"Created by: {self.created_by}")
        print(f"fid: {self.fid}")


def visualize_spatial_extents(collections: List[RdmCollection]):
    """Visualizes the spatial extent of multiple collections on a map."""

    from ipyleaflet import Map, Rectangle, basemaps

    if len(collections) == 1:
        zoom = 5
        colbbox = collections[0].spatial_extent.get("bbox", None)
        if colbbox is None:
            raise ValueError(
                f"No bounding box found for collection {collections[0].id}."
            )
        colbbox = colbbox[0]
        # compute the center of the bounding box
        center = [(colbbox[1] + colbbox[3]) / 2, (colbbox[0] + colbbox[2]) / 2]
    else:
        zoom = 1
        center = [0, 0]

    # Create the basemap
    m = Map(
        basemap=basemaps.CartoDB.Positron,
        zoom=zoom,
        center=center,
        scroll_wheel_zoom=True,
    )

    # Get the extent of each collection
    for col in collections:
        colbbox = col.spatial_extent.get("bbox", None)
        if colbbox is None:
            raise ValueError(f"No bounding box found for collection {col.id}.")
        colbbox = colbbox[0]
        bbox = [[colbbox[1], colbbox[0]], [colbbox[3], colbbox[2]]]

        # create a rectangle from the bounding box
        rectangle = Rectangle(bounds=bbox, color="green", weight=2, fill_opacity=0.1)

        # Add the rectangle to the map
        m.add_layer(rectangle)

    return m
