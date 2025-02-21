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

    def visualize_extent(self):
        """Visualizes the spatial extent of the collection on a map."""

        from ipyleaflet import Map, Rectangle, basemaps

        # Get the extent of the collection
        colbbox = self.spatial_extent.get("bbox", None)
        if colbbox is None:
            raise ValueError("No bounding box found for this collection.")
        colbbox = colbbox[0]
        bbox = [[colbbox[1], colbbox[0]], [colbbox[3], colbbox[2]]]

        # compute the center of the bounding box
        center = [(colbbox[1] + colbbox[3]) / 2, (colbbox[0] + colbbox[2]) / 2]

        # create a rectangle from the bounding box
        rectangle = Rectangle(bounds=bbox, color="green", weight=2, fill_opacity=0.1)

        # Create the basemap
        m = Map(
            basemap=basemaps.CartoDB.Positron,
            center=center,
            zoom=6,
            scroll_wheel_zoom=True,
        )

        # Add the rectangle to the map
        m.add_layer(rectangle)

        return m
