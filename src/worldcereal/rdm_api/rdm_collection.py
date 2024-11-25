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
