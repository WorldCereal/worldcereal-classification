"""Shared collation policy for training datasets and DataLoaders."""

ATTR_KEYS_ALLOW_PARTIAL_NONE = {
    "landcover_label",
    "croptype_label",
    "label_task",
    "LC10_confidence_nonoutlier",
    "CTY24_confidence_nonoutlier",
    "LC10_anomaly_flag",
    "CTY24_anomaly_flag",
}
