import numpy as np

from satio.features import Features


def elev_from_dem(dem_collection,
                  dem_settings=None,
                  resolution=10):
    names = ['DEM-alt-20m']
    dem_settings = dem_settings or {}
    features_names = dem_settings.get('features_names',
                                      names)
    altitude = dem_collection.load().astype(np.float32)

    feats = Features(altitude, features_names)

    if resolution == 10:
        feats = feats.upsample()
    elif resolution == 20:
        pass
    else:
        raise ValueError("`resolution` should be 10 or 20.")
    return feats
