import numpy as np
from satio.features import (percentile_iqr, std,
                            kurto, skewness,
                            summation)
import xarray as xr

from worldcereal.utils.masking import SCL_MASK_VALUES
from worldcereal.features.feat_sen2agri import sen2agri_temp_feat
from worldcereal.features.feat_irr import sum_div


def tsteps(x, n_steps=6):
    da = xr.DataArray(x, dims=('t', 'x', 'y'))
    dtype = da.dtype
    if dtype == 'uint16':
        da = da.astype('float32')
        da.values[da.values == 0] = np.nan
    elif dtype == 'float32':
        pass
    else:
        raise TypeError((f'Unsupported dtype `{dtype}` '
                         'for tsteps function.'))

    da_resampled = da.groupby_bins('t', bins=n_steps).mean()

    if dtype == 'uint16':
        da_resampled.values[np.isnan(da_resampled.values)] = 0
        da_resampled = da_resampled.astype(dtype)

    return da_resampled.values


def get_comp_mode_dict():
    return {
        'dewpoint_temperature': 'median',
        'precipitation_flux': 'sum',
        'solar_radiation_flux': 'median',
        'temperature_max': 'max',
        'temperature_mean': 'median',
        'temperature_min': 'min',
        'vapour_pressure': 'median',
        'wind_speed': 'median',
        'et0': 'sum'
    }


# CROPLAND

def get_default_settings():
    return {
        'OPTICAL':
            {
                'bands': ["B02", "B03", "B04", "B05", "B06",
                          "B07", "B08", "B11", "B12"],
                'rsis': ["ndvi", "evi", "anir",
                         "ndmi", "ndwi", "brightness",
                         "ndre1", "ndre2", "ndre3",
                         "ndre4", "ndre5",
                         "savi", "ndgi", "nbr"
                         ],
                'composite': {
                    'freq': 10,
                    'window': 20,
                    'start': None,
                    'end': None},
                'smooth_ndvi': False,
                'mask': {
                    'erode_r': 3,
                    'dilate_r': 13,
                    'mask_values': SCL_MASK_VALUES,
                    'multitemporal': False},
                'seasons': {'rsi': 'evi',
                            'amp_thr1': 0.15,
                            'amp_thr2': 0.35,
                            'max_window': 250,
                            'partial_start': False,
                            'partial_end': False,
                            'smooth_rsi': False}
            },
        'SAR':
            {
                'bands': ['VV', 'VH'],
                'rsis': ['vh_vv', 'rvi'],
                'composite': {
                    'freq': 12,
                    'window': 18,
                    'mode': 'mean',
                    'start': None,
                    'end': None},
                'mask': {
                    'METEOcol': '',
                    'precipitation_threshold': 10}
            },
        'METEO':
            {
                'bands': ['precipitation_flux',
                          'temperature_mean'],
                'composite': {
                    'freq': 1,  # Beware: changing this can have unexpected results for precipitation!  # NOQA
                    'window': 1,
                    'mode': get_comp_mode_dict(),
                    'start': None,
                    'end': None}
            }
    }


def get_cropland_catboost_settings():
    return {
        'OPTICAL':
            {
                'bands': ["B02", "B03", "B04", "B05",
                          "B07", "B08", "B11", "B12"],
                'rsis': ["ndvi", "anir",
                         "ndmi", "ndwi",
                         "ndre1", "ndre5", "ndgi"],
                'composite': {
                    'freq': 10,
                    'window': 20,
                    'start': None,
                    'end': None},
                'smooth_ndvi': False,
                'mask': {
                    'erode_r': 3,
                    'dilate_r': 13,
                    'mask_values': SCL_MASK_VALUES,
                    'multitemporal': False},
            },
        'SAR':
            {
                'bands': ['VV', 'VH'],
                'rsis': ['rvi'],
                'composite': {
                    'freq': 12,
                    'window': 18,
                    'mode': 'mean',
                    'start': None,
                    'end': None},
                # 'mask': {
                #     'METEOcol': '',
                #     'precipitation_threshold': 10}
            }
    }


def get_active_crop_settings():
    return {
        'OPTICAL':
            {
                'bands': [],
                'rsis': ["evi"],
                'composite': {
                    'freq': 10,
                    'window': 20,
                    'start': None,
                    'end': None},
                'mask': {
                    'erode_r': 3,
                    'dilate_r': 13,
                    'mask_values': SCL_MASK_VALUES,
                    'multitemporal': False},
                'seasons': {'rsi': 'evi',
                            'amp_thr1': 0.15,
                            'amp_thr2': 0.35,
                            'max_window': 250,
                            'partial_start': False,
                            'partial_end': False,
                            'smooth_rsi': False}
            }
    }


def calc_rgbBR(B02, B03, B04):
    return np.sqrt(np.power(B02, 2) + np.power(B03, 2)
                   + np.power(B04, 2))


def get_default_rsi_meta():
    return {
        'OPTICAL':
        {
            'rgbBR': {'bands': ['B02', 'B03', 'B04'],
                      'range': [0, 1],
                      'clamp_range': False,
                      'scale': 1,
                      'native_res': 10,
                      'func': calc_rgbBR
                      }}}


def get_cropland_features_meta():
    return {
        'OPTICAL':
            {
                "std": {
                    "function": std,
                    "names": ['std']},
                "kurt": {
                    "function": kurto,
                    "names": ['kurt']},
                "skew": {
                    "function": skewness,
                    "names": ['skew']},
                "sen2agri_temp_feat": {
                    "function": sen2agri_temp_feat,
                    "parameters": {
                        'time_start': None,
                        'time_freq': 10,
                        'w': 2,
                        'delta': 0.05,
                        'tsoil': 0.2},
                    "bands": ['ndvi'],
                    "names": ['maxdif', 'mindif', 'difminmax',
                              'peak', 'lengthpeak', 'areapeak',
                              'ascarea', 'asclength', 'ascratio',
                              'descarea', 'desclength',
                              'descratio', 'soil1', 'soil2']},
                "pheno_mult_season": {}
            },

        'SAR':
            {
                "std": {
                    "function": std,
                    "names": ['std']},
            },
        'METEO':
            {
                "sum": {
                    "function": summation,
                    "names": ['sum'],
                    "bands": ['precipitation_flux']},
                "percentile_iqr": {
                    "function": percentile_iqr,
                    "parameters": {'q': [10, 50, 90],
                                   'iqr': [25, 75]},
                    "names": ['p10', 'p50', 'p90', 'iqr'],
                    "bands": ['temperature_mean',
                              'temperature_min',
                              'temperature_max']}
            }
    }


def get_cropland_catboost_features_meta():
    return {
        'OPTICAL': {
            "percentile_iqr": {
                "function": percentile_iqr,
                "parameters": {
                    'q': [10, 50, 90],
                    'iqr': [25, 75]
                },
                "names": ['p10', 'p50', 'p90', 'iqr']
            },
            "tsteps": {
                "function": tsteps,
                "parameters": {
                    'n_steps': 6,
                },
                "names": ['ts0', 'ts1', 'ts2', 'ts3', 'ts4', 'ts5'],
                "bands": ['ndvi']
            },
            "std": {
                "function": std,
                "names": ['std']},
            "skew": {
                "function": skewness,
                "names": ['skew']},
            "sen2agri_temp_feat": {
                "function": sen2agri_temp_feat,
                "parameters": {
                    'time_start': None,
                    'time_freq': 10,
                    'w': 2,
                    'delta': 0.05,
                    'tsoil': 0.2},
                "bands": ['ndvi'],
                "names": ['maxdif', 'mindif', 'difminmax',
                          'peak', 'lengthpeak', 'areapeak',
                          'ascarea', 'asclength', 'ascratio',
                          'descarea', 'desclength',
                          'descratio', 'soil1', 'soil2']},
        },
        'SAR': {
            "percentile_iqr": {
                "function": percentile_iqr,
                "parameters": {
                    'q': [10, 50, 90],
                    'iqr': [25, 75]
                },
                "names": ['p10', 'p50', 'p90', 'iqr']
            }
        }
    }


def get_active_crop_features_meta():
    return {
        'OPTICAL': {
            "pheno_mult_season": {}
        }
    }


def get_default_ignore_def_feat():
    return {
        'OPTICAL': False,
        'SAR': False,
        'METEO': True
    }


def get_cropland_ignore_def_feat():
    return {
        'OPTICAL': True,
        'SAR': True
    }


def get_cropland_tsteps_settings():
    return {
        'OPTICAL':
        {
            'bands': ["B02", "B03", "B04",
                      "B05", "B06", "B07", "B08",
                      "B11", "B12"],
            'rsis': [],
            'composite': {
                'freq': 10,
                'window': 20,
                'start': None,
                'end': None},
            'smooth_ndvi': False,
            'interpolate': False,
            'mask': {
                'erode_r': 3,
                'dilate_r': 13,
                'mask_values': SCL_MASK_VALUES,
                'multitemporal': False},
        },
        'SAR':
        {
            'bands': ['VV', 'VH'],
            'rsis': [],
            'composite': {
                'freq': 12,
                'window': 18,
                'mode': 'mean',
                'start': None,
                'end': None,
            },
            'interpolate': False,
            # 'mask': {
            #     'METEOcol': '',
            #     'precipitation_threshold': 10}
        },
        'METEO':
        {
            'bands': [
                'temperature_mean',
            ],
            'composite': {
                'freq': 30,
                'window': 30,
                'mode': get_comp_mode_dict(),
                'start': None,
                'end': None}
        }
    }


def get_cropland_tsteps_features_meta():
    return {
        'OPTICAL':
        {
            "tsteps": {
                "function": tsteps,
                "parameters": {
                    'n_steps': 18,
                },
                "names": [f'ts{i}' for i in range(18)]
            }
        },
        'SAR':
        {
            "tsteps": {
                "function": tsteps,
                "parameters": {
                    'n_steps': 18,
                },
                "names": [f'ts{i}' for i in range(18)]
            }
        },
        'METEO':
        {
            "tsteps": {
                "function": tsteps,
                "parameters": {
                    'n_steps': 12,
                },
                "names": [f'ts{i}' for i in range(12)],
                "bands": ['temperature_mean']
            }
        }
    }


def get_cropland_tsteps_parameters():
    return {
        'settings': get_cropland_tsteps_settings(),
        'features_meta': get_cropland_tsteps_features_meta(),
        'rsi_meta': get_default_rsi_meta(),
        'ignore_def_feat': {
            'OPTICAL': True,
            'SAR': True,
            'METEO': True
        },
        'gddnormalization': False
    }


def get_cropland_catboost_parameters():
    return {
        'settings': get_cropland_catboost_settings(),
        'features_meta': get_cropland_catboost_features_meta(),
        'rsi_meta': get_default_rsi_meta(),
        'ignore_def_feat': get_cropland_ignore_def_feat(),
        'gddnormalization': False
    }


def get_active_crop_parameters():
    return {
        'settings': get_active_crop_settings(),
        'features_meta': get_active_crop_features_meta(),
        'rsi_meta': get_default_rsi_meta(),
        'ignore_def_feat': get_cropland_ignore_def_feat(),
        'gddnormalization': False
    }


# CROP TYPE


def get_croptype_features_meta():
    return {
        'OPTICAL':
            {
                "std": {
                    "function": std,
                    "names": ['std']},
                "kurt": {
                    "function": kurto,
                    "names": ['kurt']},
                "skew": {
                    "function": skewness,
                    "names": ['skew']},
                "sen2agri_temp_feat": {
                    "function": sen2agri_temp_feat,
                    "parameters": {
                        'time_start': None,
                        'time_freq': 10,
                        'w': 2,
                        'delta': 0.05,
                        'tsoil': 0.2},
                    "bands": ['ndvi'],
                    "names": ['maxdif', 'mindif', 'difminmax',
                              'peak', 'lengthpeak', 'areapeak',
                              'ascarea', 'asclength', 'ascratio',
                              'descarea', 'desclength',
                              'descratio', 'soil1', 'soil2']},
                "pheno_mult_season": {},
                "pheno_single_season": {"select_season": {"mode": "length",
                                                          "param": 1}}
            },

        'SAR':
            {
                "std": {
                    "function": std,
                    "names": ['std']}
            },
        'METEO':
            {
                "sum": {
                    "function": summation,
                    "names": ['sum'],
                    "bands": ['precipitation_flux']},
                "percentile_iqr": {
                    "function": percentile_iqr,
                    "parameters": {'q': [10, 50, 90],
                                   'iqr': [25, 75]},
                    "names": ['p10', 'p50', 'p90', 'iqr'],
                    "bands": ['temperature_mean',
                              'temperature_min',
                              'temperature_max']}
            }
    }


def get_croptype_tsteps_settings():
    return {
        'OPTICAL':
        {
            'bands': ["B02", "B03", "B04", "B05",
                      "B06", "B07", "B08", "B11", "B12"],
            'rsis': ["evi"],
            'composite': {
                'freq': 10,
                'window': 20,
                'start': None,
                'end': None},
            'smooth_ndvi': False,
            'interpolate': False,
            'mask': {
                'erode_r': 3,
                'dilate_r': 13,
                'mask_values': SCL_MASK_VALUES,
                'multitemporal': True},
            'seasons': {'rsi': 'evi',
                        'amp_thr1': 0.15,
                        'amp_thr2': 0.35,
                        'max_window': 250,
                        'partial_start': False,
                        'partial_end': False,
                        'smooth_rsi': False}
        },
        'SAR':
        {
            'bands': ['VV', 'VH'],
            'rsis': [],
            'composite': {
                'freq': 12,
                'window': 18,
                'mode': 'mean',
                'start': None,
                'end': None},
            'interpolate': False,
            # 'mask': {
            #     'METEOcol': '',
            #     'precipitation_threshold': 10}
        },
        'METEO':
        {
            'bands': [
                'temperature_mean'
            ],
            'composite': {
                'freq': 30,
                'window': 30,
                'mode': get_comp_mode_dict(),
                'start': None,
                'end': None}
        }
    }


def get_croptype_tsteps_features_meta():
    return {
        'OPTICAL':
        {
            "tsteps": {
                "function": tsteps,
                "parameters": {
                    'n_steps': 12,
                },
                "names": [f'ts{i}' for i in range(12)]
            },
            "pheno_mult_season": {},
        },
        'SAR':
        {
            "tsteps": {
                "function": tsteps,
                "parameters": {
                    'n_steps': 12,
                },
                "names": [f'ts{i}' for i in range(12)]
            }
        },
        'METEO':
        {
            "tsteps": {
                "function": tsteps,
                "parameters": {
                    'n_steps': 6,
                },
                "names": [f'ts{i}' for i in range(6)],
                "bands": ['temperature_mean']
            }
        }
    }


def get_croptype_tsteps_parameters():
    return {
        'settings': get_croptype_tsteps_settings(),
        'features_meta': get_croptype_tsteps_features_meta(),
        'rsi_meta': get_default_rsi_meta(),
        'ignore_def_feat': {
            'OPTICAL': True,
            'SAR': True,
            'METEO': True
        },
        'gddnormalization': True
    }


def get_croptype_catboost_settings():
    return {
        'OPTICAL':
            {
                'bands': ["B02", "B03", "B04", "B05",
                          "B06", "B07", "B08", "B11", "B12"],
                'rsis': ["ndvi", "evi", "anir",
                         "ndmi", "ndwi", "brightness",
                         "ndre1", "ndre2", "ndre3",
                         "ndre4", "ndre5",
                         "ndgi"],
                'composite': {
                    'freq': 10,
                    'window': 20,
                    'start': None,
                    'end': None},
                'smooth_ndvi': False,
                'mask': {
                    'erode_r': 3,
                    'dilate_r': 13,
                    'mask_values': SCL_MASK_VALUES,
                    'multitemporal': False},
                'seasons': {'rsi': 'evi',
                            'amp_thr1': 0.15,
                            'amp_thr2': 0.35,
                            'max_window': 250,
                            'partial_start': False,
                            'partial_end': False,
                            'smooth_rsi': False}
            },
        'SAR':
            {
                'bands': ['VV', 'VH'],
                'rsis': ['vh_vv', 'rvi'],
                'composite': {
                    'freq': 12,
                    'window': 18,
                    'mode': 'mean',
                    'start': None,
                    'end': None},
                # 'mask': {
                #     'METEOcol': '',
                #     'precipitation_threshold': 10}
            }
    }


def get_croptype_catboost_features_meta():
    return {
        'OPTICAL':
            {
                "std": {
                    "function": std,
                    "names": ['std']},
                "tsteps": {
                    "function": tsteps,
                    "parameters": {
                        'n_steps': 6,
                    },
                    "names": ['ts0', 'ts1', 'ts2', 'ts3', 'ts4', 'ts5'],
                    "bands": ['ndvi']
                },
                "skew": {
                    "function": skewness,
                    "names": ['skew'],
                },
                "kurt": {
                    "function": kurto,
                    "names": ['kurt'],
                },
                "sen2agri_temp_feat": {
                    "function": sen2agri_temp_feat,
                    "parameters": {
                        'time_start': None,
                        'time_freq': 10,
                        'w': 2,
                        'delta': 0.05,
                        'tsoil': 0.2},
                    "bands": ['ndvi'],
                    "names": ['maxdif', 'mindif', 'difminmax',
                              'peak', 'lengthpeak', 'areapeak',
                              'ascarea', 'asclength', 'ascratio',
                              'descarea', 'desclength',
                              'descratio', 'soil1', 'soil2']},
                "pheno_mult_season": {}
            }
    }


def get_croptype_ignore_def_feat():
    return {
        'OPTICAL': False,
        'SAR': False
    }


def get_croptype_catboost_parameters():
    return {
        'settings': get_croptype_catboost_settings(),
        'features_meta': get_croptype_catboost_features_meta(),
        'rsi_meta': get_default_rsi_meta(),
        'ignore_def_feat': get_croptype_ignore_def_feat(),
        'gddnormalization': True}


# IRRIGATION


def get_irr_parameters():
    return {
        'settings': get_irr_settings(),
        'features_meta': get_irr_features_meta(),
        'rsi_meta': get_irr_rsi_meta(),
        'ignore_def_feat': get_irr_ignore_def_feat(),
        'gddnormalization': False
    }


def get_irr_settings():
    from worldcereal.features.feat_irr import calculate_smad
    return {
        'OPTICAL': {
            'bands': [],
            'rsis': ["ndvi", "evi", "mndwi", "ndwiveg", "gvmi"],
            'composite': {
                'freq': 10,
                'window': 20,
                'start': None,
                'end': None},
            'smooth_ndvi': False,
            'mask': {
                'erode_r': 3,
                'dilate_r': 13,
                'mask_values': SCL_MASK_VALUES,
                'multitemporal': False},
            'smad': {'bands_10': ['B02', 'B03', 'B04', 'B08'],
                     'bands_20': ['B11', 'B12'],
                     'function': calculate_smad}
        },
        'METEO': {'bands': ['precipitation_flux'],
                  'rsis': ['et0', 'et', 'prdef',
                           'ssm', 'ssm_adj'],
                  'composite': {
            'freq': 10,
            'window': None,
            'mode': get_comp_mode_dict(),
            'start': None,
            'end': None},
            'et': {'method': 'ndvi_linear'},
            'prdef': {'corr': ['None']}
        },
        'TIR': {
            'bands': [],
            'rsis': ['lst_ta'],
            'composite': {
                'freq': 16,
                'window': 32,
                'mode': 'median',
                'start': None,
                'end': None
            },
            'mask': {
                'erode_r': 3,
                'dilate_r': 13,
                'max_invalid_ratio': 1}
        }
    }


def get_irr_rsi_meta():
    from worldcereal.features.feat_irr import (
        calc_gvmi, calc_ndwi_veg)

    return {
        'OPTICAL': {
            'ndwiveg': {'bands': ['B08', 'B12'],
                        'range': [-1, 1],
                        'clamp_range': False,
                        'scale': 10000,
                        'native_res': 20,
                        'func': calc_ndwi_veg},
            'gvmi': {'bands': ['B08', 'B12'],
                     'range': [-1, 1],
                     'clamp_range': False,
                     'scale': 10000,
                     'native_res': 20,
                     'func': calc_gvmi}
        },
        'METEO': {
            'ssm': {
                'bands': [],
                'range': [0, 1],
                'clamp_range': False,
                'scale': 1,
                'native_res': 20},
            'ssm_adj': {
                'bands': [],
                'range': [0, 1],
                'clamp_range': False,
                'scale': 1,
                'native_res': 20},
            'sm': {
                'bands': ['dewpoint_temperature',
                          'temperature_mean',
                          'wind_speed',
                          'vapour_pressure',
                          'solar_radiation_flux'],
                'range': [0, 1],
                'clamp_range': False,
                'scale': 1,
                'native_res': 20},
            'smstress': {
                'bands': ['dewpoint_temperature',
                          'temperature_mean',
                          'wind_speed',
                          'vapour_pressure',
                          'solar_radiation_flux'],
                'range': [0, 1],
                'clamp_range': False,
                'scale': 1,
                'native_res': 20}
        },
        'TIR': {
            'lst_ta': {'bands': ['ST-B10'],
                       'range': [200, 400],
                       'clamp_range': False,
                       'scale': 1,
                       'native_res': 10}
        }
    }


def get_irr_features_meta():
    from worldcereal.features.feat_irr import cumfeat

    return {
        'METEO': {
            "std": {
                "function": std,
                "names": ['std']},
            "cumfeat": {
                "function": cumfeat,
                "parameters": {
                    'composite_freq': 10},
                "names": ['cum_max', 'cum_min',
                          'cum_npeaks', 'cum_maxdur',
                          'cum_maxslope'],
                "bands": ['prdefNocor']},
            "sumdiv": {
                "function": sum_div,
                "names": ['sum'],
                "bands": ['precipitation_flux',
                          'et0',
                          'et'],
                "parameters": {
                    "div": None}},
            "percentile_iqr": {
                "function": percentile_iqr,
                "parameters": {
                    'q': [10, 50, 90],
                    'iqr': [25, 75]
                },
                "names": ['p10', 'p50', 'p90', 'iqr']
            }
        },
        'OPTICAL': {
            "std": {
                "function": std,
                "names": ['std']},
            "percentile_iqr": {
                "function": percentile_iqr,
                "parameters": {
                    'q': [10, 50, 90],
                    'iqr': [25, 75]
                },
                "names": ['p10', 'p50', 'p90', 'iqr']
            }
        },
        'TIR': {
            "std": {
                "function": std,
                "names": ['std']},
            "percentile_iqr": {
                "function": percentile_iqr,
                "parameters": {
                    'q': [10, 50, 90],
                    'iqr': [25, 75]
                },
                "names": ['p10', 'p50', 'p90', 'iqr']
            }
        }
    }


def get_irr_ignore_def_feat():
    return {
        'OPTICAL': True,
        'METEO': True,
        'TIR': True
    }
