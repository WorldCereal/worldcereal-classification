from pathlib import Path

BANDS_CROPLAND = [
    'OPTICAL-ndvi-p10-10m',
    'OPTICAL-ndvi-p50-10m',
    'OPTICAL-ndvi-p90-10m',
    'OPTICAL-ndvi-iqr-10m',
    'OPTICAL-ndvi-ts0-10m',
    'OPTICAL-ndvi-ts1-10m',
    'OPTICAL-ndvi-ts2-10m',
    'OPTICAL-ndvi-ts3-10m',
    'OPTICAL-ndvi-ts4-10m',
    'OPTICAL-ndvi-ts5-10m',
    'OPTICAL-ndvi-maxdif-10m',
    'OPTICAL-ndvi-mindif-10m',
    'OPTICAL-ndvi-difminmax-10m',
    'OPTICAL-ndvi-peak-10m',
    'OPTICAL-ndvi-lengthpeak-10m',
    'OPTICAL-ndvi-areapeak-10m',
    'OPTICAL-ndvi-ascarea-10m',
    'OPTICAL-ndvi-asclength-10m',
    'OPTICAL-ndvi-ascratio-10m',
    'OPTICAL-ndvi-descarea-10m',
    'OPTICAL-ndvi-desclength-10m',
    'OPTICAL-ndvi-descratio-10m',
    'OPTICAL-ndwi-p10-10m',
    'OPTICAL-ndwi-p50-10m',
    'OPTICAL-ndwi-p90-10m',
    'OPTICAL-ndwi-iqr-10m',
    'OPTICAL-ndgi-p10-10m',
    'OPTICAL-ndgi-p50-10m',
    'OPTICAL-ndgi-p90-10m',
    'OPTICAL-anir-p10-20m',
    'OPTICAL-anir-p50-20m',
    'OPTICAL-anir-p90-20m',
    'OPTICAL-anir-iqr-20m',
    'OPTICAL-ndmi-p10-20m',
    'OPTICAL-ndmi-p50-20m',
    'OPTICAL-ndmi-p90-20m',
    'OPTICAL-ndmi-iqr-20m',
    'OPTICAL-ndre1-p10-20m',
    'OPTICAL-ndre1-p50-20m',
    'OPTICAL-ndre1-p90-20m',
    'OPTICAL-ndre1-iqr-20m',
    'OPTICAL-ndre5-p10-20m',
    'OPTICAL-ndre5-p50-20m',
    'OPTICAL-ndre5-p90-20m',
    'OPTICAL-ndre5-iqr-20m',
    'OPTICAL-B11-p10-20m',
    'OPTICAL-B11-p50-20m',
    'OPTICAL-B11-p90-20m',
    'OPTICAL-B11-iqr-20m',
    'OPTICAL-B12-p10-20m',
    'OPTICAL-B12-p50-20m',
    'OPTICAL-B12-p90-20m',
    'OPTICAL-B12-iqr-20m',
    'SAR-VV-p10-20m',
    'SAR-VV-p50-20m',
    'SAR-VV-p90-20m',
    'SAR-VV-iqr-20m',
    'SAR-VH-p10-20m',
    'SAR-VH-p50-20m',
    'SAR-VH-p90-20m',
    'SAR-VH-iqr-20m',
    'SAR-rvi-p10-20m',
    'SAR-rvi-p50-20m',
    'SAR-rvi-p90-20m',
    'SAR-rvi-iqr-20m',
    'DEM-alt-20m',
    'DEM-slo-20m',
    'biome01',
    'biome02',
    'biome03',
    'biome04',
    'biome05',
    'biome06',
    'biome07',
    'biome08',
    # 'biome09',  # Too small, leads to artefacts!
    'biome10',
    'biome11',
    'biome12',
    'biome13',
    'lat',
    'lon'
]


BANDS_CROPLAND_TSTEPS = [
    'OPTICAL-B02-ts', 'OPTICAL-B03-ts', 'OPTICAL-B04-ts',
    'OPTICAL-B05-ts', 'OPTICAL-B06-ts', 'OPTICAL-B07-ts',
    'OPTICAL-B08-ts', 'OPTICAL-B11-ts', 'OPTICAL-B12-ts',
    'SAR-VH-ts', 'SAR-VV-ts', 'SAR-rvi-ts',
    'DEM-alt-20m', 'DEM-slo-20m',
    'METEO-temperature_mean-p10-100m',
    'METEO-temperature_mean-p50-100m',
    'METEO-temperature_mean-p90-100m',
]


OUTLIER_INPUTS_CROPLAND = [
    'OPTICAL-ndvi-p10-10m',
    'OPTICAL-ndvi-iqr-10m',
    'OPTICAL-ndvi-p90-10m',
    'OPTICAL-anir-p10-20m',
    'SAR-VV-p90-20m',
    'SAR-VH-iqr-20m',
    'OPTICAL-ndgi-p10-10m',
]


BANDS_CROPTYPE = [
    'OPTICAL-ndvi-ts0-10m',
    'OPTICAL-ndvi-ts1-10m',
    'OPTICAL-ndvi-ts2-10m',
    'OPTICAL-ndvi-ts3-10m',
    'OPTICAL-ndvi-ts4-10m',
    'OPTICAL-ndvi-ts5-10m',
    'OPTICAL-ndvi-maxdif-10m',
    'OPTICAL-ndvi-mindif-10m',
    'OPTICAL-ndvi-difminmax-10m',
    'OPTICAL-ndvi-peak-10m',
    'OPTICAL-ndvi-lengthpeak-10m',
    'OPTICAL-ndvi-areapeak-10m',
    'OPTICAL-ndvi-ascarea-10m',
    'OPTICAL-ndvi-asclength-10m',
    'OPTICAL-ndvi-ascratio-10m',
    'OPTICAL-ndvi-descarea-10m',
    'OPTICAL-ndvi-desclength-10m',
    'OPTICAL-ndvi-descratio-10m',
    'OPTICAL-ndvi-p10-10m',
    'OPTICAL-ndvi-p50-10m',
    'OPTICAL-ndvi-p90-10m',
    'OPTICAL-ndvi-std-10m',
    'OPTICAL-ndwi-p10-10m',
    'OPTICAL-ndwi-p50-10m',
    'OPTICAL-ndwi-p90-10m',
    'OPTICAL-ndwi-std-10m',
    'OPTICAL-ndgi-p10-10m',
    'OPTICAL-ndgi-p50-10m',
    'OPTICAL-ndgi-p90-10m',
    'OPTICAL-anir-p10-20m',
    'OPTICAL-anir-p50-20m',
    'OPTICAL-anir-p90-20m',
    'OPTICAL-anir-std-20m',
    'OPTICAL-B02-p10-10m',
    'OPTICAL-B02-p50-10m',
    'OPTICAL-B02-p90-10m',
    'OPTICAL-B02-std-10m',
    'OPTICAL-B03-p10-10m',
    'OPTICAL-B03-p50-10m',
    'OPTICAL-B03-p90-10m',
    'OPTICAL-B03-std-10m',
    'OPTICAL-B04-p10-10m',
    'OPTICAL-B04-p50-10m',
    'OPTICAL-B04-p90-10m',
    'OPTICAL-B04-std-10m',
    'OPTICAL-B12-p10-20m',
    'OPTICAL-B12-p50-20m',
    'OPTICAL-B12-p90-20m',
    'OPTICAL-B12-std-20m',
    'OPTICAL-ndre1-p10-20m',
    'OPTICAL-ndre1-p50-20m',
    'OPTICAL-ndre1-p90-20m',
    'OPTICAL-ndre1-std-20m',
    'OPTICAL-ndre5-p10-20m',
    'OPTICAL-ndre5-p50-20m',
    'OPTICAL-ndre5-p90-20m',
    'OPTICAL-ndre5-std-20m',
    'OPTICAL-evi-lSeasMax-10m',
    'OPTICAL-evi-lSeasMed-10m',
    'OPTICAL-evi-lSeasMin-10m',
    'OPTICAL-evi-aSeasMax-10m',
    'OPTICAL-evi-aSeasMed-10m',
    'OPTICAL-evi-aSeasMin-10m',
    'SAR-VV-p10-20m',
    'SAR-VV-p50-20m',
    'SAR-VV-p90-20m',
    'SAR-VV-iqr-20m',
    'SAR-VH-p10-20m',
    'SAR-VH-p50-20m',
    'SAR-VH-p90-20m',
    'SAR-VH-iqr-20m',
    'SAR-rvi-p10-20m',
    'SAR-rvi-p50-20m',
    'SAR-rvi-p90-20m',
    'SAR-rvi-iqr-20m',
    'SAR-vh_vv-p10-20m',
    'SAR-vh_vv-p50-20m',
    'SAR-vh_vv-p90-20m',
    'SAR-vh_vv-iqr-20m',
    'SAR-rvi-ts0-20m',
    'SAR-rvi-ts1-20m',
    'SAR-rvi-ts2-20m',
    'SAR-rvi-ts3-20m',
    'SAR-rvi-ts4-20m',
    'SAR-rvi-ts5-20m',
    'DEM-alt-20m',
    'DEM-slo-20m',
    'realm05'  # Africa except the North, only for maize!
]

OUTLIER_INPUTS_CROPTYPE = [
    'OPTICAL-ndvi-p10-10m',
    'OPTICAL-ndvi-iqr-10m',
    'OPTICAL-ndvi-p90-10m',
    'OPTICAL-anir-p10-20m',
    'SAR-VV-p90-20m',
    'SAR-VH-iqr-20m',
    'OPTICAL-ndgi-p10-10m',
]

BANDS_IRRIGATION = [
    'OPTICAL-smad-med-20m',
    'OPTICAL-B08-gm-20m',
    'OPTICAL-B11-gm-20m',
    'OPTICAL-evi-p90-10m',
    'OPTICAL-evi-std-10m',
    'OPTICAL-ndvi-p90-10m',
    'OPTICAL-ndvi-std-10m',
    'OPTICAL-mndwi-p90-20m',
    'OPTICAL-mndwi-std-20m',
    'OPTICAL-ndwiveg-p90-20m',
    'OPTICAL-ndwiveg-std-20m',
    'OPTICAL-gvmi-p90-20m',
    'OPTICAL-gvmi-std-20m',
    'TIR-lst_ta-p50-10m',
    'TIR-lst_ta-p90-10m',
    'TIR-lst_ta-std-10m',
    'METEO-prdefNocor-p10-20m',
    'METEO-prdefNocor-p50-20m',
    'METEO-prdefNocor-p90-20m',
    'METEO-prdefNocor-std-20m',
    'METEO-prdefNocor-cum_max-20m',
    'METEO-prdefNocor-cum_min-20m',
    'METEO-prdefNocor-cum_maxdur-20m',
    'METEO-prdefNocor-cum_maxslope-20m',
    'METEO-ssm-p50-20m',
    'METEO-ssm-p90-20m',
    'METEO-ssm-std-20m',
    'METEO-ssm_adj-p50-20m',
    'METEO-ssm_adj-p90-20m',
    'METEO-ssm_adj-std-20m',
    'METEO-precipitation_flux-sum-100m',
    'METEO-et0-sum-100m',
    'METEO-et-sum-20m'
]

TRAININGDIR_LUT = {
    'cropland': ['/data/worldcereal/features-nowhitakker-final/annual_CIB/'],
    'maize': ['/data/worldcereal/features-nowhitakker-final/summer1_CIB', '/data/worldcereal/features-nowhitakker-final/summer2_CIB'],  # NOQA
    'wintercereals': ['/data/worldcereal/features-nowhitakker-final/winter_CIB'],
    'springcereals': ['/data/worldcereal/features-nowhitakker-final/summer1_CIB'],
    'irrigation': [
        '/data/worldcereal/features/irr_features_other/irr_summer1_CIB',
        '/data/worldcereal/features/irr_features_other/irr_summer2_CIB',
        '/data/worldcereal/features/irr_features_other/irr_winter_CIB',
        '/data/worldcereal/features/irr_features_spain/irr_summer1_CIB',
    ]
}

TRAINING_SETTINGS = {
    'maize': {
        'outputlabel': 'CT',
        'targetlabels': [1200],
        'ignorelabels': [1000, 9998, 991],
        'focuslabels': [8200],  # Extra weight on sugar cane
        'focusmultiplier': 2,
        'filter_worldcover': True,
        'remove_outliers': True,
        'bands': BANDS_CROPTYPE,
        'outlierinputs': OUTLIER_INPUTS_CROPTYPE,
        'pos_neg_ratio': 0.25,
        'minsamples': [1000, 500, 500],
    },
    'wintercereals': {
        'outputlabel': 'CT',
        'targetlabels': [1110, 1510, 1610, 1910],
        'ignorelabels': [1000, 1100, 1500, 1600, 1700,
                         1800, 1900, 9998, 991],
        'focuslabels': [],
        'focusmultiplier': 1,
        'filter_worldcover': True,
        'remove_outliers': True,
        'bands': BANDS_CROPTYPE,
        'outlierinputs': OUTLIER_INPUTS_CROPTYPE,
        'pos_neg_ratio': 0.30,
        'minsamples': [1000, 500, 500]
    },
    'springcereals': {
        'outputlabel': 'CT',
        'targetlabels': [1120, 1520, 1620, 1920],
        'ignorelabels': [1000, 1100, 1500, 1600, 1700,
                         1800, 1900, 9998, 991],
        'focuslabels': [],
        'focusmultiplier': 1,
        'filter_worldcover': True,
        'remove_outliers': True,
        'bands': BANDS_CROPTYPE,
        'outlierinputs': OUTLIER_INPUTS_CROPTYPE,
        'pos_neg_ratio': 0.20,
        'minsamples': [1000, 500, 500]
    },
    'cropland': {
        'outputlabel': 'LC',
        'targetlabels': [11],
        'ignorelabels': [10],
        'focuslabels': [12, 13, 20, 30, 50, 999],
        'focusmultiplier': 3,
        'filter_worldcover': True,
        'remove_outliers': True,
        'bands': BANDS_CROPLAND,
        'outlierinputs': OUTLIER_INPUTS_CROPLAND,
        'pos_neg_ratio': 0.50
    },
    'irrigation': {
        'outputlabel': 'IRR',
        'targetlabels': [20000, 21000, 21300, 21400, 21500,
                         22000, 22300, 22400, 22500],
        'focuslabels': [],
        'ignorelabels': [],
        'focusmultiplier': 1,
        'filter_worldcover': True,
        'remove_outliers': False,
        'bands': BANDS_IRRIGATION,
        'minsamples': [500, 200, 200],
        'train_group': True,
        'train_zone': True,
        'classweight': 'balanced'
    }
}


def get_training_settings(detector, s1=True):

    if detector not in TRAINING_SETTINGS.keys():
        raise ValueError(
            f'Detector `{detector}` not in TRAINING_SETTINGS.')
    elif detector not in TRAININGDIR_LUT.keys():
        raise ValueError(
            f'Detector `{detector}` not in TRAININGDIR_LUT.')
    else:
        training_settings = TRAINING_SETTINGS[detector]

        # remove S1 bands if needed
        if not s1:
            training_settings['bands'] = [b for b
                                          in training_settings['bands']
                                          if 'SAR-' not in b]
            if training_settings.get('outlierinputs', None) is not None:
                training_settings['outlierinputs'] = [
                    b for b in training_settings['outlierinputs']
                    if 'SAR-' not in b]

        training_settings['cal_df_files'] = [
            str(Path(x) / 'CAL') for x in TRAININGDIR_LUT[detector]]
        training_settings['val_df_files'] = [
            str(Path(x) / 'VAL') for x in TRAININGDIR_LUT[detector]]
        training_settings['test_df_files'] = [
            str(Path(x) / 'TEST') for x in TRAININGDIR_LUT[detector]]

        return training_settings
