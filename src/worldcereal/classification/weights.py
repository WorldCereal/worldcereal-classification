try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

import pandas as pd

from worldcereal import resources


def load_refid_lut():
    with pkg_resources.open_text(resources, 'CIB_RefIdLUT.csv') as LUTfile:
        LUT = pd.read_csv(LUTfile, delimiter=";")

    return LUT


def load_refidweights():

    LUT = load_refid_lut()[['CIB', 'LC', 'CT', 'IRR']]

    LUT = LUT[LUT.isnull().sum(axis=1) == 0]
    LUT['CIB'] = ['_'.join(x.split('_')[:3]) for x in
                  LUT['CIB'].values]
    LUT = LUT.drop_duplicates().set_index('CIB')

    return LUT.to_dict(orient='index')


def get_refid_weight(ref_id, label, refidweights=None):
    '''
    Function to get the weight to be put on
    a particular ref_id.
    '''

    refid_weights = refidweights or load_refidweights()
    weight = refid_weights.get(ref_id, None)
    if weight is not None:
        weight = weight[label]
    else:
        weight = 90

    return weight
