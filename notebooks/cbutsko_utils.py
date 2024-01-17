import sys
sys.path.append('/home/cbutsko/Desktop/cbutsko_experiments/satclip/satclip')

import numpy as np
import pandas as pd
import itertools
import gc
import re
import torch
import glob
import rioxarray as rio
import xarray as xr
from typing import Callable, Dict, List, Optional, Union
from pathlib import Path

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression

from hiclass import LocalClassifierPerParentNode, LocalClassifierPerNode

from tqdm.auto import tqdm
tqdm.pandas()
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from load import get_satclip

# device = 'cpu'


class CatBoostClassifierWrapper(CatBoostClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, y, *args, **kwargs):
        val_fraction = 0.3
        early_stopping_rounds = 100

        _X_trn, _X_val, _y_trn, _y_val = train_test_split(
            X, y,
            stratify=y,
            test_size=val_fraction)

        return super().fit(
            _X_trn, _y_trn, 
            eval_set=(_X_val, _y_val),
            early_stopping_rounds=early_stopping_rounds)

class LocalClassifierPerNodeWrapper(LocalClassifierPerNode):
    def __init__(
        self,
        local_classifier: None,
        binary_policy: str = "siblings",
        verbose: int = 0,
        edge_list: str = None,
        replace_classifiers: bool = True,
        n_jobs: int = 1,
        bert: bool = False,
    ):
        super().__init__(
            local_classifier=local_classifier,
            verbose=verbose,
            edge_list=edge_list,
            replace_classifiers=replace_classifiers,
            n_jobs=n_jobs,
            # classifier_abbreviation="LCPN",
            bert=bert,
        )
        self.binary_policy = binary_policy

    def predict_proba(self, X):
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        if not self.bert:
            X = check_array(X, accept_sparse="csr", allow_nd=True, ensure_2d=False)
        else:
            X = np.array(X)

        # Initialize array that holds predictions
        y = np.empty((X.shape[0], self.max_levels_), dtype=self.dtype_)

        # TODO: Add threshold to stop prediction halfway if need be

        bfs = nx.bfs_successors(self.hierarchy_, source=self.root_)

        self.logger_.info("Predicting")

        # We initialize a dictionary that will hold the probabilities for each node
        probability_dict = {}

        for predecessor, successors in bfs:            
            if predecessor == self.root_:
                mask = [True] * X.shape[0]
                subset_x = X[mask]
            else:
                mask = np.isin(y, predecessor).any(axis=1)
                subset_x = X[mask]
            if subset_x.shape[0] > 0:
                probabilities = np.zeros((subset_x.shape[0], len(successors)))
                for i, successor in enumerate(successors):
                    successor_name = str(successor).split(self.separator_)[-1]
                    self.logger_.info(f"Predicting for node '{successor_name}'")
                    classifier = self.hierarchy_.nodes[successor]["classifier"]
                    positive_index = np.where(classifier.classes_ == 1)[0]
                    probabilities[:, i] = classifier.predict_proba(subset_x)[
                        :, positive_index
                    ][:, 0]

                # For each node, save the probabilities in the dictiopnary
                probability_dict[predecessor] = probabilities
                highest_probability = np.argmax(probabilities, axis=1)
                prediction = []
                for i in highest_probability:
                    prediction.append(successors[i])
                level = nx.shortest_path_length(
                    self.hierarchy_, self.root_, predecessor
                )
                prediction = np.array(prediction)
                y[mask, level] = prediction

        y = self._convert_to_1d(y)

        self._remove_separator(y)

        return y, probability_dict

class LocalClassifierPerParentNodeWrapper(LocalClassifierPerParentNode):
    def __init__(
        self,
        local_classifier: None,
        verbose: int = 0,
        edge_list: str = None,
        replace_classifiers: bool = True,
        n_jobs: int = 1,
        bert: bool = False,
    ):
        super().__init__(
            local_classifier=local_classifier,
            verbose=verbose,
            edge_list=edge_list,
            replace_classifiers=replace_classifiers,
            n_jobs=n_jobs,
            bert=bert,
        )
        
    def predict_proba(self, X):
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        if not self.bert:
            X = check_array(X, accept_sparse="csr", allow_nd=True, ensure_2d=False)
        else:
            X = np.array(X)

        # Initialize array that holds predictions
        y = np.empty((X.shape[0], self.max_levels_), dtype=self.dtype_)
         # We initialize a dictionary that will hold the probabilities for each node
        # probs = np.empty((X.shape[0], self.max_levels_, ), dtype=self.dtype_)
        probs = {}
        
        self.logger_.info("Predicting")

        # Predict first level
        classifier = self.hierarchy_.nodes[self.root_]["classifier"]
        y[:,0] = classifier.predict(X).flatten()
        probs['l0'] = classifier.predict_proba(X)

        self._predict_remaining_levels(X, y, probs)

        y = self._convert_to_1d(y)

        self._remove_separator(y)

        return y, probs

    def _predict_remaining_levels(self, X, y, probs=None):
        for level in range(1, y.shape[1]):
            predecessors = set(y[:, level - 1])
            predecessors.discard("")
            probs['l{}'.format(level)] = {}
            for predecessor in predecessors:
                mask = np.isin(y[:, level - 1], predecessor)
                predecessor_x = X[mask]
                if predecessor_x.shape[0] > 0:
                    successors = list(self.hierarchy_.successors(predecessor))
                    if len(successors) > 0:
                        classifier = self.hierarchy_.nodes[predecessor]["classifier"]
                        y[mask, level] = classifier.predict(predecessor_x).flatten()
                        if probs is not None:
                            _probs = classifier.predict_proba(predecessor_x)
                            level_probs = np.empty((y.shape[0], _probs.shape[-1]))
                            level_probs[mask,:] = _probs
                            probs['l{}'.format(level)][predecessor] = level_probs


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.2f}; misclass={:0.2f}'.format(accuracy, misclass))
    plt.show()


def patch2feats(patch: xr.Dataset, tfeatures: List, get_satclip: bool = False) -> pd.DataFrame:
    patch_df = pd.DataFrame()
    for feature_name in ['B02','B03','B04','B08']:
        _feature_colnames = ['OPTICAL-{}-ts{}-10m'.format(feature_name,xx) for xx in range(12)]
        feature_df = patch[feature_name].values.reshape(12,-1).swapaxes(0,1)
        feature_df = pd.DataFrame(feature_df, columns=_feature_colnames)
        patch_df = pd.concat((patch_df,feature_df), axis=1)

    for feature_name in ['B05','B06','B07','B8A','B11','B12']:
        _feature_colnames = ['OPTICAL-{}-ts{}-20m'.format(feature_name,xx) for xx in range(12)]
        feature_df = patch[feature_name].values.reshape(12,-1).swapaxes(0,1)
        feature_df = pd.DataFrame(feature_df, columns=_feature_colnames)
        patch_df = pd.concat((patch_df,feature_df), axis=1)

    for feature_name in ['VV','VH']:
        _feature_colnames = ['SAR-{}-ts{}-20m'.format(feature_name,xx) for xx in range(12)]
        feature_df = patch[feature_name].values.reshape(12,-1).swapaxes(0,1)
        feature_df = pd.DataFrame(feature_df, columns=_feature_colnames)
        patch_df = pd.concat((patch_df,feature_df), axis=1)

    _feature_colnames = ['METEO-temperature_mean-ts{}-100m'.format(xx) for xx in range(12)]
    feature_df = patch['temperature-mean'].values.reshape(12,-1).swapaxes(0,1)
    feature_df = pd.DataFrame(feature_df, columns=_feature_colnames)
    patch_df = pd.concat((patch_df,feature_df), axis=1)

    _feature_colnames = ['METEO-precipitation_flux-ts{}-100m'.format(xx) for xx in range(12)]
    feature_df = patch['precipitation-flux'].values.reshape(12,-1).swapaxes(0,1)
    feature_df = pd.DataFrame(feature_df, columns=_feature_colnames)
    patch_df = pd.concat((patch_df,feature_df), axis=1)

    feature_df = patch['altitude'].values[0,:,:].reshape(1,-1).swapaxes(0,1)
    feature_df = pd.DataFrame(feature_df, columns=['DEM-alt-20m'])
    patch_df = pd.concat((patch_df,feature_df), axis=1)

    feature_df = patch['slope'].values[0,:,:].reshape(1,-1).swapaxes(0,1)
    feature_df = pd.DataFrame(feature_df, columns=['DEM-slo-20m'])
    patch_df = pd.concat((patch_df,feature_df), axis=1)

    feature_df = np.repeat(patch['x'].values.reshape(-1,1), patch['y'].shape[0], axis=1).reshape(1,-1).swapaxes(0,1)
    feature_df = pd.DataFrame(feature_df, columns=['lon'])
    patch_df = pd.concat((patch_df,feature_df), axis=1)

    feature_df = np.repeat(patch['y'].values.reshape(1,-1), patch['y'].shape[0], axis=0).reshape(1,-1).swapaxes(0,1)
    feature_df = pd.DataFrame(feature_df, columns=['lat'])
    patch_df = pd.concat((patch_df,feature_df), axis=1)

    # if get_satclip:
        # ...

    patch_df = patch_df[tfeatures]

    return patch_df


def process_raw_features_input_df(
    fpath: Path, 
    wc2ec_map: pd.DataFrame,
    ec_map: pd.DataFrame,
    label_columns: List,
    ) -> pd.DataFrame:
    
    tdf = pd.read_parquet(fpath)
    tdf['CT'].replace(0, np.nan, inplace=True)
    tdf['CT'].fillna(tdf['OUTPUT'], inplace=True)

    tdf['ec_code'] = tdf['CT'].map(wc2ec_map.set_index('croptype')['ec_code'])
    tdf['CT_name'] = tdf['CT'].map(wc2ec_map.set_index('croptype')['name'])
    tdf['landcover_wc'] = tdf['CT'].map(wc2ec_map.set_index('croptype')['landcover'])
    
    tdf = tdf[tdf.isna().sum(axis=1)==0]
    for tlevel in label_columns:
        tdf[tlevel] = tdf['ec_code'].map(ec_map['{}_label'.format(tlevel)]).astype('int32')
        tdf['{}_name'.format(tlevel)] = tdf['ec_code'].map(ec_map['{}_name'.format(tlevel)])
    tdf = tdf[tdf['cropland']!=-1].reset_index(drop=True)

    for tcol in ['start_date','end_date','valid_date']:
        if tcol in tdf.columns:
            tdf[tcol] = pd.to_datetime(tdf[tcol])

    tdf['cropland_wc'] = tdf['landcover_wc']==11
    tdf['cropland_ec'] = tdf['landcover']==1

    tdf.set_index(['location_id','ref_id','pixelids'], inplace=True)

    return tdf 

def prepare_satclip_embeddings(satclip_model, target_df: pd.DataFrame, max_chunk_size: int = 10000) -> pd.DataFrame:
    df_chunks = np.array_split(target_df[['lat','lon']], np.ceil(len(target_df)/max_chunk_size))
    embeddings_df = pd.DataFrame()
    pbar = tqdm(total=len(df_chunks))
    for chunk in df_chunks:
        pbar.update(1)
        latlon_batch = torch.tensor(chunk[['lat','lon']].values).double()
        with torch.no_grad():
            emb = satclip_model(latlon_batch).numpy()
        emb = pd.DataFrame(
            emb, 
            columns=['emb{}'.format(ii) for ii in range(emb.shape[-1])],
            index=chunk.index)
        embeddings_df = pd.concat([embeddings_df,emb], axis=0)
    gc.collect()

    return embeddings_df




