"""Common functions for gphsp."""
import dill
# Chem libraries
import mordred
import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem as Chem
import scipy.stats as stats
import sklearn.metrics
import tensorflow as tf
from mordred import descriptors as mordred_descriptors

Y_COLS = ['δd', 'δp', 'δh']

def peek_df(df):
    print(df.columns)
    print(df.shape)
    display(df.head(1))


def get_isomeric_smiles(s):
    return Chem.MolToSmiles(Chem.MolFromSmiles(s))


def calculate_mordred(smis):
    """Calculate mordred features."""
    mols = np.array([Chem.MolFromSmiles(s) for s in smis])
    calc = mordred.Calculator(
        mordred_descriptors, ignore_3D=True, version='1.2.1')
    mordred_df = calc.pandas(mols.tolist())
    values = mordred_df.to_numpy(np.float32)
    return values


def calculate_mask(values):
    """Calculate mask for mordred features."""
    nan_mask = np.sum(np.isnan(values), axis=0) == 0
    const_mask = np.array([len(np.unique(v)) > 1 for v in values.T])
    mask = np.logical_and(nan_mask, const_mask)
    return mask


def save_model(model, fname):
    assert fname.endswith('.pkl'), f'Check your filename={fname}'
    with open(fname, "wb") as f:
        dill.dump(model, f)


def load_model(fname):
    assert fname.endswith('.pkl'), f'Check your filename={fname}'
    with open(fname, "rb") as f:
        model = dill.load(f)
    return model

def _flat_array(a):
    """Flatten array or tensor."""
    a = a.numpy() if tf.is_tensor(a) else a
    assert a.ndim == 1 or a.shape[1]==1, f'Expected 1D array {a.shape}'
    return a.ravel()

def evaluate(y_true, y_pred, y_std=None, info=None):
    """Evaluate predictions."""
    y_true = _flat_array(y_true)
    y_pred = _flat_array(y_pred)
    stat = info or {}
    stat['R2'] = sklearn.metrics.r2_score(y_true, y_pred)
    stat['MAE'] = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    stat['tau'] = stats.kendalltau(y_true, y_pred).correlation
    if y_std is not None:
        y_error = np.abs(y_true-y_pred)
        stat['uncertainty_tau'] = stats.kendalltau(y_error, y_std).correlation

    return stat


class SmilesMap:
    """Class to map smiles to values."""

    def __init__(self, fname):
        data = dict(np.load(fname))
        self.values = data['values']
        smi = data['smiles']
        self.mask = data['mask']
        self.index = pd.Series(np.arange(len(smi)), index=smi)

    def __call__(self, inputs):
        index = self.index.loc[inputs].values
        return self.values[index][:, self.mask]
