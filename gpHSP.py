"""Common functions for gphsp."""
import dill
# Chem libraries
import mordred
import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem as Chem
from mordred import descriptors as mordred_descriptors
from tqdm.auto import tqdm


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
    with open(fname, "wb") as f:
        dill.dump(model, f)


def load_model(fname):
    with open(fname, "rb") as f:
        model = dill.load(f)
    return model


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
