"""Common functions for gphsp."""
import dill
import gpflow as gpf
import matplotlib
# Chem libraries
import mordred
import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem as Chem
import scipy.stats as stats
import seaborn as sns
import sklearn.metrics
import tensorflow as tf
from mordred import descriptors as mordred_descriptors

Y_COLS = ['δd', 'δp', 'δh']

def notebook_context():
    gpf.config.set_default_float(np.float64)
    gpf.config.set_default_summary_fmt("notebook")
    sns.set_context('talk', font_scale=1.25)
    matplotlib.rcParams['figure.figsize'] = (12,8)
    matplotlib.rcParams['lines.linewidth'] = 2
    pd.set_option("display.precision", 3)

def print_modules(mods):
    for mod in mods:
        print(f'{mod.__name__:10s} = {mod.__version__}')

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

def save_gp(model, adir, in_dim=6):
    model.predict = tf.function(model.predict_f,
                            input_signature=[tf.TensorSpec(shape=[None, in_dim], dtype=tf.float64)])
    tf.saved_model.save(model, adir)

def load_gp(adir):
    return tf.saved_model.load(adir)

def make_gp(x, y, use_ard,  parts=None, const_mean = True):
    kernel = None
    parts = parts or {'all':None}
    for part in parts.values():
        if use_ard:
            ard = tf.convert_to_tensor(np.ones_like(parts).astype(np.float64))
            kernel_part = gpf.kernels.SquaredExponential(lengthscales=ard, active_dims=part)
        else:
            kernel_part = gpf.kernels.SquaredExponential(active_dims=part)
        kernel = kernel_part if kernel is None else kernel + kernel_part

    mean_fn = gpf.mean_functions.Constant() if const_mean else None
    model = gpf.models.GPR(data=(x, y),
                       kernel=kernel,
                       mean_function=mean_fn)
    opt = gpf.optimizers.Scipy()
    opt_logs = opt.minimize(model.training_loss,
                        model.trainable_variables,
                        options=dict(maxiter=1000))
    return model

def predictions_as_features(x, model_dict, pred_fn=None):
    default_pred_fn = lambda model, inputs: model.pred_dist(inputs)
    pred_fn = pred_fn or default_pred_fn
    new_x = np.zeros((len(x), 6), dtype=np.float64)
    for index, model in enumerate(model_dict.values()):
        y_mol_dist = pred_fn(model, x)
        new_x[:,index] = y_mol_dist.mean()
        new_x[:,index+3] = y_mol_dist.mean()
    return new_x

def cast_1d_array(a):
    """Flatten array or tensor."""
    a = a.numpy() if tf.is_tensor(a) else a
    assert a.ndim == 1 or a.shape[1]==1, f'Expected 1D array {a.shape}'
    return a.ravel()

def predict_from_model_dict(x, model_dict):
    return {key: cast_1d_array(model.predict(x)) for key, model in model_dict.items()}

def load_model(fname):
    assert fname.endswith('.pkl'), f'Check your filename={fname}'
    with open(fname, "rb") as f:
        model = dill.load(f)
    return model

def evaluate(y_true, y_pred, y_std=None, info=None):
    """Evaluate predictions."""
    y_true = cast_1d_array(y_true)
    y_pred = cast_1d_array(y_pred)
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

    def update(self, new_smi, new_values):
        needs_update = np.array([not s in self.index.index for s in new_smi])
        if needs_update.sum()!=len(new_smi):
            raise ValueError('Provide only new smiles and features')
        if len(new_smi) != len(new_values) or new_values.ndim!=2:
            raise ValueError('Inconsistent shapes')
        n = len(self.index)
        new_index = pd.Series(np.arange(n, n+len(new_smi)), index=new_smi)
        self.index = self.index.append(new_index)
        self.values = np.vstack((self.values, new_values))
        return self
