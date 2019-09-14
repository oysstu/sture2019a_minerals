"""
Wrapper around GPFlow SVGP
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''  # Force CPU
os.environ["TF_XLA_FLAGS"] = '--tf_xla_cpu_global_jit'
import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
import gpflow as gpf
import gpflow.params
import gpflow.mean_functions
import gpflow.multioutput.kernels as mok
import gpflow.multioutput.features as mof
from sklearn.preprocessing import StandardScaler


def reset_gpflow_graph():
    """
    Reset gpflow/tensorflow graph, necessary when running in loops
    """
    tf.reset_default_graph()
    graph = tf.get_default_graph()
    gpflow.reset_default_session(graph=graph)


class UHIRegression:
    """
    Wrapper around GPFlow Stochastic Variational Gaussian Process.
    Adds scaling of input/output variables to avoid problems with instability/singularities due to poorly scaled data
    """
    def __init__(self, X=None, Y=None, n_inducing=3, scaling=True, std_scaling=False, multi_output=False,
                 batch_size=2000, dtype=np.float64):
        """
        :param X: Input variables (regressor)
        :param Y: Output variables (regressand)
        :param n_inducing: Number of inducing inputs per input dimension
        :param scaling: Enable scaling of input/output variables
        :param std_scaling: Enable normalization of input/output standard deviation
        :param multi_output: Enable multi-output kernel (MoK)
        :param batch_size: The batch size during training
        :param dtype: The target dtype (float32 on GPU, float64 on CPU typically)
        """
        self.scaling = scaling
        self.multi_output = multi_output
        self.batch_size = batch_size
        self.dtype = dtype
        self.gpflow_config = gpflow.settings.get_settings()
        self.gpflow_config['dtypes']['float_type'] = dtype
        self.gpflow_config['float_type'] = dtype

        if X is not None and Y is not None:
            # Preprocess inputs (scaling, dims)
            self.X_in = (X if len(X.shape) == 2 else X[:, np.newaxis]).astype(dtype)
            self.Y_in = (Y if len(Y.shape) == 2 else Y[:, np.newaxis]).astype(dtype)

            self.X_in_min = np.min(self.X_in, axis=0)
            self.X_in_max = np.max(self.X_in, axis=0)
            self.Y_in_min = np.min(self.Y_in, axis=0)
            self.Y_in_max = np.max(self.Y_in, axis=0)

            if scaling:
                self.xscaler = StandardScaler(with_mean=True, with_std=False)
                self.yscaler = StandardScaler(with_mean=True, with_std=std_scaling)
                self.X = self.xscaler.fit_transform(self.X_in)
                self.Y = self.yscaler.fit_transform(self.Y_in)
            else:
                self.xscaler = None
                self.yscaler = None
                self.X = self.X_in
                self.Y = self.Y_in

            self.xmin = self.X.min(axis=0)
            self.xmax = self.X.max(axis=0)

            self.n_in_dims = self.X.shape[-1]
            self.n_out_dims = self.Y.shape[-1]

            # Generate initial inducing points (equidistant)
            z = [np.linspace(mi, ma, n_inducing, dtype=dtype) for mi, ma in zip(self.xmin, self.xmax)]
            gg = np.meshgrid(*z)
            self.Z = np.concatenate([g.ravel()[:, np.newaxis] for g in gg], axis=1)

            print(f'GPFlow ~ {self.n_in_dims}x{self.n_out_dims}')
            print(f'X ~ {self.X.shape[0]}x{self.X.shape[1]}')
            print(f'Y ~ {self.Y.shape[0]}x{self.Y.shape[1]}')
            print(f'Z ~ {self.Z.shape[0]}x{self.Z.shape[1]}')

        # Storage
        self.m = None
        self.kern = None
        self.lh = None
        self.mf = None
        self.feature = None

    def train(self, verbose=True, maxiter=1000):
        with gpflow.settings.temp_settings(self.gpflow_config):
            # Default parameters
            if self.kern is None:
                self.kern = gpflow.kernels.SquaredExponential(input_dim=self.n_in_dims, variance=self.dtype(0.2),
                                                              lengthscales=self.dtype(1.0))
            if self.lh is None:
                self.lh = gpflow.likelihoods.Gaussian(variance=self.dtype(0.02))

            if self.feature is None:
                self.feature = gpf.features.InducingPoints(self.Z)

            if self.multi_output:
                self.kern = mok.SharedIndependentMok(self.kern, output_dimensionality=self.n_out_dims)
                self.feature = mof.SharedIndependentMof(self.feature)

            self.m = gpflow.models.SVGP(self.X, self.Y, self.kern, likelihood=self.lh,
                                        feat=self.feature, mean_function=self.mf,
                                        minibatch_size=self.batch_size)

            opt = gpflow.train.tensorflow_optimizer.AdamOptimizer(learning_rate=0.1, beta1=0.9,
                                                                  beta2=0.999, epsilon=1e-8)

            opt.minimize(self.m, maxiter=maxiter)

            if verbose:
                pd.set_option('display.max_rows', 20)
                pd.set_option('display.max_columns', 10)
                print(self.m.as_pandas_table())
                print('Log likelihood: ', self.m.compute_log_likelihood())

    def predict_f(self, X, pred_y=False):
        """
        Predict function values at given input locations
        :param X: Input locations (N, D)
        :param pred_y: If true, the prediction is made for observations (with likelihood/noise)
        :return:
        """
        X = (X if len(X.shape) == 2 else X[:, np.newaxis]).astype(np.float64)

        if self.scaling:
            X = self.xscaler.transform(X)

        if pred_y:
            mean_f, var_f = self.m.predict_y(X)
        else:
            mean_f, var_f = self.m.predict_f(X)

        if self.scaling:
            mean_f = self.yscaler.inverse_transform(mean_f)
            if self.xscaler.with_std:
                var_f *= self.yscaler.scale_

        return mean_f, var_f

    def save_model(self, fpath):
        gpf.Saver().save(fpath, self.m)

        fbase, fname = os.path.splitext(fpath)

        params = {'scaling': self.scaling,
                  'multi_output': self.multi_output,
                  'xscaler': self.xscaler,
                  'yscaler': self.yscaler,
                  'xmin': self.xmin,
                  'xmax': self.xmax,
                  'X_in_min': self.X_in_min,
                  'X_in_max': self.X_in_max,
                  'Y_in_min': self.Y_in_min,
                  'Y_in_max': self.Y_in_max,
                  'n_in_dims': self.n_in_dims,
                  'n_out_dims': self.n_out_dims}
        with open(fbase + '.pkl', 'wb') as f:
            pickle.dump(params, f)

    def load_model(self, fpath):
        self.m = gpf.Saver().load(fpath)

        fbase, fname = os.path.splitext(fpath)

        with open(fbase + '.pkl', 'rb') as f:
            params = pickle.load(f)
            for k, v in params.items():
                setattr(self, k, v)
