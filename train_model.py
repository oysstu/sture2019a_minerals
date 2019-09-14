"""
Trains a new regression model from the raw tilted plate uhi data.
Note: This can take quite some time, and may consume quite a bit of system memory (~2.5GB).
"""
import os
import numpy as np
import gpflow

from uhi import UHIData
from gp import UHIRegression, reset_gpflow_graph


def train(model_dir):
    """
    Trains a Gaussian process regression-model.
    :param model_dir: Path to a directory which contains the files 'tiltplate_rawdata.h5' and 'tiltplate_var.npz'.

    The model is trained in two steps, first with a single band/wavelength and then refined for all wavelengths.
    The benefits of this are twofold
        1.  Shared variables can be optimized in the first pass with less data, and kept fixed in the second pass. Thus
            speeding up the optimization.
        2.  Some bands has a low signal level (especially towards the edges of the spectrum). This would
            for example negatively affect the kernel parameters, as the function would seem overly smooth.

    The likelihood variance is kept fixed for both passes, as this has been recorded while stationary over the plate.
    :return:
    """

    uhi = UHIData()
    uhi.read_hdf5(os.path.join(model_dir, 'tiltplate_rawdata.h5'))
    uhi.subtract_ambient()
    uhi.correct_gain()
    h, w, d = uhi.px.shape

    with np.load(os.path.join(model_dir, 'tiltplate_var.npz')) as f:
        tp_var = f['tp_var']  # Variance recorded whilst stationary over plate (top)

    print(tp_var.shape)

    # Regressor (input locations)
    xx, yy = np.meshgrid(uhi.alt, uhi.fov, indexing='ij')
    X = np.stack((xx.ravel(), yy.ravel()), axis=1)

    #
    # Step 1: optimize inducing point locations using a single band
    #
    i_d = np.argmax(np.mean(uhi.px[:, w//2, :], axis=0))

    # Regressand
    Y = uhi.px[:, :, i_d].reshape((-1, 1))

    m = UHIRegression(X, Y, n_inducing=5, scaling=True, multi_output=False)
    m.lh = gpflow.likelihoods.Gaussian(variance=tp_var[i_d])
    m.lh.variance.trainable = False

    # Train model
    m.train(verbose=True, maxiter=1500)

    z = m.m.feature.read_values()['SVGP/feature/Z']
    kern_ls = m.m.kern.read_values()['SVGP/kern/lengthscales']
    kern_var = m.m.kern.read_values()['SVGP/kern/variance']

    #
    # Step 2: Train model with inducing points locked and all output bands
    #
    reset_gpflow_graph()

    # New regressand with all data
    Y = uhi.px.reshape((-1, d))

    m = UHIRegression(X, Y, n_inducing=5, scaling=True, multi_output=True)

    # Likelihood
    m.lh = gpflow.likelihoods.Gaussian(variance=tp_var)
    m.lh.variance.trainable = False

    # Inducing points
    m.z = z
    m.feature = gpflow.features.InducingPoints(z)
    m.feature.trainable = False

    # Kernel
    m.kern = gpflow.kernels.SquaredExponential(input_dim=2, variance=kern_var, lengthscales=kern_ls)
    m.kern.variance.trainable = False
    m.kern.lengthscales.trainable = False

    m.train(verbose=True, maxiter=500)

    return m


if __name__ == '__main__':
    model = train('data' + os.path.sep + 'model')
    model.save_model('data' + os.path.sep + 'model' + os.path.sep + 'gpflow_tiltplate_model_new.h5')

