"""
Hyperspectral data class
"""

import h5py as h5
import numpy as np


class UHIData:
    def __init__(self):
        #
        # Flags
        #
        self.ambient_subtracted = False
        self.gain_corrected = False

        #
        # Data fields
        #

        # Hyperspectral data [counts] (h, w, d)
        self.px = None  # type: np.ndarray

        # Wavelengths [nm] (d,)
        self.wl = None  # type: np.ndarray

        # Field of view [rad] (w,)
        self.fov = None  # type: np.ndarray

        # Gain (h,)
        self.gain = None  # type: np.ndarray

        # Exposure (h,)
        self.exposure = None  # type: np.ndarray

        # Altitude [m] (h,)
        self.alt = None  # type: np.ndarray

        # Tilted reference plate model output (w, d)
        # Taken at the same height as top of the sample
        self.f_tp = None  # type: np.ndarray

        # Tilted reference plate model output (w, d)
        # Predicted at the same height as the flat reference plate
        self.f_fp = None  # type: np.ndarray

        # Spectrometer measurement of flat reference plate
        self.rho_fp = None  # type. np.ndarray (d,)

        # Ambient measurement (dark current) (w, d)
        self.px_amb = None  # type: np.ndarray

        # Measurement of flat reference plate (median across multiple lines) (w, d)
        self.e_fp = None  # type: np.ndarray

        # Calculated reflectance (h, w, d)
        self.refl = None  # type: np.ndarray

    def read_hdf5(self, fpath):
        with h5.File(fpath, 'r') as hf:
            self.px = hf['data']['px'][:].astype(np.float32)
            self.wl = hf['data']['wl'][:]
            self.fov = hf['data']['fov'][:]
            self.gain = hf['data']['gain'][:].astype(np.float64)
            self.exposure = hf['data']['exposure'][:]
            self.alt = hf['data']['alt'][:]
            self.f_fp = hf['data']['f_fp'][:]
            self.rho_fp = hf['data']['rho_fp'][:]
            self.px_amb = hf['data']['px_amb'][:]
            self.e_fp = hf['data']['e_fp'][:]

            try:
                self.f_tp = hf['data']['f_tp'][:]
            except KeyError:
                pass

    def subtract_ambient(self):
        """
        Subtract ambient measurement / dark current
        :return:
        """
        if not self.ambient_subtracted:
            self.px -= self.px_amb[np.newaxis, :, :]
            self.ambient_subtracted = True

    def correct_gain(self):
        """
        Apply gain correction (note: specific to Ecotone UHI)
        :return:
        """

        if not self.gain_corrected:
            self.subtract_ambient()
            g = 10**(self.gain / 200.0)
            self.px *= 1.0 / g[:, np.newaxis, np.newaxis]
            self.gain_corrected = True

    def calc_refl(self, model_path=None):
        h, w, d = self.px.shape

        # If model path is specified, the embedded model output is not used.
        # A new prediction of tilted plate measurement is made from the gaussian process model
        if model_path is not None:
            # Read gaussian process model
            # Note: requires tensorflow and gpflow
            try:
                from utils.gp import UHIRegression, reset_gpflow_graph

                reset_gpflow_graph()  # Necessary to run in loop
                tp_gp = UHIRegression()
                tp_gp.load_model(model_path)

                # Predict tilt-plate response for the given altitude/height and fov
                if len(np.unique(self.alt)) == 1:
                    # If constant altitude/height, the same calibration can be used for all lines
                    xx_pred, yy_pred = np.meshgrid(self.alt[0], self.fov, indexing='ij')
                    x_pred = np.stack((xx_pred.ravel(), yy_pred.ravel()), axis=1)
                    f_tp_mean, f_tp_var = tp_gp.predict_f(x_pred)
                    self.f_tp = f_tp_mean.reshape((w, d))
                else:
                    # Compute calibration data in chunks for the specified altitudes/heights
                    f_tp = np.empty_like(self.px)
                    n_chunks = self.h // 15  # Replace 15 with another number to speed up computation vs memory usage
                    for i in range(n_chunks):
                        i_start = i * h // n_chunks
                        i_end = None if i == (n_chunks - 1) else (i + 1) * h // n_chunks
                        print(f'Computing calibration data for h {i_start}-{i_end if i_end else h} of {h}')

                        # Predict f_tp
                        xx_pred, yy_pred = np.meshgrid(self.alt[i_start:i_end], self.fov, indexing='ij')
                        x_pred = np.stack((xx_pred.ravel(), yy_pred.ravel()), axis=1)
                        f_tp_mean, _ = tp_gp.predict_f(x_pred)
                        f_tp[i_start:i_end, :, :] = f_tp_mean.reshape((-1, w, d))

                    self.f_tp = f_tp

                # Predict f_fp (values of tilted plate at same height as flat reference)
                xx_pred, yy_pred = np.meshgrid(0.58 - 0.02, self.fov, indexing='ij')
                x_pred = np.stack((xx_pred.ravel(), yy_pred.ravel()), axis=1)
                f_fp_mean, _ = tp_gp.predict_f(x_pred)
                self.f_fp = f_fp_mean.reshape((w, d))

            except ImportError:
                print('Failed to import tensorflow/gpflow.')
                print('Falling back to embedded calibration data.')

        if self.f_tp is None:
            print('Calibration data not found, cannot calculate reflectance. Aborting.')
            exit()

        # Ensure broadcasting of reference plate if common height is assumed
        f_tp = self.f_tp[np.newaxis, :, :] if len(self.f_tp.shape) == 2 else self.f_tp

        # Calculate correction factor based on known reflectance of flat plate and predicted reflectance
        # from the Gaussian process sampled at the same height as the flat plate.
        fp_calib = (self.f_fp / self.e_fp) * self.rho_fp[np.newaxis, :]

        # Apply correction to the ratio between E_m (px) and Gaussian process sampled at the same height as the sample
        self.refl = self.px * (fp_calib[np.newaxis, :, :] / f_tp)

        return self.refl




