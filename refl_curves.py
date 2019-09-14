"""
Computes the reflectance curves in figures 7 and 8
"""
import os
import imageio
import numpy as np
import matplotlib.pyplot as plt

from uhi import UHIData


def fig_reflectance(sample_dir, mask_dir, model_path=None, median_centered=True):
    """
    Create reflectance figures from samples and masks contained in the given directories
    :param sample_dir: The path to the directory containing the UHI files (A.h5, B.h5.. etc)
    :param mask_dir: The path to the directory containing the masks (A-1.png, B-2.png.. etc)
    :param model_path: [Optional] Path to a gaussian process model (.h5) to use instead of the embedded calibration data
    :param median_centered: If true, each spectr is centered around the average median
    :return:
    """

    # Read masks
    mask_files = [x for x in os.listdir(mask_dir) if x.endswith('.png') and '-' in x]
    mask_files = sorted(mask_files)

    # Group samples by UHI file
    samples = {}
    for fname in mask_files:
        fbase, fext = os.path.splitext(fname)
        s_id, m_id = fbase.split('-')
        try:
            samples[s_id].append(fname)
        except (KeyError, AttributeError):
            samples[s_id] = [fname]

    # Iterate over samples and masks
    for s_id, masks in samples.items():
        # Read uhi file
        uhi = UHIData()
        uhi.read_hdf5(os.path.join(sample_dir, s_id + '.h5'))
        uhi.subtract_ambient()
        uhi.correct_gain()
        h, w, d = uhi.px.shape
        wl = np.broadcast_to(uhi.wl[np.newaxis, np.newaxis, :], shape=uhi.px.shape)

        # Calculate reflectance
        uhi.calc_refl(model_path)

        # Plot reflectance curves for masks
        for mask in masks:
            print(f'Plotting {os.path.splitext(mask)[0]} ...')
            im_mask = imageio.imread(os.path.join(mask_dir, mask))
            if len(im_mask.shape) > 2:
                im_mask = im_mask[:, :, 0]

            # Convert to boolean
            im_mask = im_mask > 127

            # Extract masked area
            s_refl = uhi.refl[im_mask, :].reshape((-1, d))
            s_wl = wl[im_mask, :].reshape((-1, d))
            s_refl_median = np.median(s_refl, axis=0)

            if median_centered:
                s_refl -= np.mean((s_refl - s_refl_median[np.newaxis, :]), axis=1)[:, np.newaxis]

            # Plot reflectance curve for mask
            plt.figure()
            plt.title(f'Sample {os.path.splitext(mask)[0]}')
            plt.scatter(s_wl.ravel(), s_refl.ravel(), c='k', s=0.05)
            plt.scatter(uhi.wl, s_refl_median, c='r', s=1.0)
            plt.ylim([np.min(s_refl[:, 30:-20]), np.max(s_refl[:, 30:-20])])
            plt.xlim([uhi.wl.min(), uhi.wl.max()])
            plt.ylabel('Reflectance [%]')
            plt.xlabel('Wavelength [nm]')

    plt.show()


if __name__ == '__main__':
    # Plot reflectance curves from masked areas and calibration data embedded in the UHI-files
    fig_reflectance(sample_dir='data' + os.path.sep + 'samples',
                    mask_dir='data' + os.path.sep + 'masks',
                    model_path=None)

    # Plot reflectance curves from masked areas and external regression model
    """
    fig_reflectance(sample_dir='data' + os.path.sep + 'samples',
                    mask_dir='data' + os.path.sep + 'masks',
                    model_path=os.path.join('data', 'model', 'gpflow_tiltplate_model'))
    """
