"""
A simple viewer for raw UHIData files
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button

from utils.uhi import UHIData


class UHIDisplay:
    def __init__(self, sample_dir):
        self.sample_dir = sample_dir
        self.sample_idx = 0
        self.uhi = None
        self.h, self.w, self.d = 0, 0, 0
        self.px_min, self.px_max, self.refl_min, self.refl_max = 0, 0, 0, 0
        self.has_refl = False
        self.show_refl = False
        self.autoscale = True

        # List sample files
        print(f'Reading sample directory... ({self.sample_dir})')
        self.sample_files = [fname for fname in os.listdir(self.sample_dir) if fname.endswith('.h5')]
        if not self.sample_files:
            print('No samples found in given directory, exiting...')
            exit()

        self.sample_files = sorted(self.sample_files)

        # Initialize first sample
        self.update_sample()

        # Create figure
        self.fig = plt.figure(figsize=(12, 4))
        plt.tight_layout()
        gridspec.GridSpec(1, 2)
        self.ax_uhi = plt.subplot2grid((3, 1), (0, 0), colspan=1, rowspan=3)
        self.image = self.ax_uhi.imshow(self.uhi.px[:, :, self.d//2], origin='lower')

        # Wavelength slider
        axfreq = plt.axes([0.1, 0.1, 0.8, 0.05], facecolor='lightgoldenrodyellow')
        self.sband = Slider(axfreq, 'Band [nm]', 0, self.d - 1, valinit=self.d//2)
        self.sband.on_changed(self.update_band)

        # Previous button
        self.btn_prev_ax = plt.axes([0.04, 0.8, 0.1, 0.075])
        self.btn_prev = Button(self.btn_prev_ax, 'Prev')
        self.btn_prev.on_clicked(self.click_prev)

        # Next button
        self.btn_next_ax = plt.axes([0.15, 0.8, 0.1, 0.075])
        self.btn_next = Button(self.btn_next_ax, 'Next')
        self.btn_next.on_clicked(self.click_next)

        # Raw data / reflectance button
        self.btn_toggle_ax = plt.axes([0.26, 0.8, 0.1, 0.075])
        self.btn_toggle = Button(self.btn_toggle_ax, 'Reflectance')
        self.btn_toggle.on_clicked(self.click_toggle)

        # Autoscaling (per-band) / uniform scaling button
        self.btn_autoscale_ax = plt.axes([0.37, 0.8, 0.1, 0.075])
        self.btn_autoscale = Button(self.btn_autoscale_ax, 'Autoscale off')
        self.btn_autoscale.on_clicked(self.click_autoscale)

        self.update_title()
        self.fig.tight_layout()
        plt.tight_layout()
        plt.show()

    def update_sample(self):
        self.uhi = UHIData()
        self.uhi.read_hdf5(os.path.join(self.sample_dir, self.sample_files[self.sample_idx]))
        (self.w, self.h, self.d) = self.uhi.px.shape

        # Calculate reflectance
        self.uhi.calc_refl()
        self.refl_min, self.refl_max = np.min(self.uhi.refl[:, :, 50:-50]), np.max(self.uhi.refl[:, :, 50:-50])
        self.px_min, self.px_max = np.min(self.uhi.px[:, :, 50:-50]), np.max(self.uhi.px[:, :, 50:-50])

        self.has_refl = self.uhi.refl is not None

    def update_band(self, val=None):
        if val is None:
            val = self.sband.val
        band = np.floor(self.sband.val).astype(np.int)
        px = self.uhi.refl[:, :, band] if self.show_refl else self.uhi.px[:, :, band]
        self.image.set_data(px)

        if self.autoscale:
            self.image.autoscale()
        else:
            self.image.set_clim(vmin=self.refl_min if self.show_refl else self.px_min,
                                vmax=self.refl_max if self.show_refl else self.px_max)

        self.sband.valtext.set_text(f'{self.uhi.wl[int(np.floor(val))]:.2f}')
        self.fig.canvas.draw_idle()

    def update_title(self):
        self.fig.suptitle(f'Sample {os.path.splitext(self.sample_files[self.sample_idx])[0]}')

    def click_prev(self, event):
        self.sample_idx -= 1
        self.sample_idx = self.sample_idx if self.sample_idx > 0 else len(self.sample_files) - 1
        self.update_sample()
        self.update_band()
        self.update_title()

    def click_next(self, event):
        self.sample_idx += 1
        self.sample_idx = self.sample_idx % len(self.sample_files)
        self.update_sample()
        self.update_band()
        self.update_title()

    def click_toggle(self, event):
        if not self.has_refl:
            print('Sample does not have calibration data embedded. Cannot show reflectance')
            self.show_refl = False
        else:
            self.show_refl = not self.show_refl

        self.btn_toggle.label.set_text('Reflectance' if not self.show_refl else 'Raw data')
        print('Displaying reflectance' if self.show_refl else 'Displaying raw data')
        self.update_band()

    def click_autoscale(self, event):
        self.autoscale = not self.autoscale
        self.btn_autoscale.label.set_text('Autoscale off' if self.autoscale else 'Autoscale on')
        print('Autoscale on' if self.autoscale else 'Autoscale off')
        self.update_band()

    def reset(self, event):
        self.sband.reset()


if __name__ == '__main__':
    sample_dir = 'data' + os.path.sep + 'samples'
    UHIDisplay(sample_dir)
