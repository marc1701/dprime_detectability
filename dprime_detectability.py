import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PyOctaveBand import *


def _spl_db_to_pa(db):
    return 10 ** (db / 20) * 2e-5


class DPrimeDetectability(object):
    def __init__(self,
                 signal,
                 masker,
                 fs=48000,
                 freq_limits=None,
                 eta=0.3,
                 delta_t=0.5,
                 alpha=3,
                 delta=14,
                 rho=1):

        if signal.ndim != 1:
            signal = signal[:, 0]

        if masker.ndim != 1:
            masker = masker[:, 0]

        self.signal = signal
        self.masker = masker
        self.fs = fs
        self.delta_t = delta_t
        self.alpha = alpha
        self.delta = delta
        self.rho = rho

        if not freq_limits: self.freq_limits = [30, 4000]
        else: self.freq_limits = freq_limits

        freq_bands = np.array(getansifrequencies(3, limits=self.freq_limits))
        self.f_c = freq_bands[0]
        self.freqband_width = np.diff(freq_bands[1:], axis=0)

        # calculate and store third-octave spectrograms
        self.signal_spectrogram = self.third_octave_spl(signal)
        self.masker_spectrogram = self.third_octave_spl(masker)

        # calculate detection efficiency curve
        self.auditory_filter_width = 24.7 * (1 + (4.37 / 1000) * self.f_c)
        self._det_eff = (eta * np.sqrt(delta_t) * np.sqrt(self.freqband_width)
                         * np.sqrt(self.freqband_width / self.auditory_filter_width)
                         * 10 ** (self._kick() / 10)).squeeze()

        # calculate equivalent auditory noise (threshold of hearing)
        thresholds = np.loadtxt('thresholds_iso389-7.csv', skiprows=1, delimiter=',')
        freq_range = (thresholds[:, 0] >= min(self.freq_limits)) & (thresholds[:, 0] <= max(self.freq_limits))
        min_audible_field = _spl_db_to_pa(thresholds[freq_range, 1]) ** 2
        self.equiv_auditory_noise = self._det_eff * (min_audible_field / 2)

    def third_octave_spl(self, x, plot=False):
        delta_t_samples = int(self.fs * self.delta_t)

        # split input into frames based on time window
        x_frames = np.lib.stride_tricks.sliding_window_view(
            x, delta_t_samples)[::delta_t_samples]
        
        # calculate third-octave band levels and store in array
        spl_frames = []
        for frame in x_frames:
            spl, _ = octavefilter(frame, self.fs,
                                      fraction=3,
                                      limits=self.freq_limits)
            spl_frames.append(spl)
        spl_frames = np.array(spl_frames)
        if plot:
            self.plot(10*np.log10(spl_frames), vmin=None)
            plt.show()
        return spl_frames

    def d_prime(self, log=False, plot=False):
        dp = self._det_eff * (self.signal_spectrogram / self.masker_spectrogram + self.equiv_auditory_noise)
        if plot:
            self.plot(10*np.log10(dp))
            plt.show()
        if log: return 10*np.log10(dp)
        else: return dp

    def d_prime_single_vals(self, plot=False):
        # integrate over frequency axis (after Buus et al. multiband model)
        freq_integration = np.sqrt(np.sum(self.d_prime()**2, 1))
        print(freq_integration)
        if plot:
            plt.plot(freq_integration)
            plt.show()

        dp_50 = float(np.percentile(freq_integration, 50))
        dp_95 = float(np.percentile(freq_integration, 95))
        dp_max = float(np.max(freq_integration))
        return dp_50, dp_95, dp_max

    def discounted_level(self, plot=False):
        l_disc = (20 * np.log10(np.sqrt(self.signal_spectrogram / 2e-5)) -
                (self.alpha / (self.d_prime(log=False) / self.delta) ** self.rho))
        if plot:
            self.plot(l_disc)
            plt.show()
        return l_disc

    def _kick(self):
        kick = -6 * (np.log10(self.f_c / 2500)) ** 2
        kick[self.f_c <= 2500] = 0
        return kick

    def plot(self, arr, vmin=0):
        freqs = [f'{f:.2f}' for f in self.f_c]
        t = np.linspace(0, len(arr) * self.delta_t - self.delta_t, len(arr))

        df = pd.DataFrame(arr.T, columns=t, index=freqs)
        df = df.iloc[::-1]

        sns.heatmap(df, vmin=vmin)


def third_octave_spl(x, fs, delta_t, freq_limits=None):
    if freq_limits is None:
        freq_limits = [30, 4000]
    delta_t_samples = int(fs * delta_t)

    # split input into frames based on time window
    x_frames = np.lib.stride_tricks.sliding_window_view(
        x, delta_t_samples)[::delta_t_samples]

    # calculate third-octave band levels and store in array
    spl_frames = []
    for frame in x_frames:
        spl, _ = octavefilter(frame, fs,
                              fraction=3,
                              limits=freq_limits)
        spl_frames.append(spl)
    spl_frames = np.array(spl_frames)
    return spl_frames