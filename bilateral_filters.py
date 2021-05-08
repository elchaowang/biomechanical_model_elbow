from scipy import signal
import numpy as np


def filter_mot(data, N=4, fs=2000, low_cut=10):
    b, a = signal.butter(N, 2 * low_cut / fs, 'lowpass')
    lpf_data = signal.filtfilt(b, a, data)
    return lpf_data


def mov_avrg(ls, N):
    n = np.ones(N)
    weights = n / N
    sma = np.convolve(weights, ls, mode='valid')
    return sma


def emg_2_activation(data, N=4, fs=2000, low_cut=2, band_cut=[20, 500]):
    b0, a0 = signal.butter(N, [(band_cut[0] * 2 / fs), (band_cut[1] * 2 / fs)], 'bandpass')
    bpf_data = signal.filtfilt(b0, a0, data)

    RECT_emg = np.abs(bpf_data)

    MovAvrg = mov_avrg(RECT_emg, 176)

    b, a = signal.butter(2, 2 * low_cut / fs, 'lowpass')

    lpf_data1 = signal.filtfilt(b, a, MovAvrg)
    lpf_data2 = signal.filtfilt(b, a, lpf_data1)
    return lpf_data2


def prediction_emg_filter(data, order=4, fs=2000, low_cut=2):
    b, a = signal.butter(order, 2 * low_cut / fs, 'lowpass')

    lpf_data = signal.filtfilt(b, a, data)
    return lpf_data


def filter_force(data, order=2, fs=2000, low_cut=10):
    b, a = signal.butter(order, 2 * low_cut / fs, 'lowpass')

    lpf_data = signal.filtfilt(b, a, data)
    return lpf_data


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
