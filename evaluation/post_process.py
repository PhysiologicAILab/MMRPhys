"""The post processing files for caluclating heart rate using FFT or peak detection.
The file also  includes helper funcs such as detrend, power2db etc.
"""

import numpy as np
import scipy.io
from scipy.signal import butter, periodogram, welch, find_peaks, filtfilt
from scipy.sparse import spdiags
from copy import deepcopy

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def _detrend(input_signal, lambda_value):
    """Detrend PPG signal."""
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal

def power2db(mag):
    """Convert power to db."""
    return 10 * np.log10(mag)

def _calculate_fft_hr(ppg_signal, fs=60, low_pass=0.6, high_pass=3.3):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = ppg_signal.shape[1]
    if N <= 30*fs:
        nfft = _next_power_of_2(N)
        f_ppg, pxx_ppg = periodogram(ppg_signal, fs=fs, nfft=nfft, detrend=False)
    else:
        f_ppg, pxx_ppg = welch(ppg_signal, fs=fs, nperseg=N//2, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr

def _calculate_peak_hr(ppg_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    ppg_peaks, _ = find_peaks(ppg_signal)
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return hr_peak


# RSP Metrics
# def _calculate_fft_rr(rsp_signal, fs=30, low_pass=0.1, high_pass=0.54):
def _calculate_fft_rr(rsp_signal, fs=30, low_pass=0.13, high_pass=0.5):
    """Calculate respiration rate using Fast Fourier transform (FFT)."""
    resp_signal = deepcopy(rsp_signal)
    avg_resp = np.mean(resp_signal)
    std_resp = np.std(resp_signal)
    resp_signal = (resp_signal - avg_resp) / std_resp   # Standardize to remove DC level - which was due to min-max normalization

    # sig_len = len(resp_signal)
    # last_zero_crossing = np.where(np.diff(np.sign(resp_signal)))[0][-1]
    # resp_signal = resp_signal[: last_zero_crossing]
    # inv_resp_signal = deepcopy(resp_signal)
    # inv_resp_signal = -1 * inv_resp_signal[::-1]
    
    # # Higher signal length is needed to reliably compute FFT for low frequencies
    # resp_signal = np.concatenate([resp_signal, inv_resp_signal[1:], resp_signal[1:],
    #                              inv_resp_signal[1:], resp_signal[1:], inv_resp_signal[1:]], axis=0)
    
    # resp_signal = resp_signal[:4*sig_len]

    resp_signal = np.expand_dims(resp_signal, 0)
    N = resp_signal.shape[1]
    if N <= 30*fs:
        nfft = _next_power_of_2(N)
        f_resp, pxx_resp = periodogram(resp_signal, fs=fs, nfft=nfft, detrend=False)
    else:
        f_resp, pxx_resp = welch(resp_signal, fs=fs, nperseg=N//2, detrend=False)

    fmask_resp = np.argwhere((f_resp >= low_pass) & (f_resp <= high_pass))
    mask_resp = np.take(f_resp, fmask_resp)
    mask_pxx = np.take(pxx_resp, fmask_resp)
    fft_rr = np.take(mask_resp, np.argmax(mask_pxx, 0))[0] * 60
    return fft_rr


def _calculate_peak_rr(resp_signal, fs):
    """Calculate respiration rate based on PPG using peak detection."""
    resp_peaks, _ = find_peaks(resp_signal)
    rr_peak = 60 / (np.mean(np.diff(resp_peaks)) / fs)
    return rr_peak


def _compute_macc(pred_signal, gt_signal):
    """Calculate maximum amplitude of cross correlation (MACC) by computing correlation at all time lags.
        Args:
            pred_signal(np.array): predicted signal 
            label_signal(np.array): ground truth, label signal
        Returns:
            MACC(float): Maximum Amplitude of Cross-Correlation
    """
    pred = deepcopy(pred_signal)
    gt = deepcopy(gt_signal)
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    min_len = np.min((len(pred), len(gt)))
    pred = pred[:min_len]
    gt = gt[:min_len]
    lags = np.arange(0, len(pred)-1, 1)
    tlcc_list = []
    for lag in lags:
        cross_corr = np.abs(np.corrcoef(
            pred, np.roll(gt, lag))[0][1])
        tlcc_list.append(cross_corr)
    macc = max(tlcc_list)
    return macc


def _calculate_SNR(pred_signal, metrics_label, fs=30, low_pass=0.6, high_pass=3.3):
    """Calculate SNR as the ratio of the area under the curve of the frequency spectrum around the first and second harmonics 
        of the ground truth HR frequency to the area under the curve of the remainder of the frequency spectrum, from 0.6 Hz
        to 3.3 Hz. 
        Ref for low_pass and high_pass filters:
        R. Cassani, A. Tiwari and T. H. Falk, "Optimal filter characterization for photoplethysmography-based pulse rate and 
        pulse power spectrum estimation," 2020 IEEE Engineering in Medicine & Biology Society (EMBC), Montreal, QC, Canada,
        doi: 10.1109/EMBC44109.2020.9175396.

        Args:
            pred_signal(np.array): predicted signal 
            fs(int or float): sampling rate of the video
        Returns:
            SNR(float): Signal-to-Noise Ratio
    """
    pred_sig = deepcopy(pred_signal)
    avg_sig = np.mean(pred_sig)
    std_sig = np.std(pred_sig)
    pred_sig = (pred_sig - avg_sig) / std_sig   # Standardize to remove DC level - which was due to min-max normalization

    # Get the first and second harmonics of the ground truth HR in Hz
    first_harmonic_freq = metrics_label / 60
    second_harmonic_freq = 2 * first_harmonic_freq
    deviation = 6 / 60  # 6 beats/min converted to Hz (1 Hz = 60 beats/min)

    # Calculate FFT
    pred_sig = np.expand_dims(pred_sig, 0)
    N = pred_sig.shape[1]
    if N < 256:
        nfft = _next_power_of_2(N)
        f_sig, pxx_sig = periodogram(pred_sig, fs=fs, nfft=nfft, detrend=False)
    else:
        f_sig, pxx_sig = welch(pred_sig, fs=fs, nperseg=N//2, detrend=False)

    # Calculate the indices corresponding to the frequency ranges
    idx_harmonic1 = np.argwhere((f_sig >= (first_harmonic_freq - deviation)) & (f_sig <= (first_harmonic_freq + deviation)))
    idx_harmonic2 = np.argwhere((f_sig >= (second_harmonic_freq - deviation)) & (f_sig <= (second_harmonic_freq + deviation)))
    idx_remainder = np.argwhere((f_sig >= low_pass) & (f_sig <= high_pass) \
     & ~((f_sig >= (first_harmonic_freq - deviation)) & (f_sig <= (first_harmonic_freq + deviation))) \
     & ~((f_sig >= (second_harmonic_freq - deviation)) & (f_sig <= (second_harmonic_freq + deviation))))

    # Select the corresponding values from the periodogram
    pxx_sig = np.squeeze(pxx_sig)
    pxx_harmonic1 = pxx_sig[idx_harmonic1]
    pxx_harmonic2 = pxx_sig[idx_harmonic2]
    pxx_remainder = pxx_sig[idx_remainder]

    # Calculate the signal power
    signal_power_hm1 = np.sum(pxx_harmonic1**2)
    signal_power_hm2 = np.sum(pxx_harmonic2**2)
    signal_power_rem = np.sum(pxx_remainder**2)

    # Calculate the SNR as the ratio of the areas
    if not signal_power_rem == 0: # catches divide by 0 runtime warning 
        SNR = power2db((signal_power_hm1 + signal_power_hm2) / signal_power_rem)
    else:
        SNR = 0
    return SNR



def calculate_metric_per_video(predictions, labels, fs=30, diff_flag=False, use_bandpass=True, hr_method='FFT'):
    """Calculate video-level HR and SNR"""
    if diff_flag:  # if the predictions and labels are 1st derivative of PPG signal.
        # predictions = _detrend(np.cumsum(predictions), 100)
        predictions = np.cumsum(predictions)
        # labels = _detrend(np.cumsum(labels), 100)
        labels = np.cumsum(labels)
    else:
        # predictions = _detrend(predictions, 100)
        # labels = _detrend(labels, 100)
        pass
    if use_bandpass:
        # bandpass filter between [0.75, 2.5] Hz, equals [45, 150] beats per min
        # bandpass filter between [0.6, 3.3] Hz, equals [36, 198] beats per min
        [b, a] = butter(2, [0.6 / fs * 2, 3.3 / fs * 2], btype='bandpass')
        predictions = filtfilt(b, a, np.double(predictions))
        labels = filtfilt(b, a, np.double(labels))
    
    macc = _compute_macc(predictions, labels)

    if hr_method == 'FFT':
        hr_pred = _calculate_fft_hr(predictions, fs=fs)
        hr_label = _calculate_fft_hr(labels, fs=fs)
    elif hr_method == 'Peak':
        hr_pred = _calculate_peak_hr(predictions, fs=fs)
        hr_label = _calculate_peak_hr(labels, fs=fs)
    else:
        raise ValueError('Please use FFT or Peak to calculate your HR.')
    SNR = _calculate_SNR(predictions, hr_label, fs=fs)
    return hr_label, hr_pred, SNR, macc


def calculate_rsp_metrics_per_video(predictions, labels, fs=30, diff_flag=False, use_bandpass=True, rr_method='FFT'):
    """Calculate video-level RR"""
    if diff_flag:  # if the predictions and labels are 1st derivative of RSP signal.
        # predictions = _detrend(np.cumsum(predictions), 100)
        # labels = _detrend(np.cumsum(labels), 100)
        predictions = np.cumsum(predictions)
        labels = np.cumsum(labels)
    else:
        # predictions = _detrend(predictions, 100)
        # labels = _detrend(labels, 100)
        pass
    if use_bandpass:
        # bandpass filter between [0.05, 0.7] Hz
        # equals [3, 42] breaths per min
        # [b, a] = butter(2, [0.05 / fs * 2, 0.7 / fs * 2], btype='bandpass')
        [b, a] = butter(2, [0.13 / fs * 2, 0.5 / fs * 2], btype='bandpass')
        predictions = filtfilt(b, a, np.double(predictions))
        labels = filtfilt(b, a, np.double(labels))
    
    macc = _compute_macc(predictions, labels)
    
    if rr_method == 'FFT':
        rr_pred = _calculate_fft_rr(predictions, fs=fs)
        rr_label = _calculate_fft_rr(labels, fs=fs)
    elif rr_method == 'Peak':
        rr_pred = _calculate_peak_rr(predictions, fs=fs)
        rr_label = _calculate_peak_rr(labels, fs=fs)
    else:
        raise ValueError('Please use FFT or Peak to calculate your RR.')
    # SNR = _calculate_SNR(predictions, rr_label, fs=fs, low_pass=0.05, high_pass=0.7)
    # SNR = _calculate_SNR(predictions, rr_label, fs=fs, low_pass=0.13, high_pass=0.5)
    SNR = _calculate_SNR(predictions, rr_label, fs=fs, low_pass=0.1, high_pass=0.54)
    return rr_label, rr_pred, SNR, macc

