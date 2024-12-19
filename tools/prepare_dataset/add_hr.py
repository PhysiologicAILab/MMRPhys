from pathlib import Path
import numpy as np
import argparse
import scipy
from scipy.signal import butter
# import mat73
import pandas as pd
import matplotlib.pyplot as plt
# import neurokit2 as nk

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _calculate_fft_hr(ppg_signal, fs=60, low_pass=0.6, high_pass=3.3):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""

    # df, info = nk.ppg_process(ppg_signal, sampling_rate=fs)
    # peaks_hr = np.median(nk.ppg_rate(info["PPG_Peaks"], sampling_rate=fs))

    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    
    # return fft_hr, peaks_hr
    return fft_hr, None


class AddHR2Labels(object):
    def __init__(self, datadir, fps, file_filter) -> None:
        self.fps = fps
        self.file_filter = file_filter
        self.files = sorted(list(Path(datadir).glob(self.file_filter)))
        [self.b, self.a] = butter(2, [0.6 / self.fps * 2, 3.3 / self.fps * 2], btype='bandpass')

    def compute_and_add_hr(self):
        hr_vals = []
        hr_vals_seg = []

        peaks_hr_vals = []
        peaks_hr_vals_seg = []
        
        seg_len = 200
        for fn in self.files:
            if "npy" in self.file_filter:
                data = np.load(fn)
                if len(data.shape) < 2:
                    data = scipy.signal.filtfilt(self.b, self.a, np.double(data))
                    hr, peaks_hr = _calculate_fft_hr(data, fs=self.fps)
                    hr = int(np.round(hr))
                    temp_data = np.expand_dims(data, 1)
                    hr_vec = hr * np.ones_like(temp_data)
                    new_data = np.concatenate([temp_data, hr_vec], axis=1)
                    print("*", hr, new_data.shape)
                    # exit()
                    np.save(str(fn), new_data)
                else:
                    bvp = data[:, 0]
                    bvp = scipy.signal.filtfilt(self.b, self.a, np.double(bvp))
                    hr, peaks_hr = _calculate_fft_hr(bvp, fs=self.fps)
                    hr = int(np.round(hr))
                    hr_vec = hr * np.ones_like(bvp)
                    hr_vec = np.expand_dims(hr_vec, 1)
                    new_data = np.concatenate([data, hr_vec], axis=1)
                    print(".", hr, new_data.shape)
                    # exit()
                    # np.save(str(fn), new_data)
            
            elif "csv" in self.file_filter:
                with open(str(fn), "r") as f:
                    data = pd.read_csv(f).to_numpy()
                bvp = np.array(data[:, 1])
                bvp = scipy.signal.filtfilt(self.b, self.a, np.double(bvp))
                hr, peaks_hr = _calculate_fft_hr(bvp, fs=self.fps)
                hr = int(np.round(hr))
                hr_vals.append(hr)
                # peaks_hr = int(np.round(peaks_hr))
                # peaks_hr_vals.append(peaks_hr)

                print_hr_seg = []
                # print_peaks_hr_seg = []
                for seg in np.arange(0, 600, seg_len):
                    hr_s, peaks_hr_s = _calculate_fft_hr(bvp[seg:seg+seg_len], fs=self.fps)
                    hr_s = int(round(hr_s))

                    print_hr_seg.append(hr_s)

                    # peaks_hr_s = int(round(peaks_hr_s))
                    # print_peaks_hr_seg.append(peaks_hr_s)
                    
                    hr_vals_seg.append(hr_s)
                    peaks_hr_vals_seg.append(peaks_hr_s)

                # print("*", hr, peaks_hr, print_hr_seg, print_peaks_hr_seg)
                print("*", hr, peaks_hr, print_hr_seg)

                # bvp = np.expand_dims(bvp, axis=1)
                # hr_vec = hr * np.ones_like(bvp)

                hr_vals = np.array(hr_vals)
                hr_vals_seg = np.array(hr_vals_seg)
                # peaks_hr_vals = np.array(peaks_hr_vals)
                # peaks_hr_vals_seg = np.array(peaks_hr_vals_seg)

                plt.hist(hr_vals, bins=np.arange(30, 200, 5))
                plt.savefig("HR_Vals.jpg")
                plt.close()

                plt.hist(hr_vals_seg, bins=np.arange(30, 200, 5))
                plt.savefig("HR_Vals_Seg.jpg")
                plt.close()

                # plt.hist(peaks_hr_vals, bins=np.arange(30, 200, 5))
                # plt.savefig("Peaks HR_Vals.jpg")
                # plt.close()

                # plt.hist(peaks_hr_vals_seg, bins=np.arange(30, 200, 5))
                # plt.savefig("peaks_hr_vals_seg.jpg")
                # plt.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default="/home/jitesh/data/PURE_Dataset/PURE_Raw_300_36x36",
                        dest="datadir", type=str, help='path of the data')
    parser.add_argument('--fps', default=30,
                        dest="fps", type=int, help='sampling rate')
    parser.add_argument('--file_filter', default="*label*.npy",
                        dest="file_filter", type=str, help='string used for filtering the data: [*label*.npy, *.mat, *.csv]')
    parser.add_argument('REMAIN', nargs='*')
    args_parser = parser.parse_args()

    utilObj = AddHR2Labels(args_parser.datadir, args_parser.fps, args_parser.file_filter)
    utilObj.compute_and_add_hr()

# fps: 30
# SCAMPS: /home/jitesh/data/SCAMPS/SCAMPS_Raw_160_72x72; --file_filter: *label*.npy
# SCAMPS Raw data: /mnt/sda/data/raw/SCAMPS/scamps_videos
# SCAMPS Raw data: /mnt/sda/data/raw/SCAMPS/scamps_waveforms_csv; --file_filter: *.csv
# UBFC-rPPG: /home/jitesh/data/UBFC-rPPG/UBFC-rPPG_Raw_160_72x72
# iBVP: /home/jitesh/data/iBVP_Dataset/iBVP_RGBT_160_72x72
# PURE: /home/jitesh/data/PURE_Dataset/PURE_Raw_160_72x72
# PURE: /home/jitesh/data/PURE_Dataset/PURE_Raw_300_36x36

# fps: 25
# BP4D: not needed as it comes with HR vector