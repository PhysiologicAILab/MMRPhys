from pathlib import Path
import numpy as np
import argparse
import scipy
from scipy.signal import butter
# import mat73
import pandas as pd
import matplotlib.pyplot as plt


def resample_sig(input_signal, target_length):
    """Samples a PPG sequence into specific length."""
    return np.interp(
        np.linspace(
            1, input_signal.shape[0], target_length), np.linspace(
            1, input_signal.shape[0], input_signal.shape[0]), input_signal)


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


# RSP Metrics
# def _calculate_fft_rr(resp_signal, fs=30, low_pass=0.05, high_pass=1.0):
# def _calculate_fft_rr(resp_signal, fs=30, low_pass=0.13, high_pass=0.5):
def _calculate_fft_rr(resp_signal, fs=30, low_pass=0.1, high_pass=0.54):
    """Calculate respiration rate based on PPG using Fast Fourier transform (FFT)."""
    resp_signal = np.expand_dims(resp_signal, 0)
    N = _next_power_of_2(resp_signal.shape[1])
    f_resp, pxx_resp = scipy.signal.periodogram(resp_signal, fs=fs, nfft=N, detrend=False)
    fmask_resp = np.argwhere((f_resp >= low_pass) & (f_resp <= high_pass))
    mask_resp = np.take(f_resp, fmask_resp)
    mask_pxx = np.take(pxx_resp, fmask_resp)
    fft_rr = np.take(mask_resp, np.argmax(mask_pxx, 0))[0] * 60
    return fft_rr


class AddRR2Labels(object):
    def __init__(self, datadir, fps, file_filter) -> None:
        self.fps = fps
        self.file_filter = file_filter
        self.files = sorted(list(Path(datadir).glob(self.file_filter)))
        # [self.b, self.a] = butter(2, [0.05 / self.fps * 2, 1.0 / self.fps * 2], btype='bandpass')
        # [self.b, self.a] = butter(2, [0.13 / self.fps * 2, 0.5 / self.fps * 2], btype='bandpass')
        [self.b, self.a] = butter(2, [0.1 / self.fps * 2, 0.54 / self.fps * 2], btype='bandpass')

    def compute_and_add_rr(self):
        rr_vals = []
        rr_vals_seg = []
        err_vals = []
        seg_len = 300
        for fn in self.files:
            if "npy" in self.file_filter:
                data = np.load(fn)
                if len(data.shape) < 2:
                    data = scipy.signal.filtfilt(self.b, self.a, np.double(data))
                    rr = _calculate_fft_rr(data, fs=self.fps)
                    rr = int(np.round(rr))
                    temp_data = np.expand_dims(data, 1)
                    rr_vec = rr * np.ones_like(temp_data)
                    new_data = np.concatenate([temp_data, rr_vec], axis=1)
                    print("*", rr, new_data.shape)
                    # exit()
                    # np.save(str(fn), new_data)
                
                else:
                    rsp = data[:, 1]
                    rsp = scipy.signal.filtfilt(self.b, self.a, np.double(rsp))
                    rr = _calculate_fft_rr(rsp, fs=self.fps)
                    rr = int(np.round(rr))
                    rr_vec = rr * np.ones_like(rsp)
                    rr_vec = np.expand_dims(rr_vec, 1)
                    new_data = np.concatenate([data, rr_vec], axis=1)
                    print(".", rr, new_data.shape)
                    # exit()
                    # np.save(str(fn), new_data)
            
            elif "csv" in self.file_filter:
                with open(str(fn), "r") as f:
                    data = pd.read_csv(f).to_numpy()
                rsp = np.asarray(data[:, 3])
                rsp = scipy.signal.filtfilt(self.b, self.a, np.double(rsp))
                rr = _calculate_fft_rr(rsp, fs=self.fps)
                rr = int(np.round(rr))
                rr_vals.append(rr)

                rr_seg = []
                err_seg = []
                for seg in np.arange(0, 600, seg_len):
                    if 600 - seg >= seg_len:
                        rsp_sig_seg = rsp[seg: seg+seg_len]
                        # # joint_seg = np.concatenate([rsp_sig_seg, rsp_sig_seg[::-1], rsp_sig_seg, rsp_sig_seg[::-1]])
                        # # rr_s = int(round(_calculate_fft_rr(joint_seg, fs=self.fps)))
                        # rr_s = int(round(_calculate_fft_rr(rsp_sig_seg, fs=self.fps)))

                        resampled_rsp = resample_sig(rsp_sig_seg, len(rsp_sig_seg)//6)
                        rr_s = int(round(_calculate_fft_rr(resampled_rsp, fs=self.fps//6)))
                        
                        rr_seg.append(rr_s)
                        rr_vals_seg.append(rr_s)
                        err_vals.append(abs(rr - rr_s))
                        err_seg.append(abs(rr - rr_s))
                print("*", rr, rr_seg, err_seg)
                
                # rsp = np.expand_dims(rsp, axis=1)
                # rr_vec = rr * np.ones_like(rsp)

        rr_vals = np.array(rr_vals)
        rr_vals_seg = np.array(rr_vals_seg)

        plt.hist(rr_vals)
        plt.savefig("RR_Vals.jpg")
        plt.close()

        plt.hist(rr_vals_seg)
        plt.savefig("RR_Vals_Seg.jpg")
        plt.close()

        plt.hist(err_vals)
        plt.savefig("Errors_Seg_300by6.jpg")
        plt.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default="/mnt/sda/data/raw/SCAMPS/scamps_waveforms_csv",
                        dest="datadir", type=str, help='path of the data')
    parser.add_argument('--fps', default=30,
                        dest="fps", type=int, help='sampling rate')
    parser.add_argument('--file_filter', default="*.csv",
                        dest="file_filter", type=str, help='string used for filtering the data: [*label*.npy, *.mat, *.csv]')
    parser.add_argument('REMAIN', nargs='*')
    args_parser = parser.parse_args()

    utilObj = AddRR2Labels(args_parser.datadir, args_parser.fps, args_parser.file_filter)
    utilObj.compute_and_add_rr()

# fps: 30
# SCAMPS: /mnt/sdc1/rppg/SCAMPS/SCAMPS_Raw_160_72x72
# SCAMPS Raw data: /mnt/sda/data/raw/SCAMPS/scamps_videos
# SCAMPS Raw data: /mnt/sda/data/raw/SCAMPS/scamps_waveforms_csv
# UBFC-rPPG: not needed as no RSP signal present
# iBVP: not needed as no RSP signal present
# PURE: not needed as no RSP signal present

# fps: 25
# BP4D: not needed as it comes with HR vector