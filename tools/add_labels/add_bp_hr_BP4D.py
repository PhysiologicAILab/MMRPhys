from pathlib import Path
import numpy as np
import argparse
import scipy
from scipy.signal import butter
# import mat73
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk
from copy import deepcopy

import scipy.signal


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _calculate_fft_hr(ppg_signal, fs=25, low_pass=0.6, high_pass=3.3):
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
    return fft_hr


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


class AddHR2Labels(object):
    def __init__(self, datadir, fps, file_filter, ref_file) -> None:
        self.fps = fps
        self.file_filter = file_filter
        self.files = sorted(list(Path(datadir).glob(self.file_filter)))
        [self.b, self.a] = butter(2, [0.6 / self.fps * 2, 3.3 / self.fps * 2], btype='bandpass')
        self.ref_file = ref_file
        with open(self.ref_file) as f:
            self.clean_data = pd.read_csv(f)
        # print("total_files:", len(self.clean_data["input_files"]))


    def add_bp_hr(self):
        duration = 60
        
        for fn in self.files:
            if self.clean_data["input_files"].str.contains(str(fn)).any():
                data = np.load(str(fn))
                label_bvp = data[:, 0]
                label_rsp = data[:, 1]
                label_hr = data[:, 4]
                label_rr = data[:, 5]
                label_sysBP = data[:, 6]
                label_avgBP = data[:, 7]
                label_diaBP = data[:, 8]
                label_bp = data[:, 9]

                N = len(label_bvp)
                HR = int(np.round(np.median(label_hr)))
                SBP = int(np.round(np.median(label_sysBP)))
                DBP = int(np.round(np.median(label_diaBP)))

                new_label_bvp = nk.ppg_simulate(duration=duration, sampling_rate=self.fps, heart_rate=HR, drift=0, powerline_amplitude=0, burst_amplitude=0, motion_amplitude=0, random_state=1, random_state_distort=1)

                macc_list = []
                for i in np.arange(0, 5*self.fps, 5):
                    new_label_bvp_seg = new_label_bvp[i: i + N]
                    macc_list.append(_compute_macc(new_label_bvp_seg, label_bvp))

                macc_list = np.array(macc_list)
                # print("MACC:", np.max(macc_list))
                # print("MACC_List:", macc_list)
                opt_seg_index = np.argmax(macc_list)

                new_label_bvp_seg = new_label_bvp[opt_seg_index: opt_seg_index + N]
                mn = np.min(new_label_bvp_seg)
                mx = np.max(new_label_bvp_seg)
                new_label_bvp_seg = (new_label_bvp_seg - mn) / (mx - mn)
                new_label_bvp_seg = (new_label_bvp_seg * (SBP - DBP)) + DBP

                # plt.plot(label_bvp)
                # plt.plot(new_label_bvp_seg)
                # plt.savefig(Path("./zz_plots").joinpath(fn.stem + ".jpg"))
                # plt.close()

                new_label_bvp_seg = np.expand_dims(new_label_bvp_seg, 1)
                new_data = np.concatenate([data, new_label_bvp_seg], axis=1)
                print(fn.stem, ": HR, SBP, DBP, MACC:", [HR, SBP, DBP, np.round(np.max(macc_list), 2)], "; New Data Shape:", new_data.shape)

                # exit()
                np.save(str(fn), new_data)

            else:
                print("Removing:", str(fn))
                fn.unlink(missing_ok=True)

                input_fn =  fn.parent.joinpath(fn.name.replace("label", "input"))
                print("Removing:", str(input_fn))
                input_fn.unlink(missing_ok=True)


    def add_bvp(self):
        for fn in self.files:
            if self.clean_data["input_files"].str.contains(str(fn)).any():
                data = np.load(str(fn))
                label_bp = data[:, 10]
                avg_bvp = np.mean(label_bp)
                std_bvp = np.std(label_bp)
                norm_bvp = (label_bp - avg_bvp) / std_bvp

                norm_bvp = np.expand_dims(norm_bvp, 1)
                new_data = np.concatenate([data, norm_bvp], axis=1)
                np.save(str(fn), new_data)

    def plot_bvp(self):
        for fn in self.files:
            if self.clean_data["input_files"].str.contains(str(fn)).any():
                data = np.load(str(fn))
                label_bvp1 = data[:, 0]
                label_bvp2 = data[:, 11]

                plt.plot(label_bvp1)
                plt.plot(label_bvp2)
                plt.savefig(Path("./zz_plots").joinpath(fn.stem + ".jpg"))
                plt.close()


    def remove_noisy_input_files(self):
        for fn in self.files:
            if not self.clean_data["input_files"].str.contains(str(fn)).any():
                print("Removing:", str(fn))
                fn.unlink()

                # input_fn =  fn.parent.joinpath(fn.name.replace("label", "input"))
                # print("Removing:", str(input_fn))
                # input_fn.unlink(missing_ok=True)


    def update_bvp_bp(self):
        duration = 60

        for fn in self.files:
            if self.clean_data["input_files"].str.contains(str(fn)).any():
                data = np.load(str(fn))
                label_bvp1 = data[:, 0]
                label_bp = data[:, 10]
                label_bvp2 = data[:, 11]
                label_sysBP = data[:, 6]
                label_diaBP = data[:, 8]


                N = len(label_bvp1)
                HR = int(np.round(_calculate_fft_hr(scipy.signal.filtfilt(self.b, self.a, np.double(label_bvp1)), self.fps)))
                SBP = int(np.round(np.median(label_sysBP)))
                DBP = int(np.round(np.median(label_diaBP)))

                generated_bvp = nk.ppg_simulate(duration=duration, sampling_rate=self.fps, heart_rate=HR, drift=0, powerline_amplitude=0, burst_amplitude=0, motion_amplitude=0, random_state=1, random_state_distort=1)

                macc_list = []
                for i in np.arange(0, 2*self.fps, 5):
                    gen_bvp_seg = generated_bvp[i: i + N]
                    macc_list.append(_compute_macc(gen_bvp_seg, label_bvp1))

                macc_list = np.array(macc_list)
                # print("MACC:", np.max(macc_list))
                # print("MACC_List:", macc_list)
                opt_seg_index = np.argmax(macc_list)

                gen_bvp_seg = generated_bvp[opt_seg_index: opt_seg_index + N]
                gen_bvp_seg = scipy.signal.filtfilt(self.b, self.a, np.double(gen_bvp_seg))

                mn = np.min(gen_bvp_seg)
                mx = np.max(gen_bvp_seg)
                avg_bvp = np.mean(gen_bvp_seg)
                std_bvp = np.std(gen_bvp_seg)
                gen_bp_seg = (gen_bvp_seg - mn) / (mx - mn)
                gen_bp_seg = (gen_bp_seg * (SBP - DBP)) + DBP

                norm_bvp_seg = (gen_bvp_seg - avg_bvp) / std_bvp

                data[:, 10] = gen_bp_seg
                data[:, 11] = norm_bvp_seg

                # plt.plot(gen_bp_seg)
                # plt.plot(norm_bvp_seg)
                # plt.savefig(Path("./zz_plots").joinpath(fn.stem + ".jpg"))
                # plt.close()

                print(fn.stem, ": HR, SBP, DBP, MACC:", [HR, SBP, DBP, np.round(np.max(macc_list), 2)], ";Data Shape:", data.shape)

                # exit()
                np.save(str(fn), data)



    def update_bvp_bp_new(self):

        for fn in self.files:
            if self.clean_data["input_files"].str.contains(str(fn)).any():
                data = np.load(str(fn))
                label_bvp1 = data[:, 0]
                # label_bp = data[:, 10]
                label_sysBP = data[:, 6]
                label_diaBP = data[:, 8]

                SBP = int(np.round(np.median(label_sysBP)))
                DBP = int(np.round(np.median(label_diaBP)))

                mn = np.min(label_bvp1)
                mx = np.max(label_bvp1)
                bp_seg = (label_bvp1 - mn) / (mx - mn)
                bp_seg = (bp_seg * (SBP - DBP)) + DBP

                data[:, 10] = bp_seg

                print(fn.stem, ": SBP, DBP:", [SBP, DBP])

                # exit()
                np.save(str(fn), data)





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default="/home/jitesh/data/BP4D/BP4D_RGBT_500_36x36",
                        dest="datadir", type=str, help='path of the data')
    parser.add_argument('--fps', default=25,
                        dest="fps", type=int, help='sampling rate')
    parser.add_argument('--file_filter', default="*.npy",
                        dest="file_filter", type=str, help='string used for filtering the data: [*label*.npy, *.mat, *.csv]')
    # parser.add_argument('--file_filter', default="*label*.npy",
    #                     dest="file_filter", type=str, help='string used for filtering the data: [*label*.npy, *.mat, *.csv]')
    # parser.add_argument('--file_filter', default="*input*.npy",
    #                     dest="file_filter", type=str, help='string used for filtering the data: [*label*.npy, *.mat, *.csv]')
    parser.add_argument('--ref_file', default="/home/jitesh/data/BP4D/BP4D_500_36_Clean.csv",
                        dest="ref_file", type=str, help='file with datalist used for removing the noisy data')


    parser.add_argument('REMAIN', nargs='*')
    args_parser = parser.parse_args()

    utilObj = AddHR2Labels(args_parser.datadir, args_parser.fps, args_parser.file_filter)
    # utilObj.add_bp_hr()
    # utilObj.add_bvp()
    utilObj.remove_noisy_input_files()
    # utilObj.plot_bvp()
    # utilObj.update_bvp_bp()
    # utilObj.update_bvp_bp_new()


# fps: 25
# BP4D: /home/jitesh/data/BP4D/BP4D_RGBT_500_72x72; '/home/jitesh/data/BP4D/BP4D_500_72_Clean.csv'
# BP4D: /home/jitesh/data/BP4D/BP4D_RGBT_500_36x36; '/home/jitesh/data/BP4D/BP4D_500_36_Clean.csv'
# BP4D: /home/jitesh/data/BP4D/BP4D_RGBT_500_9x9; '/home/jitesh/data/BP4D/BP4D_500_9_Clean.csv'
