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


bp_outlier_list = ["F001T02", "F001T03", "F001T04", "F001T05", "F001T06", "F001T07", "F001T08", "F001T09", "F001T10", "F003T09", "F006T08", "F009T01", "F009T03", "F009T04", "F009T05", "F009T06", "F009T07", "F009T08", "F009T09", "F011T01", "F011T06", "F011T07", "F011T09", "F012T10", "F013T10", "F014T01", "F014T02", "F014T03", "F014T04", "F014T05", "F014T06", "F014T07", "F014T08", "F014T09", "F014T10", "F016T02", "F016T03", "F016T05", "F016T08", "F016T10", "F017T03", "F017T04", "F017T05", "F017T06", "F017T07", "F017T09", "F017T10", "F018T01", "F018T02", "F018T03", "F018T04", "F018T05", "F018T06", "F018T07", "F019T03", "F023T01", "F023T02", "F023T04", "F023T06", "F023T07", "F023T08", "F023T09", "F024T01", "F024T02", "F024T03", "F024T04", "F024T05", "F024T06", "F024T07", "F024T08", "F024T10", "F027T04", "F027T05", "F027T06", "F027T07", "F027T09", "F027T10", "F028T01", "F028T02", "F028T06", "F028T08", "F028T09", "F028T10", "F029T01", "F029T03", "F029T05", "F029T06", "F029T07", "F029T08", "F030T01", "F030T04", "F030T06", "F030T07", "F030T10", "F032T08", "F032T10", "F033T01", "F033T02", "F033T03", "F033T04", "F033T07", "F033T08", "F033T09", "F036T03", "F037T01", "F037T02", "F037T03", "F037T04", "F037T05", "F037T06", "F037T07", "F037T10", "F039T01", "F039T02", "F039T03", "F039T04", "F039T05", "F039T06", "F039T07", "F039T08", "F039T10", "F040T09", "F041T09", "F041T10", "F042T02", "F042T09", "F043T01", "F043T04", "F043T06", "F043T07", "F044T03", "F044T04", "F044T05", "F044T07", "F044T09", "F044T10", "F045T06", "F045T10", "F046T02", "F047T01", "F047T02", "F047T03", "F047T04", "F047T05", "F047T06", "F047T07", "F047T08", "F047T09", "F047T10", "F050T03", "F051T02", "F053T01", "F053T02", "F053T03", "F053T05", "F053T06", "F053T07", "F053T10", "F056T02", "F056T03", "F056T04", "F056T05", "F056T06", "F056T07", "F056T09", "F058T01", "F058T02", "F058T03", "F058T07", "F058T10", "F059T02", "F059T03", "F059T04", "F059T05", "F059T07", "F059T08", "F059T09", "F059T10", "F060T05", "F060T06", "F060T07", "F060T08", "F060T09", "F060T10", "F062T07", "F065T01", "F065T02", "F065T03", "F065T08", "F065T10", "F066T03", "F066T06", "F066T08", "F067T01", "F067T02", "F067T03", "F067T04", "F067T06", "F067T09", "F067T10", "F068T03", "F068T04", "F068T05", "F068T06", "F068T07", "F071T01", "F071T02", "F071T07", "F074T07", "F074T08", "F074T09", "F074T10", "F077T03", "F078T08", "F082T02", "F082T03", "F082T04", "F082T05", "F082T07", "F082T08", "F082T09", "F082T10", "M004T10", "M006T01", "M006T02", "M006T03", "M006T04", "M006T05", "M006T06", "M006T07", "M006T08", "M006T09", "M006T10", "M007T01", "M007T02", "M007T03", "M007T04", "M007T05", "M007T06", "M009T07", "M009T08", "M009T09", "M010T02", "M010T03", "M010T04", "M010T05", "M010T06", "M010T07", "M014T05", "M016T01", "M016T04", "M016T05", "M016T06", "M016T07", "M017T01", "M017T02", "M017T03", "M017T04", "M017T05", "M017T07", "M017T10", "M018T01", "M019T02", "M019T03", "M019T04", "M020T05", "M020T06", "M020T07", "M020T09", "M020T10", "M021T02", "M021T05", "M021T10", "M022T08", "M022T10", "M024T01", "M025T01", "M025T10", "M032T03", "M032T04", "M032T05", "M032T06", "M032T07", "M032T08", "M033T01", "M033T02", "M033T08", "M034T01", "M034T09", "M035T03", "M035T06", "M036T01", "M036T02", "M036T03", "M036T04", "M036T05", "M036T10", "M037T10", "M038T01", "M038T02", "M038T03", "M038T04", "M038T05", "M038T06", "M038T07", "M038T10", "M039T02", "M039T03", "M039T04", "M039T08", "M043T04", "M044T01", "M044T03", "M044T07", "M046T10", "M047T01", "M047T02", "M047T03", "M047T04", "M047T05", "M047T06", "M047T07", "M047T08", "M047T10", "M048T10", "M049T01", "M049T02", "M049T03", "M049T06", "M049T10", "M050T02", "M051T02", "M051T06", "M051T07", "M052T01", "M052T10", "M053T07", "M053T08", "M055T07", "M055T08", "M055T10", "M056T01", "M056T03", "M056T08"]

bp_outlier_list = ["F001T02","F001T03","F001T04","F001T05","F001T06","F001T07","F001T08","F001T09","F001T10","F003T09","F004T07","F006T08","F006T09","F009T01","F009T03","F009T04","F009T05","F009T06","F009T07","F009T08","F009T09","F011T01","F011T06","F011T07","F011T09","F012T10","F013T09","F013T10","F014T01","F014T02","F014T03","F014T04","F014T05","F014T06","F014T07","F014T08","F014T09","F014T10","F016T02","F016T03","F016T05","F016T08","F016T10","F017T03","F017T04","F017T05","F017T06","F017T07","F017T09","F017T10","F018T01","F018T02","F018T03","F018T04","F018T05","F018T06","F018T07","F019T03","F020T09","F023T01","F023T02","F023T04","F023T06","F023T07","F023T08","F023T09","F024T01","F024T02","F024T03","F024T04","F024T05","F024T06","F024T07","F024T08","F024T10","F025T08","F025T09","F027T04","F027T05","F027T06","F027T07","F027T09","F027T10","F028T01","F028T02","F028T04","F028T06","F028T08","F028T09","F028T10","F029T01","F029T03","F029T05","F029T06","F029T07","F029T08","F030T01","F030T04","F030T06","F030T07","F030T10","F032T04","F032T08","F032T09","F032T10","F033T01","F033T02","F033T03","F033T04","F033T07","F033T08","F033T09","F036T03","F037T01","F037T02","F037T03","F037T04","F037T05","F037T06","F037T07","F037T10","F038T03","F039T01","F039T02","F039T03","F039T04","F039T05","F039T06","F039T07","F039T08","F039T10","F040T09","F041T09","F041T10","F042T02","F042T09","F043T01","F043T04","F043T06","F043T07","F044T03","F044T04","F044T05","F044T07","F044T09","F044T10","F045T06","F045T10","F046T02","F046T03","F047T01","F047T02","F047T03","F047T04","F047T05","F047T06","F047T07","F047T08","F047T09","F047T10","F049T09","F050T03","F051T02","F051T09","F053T01","F053T02","F053T03","F053T05","F053T06","F053T07","F053T09","F053T10","F056T02","F056T03","F056T04","F056T05","F056T06","F056T07","F056T09","F058T01","F058T02","F058T03","F058T07","F058T10","F059T02","F059T03","F059T04","F059T05","F059T07","F059T08","F059T09","F059T10","F060T05","F060T06","F060T07","F060T08","F060T09","F060T10","F061T05","F061T09","F062T07","F063T09","F065T01","F065T02","F065T03","F065T08","F065T10","F066T03","F066T06","F066T08","F067T01","F067T02","F067T03","F067T04","F067T06","F067T09","F067T10","F068T03","F068T04","F068T05","F068T06","F068T07","F068T09","F071T01","F071T02","F071T07","F072T09","F074T07","F074T08","F074T09","F074T10","F076T08","F077T03","F078T08","F082T02","F082T03","F082T04","F082T05","F082T07","F082T08","F082T09","F082T10","M003T01","M004T10","M006T01","M006T02","M006T03","M006T04","M006T05","M006T06","M006T07","M006T08","M006T09","M006T10","M007T01","M007T02","M007T03","M007T04","M007T05","M007T06","M009T07","M009T08","M009T09","M010T02","M010T03","M010T04","M010T05","M010T06","M010T07","M012T09","M014T02","M014T05","M016T01","M016T04","M016T05","M016T06","M016T07","M017T01","M017T02","M017T03","M017T04","M017T05","M017T07","M017T10","M018T01","M019T02","M019T03","M019T04","M020T05","M020T06","M020T07","M020T09","M020T10","M021T02","M021T05","M021T10","M022T08","M022T10","M024T01","M025T01","M025T10","M032T03","M032T04","M032T05","M032T06","M032T07","M032T08","M032T09","M033T01","M033T02","M033T08","M033T09","M034T01","M034T09","M035T03","M035T06","M036T01","M036T02","M036T03","M036T04","M036T05","M036T10","M037T02","M037T10","M038T01","M038T02","M038T03","M038T04","M038T05","M038T06","M038T07","M038T09","M038T10","M039T02","M039T03","M039T04","M039T08","M043T04","M044T01","M044T03","M044T07","M045T09","M046T10","M047T01","M047T02","M047T03","M047T04","M047T05","M047T06","M047T07","M047T08","M047T10","M048T10","M049T01","M049T02","M049T03","M049T06", "M049T10","M050T02","M051T02","M051T06","M051T07","M052T01","M052T10","M053T07","M053T08","M053T09","M055T07","M055T08","M055T10","M056T01","M056T03","M056T08"]

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
            fname = fn.name.split("_")[0]
            if not self.clean_data["input_files"].str.contains(str(fn)).any() or fname in bp_outlier_list:
                print("Removing:", str(fn))
                fn.unlink()

                # input_fn =  fn.parent.joinpath(fn.name.replace("label", "input"))
                # print("Removing:", str(input_fn))
                # input_fn.unlink(missing_ok=True)


                # also remove from data_files
                


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
    parser.add_argument('--datadir', default="data/BP4D/BP4D_RGBT_500_72x72",
                        dest="datadir", type=str, help='path of the data')
    parser.add_argument('--fps', default=25,
                        dest="fps", type=int, help='sampling rate')
    parser.add_argument('--file_filter', default="*.npy",
                        dest="file_filter", type=str, help='string used for filtering the data: [*label*.npy, *.mat, *.csv]')
    # parser.add_argument('--file_filter', default="*label*.npy",
    #                     dest="file_filter", type=str, help='string used for filtering the data: [*label*.npy, *.mat, *.csv]')
    # parser.add_argument('--file_filter', default="*input*.npy",
    #                     dest="file_filter", type=str, help='string used for filtering the data: [*label*.npy, *.mat, *.csv]')
    parser.add_argument('--ref_file', default="data/BP4D/BP4D_500_72_Clean.csv",
                        dest="ref_file", type=str, help='file with datalist used for removing the noisy data')


    parser.add_argument('REMAIN', nargs='*')
    args_parser = parser.parse_args()

    utilObj = AddHR2Labels(args_parser.datadir, args_parser.fps, args_parser.file_filter, args_parser.ref_file)
    # utilObj.add_bp_hr()
    # utilObj.add_bvp()
    utilObj.remove_noisy_input_files()
    # utilObj.plot_bvp()
    # utilObj.update_bvp_bp()
    # utilObj.update_bvp_bp_new()


# fps: 25
# BP4D: data/BP4D/BP4D_RGBT_500_72x72; 'data/BP4D/BP4D_500_72_Clean.csv'
# BP4D: data/BP4D/BP4D_RGBT_500_36x36; 'data/BP4D/BP4D_500_36_Clean.csv'
# BP4D: data/BP4D/BP4D_RGBT_500_9x9; 'data/BP4D/BP4D_500_9_Clean.csv'
