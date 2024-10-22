from pathlib import Path
import numpy as np
import argparse
import scipy
from scipy.signal import butter
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sg


def resample_sig(input_signal, target_length):
    """Samples a PPG sequence into specific length."""
    return np.interp(
        np.linspace(
            1, input_signal.shape[0], target_length), np.linspace(
            1, input_signal.shape[0], input_signal.shape[0]), input_signal)


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _calculate_fft_hr(ppg_signal, fs=15, low_pass=0.6, high_pass=3.3):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""

    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr


class DownSampleData(object):
    def __init__(self, datadir, plotdir, fps, add_metrics) -> None:
        video_filter = "*input*.npy"
        label_filter = "*label*.npy"
        self.video_files = sorted(list(Path(datadir).glob(video_filter)))
        self.label_files = sorted(list(Path(datadir).glob(label_filter)))
        self.fps = fps
        self.add_metrics = add_metrics

        self.plotdir = Path(plotdir)
        self.plotdir.mkdir(parents=True, exist_ok=True)

        [self.bvp_b, self.bvp_a] = butter(2, [0.6 / self.fps * 2, 3.3 / self.fps * 2], btype='bandpass')
        # [self.rsp_b, self.rsp_a] = butter(2, [0.05 / self.fps * 2, 0.7 / self.fps * 2], btype='bandpass')
        [self.rsp_b, self.rsp_a] = butter(2, [0.13 / self.fps * 2, 0.5 / self.fps * 2], btype='bandpass')
        [self.eda_b, self.eda_a] = butter(2, [0.02 / self.fps * 2, 5.0 / self.fps * 2], btype='bandpass')

        self.target_fps = fps // 2
        [self.bvp_b_target, self.bvp_a_target] = butter(2, [0.6 / self.target_fps * 2, 3.3 / self.target_fps * 2], btype='bandpass')
        # [self.rsp_b_target, self.rsp_a_target] = butter(2, [0.05 / self.target_fps * 2, 0.7 / self.target_fps * 2], btype='bandpass')
        [self.rsp_b_target, self.rsp_a_target] = butter(2, [0.13 / self.target_fps * 2, 0.5 / self.target_fps * 2], btype='bandpass')
        [self.eda_b_target, self.eda_a_target] = butter(2, [0.02 / self.target_fps * 2, 5.0 / self.target_fps * 2], btype='bandpass')

        if "ubfc" in datadir.lower():
            self.data_lists = [
                "/home/jitesh/data/UBFC-rPPG/DataFileLists/UBFC-rPPG_Raw_160_72x72_15FPS_0.0_0.7.csv", 
                "/home/jitesh/data/UBFC-rPPG/DataFileLists/UBFC-rPPG_Raw_160_72x72_15FPS_0.0_0.8.csv",
                "/home/jitesh/data/UBFC-rPPG/DataFileLists/UBFC-rPPG_Raw_160_72x72_15FPS_0.0_1.0.csv", 
                "/home/jitesh/data/UBFC-rPPG/DataFileLists/UBFC-rPPG_Raw_160_72x72_15FPS_0.7_1.0.csv", 
                "/home/jitesh/data/UBFC-rPPG/DataFileLists/UBFC-rPPG_Raw_160_72x72_15FPS_0.8_1.0.csv"
                ]

        elif "pure" in datadir.lower():
            self.data_lists = [
                "/home/jitesh/data/PURE_Dataset/DataFileLists/PURE_Raw_160_72x72_15FPS_0.0_0.7.csv",
                "/home/jitesh/data/PURE_Dataset/DataFileLists/PURE_Raw_160_72x72_15FPS_0.0_0.8.csv",
                "/home/jitesh/data/PURE_Dataset/DataFileLists/PURE_Raw_160_72x72_15FPS_0.0_1.0.csv",
                "/home/jitesh/data/PURE_Dataset/DataFileLists/PURE_Raw_160_72x72_15FPS_0.7_1.0.csv",
                "/home/jitesh/data/PURE_Dataset/DataFileLists/PURE_Raw_160_72x72_15FPS_0.8_1.0.csv",
                ]

        elif "ibvp" in datadir.lower():
            self.data_lists = []

        elif "scamps" in datadir.lower():
            self.data_lists = []

        elif "bp4d" in datadir.lower():
            self.data_lists = []


    def find_sessions_and_pair_data(self):
        self.sessions_info = {}
        for i in range(len(self.video_files)):
            video_fn = self.video_files[i]
            label_fn = self.label_files[i]
            sess = video_fn.stem.split("_")[0]
            if sess not in self.sessions_info:
                self.sessions_info[sess] = {}
                self.sessions_info[sess]["video"] = []
                self.sessions_info[sess]["label"] = []
            self.sessions_info[sess]["video"].append(str(video_fn))
            self.sessions_info[sess]["label"].append(str(label_fn))

        for sess in self.sessions_info:
            tmp_video_index_list = []
            print("sess:", sess)
            for i in range(len(self.sessions_info[sess]["video"])):
                num = int(Path(self.sessions_info[sess]["video"][i]).stem.split("_")[-1].replace("input", ""))
                tmp_video_index_list.append(num)
            ordered_list = np.argsort(np.array(tmp_video_index_list))
            # print("tmp_video_index_list:", tmp_video_index_list)
            # print("ordered_list:", ordered_list)
            self.sessions_info[sess]["video"] = np.array(self.sessions_info[sess]["video"])[ordered_list]
            self.sessions_info[sess]["label"] = np.array(self.sessions_info[sess]["label"])[ordered_list]


        for sess in self.sessions_info:
            if len(self.sessions_info[sess]["video"]) % 2 != 0:
                print("deleting:", self.sessions_info[sess]["video"][-1], self.sessions_info[sess]["label"][-1])

                # remove the file which can't be paired - whereever they are from self.data_lists
                for ls in self.data_lists:
                    df = pd.read_csv(ls)
                    loc = df[df.input_files == self.sessions_info[sess]["video"][-1]].index
                    # print(loc)
                    df = df.drop(loc)
                    df.to_csv(ls, header=["","input_files"], index=False)

                # delete the file which can't be paired
                Path(self.sessions_info[sess]["video"][-1]).unlink()
                self.sessions_info[sess]["video"] = np.delete(self.sessions_info[sess]["video"], -1)
                self.sessions_info[sess]["label"] = np.delete(self.sessions_info[sess]["label"], -1)


    def downsample_video(self):
        for sess in self.sessions_info:
            for i in np.arange(0, len(self.sessions_info[sess]["video"]), 2):
                vid1_fn = self.sessions_info[sess]["video"][i]
                vid2_fn = self.sessions_info[sess]["video"][i+1]
                labl1_fn = self.sessions_info[sess]["label"][i]
                labl2_fn = self.sessions_info[sess]["label"][i+1]

                print("paired videos:", vid1_fn, vid2_fn)
                print("paired labels:", labl1_fn, labl2_fn)

                vid1 = np.load(vid1_fn)
                vid2 = np.load(vid2_fn)
                labl1 = np.load(labl1_fn)
                labl2 = np.load(labl2_fn)

                assert vid1.shape[0] == labl1.shape[0]
                total_frames = vid1.shape[0]

                new_vid1 = np.concatenate([vid1[np.arange(0, total_frames, 2), ...], vid2[np.arange(0, total_frames, 2), ...]])
                new_vid2 = np.concatenate([vid1[np.arange(1, total_frames, 2), ...], vid2[np.arange(1, total_frames, 2), ...]])
                
                # new_labl1 = np.concatenate([labl1[np.arange(0, total_frames, 2), ...], labl2[np.arange(0, total_frames, 2), ...]])
                # new_labl2 = np.concatenate([labl1[np.arange(1, total_frames, 2), ...], labl2[np.arange(1, total_frames, 2), ...]])
                if len(labl1.shape) > 1:
                    bvp1 = labl1[:, 0]
                    bvp2 = labl2[:, 0]
                else:
                    bvp1 = labl1
                    bvp2 = labl2

                merged_bvp = np.concatenate([bvp1, bvp2])
                merged_bvp = sg.filtfilt(self.bvp_b, self.bvp_a, np.double(merged_bvp))
                # merged_bvp = resample_sig(merged_bvp, total_frames)
                merged_bvp = sg.resample(merged_bvp, total_frames)
                merged_bvp = sg.filtfilt(self.bvp_b_target, self.bvp_a_target, np.double(merged_bvp))
                
                if self.add_metrics:
                    hr = _calculate_fft_hr(merged_bvp, self.target_fps)
                    hr = int(np.round(hr))
                    # if len(labl1.shape) == 1:
                    merged_bvp = np.expand_dims(merged_bvp, 1)
                    hr_vec = hr * np.ones_like(merged_bvp)
                    new_labl = np.concatenate([merged_bvp, hr_vec], axis=1)
                    # else:
                    #     new_labl = np.concatenate([merged_bvp, hr_vec], axis=1)
                else:
                    new_labl = merged_bvp

                # print(new_vid1.shape, new_vid2.shape)
                print("Heart Rate:", hr, "; label_shape:", new_labl.shape)

                fig, ax = plt.subplots(3, 1)
                ax[0].plot(bvp1)
                ax[1].plot(bvp2)
                ax[2].plot(merged_bvp)
                plot_fn = self.plotdir.joinpath(sess + "_" + str(i) + ".jpg")
                plt.suptitle("HR: " + str(hr))
                plt.savefig(plot_fn)
                plt.close()

                # exit()

                # If more signals, --> concatenate them all before saving

                np.save(vid1_fn, new_vid1)
                np.save(vid2_fn, new_vid2)
                if self.add_metrics:
                    np.save(labl1_fn, new_labl)
                    np.save(labl2_fn, new_labl)
                else:
                    np.save(labl1_fn, merged_bvp)
                    np.save(labl2_fn, merged_bvp)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default="/home/jitesh/data/PURE_Dataset/PURE_Raw_160_72x72_15FPS",
                        dest="datadir", type=str, help='path of the data')
    parser.add_argument('--plotdir', default="/mnt/sdc1/rppg/Review15FPSData/PURE",
                        dest="plotdir", type=str, help='path of the plots')
    parser.add_argument('--fps', default=30, dest="fps",
                        type=int, help='original video frame rate')
    parser.add_argument('--add_metrics', default=1, dest="add_metrics",
                        type=int, help='0:No; [1]:Yes')

    parser.add_argument('REMAIN', nargs='*')
    args_parser = parser.parse_args()

    dsObj = DownSampleData(args_parser.datadir, args_parser.plotdir,
                           args_parser.fps, args_parser.add_metrics)
    dsObj.find_sessions_and_pair_data()
    dsObj.downsample_video()


# Do following before executing this code
# 1. Copy the prepared data at 30 FPS, and specify the location where it is pasted
# 2. Also copy the Datalist files for all the splits which are present. 
# 3. Rename both the folder and the datalist files. 
# 4. Update the paths within each datalist files
# 5. Add these files in the list of respective dataset files "self.data_lists" in the init function.


# BP4D = "/home/jitesh/data/BP4D/BP4D_RGBT_180_72x72"
# UBFC-rPPG = "/home/jitesh/data/UBFC-rPPG/UBFC-rPPG_Raw_160_72x72_15FPS"
# PURE = "/home/jitesh/data/PURE_Dataset/PURE_Raw_160_72x72_15FPS"
