"""The dataloader for BP4D dataset.

This data loader is specifically prepapred for MMRPhys work
"""
import glob
import os
import re

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
import pandas as pd


class BP4DLoader(BaseLoader):
    """The data loader for the BP4D dataset."""

    def __init__(self, name, data_path, config_data, device=None):
        """Initializes an BP4D dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "BP4D" for below dataset structure:
                -----------------
                     BP4D/
                     |   |-- F001_T01/
                     |      |-- F001_T01_rgb/
                     |      |-- F001_T01_t/
                     |      |-- F001_T01_phys.csv
                     |   |-- F001_T02/
                     |      |-- F001_T02_rgb/
                     |      |-- F001_T02_t/
                     |      |-- F001_T02_phys.csv
                     |...
                     |   |-- piii_xx/
                     |      |-- piii_xx_rgb/
                     |      |-- piii_xx_t/
                     |      |-- piii_xx_phys.csv

                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        self.config_data = config_data

        self.HR_min = 35
        self.HR_max = 185
        self.RR_min = 4
        self.RR_max = 32
        self.rawBP_min = 20
        self.SBP_min = 90
        self.SBP_max = 180
        self.DBP_min = 60
        self.DBP_max = 120

        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For BP4D dataset)."""

        data_dirs = glob.glob(data_path + os.sep + "*_*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        for data_dir in data_dirs:
            subject_trail_val = os.path.split(data_dir)[-1].replace('_', '')
            index = subject_trail_val
            subject = subject_trail_val[0:4]
            dirs.append({"index": index, "path": data_dir, "subject": subject})
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits"""

        if self.config_data.FOLD.FOLD_NAME and self.config_data.FOLD.FOLD_PATH:
            data_dirs_new = self.split_raw_data_by_fold(data_dirs, self.config_data.FOLD.FOLD_PATH)
            return data_dirs_new

        # return the full directory
        if begin == 0 and end == 1:
            return data_dirs
        
        # get info about the dataset: subject list and num vids per subject
        data_info = dict()
        for data in data_dirs:
            subject = data['subject']
            data_dir = data['path']
            index = data['index']
            # creates a dictionary of data_dirs indexed by subject number
            if subject not in data_info:  # if subject not in the data info dictionary
                data_info[subject] = []  # make an emplty list for that subject
            # append a tuple of the filename, subject num, trial num, and chunk num
            data_info[subject].append({"index": index, "path": data_dir, "subject": subject})

        subj_list = list(data_info.keys())  # all subjects by number ID
        subj_list = sorted(subj_list)
        num_subjs = len(subj_list)  # number of unique subjects

        # get split of data set (depending on start / end)
        subj_range = list(range(0, num_subjs))
        if begin != 0 or end != 1:
            subj_range = list(range(int(begin * num_subjs), int(end * num_subjs)))

        # compile file list
        data_dirs_new = []
        for i in subj_range:
            subj_num = subj_list[i]
            subj_files = data_info[subj_num]
            data_dirs_new += subj_files  # add file information to file_list (tuple of fname, subj ID, trial num,
            # chunk num)

        return data_dirs_new


    def split_raw_data_by_fold(self, data_dirs, fold_path):

        fold_df = pd.read_csv(fold_path)
        fold_subjs = list(set(list(fold_df.subjects)))

        fold_data_dirs = []
        for d in data_dirs:
            idx = d['index']
            subj = idx[0:4]

            if subj in fold_subjs: # if trial has already been processed
                fold_data_dirs.append(d)

        return fold_data_dirs


    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ Invoked by preprocess_dataset for multi_process. """
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']
        process_frames = config_preprocess.PREPROCESS_FRAMES

        # Read Frames
        if 'None' in config_preprocess.DATA_AUG:
            # Utilize dataset-specific function to read video
            
            if config_preprocess.BP4D.DATA_MODE == "T":
                frames = self.read_video(
                    os.path.join(data_dirs[i]['path'], "{0}_t".format(filename), ""))
            
            elif config_preprocess.BP4D.DATA_MODE == "RGBT":
                rgb_frames = self.read_video(
                    os.path.join(data_dirs[i]['path'], "{0}_rgb".format(filename), ""))

                thermal_frames = self.read_thermal_video(
                    os.path.join(data_dirs[i]['path'], "{0}_t".format(filename), ""))
            else:
                frames = self.read_video(
                    os.path.join(data_dirs[i]['path'], "{0}_rgb".format(filename), ""))

        elif 'Motion' in config_preprocess.DATA_AUG:
            # Utilize general function to read video in .npy format
            frames = self.read_npy_video(
                glob.glob(os.path.join(data_dirs[i]['path'], filename, '*.npy')))
        else:
            raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')

        # Read Labels
        # "BVP", "RSP", "EDA", "ECG", "HR", "RR", "SysBP", "MeanBP", "DiaBP", 
        # e.g. after normalization: ['-0.45', '1.37', '-0.15', '0.0', '97.4', '18.55', '121.61', '115.85', '94.2']
        phys, sq_vec = self.read_wave(os.path.join(data_dirs[i]['path'], "{0}_phys.csv".format(filename)))

        if "RGBT" in config_preprocess.BP4D.DATA_MODE:
            rgb_length, rgb_height, rgb_width, rgb_ch = rgb_frames.shape
            thermal_length, t_height, t_width, t_ch = thermal_frames.shape
            target_length = min(rgb_length, thermal_length)
            rgb_frames = rgb_frames[:target_length, ...]
            thermal_frames = thermal_frames[:target_length, :, :, :]
            frames = np.concatenate([rgb_frames, thermal_frames], axis=-1)
        else:
            target_length = frames.shape[0]

        bvp = phys[:, 0]
        rsp = phys[:, 1]
        eda = phys[:, 2]
        ecg = phys[:, 3]
        hr = phys[:, 4]
        rr = phys[:, 5]
        sysBP = phys[:, 6]
        avgBP = phys[:, 7]
        diaBP = phys[:, 8]
        rawBP = phys[:, 9]

        # print("filename:", filename)
        # print("eda:", np.min(eda), np.max(eda), np.mean(eda))
        # print("hr:", np.min(hr), np.max(hr), np.mean(hr))
        # print("rr:", np.min(rr), np.max(rr), np.mean(rr))
        # print("sysBP:", np.min(sysBP), np.max(sysBP), np.mean(sysBP))
        # print("avgBP:", np.min(avgBP), np.max(avgBP), np.mean(avgBP))
        # print("diaBP:", np.min(diaBP), np.max(diaBP), np.mean(diaBP))
        # print("rawBP:", np.min(rawBP), np.max(rawBP), np.mean(rawBP))
        
        # # REMOVE BP OUTLIERS
        # sysBP[sysBP < 5] = 5
        # sysBP[sysBP > 250] = 250
        # avgBP[avgBP < 5] = 5
        # avgBP[avgBP > 250] = 250
        # diaBP[diaBP < 5] = 5
        # diaBP[diaBP > 200] = 200
        # rawBP[rawBP < 5] = 5
        # rawBP[rawBP > 200] = 200

        # # REMOVE EDA OUTLIERS
        # eda[eda < 1] = 1
        # eda[eda > 40] = 40

        # # REMOVE HR OUTLIERS
        # hr[hr < 30] = 30
        # hr[hr > 200] = 200

        # # REMOVE RR OUTLIERS
        # rr[rr < 3] = 3
        # rr[rr > 42] = 42

        bvp = np.expand_dims(bvp, 1)
        rsp = np.expand_dims(rsp, 1)
        eda = np.expand_dims(eda, 1)
        ecg = np.expand_dims(ecg, 1)
        hr = np.expand_dims(hr, 1)
        rr = np.expand_dims(rr, 1)
        sysBP = np.expand_dims(sysBP, 1)
        avgBP = np.expand_dims(avgBP, 1)
        diaBP = np.expand_dims(diaBP, 1)
        rawBP = np.expand_dims(rawBP, 1)

        phys = np.concatenate([bvp, rsp, eda, ecg, hr, rr, sysBP, avgBP, diaBP, rawBP], axis=1)
        # print(type(frames), frames.shape)
        # print(type(phys), phys.shape)
        # exit()

        # # Discard frames based on Signal Quality
        # del_idx = sq_vec <= 0.3
        # frames = np.delete(frames, del_idx, axis=0)
        # phys = np.delete(phys, del_idx, axis=0)
        # sq_vec = np.delete(sq_vec, del_idx, axis=0)

        frames_clips, phys_clips = self.preprocess(frames, phys, config_preprocess, phys_axis=[0, 1, 2, 3], process_frames=process_frames)
        num_clips = phys_clips.shape[0]
        del_idx = []

        for n_clip in range(num_clips):
            HR = int(round(np.median(phys_clips[n_clip, :, 4])))
            RR = int(round(np.median(phys_clips[n_clip, :, 5])))
            SBP = int(round(np.median(phys_clips[n_clip, :, 6])))
            DBP = int(round(np.median(phys_clips[n_clip, :, 7])))
            min_BP = int(round(np.min(phys_clips[n_clip, :, 9])))

            # print("[HR, RR, SBP, DBP, min_BP] = " + str([HR, RR, SBP, DBP, min_BP]))

            if SBP < self.SBP_min or SBP > self.SBP_max or DBP < self.DBP_min or DBP > self.DBP_max or min_BP < self.rawBP_min or RR < self.RR_min or RR > self.RR_max or HR < self.HR_min or HR > self.HR_max:
                del_idx.append(n_clip)
                # pass  # noisy
            else:
                pass  # good
        
        if len(del_idx) > 0:
            phys_clips = np.delete(phys_clips, del_idx, axis=0)

        if phys_clips.shape[0] > 0:
            input_name_list, label_name_list = self.save_multi_process(frames_clips, phys_clips, saved_filename)
            file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        frames = list()
        all_rgb = sorted(glob.glob(video_file + '*.npy'))
        for img_path in all_rgb:
            img = np.load(img_path)
            frames.append(img)
        return np.asarray(frames)

    @staticmethod
    def read_thermal_video(video_file):
        """Reads a video file, returns frames(T, H, W, 1) """
        frames = list()
        all_t = sorted(glob.glob(video_file + '*.npy'))
        for t_path in all_t:
            thermal_matrix = np.load(t_path)
            frames.append(thermal_matrix)
        frames = np.expand_dims(frames, axis=-1)
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            waves = pd.read_csv(f).to_numpy()
            # sq_vec = waves[:, -1]   #Signal quality
            # waves = waves[:, 0]
        return waves, None
