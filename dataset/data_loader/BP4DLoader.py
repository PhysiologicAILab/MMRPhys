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

        # # print(type(frames), frames.shape)
        # # print(type(phys), phys.shape)
        # # exit()
        # # Discard frames based on Signal Quality
        # del_idx = sq_vec <= 0.3
        # frames = np.delete(frames, del_idx, axis=0)
        # phys = np.delete(phys, del_idx, axis=0)
        # sq_vec = np.delete(sq_vec, del_idx, axis=0)

        # TODO: Add a code to resample and then filter appropriately all the signals. For metrics = rounding of values may be needed.
        # bvps = BaseLoader.resample_ppg(bvps, target_length)

        frames_clips, phys_clips = self.preprocess(frames, phys, config_preprocess, phys_axis=[0, 1, 2, 3], process_frames=process_frames)
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
