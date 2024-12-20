"""The dataloader for SCAMPS dataset.

This data loader is specifically prepapred for MMRPhys work
"""
import glob
import json
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import mat73
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
import pandas as pd
import pickle


class SCAMPSLoaderBigSmall(BaseLoader):
    """The data loader for the SCAMPS Processed dataset."""

    def __init__(self, name, data_path, config_data, device=None):
        """Initializes an SCAMPS Processed dataloader.
            Args:
                data_path(string): path of a folder which stores raw video and ground truth biosignal in mat files.
                Each mat file contains a video sequence of resolution of 72x72 and various ground trugh signal.
                e.g., dXsub -> raw/diffnormalized data; d_ppg -> pulse signal, d_br -> resp signal
                -----------------
                     ProcessedData/
                     |   |-- P000001.mat/
                     |   |-- P000002.mat/
                     |   |-- P000003.mat/
                     ...
                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        self.config_data = config_data
        super().__init__(name, data_path, config_data, device)
        self.cached_path = config_data.CACHED_PATH + "_" + self.dataset_name
        self.file_list_path = config_data.FILE_LIST_PATH.split('.')[0] + "_" + self.dataset_name \
            + os.path.basename(config_data.FILE_LIST_PATH)  # append split name before .csv ext


    def get_raw_data(self, data_path):
        """Returns data directories under the path(For SCAMPS dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "*.mat")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        for data_dir in data_dirs:
            subject = os.path.split(data_dir)[-1]
            dirs.append({"index": subject, "path": data_dir})
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values"""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs
        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []
        for i in choose_range:
            data_dirs_new.append(data_dirs[i])
        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ Invoked by preprocess_dataset() for multi_process. """
        matfile_path = data_dirs[i]['path']
        saved_filename = data_dirs[i]['index']

        process_frames = config_preprocess.PREPROCESS_FRAMES

        # Read Frames
        if process_frames:
            frames = self.read_video(matfile_path)
            frames = (np.round(frames * 255)).astype(np.uint8)
        else:
            frames = np.empty(0)

        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            phys = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            phys = self.read_wave(matfile_path)

        frames_clips, phys_clips = self.preprocess(frames, phys, config_preprocess, phys_axis=[0, 1], process_frames=process_frames)

        input_name_list, label_name_list = self.save_multi_process(frames_clips, phys_clips, saved_filename, process_frames=process_frames)
        file_list_dict[i] = input_name_list

    def preprocess_dataset_backup(self, data_dirs, config_preprocess):
        """Preprocesses the raw data."""
        file_num = len(data_dirs)
        pbar = tqdm(list(range(file_num)))
        for i in pbar:
            matfile_path = data_dirs[i]['path']
            pbar.set_description("Processing %s" % matfile_path)

            # Read Frames
            frames = self.read_video(matfile_path)

            # Read Labels
            if config_preprocess.USE_PSUEDO_PPG_LABEL:
                bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
            else:
                bvps = self.read_wave(matfile_path)
                
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
            self.preprocessed_data_len += self.save(frames_clips, bvps_clips, data_dirs[i]['index'])

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3). """
        mat = mat73.loadmat(video_file)
        frames = mat['Xsub']  # load raw frames
        return np.asarray(frames)

    @staticmethod
    def read_wave(wave_file):
        """Reads a bvp and resp signal file."""
        mat = mat73.loadmat(wave_file)
        
        ppg = mat['d_ppg']  # load ppg signal
        ppg = np.asarray(ppg)
        ppg = np.expand_dims(ppg, axis=1)
        
        resp = mat['d_br']  # load resp signal
        resp = np.asarray(resp)
        resp = np.expand_dims(resp, axis=1)
        
        data = np.concatenate([ppg, resp], axis=1)
        return data


    def load(self):
        """ Loads the preprocessed data listed in the file list.

        Args:
            None
        Returns:
            None
        """
        file_list_path = self.file_list_path  # get list of files in
        file_list_df = pd.read_csv(file_list_path)
        inputs = file_list_df['input_files'].tolist()
        if not inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        inputs = sorted(inputs)  # sort input file name list
        labels = [input_file.replace("input", "label").replace('.pickle', '.npy') for input_file in inputs]
        self.inputs = inputs
        self.labels = labels
        self.preprocessed_data_len = len(inputs)


    def __getitem__(self, index):
        """Returns a clip of video(3,T,W,H) and it's corresponding signals(T)."""

        with open(self.inputs[index], 'rb') as handle:
            data = pickle.load(handle)

        # format data shapes
        if self.data_format == 'NDCHW':
            data[0] = np.float32(np.transpose(data[0], (0, 3, 1, 2)))
            data[1] = np.float32(np.transpose(data[1], (0, 3, 1, 2)))
        elif self.data_format == 'NCDHW':
            data[0] = np.float32(np.transpose(data[0], (3, 0, 1, 2)))
            data[1] = np.float32(np.transpose(data[1], (3, 0, 1, 2)))
        elif self.data_format == 'NDHWC':
            pass
        else:
            raise ValueError('Unsupported Data Format!')

        label = np.load(self.labels[index])
        label = np.float32(label)
        
        item_path = self.inputs[index]
        item_path_filename = item_path.split(os.sep)[-1]
        split_idx = item_path_filename.index('_')
        filename = item_path_filename[:split_idx]
        chunk_id = item_path_filename[split_idx + 6:].split('.')[0]
        return data, label, filename, chunk_id

