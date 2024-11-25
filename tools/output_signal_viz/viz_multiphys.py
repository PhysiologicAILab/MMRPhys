import argparse
import io
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from scipy.signal import butter, filtfilt
from scipy.sparse import spdiags
from copy import deepcopy

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=device)
        else:
            return super().find_class(module, name)


# path_dict_within_dataset = {
#     "test_datasets": {
#         "BP4D_RSP_Tx72_Base_Fold1": {
#             "root": "runs/exp/BP4D_RGBT_180_72x72/saved_test_outputs/",
#             "exp": {
#                 "MMRPhys": "BP4D_MMRPhys_RSP_Tx72_Base_Fold1_rsp_outputs.pickle",
#             }
#         }
#     }
# }


# path_dict_within_dataset = {
#     "test_datasets": {
#         "BP4D_300x36_Fold1": {
#             "root": "runs/exp/BP4D_RGBT_300_36x36/saved_test_outputs/",
#             "exp" : {
#                 "MMRPhys_BVP_RSP_RGBT_FuseM_SFSAM_Label":
#                 {
#                     "bvp": "BP4D_MMRPhys_BVP_RSP_RGBT_FuseMx300x36_SFSAM_Label_Fold1_bvp_outputs.pickle",
#                     "rsp": "BP4D_MMRPhys_BVP_RSP_RGBT_FuseMx300x36_SFSAM_Label_Fold1_rsp_outputs.pickle",
#                 }
#             },
#         }
#     }
# }


path_dict_within_dataset = {
    "test_datasets": {
        "BP4D_180x72_Fold1": {
            "root": "runs/exp/BP4D_RGBT_180_72x72/saved_test_outputs/",
            "exp": {
                "MMRPhysLEF_RGB_Base":
                {
                    "bvp": "BP4D_MMRPhysLEF_BVP_RSP_RGBx180x72_Base_Fold1_bvp_outputs.pickle",
                    "rsp": "BP4D_MMRPhysLEF_BVP_RSP_RGBx180x72_Base_Fold1_rsp_outputs.pickle",
                },
                "MMRPhysLEF_RGBT_Base":
                {
                    "bvp": "BP4D_MMRPhysLEF_BVP_RSP_RGBTx180x72_Base_Fold1_bvp_outputs.pickle",
                    "rsp": "BP4D_MMRPhysLEF_BVP_RSP_RGBTx180x72_Base_Fold1_rsp_outputs.pickle",
                },
            },
        },

        # "BP4D_500x72_Fold1": {
        #     "root": "runs/exp/BP4D_RGBT_500_72x72/saved_test_outputs/",
        #     "exp" : {
                # "MMRPhys_FuseL_SFSAM_Label":
                # {
                #     "bvp": "BP4D_MMRPhys_All_RGBT_FuseLx500x72_SFSAM_Label_Fold1_bvp_outputs.pickle",
                #     "rsp": "BP4D_MMRPhys_All_RGBT_FuseLx500x72_SFSAM_Label_Fold1_rsp_outputs.pickle",
                #     "bp": "BP4D_MMRPhys_All_RGBT_FuseLx500x72_SFSAM_Label_Fold1_bp_outputs.pickle",
                #     # "eda": "BP4D_MMRPhys_All_RGBT_FuseLx500x72_SFSAM_Label_Fold1_eda_outputs.pickle",
                # },
                # "MMRPhys_FuseL_Base":
                # {
                #     "bvp":"BP4D_MMRPhys_All_RGBT_FuseLx500x72_Base_Fold1_bvp_outputs.pickle",
                #     "rsp":"BP4D_MMRPhys_All_RGBT_FuseLx500x72_Base_Fold1_rsp_outputs.pickle",
                #     "bp":"BP4D_MMRPhys_All_RGBT_FuseLx500x72_Base_Fold1_bp_outputs.pickle",
                #     # "eda":"BP4D_MMRPhys_All_RGBT_FuseLx500x72_Base_Fold1_eda_outputs.pickle",
                # },
                # "MMRPhys_LEF_SFSAM_Thermal":
                # {
                #     "rsp": "BP4D_MMRPhysLEF_RSP_Tx72_SFSAM_Label_Fold1_rsp_outputs.pickle"
                # },
                # "MMRPhys_LLF_SFSAM":
                # {
                #     "bvp": "BP4D_MMRPhysLLF_BVP_RSP_RGBTx72_SFSAM_Label_Fold1_bvp_outputs.pickle",
                #     "rsp": "BP4D_MMRPhysLLF_BVP_RSP_RGBTx72_SFSAM_Label_Fold1_rsp_outputs.pickle",
                # },
                # "MMRPhys_LEF_SFSAM_RGB":
                # {
                #     "rsp": "BP4D_MMRPhysLEF_RSP_RGBx72_SFSAM_Label_Fold1_rsp_outputs.pickle"
                # },

                # "MMRPhysLNF_Base_Fold1":{
                #     "bvp": "BP4D_MMRPhysLNF_BVP_RSP_RGBTx72_Base_Fold1_bvp_outputs.pickle",
                #     "rsp": "BP4D_MMRPhysLNF_BVP_RSP_RGBTx72_Base_Fold1_rsp_outputs.pickle"
                # },
                # "MMRPhysLNF_SFSAM_Fold1": {
                #     "bvp": "BP4D_MMRPhysLNF_BVP_RSP_RGBTx72_SFSAM_Label_Fold1_bvp_outputs.pickle",
                #     "rsp": "BP4D_MMRPhysLNF_BVP_RSP_RGBTx72_SFSAM_Label_Fold1_rsp_outputs.pickle"
                # },
                # "MMRPhys_RGB":{
                #     "bvp": "BP4D_MMRPhysLEF_BVP_RSP_RGBx72_SFSAM_Label_Fold1_bvp_outputs.pickle",
                #     "rsp": "BP4D_MMRPhysLEF_BVP_RSP_RGBx72_SFSAM_Label_Fold1_rsp_outputs.pickle",
                # },
                # "MMRPhys_BVP-RGB_RSP-Thermal":{
                #     "bvp": "BP4D_MMRPhysLNF_BVP_RSP_RGBTx72_SFSAM_Label_Fold1_bvp_outputs.pickle",
                #     "rsp": "BP4D_MMRPhysLNF_BVP_RSP_RGBTx72_SFSAM_Label_Fold1_rsp_outputs.pickle"
                # },

                # "MMRPhysLNF_SFSAM": {
                #     "bvp": "BP4D_MMRPhysLNF_BP_RGBTx72_SFSAM_Label_Fold1_Epoch99_BP4D_bvp_outputs.pickle",
                #     "rsp": "BP4D_MMRPhysLNF_BP_RGBTx72_SFSAM_Label_Fold1_Epoch99_BP4D_rsp_outputs.pickle",
                #     "sbp": "BP4D_MMRPhysLNF_BP_RGBTx72_SFSAM_Label_Fold1_Epoch99_BP4D_SBP_outputs.pickle",
                #     "dbp": "BP4D_MMRPhysLNF_BP_RGBTx72_SFSAM_Label_Fold1_Epoch99_BP4D_DBP_outputs.pickle",
                # },
                # "MMRPhysLNF_Base" : {
                #     "bvp": "BP4D_MMRPhysLNF_BP_RGBTx72_Base_Fold1_Epoch99_BP4D_bvp_outputs.pickle",
                #     "rsp": "BP4D_MMRPhysLNF_BP_RGBTx72_Base_Fold1_Epoch99_BP4D_rsp_outputs.pickle",
                #     "sbp": "BP4D_MMRPhysLNF_BP_RGBTx72_Base_Fold1_Epoch99_BP4D_SBP_outputs.pickle",
                #     "dbp": "BP4D_MMRPhysLNF_BP_RGBTx72_Base_Fold1_Epoch99_BP4D_DBP_outputs.pickle",
                # },
            # },
        # },
        # "BP4D_500x72_Fold2": {
        #     "root": "runs/exp/BP4D_RGBT_500_72x72/saved_test_outputs/",
        #     "exp": {
        #         "MMRPhysLNF_Base_Fold2": {
        #             "bvp": "BP4D_MMRPhysLNF_BVP_RSP_RGBTx72_Base_Fold2_bvp_outputs.pickle",
        #             "rsp": "BP4D_MMRPhysLNF_BVP_RSP_RGBTx72_Base_Fold2_rsp_outputs.pickle"
        #         },
        #         "MMRPhysLNF_SFSAM_Fold2": {
        #             "bvp": "BP4D_MMRPhysLNF_BVP_RSP_RGBTx72_SFSAM_Label_Fold2_bvp_outputs.pickle",
        #             "rsp": "BP4D_MMRPhysLNF_BVP_RSP_RGBTx72_SFSAM_Label_Fold2_rsp_outputs.pickle"
        #         },
        #     }
        # },
        # "BP4D_500x72_Fold3": {
        #     "root": "runs/exp/BP4D_RGBT_500_72x72/saved_test_outputs/",
        #     "exp": {
        #         # "MMRPhysLNF_Base_Fold3": {
        #         #     "bvp": "BP4D_MMRPhysLNF_BVP_RSP_RGBTx72_Base_Fold3_bvp_outputs.pickle",
        #         #     "rsp": "BP4D_MMRPhysLNF_BVP_RSP_RGBTx72_Base_Fold3_rsp_outputs.pickle"
        #         # },
        #         # "MMRPhysLNF_SFSAM_Fold3": {
        #         #     "bvp": "BP4D_MMRPhysLNF_BVP_RSP_RGBTx72_SFSAM_Label_Fold3_bvp_outputs.pickle",
        #         #     "rsp": "BP4D_MMRPhysLNF_BVP_RSP_RGBTx72_SFSAM_Label_Fold3_rsp_outputs.pickle"
        #         # },
        #         "MMRPhysLNF_SFSAM_Fold3":{
        #             "bvp": "BP4D_MMRPhysLNF_BVP_BP_RSP_RGBTx72_SFSAM_Label_Fold3_bvp_outputs.pickle",
        #             "rsp": "BP4D_MMRPhysLNF_BVP_BP_RSP_RGBTx72_SFSAM_Label_Fold3_rsp_outputs.pickle",
        #             "SBP": "BP4D_MMRPhysLNF_BVP_BP_RSP_RGBTx72_SFSAM_Label_Fold3_SBP_outputs.pickle",
        #             "DBP": "BP4D_MMRPhysLNF_BVP_BP_RSP_RGBTx72_SFSAM_Label_Fold3_DBP_outputs.pickle",
        #         },
        #     }
        # },

        # "BP4D_500x9_Fold1_RGBT": {
        #     "root": "runs/exp/BP4D_RGBT_500_9x9/saved_test_outputs/",
        #     "exp": {
        #         "MMRPhys_FuseS_SFSAM_Label":
        #         {
        #             "bvp": "BP4D_MMRPhys_BVP_RSP_RGBT_FuseSx500x9_SFSAM_Label_Fold1_bvp_outputs.pickle",
        #             "rsp": "BP4D_MMRPhys_BVP_RSP_RGBT_FuseSx500x9_SFSAM_Label_Fold1_rsp_outputs.pickle",
        #         },
        #     }
        # }
    }
}

# HELPER FUNCTIONS

def _reform_data_from_dict(data, flatten=True):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)

    if flatten:
        sort_data = np.reshape(sort_data.cpu(), (-1))
    else:
        sort_data = np.array(sort_data.cpu())

    return sort_data


def _reform_BP_data_from_dict(data):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = float(sort_data[0].cpu())
    return sort_data


def _process_bvp_signal(signal, fs=25, diff_flag=False):
    # Detrend and filter
    use_bandpass = True
    if diff_flag:  # if the predictions and labels are 1st derivative of RSP signal.
        # signal = _detrend(np.cumsum(signal), 100)
        signal = np.cumsum(signal)
    else:
        # signal = _detrend(signal, 100)
        pass
    if use_bandpass:
        [b, a] = butter(2, [0.6 / fs * 2, 3.3 / fs * 2], btype='bandpass')
        signal = filtfilt(b, a, np.double(signal))
    return signal


def _process_rsp_signal(signal, fs=25, diff_flag=False):
    # Detrend and filter
    use_bandpass = True
    if diff_flag:  # if the predictions and labels are 1st derivative of RSP signal.
        # signal = _detrend(np.cumsum(signal), 100)
        signal = np.cumsum(signal)
    else:
        # signal = _detrend(signal, 100)
        pass
    if use_bandpass:
        # [b, a] = butter(2, [0.05 / fs * 2, 0.6 / fs * 2], btype='bandpass')
        [b, a] = butter(2, [0.13 / fs * 2, 0.5 / fs * 2], btype='bandpass')
        signal = filtfilt(b, a, np.double(signal))
    return signal



def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _calculate_fft_hr(ppg_signal, fs=60, low_pass=0.6, high_pass=3.3):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr


# RSP Metrics
# def _calculate_fft_rr(rsp_signal, fs=25, low_pass=0.05, high_pass=0.7):
# def _calculate_fft_rr(rsp_signal, fs=25, low_pass=0.13, high_pass=0.5):
def _calculate_fft_rr(rsp_signal, fs=25, low_pass=0.1, high_pass=0.54):
    """Calculate respiration rate using Fast Fourier transform (FFT)."""
    resp_signal = deepcopy(rsp_signal)
    sig_len = len(resp_signal)
    avg_resp = np.mean(resp_signal)
    std_resp = np.std(resp_signal)
    resp_signal = (resp_signal - avg_resp) / std_resp   # Standardize to remove DC level - which was due to min-max normalization

    last_zero_crossing = np.where(np.diff(np.sign(resp_signal)))[0][-1]
    resp_signal = resp_signal[: last_zero_crossing]
    inv_resp_signal = deepcopy(resp_signal)
    inv_resp_signal = -1 * inv_resp_signal[::-1]
    
    # Higher signal length is needed to reliably compute FFT for low frequencies
    resp_signal = np.concatenate([resp_signal, inv_resp_signal[1:], resp_signal[1:],
                                 inv_resp_signal[1:], resp_signal[1:], inv_resp_signal[1:]], axis=0)
    
    resp_signal = resp_signal[:4*sig_len]
    resp_signal = np.expand_dims(resp_signal, 0)
    N = _next_power_of_2(resp_signal.shape[1])
    f_resp, pxx_resp = scipy.signal.periodogram(resp_signal, fs=fs, nfft=N, detrend=False)
    fmask_resp = np.argwhere((f_resp >= low_pass) & (f_resp <= high_pass))
    mask_resp = np.take(f_resp, fmask_resp)
    mask_pxx = np.take(pxx_resp, fmask_resp)
    fft_rr = np.take(mask_resp, np.argmax(mask_pxx, 0))[0] * 60
    return fft_rr

# # def _calculate_fft_rr(rsp_signal, fs=25, low_pass=0.05, high_pass=0.7):
# def _calculate_fft_rr(rsp_signal, fs=25, low_pass=0.13, high_pass=0.5):
#     """Calculate respiration rate based on RSP using Fast Fourier transform (FFT)."""
#     rsp_signal = np.expand_dims(rsp_signal, 0)
#     N = _next_power_of_2(rsp_signal.shape[1])
#     f_rsp, pxx_rsp = scipy.signal.periodogram(rsp_signal, fs=fs, nfft=N, detrend=False)
#     fmask_rsp = np.argwhere((f_rsp >= low_pass) & (f_rsp <= high_pass))
#     mask_rsp = np.take(f_rsp, fmask_rsp)
#     mask_pxx = np.take(pxx_rsp, fmask_rsp)
#     fft_rr = np.take(mask_rsp, np.argmax(mask_pxx, 0))[0] * 60
#     return fft_rr


def _calculate_peak_hr(ppg_signal, fs=25):
    """Calculate heart rate based on PPG using peak detection."""
    ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return hr_peak


def _calculate_peak_rr(resp_signal, fs=25):
    """Calculate respiration rate based on PPG using peak detection."""
    resp_peaks, _ = scipy.signal.find_peaks(resp_signal)
    rr_peak = 60 / (np.mean(np.diff(resp_peaks)) / fs)
    return rr_peak


# Main functions

def compare_estimated_phys_within_dataset(tasks=0, save_plot=1):

    # if save_plot:
    plot_dir = Path.cwd().joinpath("plots").joinpath("BP4D_MultiPhys")
    plot_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = 180 #180 #500 #300  # size of chunk to visualize: -1 will plot the entire signal

    for test_dataset in path_dict_within_dataset["test_datasets"]:
        print("*"*50)
        print("Test Data:", test_dataset)
        print("*"*50)
        if tasks in [0, 1, 3]:
            bvp_dict = {}
            if tasks == 3:
                sbp_dict = {}
                dbp_dict = {}

        if tasks in [0, 2, 3]:
            rsp_dict = {}
        if tasks not in [0, 1, 2, 3]:
            print("Unsupported task option")
            exit()

        root_dir = Path(path_dict_within_dataset["test_datasets"][test_dataset]["root"])
        if not root_dir.exists():
            print("Data path does not exists:", str(root_dir))
            exit()

        # if save_plot:
        plot_test_dir = plot_dir.joinpath(test_dataset)
        plot_test_dir.mkdir(parents=True, exist_ok=True)
        plot_test_dir_detailed = plot_test_dir.joinpath("Detailed")
        plot_test_dir_detailed.mkdir(parents=True, exist_ok=True)

        for train_model in path_dict_within_dataset["test_datasets"][test_dataset]["exp"]:
            print("Model:", train_model)
            if tasks in [0, 1, 3]:
                bvp_fn = root_dir.joinpath(path_dict_within_dataset["test_datasets"][test_dataset]["exp"][train_model]["bvp"])
                bvp_dict[train_model] = CPU_Unpickler(open(bvp_fn, "rb")).load()
                if tasks == 3:
                    sbp_fn = root_dir.joinpath(path_dict_within_dataset["test_datasets"][test_dataset]["exp"][train_model]["sbp"])
                    sbp_dict[train_model] = CPU_Unpickler(open(sbp_fn, "rb")).load()
                    dbp_fn = root_dir.joinpath(path_dict_within_dataset["test_datasets"][test_dataset]["exp"][train_model]["dbp"])
                    dbp_dict[train_model] = CPU_Unpickler(open(dbp_fn, "rb")).load()

            if tasks in [0, 2, 3]:
                rsp_fn = root_dir.joinpath(path_dict_within_dataset["test_datasets"][test_dataset]["exp"][train_model]["rsp"])
                rsp_dict[train_model] = CPU_Unpickler(open(rsp_fn, "rb")).load()

        print("-"*50)

        if tasks in [0, 1, 3]:
            model_names = list(bvp_dict.keys())
        else:
            model_names = list(rsp_dict.keys())
        print("Model Names:", model_names)
        # print(bvp_dict[model_names[0]].keys())
        # exit()

        # List of all video trials
        if tasks in [0, 1, 3]:
            trial_list = list(bvp_dict[model_names[0]]['predictions'].keys())
        else:
            trial_list = list(rsp_dict[model_names[0]]['predictions'].keys())
        print('Num Trials', len(trial_list))

        if tasks in [0, 1, 3]:
            all_hr_labels = {}
            all_hr_preds = {}
        if tasks in [0, 2, 3]:
            all_rr_labels = {}
            all_rr_preds = {}
        if tasks == 3:
            all_sbp_labels = {}
            all_sbp_pred = {}
            all_dbp_labels = {}
            all_dbp_pred = {}


        for trial_ind in range(len(trial_list)):
            # print("."*25)

            # Read in meta-data from pickle file
            if tasks in [0, 1, 3]:               
                gt_bvp = np.array(_reform_data_from_dict(bvp_dict[model_names[0]]['predictions'][trial_list[trial_ind]]))
                total_samples = len(gt_bvp)
                fs = bvp_dict[model_names[0]]['fs'] # Video Frame Rate
                label_type = bvp_dict[model_names[0]]['label_type'] # BVP Signal Transformation: `DiffNormalized` or `Standardized`
                diff_flag = (label_type == 'DiffNormalized')
            else:
                # Read in meta-data from pickle file
                gt_rsp = np.array(_reform_data_from_dict(rsp_dict[model_names[0]]['predictions'][trial_list[0]]))
                total_samples = len(gt_rsp)
                fs = rsp_dict[model_names[0]]['fs'] # Video Frame Rate
                label_type = rsp_dict[model_names[0]]['label_type'] # RSP Signal Transformation: `DiffNormalized` or `Standardized`
                diff_flag = (label_type == 'DiffNormalized')

            total_chunks = total_samples // chunk_size
            # print('Chunk size', chunk_size)
            # print('Total chunks', total_chunks)

            if tasks in [0, 1, 3]:
                bvp_label = np.array(_reform_data_from_dict(bvp_dict[model_names[0]]['labels'][trial_list[trial_ind]]))
                hr_pred = {}

            if tasks in [0, 2, 3]:
                rsp_label = np.array(_reform_data_from_dict(rsp_dict[model_names[0]]['labels'][trial_list[trial_ind]]))
                rr_pred = {}

            trial_dict = {}
            
            for c_ind in range(total_chunks):
                # print("*"*25)
                # try:
                if save_plot:
                    if tasks in [0, 3]:
                        fig, ax = plt.subplots(2, 1, figsize=(16, 12))
                    else:
                        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
                # fig.tight_layout()

                start = (c_ind)*chunk_size
                stop = (c_ind+1)*chunk_size
                samples = stop - start
                
                # print("start, stop, samples: ", [start, stop, len(bvp_label[start: stop])])
                if tasks in [0, 1, 3]:
                    bvp_seg = _process_bvp_signal(bvp_label[start: stop], fs, diff_flag=diff_flag)
                    hr_label = _calculate_fft_hr(bvp_seg, fs=fs)
                    hr_label = int(np.round(hr_label))
                    # print("hr_label:", hr_label)

                if tasks in [0, 2, 3]:
                    rsp_seg = _process_rsp_signal(rsp_label[start: stop], fs, diff_flag=diff_flag)
                    
                    try:
                        rr_label = _calculate_peak_rr(rsp_seg, fs=fs)
                        rr_label = int(np.round(rr_label))
                    except:
                        rr_label = _calculate_fft_rr(rsp_seg, fs=fs)
                        rr_label = int(np.round(rr_label))
                
                    # print("rr_label:", rr_label)

                x_time = np.linspace(0, samples/fs, num=samples)

                for m_ind in range(len(model_names)):

                    if model_names[m_ind] not in trial_dict:
                        trial_dict[model_names[m_ind]] = {}

                        # Reform label and prediction vectors from multiple trial chunks
                        if tasks in [0, 1, 3]:
                            trial_dict[model_names[m_ind]]["bvp_pred"] = np.array(_reform_data_from_dict(bvp_dict[model_names[m_ind]]['predictions'][trial_list[trial_ind]]))                        

                        if tasks in [0, 2, 3]:
                            trial_dict[model_names[m_ind]]["rsp_pred"] = np.array(_reform_data_from_dict(rsp_dict[model_names[m_ind]]['predictions'][trial_list[trial_ind]]))

                        if tasks in [0, 1, 3]:
                            if model_names[m_ind] not in all_hr_labels:
                                all_hr_labels[model_names[m_ind]] = []
                                all_hr_preds[model_names[m_ind]] = []
                        if tasks in [0, 2, 3]:
                            if model_names[m_ind] not in all_rr_labels:
                                all_rr_labels[model_names[m_ind]] = []
                                all_rr_preds[model_names[m_ind]] = []

                        if tasks == 3:
                            if model_names[m_ind] not in all_sbp_labels:
                                all_sbp_labels[model_names[m_ind]] = []
                                all_sbp_pred[model_names[m_ind]] = []
                                all_dbp_labels[model_names[m_ind]] = []
                                all_dbp_pred[model_names[m_ind]] = []

                    # Process label and prediction signals
                    if tasks in [0, 1, 3]:
                        bvp_pred_seg = _process_bvp_signal(trial_dict[model_names[m_ind]]["bvp_pred"][start: stop], fs, diff_flag=diff_flag)
                    if tasks in [0, 2, 3]:
                        rsp_pred_seg = _process_rsp_signal(trial_dict[model_names[m_ind]]["rsp_pred"][start: stop], fs, diff_flag=diff_flag)

                    if tasks in [0, 1, 3]:
                        hr_pred[model_names[m_ind]] = _calculate_fft_hr(bvp_pred_seg, fs=fs)
                        hr_pred[model_names[m_ind]] = int(np.round(hr_pred[model_names[m_ind]]))
                        # print("m_ind:", m_ind, "; hr_pred: ", hr_pred[model_names[m_ind]])

                    if tasks in [0, 2, 3]:
                        try:
                            rr_pred[model_names[m_ind]] = _calculate_peak_rr(rsp_pred_seg, fs=fs)
                            rr_pred[model_names[m_ind]] = int(np.round(rr_pred[model_names[m_ind]]))
                        except:
                            rr_pred[model_names[m_ind]] = _calculate_fft_rr(rsp_pred_seg, fs=fs)
                            rr_pred[model_names[m_ind]] = int(np.round(rr_pred[model_names[m_ind]]))

                        # print("m_ind:", m_ind, "; rr_pred: ", rr_pred[model_names[m_ind]])

                    if tasks in [0, 1, 3]:
                        all_hr_labels[model_names[m_ind]].append(hr_label)
                        all_hr_preds[model_names[m_ind]].append(hr_pred[model_names[m_ind]])

                    if tasks in [0, 2, 3]:
                        all_rr_labels[model_names[m_ind]].append(rr_label)
                        all_rr_preds[model_names[m_ind]].append(rr_pred[model_names[m_ind]])

                    if tasks == 3:
                        all_sbp_labels[model_names[m_ind]].append(_reform_BP_data_from_dict(sbp_dict[model_names[m_ind]]['labels'][trial_list[trial_ind]]))
                        all_sbp_pred[model_names[m_ind]].append(_reform_BP_data_from_dict(sbp_dict[model_names[m_ind]]['predictions'][trial_list[trial_ind]]))
                        all_dbp_labels[model_names[m_ind]].append(_reform_BP_data_from_dict(dbp_dict[model_names[m_ind]]['labels'][trial_list[trial_ind]]))
                        all_dbp_pred[model_names[m_ind]].append(_reform_BP_data_from_dict(dbp_dict[model_names[m_ind]]['predictions'][trial_list[trial_ind]]))

                    if save_plot:
                        if tasks in [0, 3]:
                            ax[0].plot(x_time, bvp_pred_seg, label=model_names[m_ind] + "; HR = " + str(hr_pred[model_names[m_ind]]))
                            ax[1].plot(x_time, rsp_pred_seg, label=model_names[m_ind] + "; RR = " + str(rr_pred[model_names[m_ind]]))
                        else:
                            if tasks in [1, 3]:
                                ax.plot(x_time, bvp_pred_seg, label=model_names[m_ind] + "; HR = " + str(hr_pred[model_names[m_ind]]))
                            else:
                                ax.plot(x_time, rsp_pred_seg, label=model_names[m_ind] + "; RR = " + str(rr_pred[model_names[m_ind]]))


                if save_plot:
                    if tasks in [0, 3]:
                        ax[0].plot(x_time, bvp_label[start: stop], label="GT ; HR = " + str(hr_label), color='black')
                        ax[0].legend(loc="upper right")
                        ax[1].plot(x_time, rsp_label[start: stop], label="GT ; RR = " + str(rr_label), color='black')
                        ax[1].legend(loc="upper right")
                    else:
                        if tasks in [1, 3]:
                            ax.plot(x_time, bvp_label[start: stop], label="GT ; HR = " + str(hr_label), color='black')
                            ax.legend(loc="upper right")
                        else:
                            ax.plot(x_time, rsp_label[start: stop], label="GT ; RR = " + str(rr_label), color='black')
                            ax.legend(loc="upper right")


                    # ax[2].plot(x_time, bp_label[start: stop], label="GT", color='black')
                    # ax[2].legend(loc="upper right")
                    # ax[3].plot(x_time, eda_label[start: stop], label="GT", color='black')
                    # ax[3].legend(loc="upper right")
                
                    plt.suptitle("Dataset: " + test_dataset + '; Trial: ' + trial_list[trial_ind] + '; Chunk: ' + str(c_ind), fontsize=14)

                    # plt.show()
                    save_fn = plot_test_dir_detailed.joinpath(str(trial_list[trial_ind]) + "_" + str(c_ind) + ".jpg")

                    plt.xlabel('Time (s)')
                    plt.savefig(save_fn)
                    plt.close()

                    # except Exception as e:
                    #     print("Encoutered error:", e)
                    #     exit()


        all_test_data = {}
        for m_ind in range(len(model_names)):
            if model_names[m_ind] not in all_test_data:
                all_test_data[model_names[m_ind]] = {}

            if tasks in [0, 1, 3]:
                hr_labels_array = np.array(all_hr_labels[model_names[m_ind]])
                hr_pred_array = np.array(all_hr_preds[model_names[m_ind]])
                hr_labels_clean_array = deepcopy(hr_labels_array)
                hr_pred_clean_array = deepcopy(hr_pred_array)

                for ind in range(len(hr_labels_array)):
                    if hr_labels_array[ind]==0 or np.isnan(hr_labels_array[ind]) or hr_pred_array[ind]==0 or np.isnan(hr_pred_array[ind]):
                        hr_labels_clean_array = np.delete(hr_labels_array, ind)
                        hr_pred_clean_array = np.delete(hr_pred_array, ind)
                all_test_data[model_names[m_ind]]["hr_labels"] = hr_labels_clean_array
                all_test_data[model_names[m_ind]]["hr_pred"] = hr_pred_clean_array

            if tasks in [0, 2, 3]:
                rr_labels_array = np.array(all_rr_labels[model_names[m_ind]])
                rr_pred_array = np.array(all_rr_preds[model_names[m_ind]])
                rr_labels_clean_array = deepcopy(rr_labels_array)
                rr_pred_clean_array = deepcopy(rr_pred_array)

                for ind in range(len(rr_labels_array)):
                    if rr_labels_array[ind]==0 or np.isnan(rr_labels_array[ind]) or rr_pred_array[ind]==0 or np.isnan(rr_pred_array[ind]):
                        rr_labels_clean_array = np.delete(rr_labels_array, ind)
                        rr_pred_clean_array = np.delete(rr_pred_array, ind)
                all_test_data[model_names[m_ind]]["rr_labels"] = rr_labels_clean_array
                all_test_data[model_names[m_ind]]["rr_pred"] = rr_pred_clean_array

            if tasks == 3:
                sbp_label = np.array(all_sbp_labels[model_names[m_ind]])
                sbp_pred = np.array(all_sbp_pred[model_names[m_ind]])
                dbp_label = np.array(all_dbp_labels[model_names[m_ind]])
                dbp_pred = np.array(all_dbp_pred[model_names[m_ind]])

                # Handle out of range predictions and labels
                sbp_label[sbp_label < 90] = 90
                sbp_label[sbp_label > 180] = 180
                sbp_pred[sbp_pred < 90] = 90
                sbp_pred[sbp_pred > 180] = 180

                dbp_label[dbp_label < 60] = 60
                dbp_label[dbp_label > 120] = 120
                dbp_pred[dbp_pred < 60] = 60
                dbp_pred[dbp_pred > 120] = 120

                all_test_data[model_names[m_ind]]["sbp_labels"] = sbp_label
                all_test_data[model_names[m_ind]]["sbp_pred"] = sbp_pred
                all_test_data[model_names[m_ind]]["dbp_labels"] = dbp_label
                all_test_data[model_names[m_ind]]["dbp_pred"] = dbp_pred


        data_fn = plot_test_dir.joinpath(plot_test_dir.name + "_eval_data.npy")
        np.save(data_fn, all_test_data)

        print("-*-"*50)
        print("Performance of different models for test dataset:", test_dataset)
        print("-*-"*50)
        for m_ind in range(len(model_names)):
            print("-"*50)
            print("Model:", model_names[m_ind])
            print("-"*50)
            
            if tasks in [0, 1, 3]:
                gt_hr = np.array(all_hr_labels[model_names[m_ind]])
                pred_hr = np.array(all_hr_preds[model_names[m_ind]])
                num_test_samples = len(gt_hr)
            
            if tasks in [0, 2, 3]:
                gt_rr = np.array(all_rr_labels[model_names[m_ind]])
                pred_rr = np.array(all_rr_preds[model_names[m_ind]])
                num_test_samples = len(gt_rr)

            if tasks == 3:
                sbp_label = np.array(all_sbp_labels[model_names[m_ind]])
                sbp_pred = np.array(all_sbp_pred[model_names[m_ind]])
                dbp_label = np.array(all_dbp_labels[model_names[m_ind]])
                dbp_pred = np.array(all_dbp_pred[model_names[m_ind]])
                num_test_samples = len(sbp_label)

            print("Total Samples: ", num_test_samples)
            print("."*50)
            
            if tasks in [0, 1, 3]:
                print("HR Metrics")
                print("."*50)

                MAE_HR = np.mean(np.abs(pred_hr - gt_hr))
                standard_error = np.std(np.abs(pred_hr - gt_hr)) / np.sqrt(num_test_samples)
                print("HR MAE: {0} +/- {1}".format(MAE_HR, standard_error))

                RMSE_HR = np.sqrt(np.mean(np.square(pred_hr - gt_hr)))
                standard_error = np.sqrt(np.std(np.square(pred_hr - gt_hr))) / np.sqrt(num_test_samples)
                print("HR RMSE: {0} +/- {1}".format(RMSE_HR, standard_error))

                MAPE_HR = np.mean(np.abs((pred_hr - gt_hr) / gt_hr)) * 100
                standard_error = np.std(np.abs((pred_hr - gt_hr) / gt_hr)) / np.sqrt(num_test_samples) * 100
                print("HR MAPE: {0} +/- {1}".format(MAPE_HR, standard_error))

                Pearson_HR = np.corrcoef(pred_hr, gt_hr)
                corr_HR = Pearson_HR[0][1]
                standard_error = np.sqrt((1 - corr_HR**2) / (num_test_samples - 2))
                print("HR Pearson: {0} +/- {1}".format(corr_HR, standard_error))

                scatter_plot_hr_fn = plot_test_dir.joinpath(plot_test_dir.name + "_" + model_names[m_ind] + "_scatter_plot_HR.jpg")
                plt.scatter(hr_labels_clean_array, hr_pred_clean_array)
                plt.xlabel("Ground Truth HR")
                plt.ylabel("Estimated HR")
                plt.title("HR | MAE: " + str(np.round(MAE_HR, 2)) + "; RMSE: " + str(np.round(RMSE_HR, 2)) + "; MAPE: " + str(np.round(MAPE_HR, 2)) + "; Corr: " + str(np.round(corr_HR, 2)))
                plt.savefig(scatter_plot_hr_fn)
                plt.close()

            if tasks in [0, 2, 3]:
                print("."*50)
                print("RR Metrics")
                print("."*50)

                MAE_RR = np.mean(np.abs(pred_rr - gt_rr))
                standard_error = np.std(np.abs(pred_rr - gt_rr)) / np.sqrt(num_test_samples)
                print("RR MAE: {0} +/- {1}".format(MAE_RR, standard_error))

                RMSE_RR = np.sqrt(np.mean(np.square(pred_rr - gt_rr)))
                standard_error = np.sqrt(np.std(np.square(pred_rr - gt_rr))) / np.sqrt(num_test_samples)
                print("RR RMSE: {0} +/- {1}".format(RMSE_RR, standard_error))

                MAPE_RR = np.mean(np.abs((pred_rr - gt_rr) / gt_rr)) * 100
                standard_error = np.std(np.abs((pred_rr - gt_rr) / gt_rr)) / np.sqrt(num_test_samples) * 100
                print("RR MAPE: {0} +/- {1}".format(MAPE_RR, standard_error))

                Pearson_RR = np.corrcoef(pred_rr, gt_rr)
                corr_RR = Pearson_RR[0][1]
                standard_error = np.sqrt((1 - corr_RR**2) / (num_test_samples - 2))
                print("RR Pearson: {0} +/- {1}".format(corr_RR, standard_error))

                scatter_plot_rr_fn = plot_test_dir.joinpath(plot_test_dir.name + "_" + model_names[m_ind] + "_scatter_plot_RR.jpg")
                plt.scatter(rr_labels_clean_array, rr_pred_clean_array)
                plt.xlabel("Ground Truth RR")
                plt.ylabel("Estimated RR")
                plt.title("RR | MAE: " + str(np.round(MAE_RR, 2)) + "; RMSE: " + str(np.round(RMSE_RR, 2)) + "; MAPE: " + str(np.round(MAPE_RR, 2)) + "; Corr: " + str(np.round(corr_RR, 2)))
                plt.savefig(scatter_plot_rr_fn)
                plt.close()

            if tasks == 3:
                print("."*50)
                print("BP Metrics")
                print("."*50)

                # Handle out of range predictions and labels
                sbp_label[sbp_label < 90] = 90
                sbp_label[sbp_label > 180] = 180
                sbp_pred[sbp_pred < 90] = 90
                sbp_pred[sbp_pred > 180] = 180

                dbp_label[dbp_label < 60] = 60
                dbp_label[dbp_label > 120] = 120
                dbp_pred[dbp_pred < 60] = 60
                dbp_pred[dbp_pred > 120] = 120

                MAE_SBP = np.mean(np.abs(sbp_pred - sbp_label))
                standard_error = np.std(np.abs(sbp_pred - sbp_label)) / np.sqrt(num_test_samples)
                print("SBP MAE: {0} +/- {1}".format(MAE_SBP, standard_error))
                MAE_DBP = np.mean(np.abs(dbp_pred - dbp_label))
                standard_error = np.std(np.abs(dbp_pred - dbp_label)) / np.sqrt(num_test_samples)
                print("DBP MAE: {0} +/- {1}".format(MAE_DBP, standard_error))
                
                RMSE_SBP = np.sqrt(np.mean(np.square(sbp_pred - sbp_label)))
                standard_error = np.sqrt(np.std(np.square(sbp_pred - sbp_label))) / np.sqrt(num_test_samples)
                print("SBP RMSE: {0} +/- {1}".format(RMSE_SBP, standard_error))
                RMSE_DBP = np.sqrt(np.mean(np.square(dbp_pred - dbp_label)))
                standard_error = np.sqrt(np.std(np.square(dbp_pred - dbp_label))) / np.sqrt(num_test_samples)
                print("DBP RMSE: {0} +/- {1}".format(RMSE_DBP, standard_error))

                MAPE_SBP = np.mean(np.abs((sbp_pred - sbp_label) / sbp_label)) * 100
                standard_error = np.std(np.abs((sbp_pred - sbp_label) / sbp_label)) / np.sqrt(num_test_samples) * 100
                print("SBP MAPE: {0} +/- {1}".format(MAPE_SBP, standard_error))
                MAPE_DBP = np.mean(np.abs((dbp_pred - dbp_label) / dbp_label)) * 100
                standard_error = np.std(np.abs((dbp_pred - dbp_label) / dbp_label)) / np.sqrt(num_test_samples) * 100
                print("DBP MAPE: {0} +/- {1}".format(MAPE_DBP, standard_error))

                Pearson_SBP = np.corrcoef(sbp_pred, sbp_label)
                corr_SBP = Pearson_SBP[0][1]
                standard_error = np.sqrt((1 - corr_SBP**2) / (num_test_samples - 2))
                print("SBP Pearson: {0} +/- {1}".format(corr_SBP, standard_error))
                Pearson_DBP = np.corrcoef(dbp_pred, dbp_label)
                corr_DBP = Pearson_DBP[0][1]
                standard_error = np.sqrt((1 - corr_DBP**2) / (num_test_samples - 2))
                print("DBP Pearson: {0} +/- {1}".format(corr_DBP, standard_error))

                scatter_plot_SBP_fn = plot_test_dir.joinpath(plot_test_dir.name + "_" + model_names[m_ind] + "_scatter_plot_SBP.jpg")
                plt.scatter(sbp_label, sbp_pred)
                plt.xlabel("Ground Truth SBP")
                plt.ylabel("Estimated SBP")
                plt.title("SBP | MAE: " + str(np.round(MAE_SBP, 2)) + "; RMSE: " + str(np.round(RMSE_SBP, 2)) + "; MAPE: " + str(np.round(MAPE_SBP, 2)) + "; Corr: " + str(np.round(corr_SBP, 2)))
                plt.savefig(scatter_plot_SBP_fn)
                plt.close()

                scatter_plot_DBP_fn = plot_test_dir.joinpath(plot_test_dir.name + "_" + model_names[m_ind] + "_scatter_plot_DBP.jpg")
                plt.scatter(dbp_label, dbp_pred)
                plt.xlabel("Ground Truth DBP")
                plt.ylabel("Estimated DBP")
                plt.title("DBP | MAE: " + str(np.round(MAE_DBP, 2)) + "; RMSE: " + str(np.round(RMSE_DBP, 2)) + "; MAPE: " + str(np.round(MAPE_DBP, 2)) + "; Corr: " + str(np.round(corr_DBP, 2)))
                plt.savefig(scatter_plot_DBP_fn)
                plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', default="0", dest="tasks", type=int,
                        help='physiological signals to analyze: [0: BVP and RSP; 1:BVP; 2: RSP; 3: All (BVP, RSP, BP)')
    parser.add_argument('--plot', default="1", dest="plot", type=int,
                        help='whether to save plots: [0: No; 1:Yes')
    parser.add_argument('REMAIN', nargs='*')
    args_parser = parser.parse_args()

    compare_estimated_phys_within_dataset(args_parser.tasks, args_parser.plot)
