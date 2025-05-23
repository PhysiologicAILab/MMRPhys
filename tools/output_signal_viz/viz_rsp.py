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


path_dict_within_dataset = {
    "test_datasets": {
        # "BP4D_500x36_Fold1_RGBT": {
        #     "root": "runs/exp/BP4D_RGBT_500_36x36/saved_test_outputs/",
        #     "exp" : {
        #         "MMRPhys_FuseM_SFSAM_Label":
        #         {
        #             "rsp": "BP4D_MMRPhys_BVP_RSP_RGBT_FuseMx500x36_SFSAM_Label_Fold1_rsp_outputs.pickle",
        #         }
        #     },
        # },
        # "BP4D_500x9_Fold1_RGBT": {
        #     "root": "runs/exp/BP4D_RGBT_500_9x9/saved_test_outputs/",
        #     "exp": {
        #         "MMRPhys_FuseS_SFSAM_Label":
        #         {
        #             "rsp": "BP4D_MMRPhys_BVP_RSP_RGBT_FuseSx500x9_SFSAM_Label_Fold1_rsp_outputs.pickle",
        #         },
        #     }
        # },
        "BP4D_180x9_Fold1_RGBT": {
            "root": "runs/exp/BP4D_RGBT_180_9x9/saved_test_outputs/",
            "exp": {
                "MMRPhys_Small_Base_Label":
                {
                    "rsp": "BP4D_MMRPhys_RSP_RGBTx9_Base_Fold1_rsp_outputs.pickle",
                },
            }
        }
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


def _process_rsp_signal(signal, fs=25, diff_flag=True):
    # Detrend and filter
    use_bandpass = True
    if diff_flag:  # if the predictions and labels are 1st derivative of RSP signal.
        # signal = _detrend(np.cumsum(signal), 100)
        signal = np.cumsum(signal)
    else:
        # signal = _detrend(signal, 100)
        pass
    if use_bandpass:
        # [b, a] = butter(2, [0.05 / fs * 2, 0.7 / fs * 2], btype='bandpass')
        # [b, a] = butter(2, [0.08 / fs * 2, 0.6 / fs * 2], btype='bandpass')
        # [b, a] = butter(2, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
        [b, a] = butter(2, [0.13 / fs * 2, 0.5 / fs * 2], btype='bandpass')
        signal = filtfilt(b, a, np.double(signal))
    return signal

# def _detrend(input_signal, lambda_value):
#     """Detrend RSP signal."""
#     signal_length = input_signal.shape[0]
#     # observation matrix
#     H = np.identity(signal_length)
#     ones = np.ones(signal_length)
#     minus_twos = -2 * np.ones(signal_length)
#     diags_data = np.array([ones, minus_twos, ones])
#     diags_index = np.array([0, 1, 2])
#     D = spdiags(diags_data, diags_index,
#                 (signal_length - 2), signal_length).toarray()
#     detrended_signal = np.dot(
#         (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
#     return detrended_signal


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


# def _calculate_fft_rr(rsp_signal, fs=25, low_pass=0.05, high_pass=0.7):
# def _calculate_fft_rr(rsp_signal, fs=25, low_pass=0.08, high_pass=0.6):
# def _calculate_fft_rr(rsp_signal, fs=25, low_pass=0.08, high_pass=0.5):
# def _calculate_fft_rr(rsp_signal, fs=25, low_pass=0.13, high_pass=0.5):
def _calculate_fft_rr(rsp_signal, fs=25, low_pass=0.1, high_pass=0.54):
    """Calculate respiration rate based on RSP using Fast Fourier transform (FFT)."""
    rsp_signal = np.expand_dims(rsp_signal, 0)
    N = _next_power_of_2(rsp_signal.shape[1])
    f_rsp, pxx_rsp = scipy.signal.periodogram(rsp_signal, fs=fs, nfft=N, detrend=False)
    fmask_rsp = np.argwhere((f_rsp >= low_pass) & (f_rsp <= high_pass))
    mask_rsp = np.take(f_rsp, fmask_rsp)
    mask_pxx = np.take(pxx_rsp, fmask_rsp)
    fft_rr = np.take(mask_rsp, np.argmax(mask_pxx, 0))[0] * 60
    return fft_rr


def _calculate_peak_rr(resp_signal, fs=25):
    """Calculate respiration rate based on PPG using peak detection."""
    resp_peaks, _ = scipy.signal.find_peaks(resp_signal)
    rr_peak = 60 / (np.mean(np.diff(resp_peaks)) / fs)
    return rr_peak


# Main functions

def compare_estimated_phys_within_dataset(save_plot=1):

    if save_plot:
        plot_dir = Path.cwd().joinpath("plots").joinpath("BP4D_RSP")
        plot_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = 180 #300  # size of chunk to visualize: -1 will plot the entire signal

    for test_dataset in path_dict_within_dataset["test_datasets"]:
        print("*"*50)
        print("Test Data:", test_dataset)
        print("*"*50)
        rsp_dict = {}

        root_dir = Path(path_dict_within_dataset["test_datasets"][test_dataset]["root"])
        if not root_dir.exists():
            print("Data path does not exists:", str(root_dir))
            exit()

        if save_plot:
            plot_test_dir = plot_dir.joinpath(test_dataset)
            plot_test_dir.mkdir(parents=True, exist_ok=True)

        for train_model in path_dict_within_dataset["test_datasets"][test_dataset]["exp"]:
            print("Model:", train_model)
            rsp_fn = root_dir.joinpath(path_dict_within_dataset["test_datasets"][test_dataset]["exp"][train_model]["rsp"])
            rsp_dict[train_model] = CPU_Unpickler(open(rsp_fn, "rb")).load()

        print("-"*50)

        model_names = list(rsp_dict.keys())
        print("Model Names:", model_names)
        # print(bvp_dict[model_names[0]].keys())
        # exit()

        # List of all video trials
        trial_list = list(rsp_dict[model_names[0]]['predictions'].keys())
        print('Num Trials', len(trial_list))
        all_rr_labels = {}
        all_rr_preds = {}

        for trial_ind in range(len(trial_list)):
            # print("."*25)
            gt_rsp = np.array(_reform_data_from_dict(rsp_dict[model_names[0]]['predictions'][trial_list[trial_ind]]))

            total_samples = len(gt_rsp)
            total_chunks = total_samples // chunk_size
            # print('Chunk size', chunk_size)
            # print('Total chunks', total_chunks)

            # Read in meta-data from pickle file
            fs = rsp_dict[model_names[0]]['fs'] # Video Frame Rate
            label_type = rsp_dict[model_names[0]]['label_type'] # RSP Signal Transformation: `DiffNormalized` or `Standardized`
            diff_flag = (label_type == 'DiffNormalized')
            rsp_label = np.array(_reform_data_from_dict(rsp_dict[model_names[0]]['labels'][trial_list[trial_ind]]))

            trial_dict = {}
            rr_pred = {}

            for c_ind in range(total_chunks):
                # print("*"*25)
                # try:
                if save_plot:
                    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
                # fig.tight_layout()

                start = (c_ind)*chunk_size
                stop = (c_ind+1)*chunk_size
                samples = stop - start

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
                        trial_dict[model_names[m_ind]]["rsp_pred"] = np.array(_reform_data_from_dict(rsp_dict[model_names[m_ind]]['predictions'][trial_list[trial_ind]]))

                    if model_names[m_ind] not in all_rr_labels:
                        all_rr_labels[model_names[m_ind]] = []
                        all_rr_preds[model_names[m_ind]] = []


                    # Process label and prediction signals
                    rsp_pred_seg = _process_rsp_signal(trial_dict[model_names[m_ind]]["rsp_pred"][start: stop], fs, diff_flag=diff_flag)

                    try:
                        rr_pred[model_names[m_ind]] = _calculate_peak_rr(rsp_pred_seg, fs=fs)
                        rr_pred[model_names[m_ind]] = int(np.round(rr_pred[model_names[m_ind]]))
                    except:
                        rr_pred[model_names[m_ind]] = _calculate_fft_rr(rsp_pred_seg, fs=fs)
                        rr_pred[model_names[m_ind]] = int(np.round(rr_pred[model_names[m_ind]]))

                    # print("m_ind:", m_ind, "; rr_pred: ", rr_pred[model_names[m_ind]])
                    all_rr_labels[model_names[m_ind]].append(rr_label)
                    all_rr_preds[model_names[m_ind]].append(rr_pred[model_names[m_ind]])

                    if save_plot:
                        plt.plot(x_time, rsp_pred_seg, label=model_names[m_ind] + "; RR = " + str(rr_pred[model_names[m_ind]]))

                if save_plot:
                    plt.plot(x_time, rsp_label[start: stop], label="GT ; RR = " + str(rr_label), color='black')
                    plt.legend(loc="upper right")
                
                    plt.suptitle("Dataset: " + test_dataset + '; Trial: ' + trial_list[trial_ind] + '; Chunk: ' + str(c_ind), fontsize=14)

                    # plt.show()
                    save_fn = plot_test_dir.joinpath(str(trial_list[trial_ind]) + "_" + str(c_ind) + ".jpg")
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

        data_fn = plot_test_dir.parent.joinpath(plot_test_dir.name + "_eval_data.npy")
        np.save(data_fn, all_test_data)

        print("-*-"*50)
        print("Performance of different models for test dataset:", test_dataset)
        print("-*-"*50)
        for m_ind in range(len(model_names)):
            print("-"*50)
            print("Model:", model_names[m_ind])
            print("-"*50)
            
            gt_rr = np.array(all_rr_labels[model_names[m_ind]])
            pred_rr = np.array(all_rr_preds[model_names[m_ind]])
            num_test_samples = len(gt_rr)

            print("Total Samples: ", num_test_samples)
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
            correlation_coefficient = Pearson_RR[0][1]
            standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
            print("RR Pearson: {0} +/- {1}".format(correlation_coefficient, standard_error))

            scatter_plot_rr_fn = plot_test_dir.parent.joinpath(plot_test_dir.name + "_" + model_names[m_ind] + "_scatter_plot_RR.jpg")
            plt.scatter(rr_labels_clean_array, rr_pred_clean_array)
            plt.title("RR | MAE: " + str(np.round(MAE_RR, 2)) + "; RMSE: " + str(np.round(RMSE_RR, 2)) + "; MAPE: " + str(np.round(MAPE_RR, 2)) + "; Corr: " + str(np.round(Pearson_RR, 2)))
            plt.savefig(scatter_plot_rr_fn)
            plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', default="1", dest="plot", type=int,
                        help='whether to save plots: [0: No; 1:Yes')
    parser.add_argument('REMAIN', nargs='*')
    args_parser = parser.parse_args()

    compare_estimated_phys_within_dataset(args_parser.plot)