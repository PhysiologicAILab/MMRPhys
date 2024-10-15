import numpy as np
import torch
import pickle
import scipy
from scipy.signal import filtfilt, butter
from scipy.sparse import spdiags
import argparse
from pathlib import Path
import io
import matplotlib.pyplot as plt

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
        "BP4D_RSP_Tx72_Base_Fold1": {
            "root": "runs/exp/BP4D_RGBT_180_72x72/saved_test_outputs/",
            "exp": {
                "MMRPhys": "BP4D_MMRPhys_RSP_Tx72_Base_Fold1_rsp_outputs.pickle",
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

def _process_signal(signal, fs=25, diff_flag=True):
    # Detrend and filter
    use_bandpass = True
    if diff_flag:  # if the predictions and labels are 1st derivative of RSP signal.
        # signal = _detrend(np.cumsum(signal), 100)
        signal = np.cumsum(signal)
    else:
        # signal = _detrend(signal, 100)
        pass
    if use_bandpass:
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


def _calculate_fft_rr(rsp_signal, fs=25, low_pass=0.13, high_pass=0.5):
    """Calculate respiration rate based on RSP using Fast Fourier transform (FFT)."""
    rsp_signal = np.expand_dims(rsp_signal, 0)
    N = _next_power_of_2(rsp_signal.shape[1])
    f_rsp, pxx_rsp = scipy.signal.periodogram(rsp_signal, fs=fs, nfft=N, detrend=False)
    fmask_rsp = np.argwhere((f_rsp >= low_pass) & (f_rsp <= high_pass))
    mask_rsp = np.take(f_rsp, fmask_rsp)
    mask_pxx = np.take(pxx_rsp, fmask_rsp)
    fft_hr = np.take(mask_rsp, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr


# Main functions

def compare_estimated_rsp_within_dataset():

    plot_dir = Path.cwd().joinpath("plots").joinpath("BP4D_rsp")
    plot_dir.mkdir(parents=True, exist_ok=True)

    for test_dataset in path_dict_within_dataset["test_datasets"]:
        print("*"*50)
        print("Test Data:", test_dataset)
        print("*"*50)
        data_dict = {}

        root_dir = Path(path_dict_within_dataset["test_datasets"][test_dataset]["root"])
        if not root_dir.exists():
            print("Data path does not exists:", str(root_dir))
            exit()

        plot_test_dir = plot_dir.joinpath(test_dataset)
        plot_test_dir.mkdir(parents=True, exist_ok=True)

        for train_model in path_dict_within_dataset["test_datasets"][test_dataset]["exp"]:
            print("Model:", train_model)

            fn = root_dir.joinpath(path_dict_within_dataset["test_datasets"][test_dataset]["exp"][train_model])
            data_dict[train_model] = CPU_Unpickler(open(fn, "rb")).load()
        
        print("-"*50)

        model_names = list(data_dict.keys())
        print("Model Names:", model_names)
        # print(data_dict[model_names[0]].keys())
        # exit()

        # List of all video trials
        trial_list = list(data_dict[model_names[0]]['predictions'].keys())
        print('Num Trials', len(trial_list))

        gt_bvp = np.array(_reform_data_from_dict(
            data_dict[model_names[0]]['predictions'][trial_list[0]]))

        total_samples = len(gt_bvp)
        chunk_size = 180  # size of chunk to visualize: -1 will plot the entire signal
        total_chunks = total_samples // chunk_size
        print('Chunk size', chunk_size)
        print('Total chunks', total_chunks)

        for trial_ind in range(len(trial_list)):
            
            # Read in meta-data from pickle file
            fs = data_dict[model_names[0]]['fs'] # Video Frame Rate
            label_type = data_dict[model_names[0]]['label_type'] # RSP Signal Transformation: `DiffNormalized` or `Standardized`
            diff_flag = (label_type == 'DiffNormalized')

            trial_dict = {}

            rsp_label = np.array(_reform_data_from_dict(data_dict[model_names[0]]['labels'][trial_list[trial_ind]]))
            rsp_label = _process_signal(rsp_label, fs, diff_flag=diff_flag)

            rr_label = _calculate_fft_rr(rsp_label, fs=fs)
            rr_label = int(np.round(rr_label))
            print("rr_label", rr_label)
            rr_pred = {}


            for c_ind in range(total_chunks):
                try:
                    fig = plt.figure(figsize=(15, 5))
                    # fig.tight_layout()

                    start = (c_ind)*chunk_size
                    stop = (c_ind+1)*chunk_size
                    samples = stop - start

                    x_time = np.linspace(0, samples/fs, num=samples)

                    for m_ind in range(len(model_names)):

                        if model_names[m_ind] not in trial_dict:
                            trial_dict[model_names[m_ind]] = {}

                            # Reform label and prediction vectors from multiple trial chunks
                            trial_dict[model_names[m_ind]]["prediction"] = np.array(_reform_data_from_dict(
                                data_dict[model_names[m_ind]]['predictions'][trial_list[trial_ind]]))

                            # Process label and prediction signals
                            trial_dict[model_names[m_ind]]["prediction"] = _process_signal(
                                trial_dict[model_names[m_ind]]["prediction"], fs, diff_flag=diff_flag)

                            rr_pred[model_names[m_ind]] = _calculate_fft_rr(trial_dict[model_names[m_ind]]["prediction"], fs=fs)
                            rr_pred[model_names[m_ind]] = int(np.round([model_names[m_ind]]))
        
                        plt.plot(x_time, trial_dict[model_names[m_ind]]
                                ["prediction"][start: stop], label=model_names[m_ind] + "; RR = " + str(rr_pred[model_names[m_ind]]))

                    plt.plot(x_time, rsp_label[start: stop], label="GT ; RR = " + str(rr_label), color='black')
                    plt.legend(loc = "upper right")
                    plt.title("Dataset: " + test_dataset + '; Trial: ' +
                                trial_list[trial_ind] + '; Chunk: ' + str(c_ind), fontsize=14)

                    # plt.show()
                    save_fn = plot_test_dir.joinpath(str(trial_list[trial_ind]) + "_" + str(c_ind) + ".jpg")
                    plt.xlabel('Time (s)')
                    plt.savefig(save_fn)
                    plt.close()

                except Exception as e:
                    print("Encoutered error:", e)

                plt.close(fig)


if __name__ == "__main__":
    compare_estimated_rsp_within_dataset()