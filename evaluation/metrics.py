import numpy as np
import pandas as pd
import torch
from evaluation.post_process import *
from tqdm import tqdm
from evaluation.BlandAltmanPy import BlandAltman

def read_label(dataset):
    """Read manually corrected labels."""
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}
    return out_dict


def read_hr_label(feed_dict, index):
    """Read manually corrected UBFC labels."""
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr


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
    sort_data = float(sort_data[0])
    return sort_data


def calculate_metrics(predictions, labels, config):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    SNR_all = list()
    MACC_all = list()
    print("Calculating metrics!")
    for index in tqdm(predictions.keys(), ncols=80):
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        video_frame_size = prediction.shape[0]
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
            if window_frame_size > video_frame_size:
                window_frame_size = video_frame_size
        else:
            window_frame_size = video_frame_size

        for i in range(0, len(prediction), window_frame_size):
            pred_window = prediction[i:i+window_frame_size]
            label_window = label[i:i+window_frame_size]

            if len(pred_window) < 9:
                print(f"Window frame size of {len(pred_window)} is smaller than minimum pad length of 9. Window ignored!")
                continue

            if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                    config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
                diff_flag_test = False
            elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
                diff_flag_test = True
            else:
                raise ValueError("Unsupported label type in testing!")
            
            if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                gt_hr_peak, pred_hr_peak, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='Peak')
                gt_hr_peak_all.append(gt_hr_peak)
                predict_hr_peak_all.append(pred_hr_peak)
                SNR_all.append(SNR)
                MACC_all.append(macc)
            elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                gt_hr_fft, pred_hr_fft, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='FFT')
                gt_hr_fft_all.append(gt_hr_fft)
                predict_hr_fft_all.append(pred_hr_fft)
                SNR_all.append(SNR)
                MACC_all.append(macc)
            else:
                raise ValueError("Inference evaluation method name wrong!")
    
    # Filename ID to be used in any results files (e.g., Bland-Altman plots) that get saved
    if config.TOOLBOX_MODE == 'train_and_test' or config.TOOLBOX_MODE == 'only_train':
        filename_id = config.TRAIN.MODEL_FILE_NAME
    elif config.TOOLBOX_MODE == 'only_test':
        model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
        filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
    else:
        raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')

    if config.INFERENCE.EVALUATION_METHOD == "FFT":
        gt_hr_fft_all = np.array(gt_hr_fft_all)
        predict_hr_fft_all = np.array(predict_hr_fft_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_fft_all)
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                standard_error = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, standard_error))
            elif metric == "RMSE":
                RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
                standard_error = np.sqrt(np.std(np.square(predict_hr_fft_all - gt_hr_fft_all))) / np.sqrt(num_test_samples)
                print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, standard_error))
            elif metric == "MAPE":
                MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                standard_error = np.std(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(num_test_samples) * 100
                print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error))
            elif metric == "Pearson":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                correlation_coefficient = Pearson_FFT[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_FFT = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_FFT, standard_error))
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("FFT MACC (FFT Label): {0} +/- {1}".format(MACC_avg, standard_error))
            elif "AU" in metric:
                pass
            elif "BA" in metric:  
                compare = BlandAltman(gt_hr_fft_all, predict_hr_fft_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT PPG HR [bpm]',
                    y_label='rPPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_BVP_FFT_BlandAltman_ScatterPlot',
                    file_name=f'{filename_id}_BVP_FFT_BlandAltman_ScatterPlot.pdf')
                compare.difference_plot(
                    x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                    y_label='Average of rPPG HR and GT PPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_BVP_FFT_BlandAltman_DifferencePlot',
                    file_name=f'{filename_id}_BVP_FFT_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")
    elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
        gt_hr_peak_all = np.array(gt_hr_peak_all)
        predict_hr_peak_all = np.array(predict_hr_peak_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_peak_all)
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                standard_error = np.std(np.abs(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_PEAK, standard_error))
            elif metric == "RMSE":
                RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
                standard_error = np.sqrt(np.std(np.square(predict_hr_peak_all - gt_hr_peak_all))) / np.sqrt(num_test_samples)
                print("PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, standard_error))
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                standard_error = np.std(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) / np.sqrt(num_test_samples) * 100
                print("PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, standard_error))
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                correlation_coefficient = Pearson_PEAK[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("PEAK Pearson (Peak Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_PEAK = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("PEAK SNR (Peak Label): {0} +/- {1} (dB)".format(SNR_PEAK, standard_error))
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("PEAK MACC (Peak Label): {0} +/- {1}".format(MACC_avg, standard_error))
            elif "AU" in metric:
                pass
            elif "BA" in metric:
                compare = BlandAltman(gt_hr_peak_all, predict_hr_peak_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT PPG HR [bpm]',
                    y_label='rPPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_BVP_Peak_BlandAltman_ScatterPlot',
                    file_name=f'{filename_id}_BVP_Peak_BlandAltman_ScatterPlot.pdf')
                compare.difference_plot(
                    x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                    y_label='Average of rPPG HR and GT PPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_BVP_Peak_BlandAltman_DifferencePlot',
                    file_name=f'{filename_id}_BVP_Peak_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")
    else:
        raise ValueError("Inference evaluation method name wrong!")


def calculate_rsp_metrics(predictions, labels, config):
    """Calculate Respiration Metrics (MAE, RMSE, MAPE, Pearson Coef., SNR)."""

    print('=====================')
    print('==== RSP Metrics ===')
    print('=====================')

    predict_rr_fft_all = list()
    gt_rr_fft_all = list()
    predict_rr_peak_all = list()
    gt_rr_peak_all = list()
    SNR_all = list()
    MACC_all = list()
    for index in tqdm(predictions.keys(), ncols=80):
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        video_frame_size = prediction.shape[0]
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
            if window_frame_size > video_frame_size:
                window_frame_size = video_frame_size
        else:
            window_frame_size = video_frame_size

        for i in range(0, len(prediction), window_frame_size):
            pred_window = prediction[i:i+window_frame_size]
            label_window = label[i:i+window_frame_size]

            if len(pred_window) < 9:
                print(f"Window frame size of {len(pred_window)} is smaller than minimum pad length of 9. Window ignored!")
                continue

            if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                    config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
                diff_flag_test = False
            elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
                diff_flag_test = True
            else:
                raise ValueError("Unsupported label type in testing!")
            
            if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                gt_rr_peak, pred_rr_peak, SNR, macc = calculate_rsp_metrics_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, rr_method='Peak')
                gt_rr_peak_all.append(gt_rr_peak)
                predict_rr_peak_all.append(pred_rr_peak)
                SNR_all.append(SNR)
                MACC_all.append(macc)
            elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                gt_rr_fft, pred_rr_fft, SNR, macc = calculate_rsp_metrics_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, rr_method='FFT')
                gt_rr_fft_all.append(gt_rr_fft)
                predict_rr_fft_all.append(pred_rr_fft)
                SNR_all.append(SNR)
                MACC_all.append(macc)
            else:
                raise ValueError("Inference evaluation method name wrong!")

    # Filename ID to be used in any results files (e.g., Bland-Altman plots) that get saved
    if config.TOOLBOX_MODE == 'train_and_test' or config.TOOLBOX_MODE == 'only_train':
        filename_id = config.TRAIN.MODEL_FILE_NAME
    elif config.TOOLBOX_MODE == 'only_test':
        model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
        filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
    else:
        raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')

    if config.INFERENCE.EVALUATION_METHOD == "FFT":
        gt_rr_fft_all = np.array(gt_rr_fft_all)
        predict_rr_fft_all = np.array(predict_rr_fft_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_rr_fft_all)
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_FFT = np.mean(np.abs(predict_rr_fft_all - gt_rr_fft_all))
                standard_error = np.std(np.abs(predict_rr_fft_all - gt_rr_fft_all)) / np.sqrt(num_test_samples)
                print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, standard_error))
            elif metric == "RMSE":
                RMSE_FFT = np.sqrt(np.mean(np.square(predict_rr_fft_all - gt_rr_fft_all)))
                standard_error = np.sqrt(np.std(np.square(predict_rr_fft_all - gt_rr_fft_all))) / np.sqrt(num_test_samples)
                print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, standard_error))
            elif metric == "MAPE":
                MAPE_FFT = np.mean(np.abs((predict_rr_fft_all - gt_rr_fft_all) / gt_rr_fft_all)) * 100
                standard_error = np.std(np.abs((predict_rr_fft_all - gt_rr_fft_all) / gt_rr_fft_all)) / np.sqrt(num_test_samples) * 100
                print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error))
            elif metric == "Pearson":
                Pearson_FFT = np.corrcoef(predict_rr_fft_all, gt_rr_fft_all)
                correlation_coefficient = Pearson_FFT[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_FFT = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1}".format(SNR_FFT, standard_error))
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("MACC: {0} +/- {1}".format(MACC_avg, standard_error))
            elif "AU" in metric:
                pass
            elif "BA" in metric:
                compare = BlandAltman(gt_rr_fft_all, predict_rr_fft_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT RR [bpm]',
                    y_label='Predicted RR [bpm]', 
                    show_legend=True, figure_size=(5, 5), 
                    the_title=f'{filename_id}_RSP_FFT_BlandAltman_ScatterPlot',
                    file_name=f'{filename_id}_RSP_FFT_BlandAltman_ScatterPlot.pdf',
                    measure_lower_lim=3, 
                    measure_upper_lim=45)
                compare.difference_plot(
                    x_label='Difference between Predicted RR and GT RR [bpm]', 
                    y_label='Average of Predicted RR and GT RR [bpm]', 
                    show_legend=True, figure_size=(5, 5), 
                    the_title=f'{filename_id}_RSP_FFT_BlandAltman_DifferencePlot',
                    file_name=f'{filename_id}_RSP_FFT_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")
    elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
        gt_rr_peak_all = np.array(gt_rr_peak_all)
        predict_rr_peak_all = np.array(predict_rr_peak_all)
        SNR_all = np.array(SNR_all)
        num_test_samples = len(predict_rr_peak_all)
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_rr_peak_all - gt_rr_peak_all))
                standard_error = np.std(np.abs(predict_rr_peak_all - gt_rr_peak_all)) / np.sqrt(num_test_samples)
                print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_PEAK, standard_error))
            elif metric == "RMSE":
                RMSE_PEAK = np.sqrt(np.mean(np.square(predict_rr_peak_all - gt_rr_peak_all)))
                standard_error = np.sqrt(np.std(np.square(predict_rr_peak_all - gt_rr_peak_all))) / np.sqrt(num_test_samples)
                print("PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, standard_error))
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(np.abs((predict_rr_peak_all - gt_rr_peak_all) / gt_rr_peak_all)) * 100
                standard_error = np.std(np.abs((predict_rr_peak_all - gt_rr_peak_all) / gt_rr_peak_all)) / np.sqrt(num_test_samples) * 100
                print("PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, standard_error))
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_rr_peak_all, gt_rr_peak_all)
                correlation_coefficient = Pearson_PEAK[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("PEAK Pearson (Peak Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_PEAK = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1}".format(SNR_PEAK, standard_error))
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("MACC: {0} +/- {1}".format(MACC_avg, standard_error))
            elif "AU" in metric:
                pass
            elif "BA" in metric:
                compare = BlandAltman(gt_rr_peak_all, predict_rr_peak_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT RR [bpm]',
                    y_label='Predicted RR [bpm]', 
                    show_legend=True, figure_size=(5, 5), 
                    the_title=f'{filename_id}_RSP_Peak_BlandAltman_ScatterPlot',
                    file_name=f'{filename_id}_RSP_Peak_BlandAltman_ScatterPlot.pdf',
                    measure_lower_lim=10, 
                    measure_upper_lim=60)
                compare.difference_plot(
                    x_label='Difference between Predicted RR and GT RR [bpm]', 
                    y_label='Average of Predicted RR and GT RR [bpm]', 
                    show_legend=True, figure_size=(5, 5), 
                    the_title=f'{filename_id}_RSP_Peak_BlandAltman_DifferencePlot',
                    file_name=f'{filename_id}_RSP_Peak_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")
    else:
        raise ValueError("Inference evaluation method name wrong!")
    

def calculate_bp_metrics(predictions_sbp, labels_sbp, predictions_dbp, labels_dbp, config):
    """Calculate BP Metrics - SBP (MAE, RMSE, MAPE, Pearson Coef.), DBP (MAE, RMSE, MAPE, Pearson Coef.)."""

    print('=====================')
    print('==== BP Metrics ===')
    print('=====================')

    predict_sbp_all = list()
    gt_sbp_all = list()
    predict_dbp_all = list()
    gt_dbp_all = list()

    for index in tqdm(predictions_sbp.keys(), ncols=80):
        sbp_pred = _reform_BP_data_from_dict(predictions_sbp[index])
        dbp_pred = _reform_BP_data_from_dict(predictions_dbp[index])
        sbp_label = _reform_BP_data_from_dict(labels_sbp[index])
        dbp_label = _reform_BP_data_from_dict(labels_dbp[index])

        # sbp_pred = predictions_sbp[index]
        # dbp_pred = predictions_dbp[index]
        # sbp_label = labels_sbp[index]
        # dbp_label = labels_dbp[index]

        # print("[sbp_pred, sbp_label, dbp_pred, dbp_label]:", [sbp_pred, sbp_label, dbp_pred, dbp_label])
        predict_sbp_all.append(sbp_pred)
        gt_sbp_all.append(sbp_label)
        predict_dbp_all.append(dbp_pred)
        gt_dbp_all.append(dbp_label)

    # Filename ID to be used in any results files (e.g., Bland-Altman plots) that get saved
    if config.TOOLBOX_MODE == 'train_and_test' or config.TOOLBOX_MODE == 'only_train':
        filename_id = config.TRAIN.MODEL_FILE_NAME
    elif config.TOOLBOX_MODE == 'only_test':
        model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
        filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
    else:
        raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')

    predict_sbp_all = np.array(predict_sbp_all)
    gt_sbp_all = np.array(gt_sbp_all)
    predict_dbp_all = np.array(predict_dbp_all)
    gt_dbp_all = np.array(gt_dbp_all)
    num_test_samples = len(gt_sbp_all)
    
    for metric in config.TEST.METRICS:
        if metric == "MAE":
            MAE_SBP = np.mean(np.abs(predict_sbp_all - gt_sbp_all))
            standard_error = np.std(np.abs(predict_sbp_all - gt_sbp_all)) / np.sqrt(num_test_samples)
            print("SBP MAE: {0} +/- {1}".format(MAE_SBP, standard_error))
            MAE_DBP = np.mean(np.abs(predict_dbp_all - gt_dbp_all))
            standard_error = np.std(np.abs(predict_dbp_all - gt_dbp_all)) / np.sqrt(num_test_samples)
            print("DBP MAE: {0} +/- {1}".format(MAE_DBP, standard_error))
        
        elif metric == "RMSE":
            RMSE_SBP = np.sqrt(np.mean(np.square(predict_sbp_all - gt_sbp_all)))
            standard_error = np.sqrt(np.std(np.square(predict_sbp_all - gt_sbp_all))) / np.sqrt(num_test_samples)
            print("SBP RMSE: {0} +/- {1}".format(RMSE_SBP, standard_error))
            RMSE_DBP = np.sqrt(np.mean(np.square(predict_dbp_all - gt_dbp_all)))
            standard_error = np.sqrt(np.std(np.square(predict_dbp_all - gt_dbp_all))) / np.sqrt(num_test_samples)
            print("DBP RMSE: {0} +/- {1}".format(RMSE_DBP, standard_error))

        elif metric == "MAPE":
            MAPE_SBP = np.mean(np.abs((predict_sbp_all - gt_sbp_all) / gt_sbp_all)) * 100
            standard_error = np.std(np.abs((predict_sbp_all - gt_sbp_all) / gt_sbp_all)) / np.sqrt(num_test_samples) * 100
            print("SBP MAPE: {0} +/- {1}".format(MAPE_SBP, standard_error))
            MAPE_DBP = np.mean(np.abs((predict_dbp_all - gt_dbp_all) / gt_dbp_all)) * 100
            standard_error = np.std(np.abs((predict_dbp_all - gt_dbp_all) / gt_dbp_all)) / np.sqrt(num_test_samples) * 100
            print("DBP MAPE: {0} +/- {1}".format(MAPE_DBP, standard_error))

        elif metric == "Pearson":
            Pearson_SBP = np.corrcoef(predict_sbp_all, gt_sbp_all)
            correlation_coefficient = Pearson_SBP[0][1]
            standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
            print("SBP Pearson: {0} +/- {1}".format(correlation_coefficient, standard_error))
            Pearson_DBP = np.corrcoef(predict_dbp_all, gt_dbp_all)
            correlation_coefficient = Pearson_DBP[0][1]
            standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
            print("DBP Pearson: {0} +/- {1}".format(correlation_coefficient, standard_error))


        elif "BA" in metric:
            compare = BlandAltman(gt_sbp_all, predict_sbp_all, config, averaged=True)
            compare.scatter_plot(
                x_label='GT SBP [mmHg]',
                y_label='Predicted SBP [mmHg]', 
                show_legend=True, figure_size=(5, 5), 
                the_title=f'{filename_id}_SBP_BlandAltman_ScatterPlot',
                file_name=f'{filename_id}_SBP_BlandAltman_ScatterPlot.pdf',
                measure_lower_lim=50, 
                measure_upper_lim=250)
            compare.difference_plot(
                x_label='Difference between Predicted SBP and GT SBP [mmHg]', 
                y_label='Average of Predicted SBP and GT SBP [mmHg]', 
                show_legend=True, figure_size=(5, 5), 
                the_title=f'{filename_id}_SBP_BlandAltman_DifferencePlot',
                file_name=f'{filename_id}_SBP_BlandAltman_DifferencePlot.pdf')

            compare = BlandAltman(gt_dbp_all, predict_dbp_all, config, averaged=True)
            compare.scatter_plot(
                x_label='GT DBP [mmHg]',
                y_label='Predicted DBP [mmHg]', 
                show_legend=True, figure_size=(5, 5), 
                the_title=f'{filename_id}_DBP_BlandAltman_ScatterPlot',
                file_name=f'{filename_id}_DBP_BlandAltman_ScatterPlot.pdf',
                measure_lower_lim=40, 
                measure_upper_lim=200)
            compare.difference_plot(
                x_label='Difference between Predicted DBP and GT DBP [mmHg]', 
                y_label='Average of Predicted DBP and GT DBP [mmHg]', 
                show_legend=True, figure_size=(5, 5), 
                the_title=f'{filename_id}_DBP_BlandAltman_DifferencePlot',
                file_name=f'{filename_id}_DBP_BlandAltman_DifferencePlot.pdf')
        else:
            pass
            # raise ValueError("Wrong Test Metric Type")



