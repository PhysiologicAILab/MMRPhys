# %%
from pathlib import Path

import mat73
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
import glob
import os
import torch
import scipy

# %%
a = np.random.random((4, 500))
a = np.delete(a, [0, 3], axis=0)
print(a.shape)


# %%
a = torch.rand((2, 4, 3, 3,))
b = torch.mean(a, dim=(2,3))
print(b)
b_max_idx = torch.max(b, dim=1).indices
b_min_idx = torch.min(b, dim=1).indices

print(b_max_idx.shape)

diff_a = torch.zeros((2, 1, 3, 3))
for bt in range(2):
    diff_a[bt, ...] = a[bt, b_max_idx[bt], :, :] - a[bt, b_min_idx[bt], :, :]
print(diff_a.shape)

# %%

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

fs = 25
# %%
rsp = nk.rsp_simulate(120, sampling_rate=fs, respiratory_rate=5)
rsp = rsp[0: 500]
rsp_tensor = torch.from_numpy(rsp).unsqueeze(0)
rsp_tensor = rsp_tensor.repeat(4, 1)

print(rsp_tensor.shape)
last_zero_crossing = torch.zeros_like(rsp_tensor)
sign = torch.sign(rsp_tensor, out=last_zero_crossing)
last_zero_crossing = torch.where(torch.diff(last_zero_crossing))
index_tensor = torch.stack(last_zero_crossing, dim=1)

print(index_tensor.shape)
# plt.plot(last_zero_crossing[0, ])

# %%
rsp = nk.rsp_simulate(120, sampling_rate=fs, respiratory_rate=5)
rsp = rsp[0: 500]
rsp_tensor = torch.from_numpy(rsp)

last_zero_crossing = torch.where(torch.diff(torch.sign(rsp_tensor)))[0][-1].numpy()
print("last_zero_crossing:", last_zero_crossing)
new_rsp_tensor = rsp_tensor[0:last_zero_crossing].unsqueeze(0)

rsp_tensor_rep = -1 * new_rsp_tensor.fliplr()
long_rsp_tensor = torch.concat([new_rsp_tensor, rsp_tensor_rep[:,1:],
                               new_rsp_tensor[:,1:], rsp_tensor_rep[:,1:], new_rsp_tensor[:,1:]], dim=1)
long_rsp_tensor = long_rsp_tensor[:, 0:2000]
rsp_nfft = _next_power_of_2(2000)
rsp_fft_freq = (60 * fs * torch.fft.rfftfreq(rsp_nfft))
rsp_freq_idx = torch.argwhere((rsp_fft_freq > 5) & (rsp_fft_freq < 33))
rsp_freq_idx_min = rsp_freq_idx.min()
rsp_freq_idx_max = rsp_freq_idx.max()
print(rsp_freq_idx.min(), rsp_freq_idx.max())

rsp_fft = torch.fft.rfft(long_rsp_tensor[0, :]).real.abs()
rsp_fft_foi = rsp_fft[rsp_freq_idx_min: rsp_freq_idx_max]
rsp_freq_foi = rsp_fft_freq[rsp_freq_idx_min: rsp_freq_idx_max]
plt.plot(rsp_freq_foi, rsp_fft_foi)

# %%
plt.plot(long_rsp_tensor[0, :])

# %%
zero_crossings = torch.where(torch.diff(torch.sign(rsp_tensor[0,:])))[0]
zero_crossings[-1]

# %%
fs = 25
bvp_nfft = _next_power_of_2(2000)
bvp_fft_freq = (60 * fs * torch.fft.rfftfreq(bvp_nfft))
bvp_freq_idx = torch.argwhere((bvp_fft_freq > 35) & (bvp_fft_freq < 185))

bvp_freq_idx_min = bvp_freq_idx.min()
bvp_freq_idx_max = bvp_freq_idx.max()
print(bvp_freq_idx.min(), bvp_freq_idx.max())

ppg_gen = nk.ppg_simulate(120, sampling_rate=fs, heart_rate=70)
ppg_tensor = torch.from_numpy(ppg_gen[200: 2200])
ppg_fft = torch.fft.rfft(ppg_tensor).real.abs()
ppg_fft_foi = ppg_fft[bvp_freq_idx_min: bvp_freq_idx_max]
bvp_freq_foi = bvp_fft_freq[bvp_freq_idx_min: bvp_freq_idx_max]
plt.plot(bvp_freq_foi, ppg_fft_foi)


bvp_win = torch.hann_window(400)
bvp_stft = torch.stft(ppg_tensor, n_fft=bvp_nfft, win_length=400, hop_length=100, window=bvp_win, return_complex=True)

bvp_stft_mag = bvp_stft.real[bvp_freq_idx_min:bvp_freq_idx_max, :].abs()

bvp_stft_mag_min = torch.min(bvp_stft_mag, dim=0, keepdim=True).values
bvp_stft_mag_max = torch.max(bvp_stft_mag, dim=0, keepdim=True).values
bvp_stft_mag_norm = (bvp_stft_mag - bvp_stft_mag_min) / (bvp_stft_mag_max - bvp_stft_mag_min)


# print(bvp_stft_phase.min(), bvp_stft_phase.max())

# Imp Note: Normalize the fft mag for BVP and RSP; Normalize Phase angle.
# Normalize all inputs to BP estimation head.

plt.imshow(bvp_stft_mag_norm, cmap="coolwarm")


# %%
thresh_mag = 0.7
bvp_stft_phase = bvp_stft.angle()[bvp_freq_idx_min:bvp_freq_idx_max, :]   # torch.rad2deg(bvp_stft.angle())
bvp_stft_phase_min = torch.min(bvp_stft_phase, dim=0, keepdim=True).values
bvp_stft_phase_max = torch.max(bvp_stft_phase, dim=0, keepdim=True).values
bvp_stft_phase_norm = (bvp_stft_phase - bvp_stft_phase_min) / (bvp_stft_phase_max - bvp_stft_phase_min)

bvp_stft_mag_mask = torch.ones_like(bvp_stft_mag_norm)
bvp_stft_mag_mask[bvp_stft_mag_norm < thresh_mag] = 0
bvp_stft_phase_norm = bvp_stft_mag_mask * bvp_stft_phase_norm
print(bvp_stft_phase_norm.shape)
fig, ax = plt.subplots(1, 3)
ax[0].imshow(bvp_stft_mag_norm, cmap="coolwarm")
ax[1].imshow(bvp_stft_phase_norm, cmap="coolwarm")

# %%
rsp = nk.rsp_simulate(120, sampling_rate=fs, respiratory_rate=12)
rsp = rsp[0: 500]
rsp = torch.from_numpy(rsp)
rsp = rsp.unsqueeze(0)
rsp = rsp.repeat(2, 1)
print(rsp.shape)

rsp_win = torch.hann_window(400)
# rsp_stft = torch.stft(rsp, n_fft=_next_power_of_2(800), return_complex=True)
rsp_stft = torch.stft(rsp, n_fft=_next_power_of_2(800), win_length=400, hop_length=100, window=rsp_win, return_complex=True)

rsp_stft_mag = rsp_stft.real[0, 4:22, :].abs()
print("rsp_stft_mag.shape", rsp_stft_mag.shape)
rsp_stft_mag_min = torch.min(rsp_stft_mag, dim=0, keepdim=True).values
rsp_stft_mag_max = torch.max(rsp_stft_mag, dim=0, keepdim=True).values
rsp_stft_mag_norm = (rsp_stft_mag - rsp_stft_mag_min) / (rsp_stft_mag_max - rsp_stft_mag_min)

print(rsp_stft_mag_norm.shape)
ax[2].imshow(rsp_stft_mag_norm, cmap="coolwarm")


# %%

# %%
f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_gen[200: 700], fs=fs, nfft=bvp_nfft, detrend=False)
fmask_ppg = np.argwhere((f_ppg >= 0.6) & (f_ppg <= 3.3))
print(len(fmask_ppg))

# %%
ppg = torch.from_numpy(ppg_gen[200: 700])

# %%
run_cell = 12

# %% 
if run_cell == 1:
    a = torch.rand((4, 10))
    b = a.roll(1)
    plt.plot(a[0,:])
    plt.plot(b[0,:])
    plt.show()

    plt.plot(a[1, :])
    plt.plot(b[1, :])
    plt.show()

    # print(a.shape)
    # b = []
    # for s in range(10):
    #     b.append(a.roll(s))
    # b = torch.from_numpy(np.array(b))
    # print(b.shape)

# %%
if run_cell == 2:
    wave_file = "/mnt/sda/data/raw/SCAMPS/scamps_videos/P000007.mat"
    opt = "bvp_rsp"
    """Reads a bvp and resp signal file."""
    mat = mat73.loadmat(wave_file)
    ppg = mat['d_ppg']  # load ppg signal
    ppg = np.asarray(ppg)
    ppg = np.expand_dims(ppg, axis=1)
    if "rsp" in opt:
        resp = mat['d_br']  # load resp signal
        resp = np.asarray(resp)
        resp = np.expand_dims(resp, axis=1)

    print(ppg.shape, resp.shape)
    data = np.concatenate([ppg, resp], axis=1)
    print(data.shape)

# %% Simulated PPG for Smooth NMF estimators
if run_cell == 3:
    fs = 25
    seg_len = 180
    sig_type = "hr" # "hr" or "rr"

    if sig_type == "hr":
        hr = 72
        max_samples_in_lowest_hr = 2*fs  # 30 BPM
        max_samples = 4 * max_samples_in_lowest_hr
        iters = np.arange(0, max_samples, 1)
        duration = ((max_samples + seg_len) // fs) + 1
        total_estimators = len(iters)
    
    else:
        rr = 12
        max_samples_in_lowest_rr = 10*fs  # 6 BPM
        max_samples = 4 * max_samples_in_lowest_rr
        iters = np.arange(0, max_samples, 5)
        duration = ((max_samples + seg_len) // fs) + 1
        total_estimators = len(iters)
    
    print("Duration:", duration)
    print("Total SNMF Estimators:", total_estimators)
    
    for iter in iters:
        if sig_type == "hr":
            sig = nk.ppg_simulate(duration=duration, sampling_rate=fs, heart_rate=hr, frequency_modulation=0.1, ibi_randomness=0.03)
        else:
            sig = nk.rsp_simulate(duration=duration, sampling_rate=fs, respiratory_rate=rr)
        sig_seg = sig[iter: iter + seg_len]
        mx = np.max(sig_seg)
        mn = np.min(sig_seg)
        sig_seg = (sig_seg - mn)/(mx - mn)
        plt.plot(sig_seg)

    plt.show()

# %%
if run_cell == 4:
    pth = "/mnt/sda/data/prep/UBFC-rPPG/UBFC-rPPG_Raw_160_72x72/subject1_label3.npy"
    data = np.load(pth)
    print(data.shape)

# %%
if len(data.shape) < 2:
    temp_data = np.expand_dims(data, 1)
    hr_vec = 72 * np.ones_like(temp_data)
    print(hr_vec.shape)
    new_data = np.concatenate([temp_data, hr_vec], axis=1)
    print(new_data.shape)
# %%
pth = Path("/mnt/sda/data/prep/UBFC-rPPG/UBFC-rPPG_Raw_160_72x72")
fns = sorted(list(pth.glob("*label*.npy")))
for fn in fns:
    print(fn.name)


# %% check added HR to prepared data
if run_cell == 4:
    pth = Path("/home/jitesh/data/UBFC-rPPG/UBFC-rPPG_Raw_160_72x72/subject1_label1.npy")
    data = np.load(str(pth))
    print(data.shape)
    plt.plot(data[:, 0])
    plt.plot(data[:, 1])



# %% Read CSV and find a specific file, remove the row.
if run_cell == 5:
    csv_path = "/home/jitesh/data/UBFC-rPPG/DataFileLists/UBFC-rPPG_Raw_160_72x72_0.0_1.0_15FPS.csv"
    video_fn = "/home/jitesh/data/UBFC-rPPG/UBFC-rPPG_Raw_160_72x72_15FPS/subject1_input8.npy"

    save_csv_path = "/home/jitesh/data/UBFC-rPPG/DataFileLists/UBFC-rPPG_Raw_160_72x72_0.0_1.0_15FPS_new.csv"
    
    df = pd.read_csv(csv_path)
    loc = df[df.input_files == video_fn].index
    print(loc)
    df = df.drop(loc)
    df.to_csv(save_csv_path, header=["","input_files"], index=False)

# %% merge selective frames of video
if run_cell == 6:
    # a = np.random.random((10, 3, 4, 5))
    # b = np.random.random((10, 3, 4, 5))
    # c = np.concatenate(a[])
    a = np.array([1, 2, 3, 4, 5, 6])
    b = np.array([7, 8, 9, 10, 11, 12])
    c = np.concatenate([a[np.arange(0, 6, 2)], b[np.arange(0, 6, 2)]])
    d = np.concatenate([a[np.arange(1, 6, 2)], b[np.arange(1, 6, 2)]])

    print(c)
    print(d)
    
# %% check labels for BP4D data
if run_cell == 7:
    old_prep_label_fn = "/home/jitesh/data/BP4D/BP4D_RGBT_180_36x36/F001T01_label0.npy"
    new_prep_label_fn = "/home/jitesh/data/BP4D/BP4D_RGBT_300_36x36/F001T01_label0.npy"

    l_old = np.load(old_prep_label_fn)
    l_new = np.load(new_prep_label_fn)

    bvp_old = l_old[:, 0]
    bvp_new = l_new[:, 0]

    rsp_old = l_old[:, 1]
    rsp_new = l_new[:, 1]

    bvp_bpm_old = l_old[:, 4]
    bvp_bpm_new = l_new[:, 4]

    rsp_bpm_old = l_old[:, 5]
    rsp_bpm_new = l_new[:, 5]

    fig, ax = plt.subplots(8, 1)
    ax[0].plot(bvp_old)
    ax[1].plot(bvp_new)
    ax[2].plot(bvp_bpm_old)
    ax[3].plot(bvp_bpm_new)

    ax[4].plot(rsp_old)
    ax[5].plot(rsp_new)
    ax[6].plot(rsp_bpm_old)
    ax[7].plot(rsp_bpm_new)

    plt.savefig("test1.jpg")
    plt.close()
 
# %% 
if run_cell == 8:
    fn = "/home/jitesh/dev/repos/vis/mmrPhys/BP4D_500x9_Fold1_RGBT.npy"
    test_data = np.load(fn, allow_pickle=True).item()
    for models in test_data.keys():
        hr_labels = test_data[models]["hr_labels"]
        hr_pred = test_data[models]["hr_pred"]
        rr_labels = test_data[models]["rr_labels"]
        rr_pred = test_data[models]["rr_pred"]

        plt.scatter(hr_labels, hr_pred)
        plt.savefig("HR.jpg")
        plt.close()

        plt.scatter(rr_labels, rr_pred)
        plt.savefig("RR.jpg")
        plt.close()

# %%
if run_cell == 9:
    fold_path = "dataset/BP4D_BigSmall_Subject_Splits/Split1_Test_Subjects.csv"
    data_path = "/mnt/sda/data/raw/BP4D_9x9"
    data_dirs = glob.glob(data_path + os.sep + "*_*")

    dirs = list()
    for data_dir in data_dirs:
        subject_trail_val = os.path.split(data_dir)[-1].replace('_', '')
        index = subject_trail_val
        subject = subject_trail_val[0:4]
        dirs.append({"index": index, "path": data_dir, "subject": subject})

    print("data_dirs:", data_dirs)

    fold_df = pd.read_csv(fold_path)
    fold_subjs = list(set(list(fold_df.subjects)))

    fold_data_dirs = []
    for d in dirs:
        idx = d['index']
        subj = idx[0:4]

        if subj in fold_subjs:  # if trial has already been processed
            fold_data_dirs.append(d)

    print("fold_data_dirs:", fold_data_dirs)

# %%
if run_cell == 10:
    a = 10 + torch.rand((10, 1))
    b = torch.rand((10, 1))
    crit = torch.nn.SmoothL1Loss()
    loss = crit(a, b)
    print(loss)

# %%
if run_cell == 11:
    SBP = 120
    DBP = 80
    hr_bpm = 70
    fs = 25
    duration = 20
    hr_bpm = 30 if hr_bpm < 30 else hr_bpm
    hr_bpm = 200 if hr_bpm > 200 else hr_bpm
    sig = nk.ppg_simulate(duration=duration, sampling_rate=fs, heart_rate=hr_bpm,
                          frequency_modulation=0, ibi_randomness=0, motion_amplitude=0, powerline_amplitude=0, burst_amplitude=0, random_state=1, random_state_distort=1)
    mn = np.min(sig)
    mx = np.max(sig)
    sig = (sig - mn)/ (mx - mn)
    sig = (sig * (SBP - DBP)) + DBP
    plt.plot(sig)


# %%
if run_cell == 12:
    pth = Path("/home/jitesh/data/BP4D/BP4D_RGBT_500_72x72/F001T01_label0.npy")
    data = np.load(pth)
    bvp = data[:, 11]
    # avg_bvp = np.mean(bvp)
    # std_bvp = np.std(bvp)
    # norm_bvp = (bvp - avg_bvp) / std_bvp

    plt.plot(bvp)
    


# %%

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

nfft = _next_power_of_2(500)
fs = 25
freqs = 60 * (np.arange(0, nfft)) * (fs / nfft / 2.)
min_hr = 30
max_hr = 200
idx_hr = np.argwhere((freqs > min_hr) & (freqs < max_hr))
print(idx_hr)

min_rr = 6
max_rr = 33
idx_rr = np.argwhere((freqs > min_rr) & (freqs < max_rr))
print(idx_rr)

# rsp_fft_freq = 60 * fs * torch.fft.rfftfreq(nfft)
# rsp_foi = torch.argwhere((rsp_fft_freq > 3) & (rsp_fft_freq < 33))
# print(rsp_foi)

# %%

ppg = nk.ppg_simulate(120, sampling_rate=fs, heart_rate=40)
ppg = ppg[200: 700]
ppg_tensor = torch.tensor(ppg)
ppg_fft_c = torch.fft.fft(ppg_tensor)
ppg_fft = torch.fft.rfft(ppg_tensor)
ppg_fft_angle = torch.angle(ppg_fft_c)
ppg_fft_freq = 60 * 25 * torch.fft.rfftfreq(nfft)
# ppg_fft_freq 
ppg_foi = torch.argwhere((ppg_fft_freq > 40) & (ppg_fft_freq < 200))

plt.plot(ppg_fft_freq[ppg_foi], ppg_fft[ppg_foi])
plt.plot(ppg_fft_freq[ppg_foi], ppg_fft_angle[ppg_foi])

ppg_fft_foi = ppg_fft[ppg_foi]
foi_idx = torch.argsort(ppg_fft_foi)

ppg_fft_peaks = [ppg_fft_freq[foi_idx[0]], ppg_fft_freq[foi_idx[1]], ppg_fft_freq[foi_idx[2]]]
ppg_fft_peaks_angle = [torch.angle(ppg_fft_foi[foi_idx[0]]), torch.angle(ppg_fft_foi[foi_idx[1]]), torch.angle(ppg_fft_foi[foi_idx[2]])]

print("ppg_fft_peaks:", ppg_fft_peaks)
print("ppg_fft_peaks_angle:", ppg_fft_peaks_angle)

'''
ppg_fft_sorted_idx = torch.argsort(ppg_fft, descending=True)
ppg_peak_1 = ppg_fft[ppg_fft_sorted_idx[0]]
ppg_freq_1 = 60 * ppg_fft_sorted_idx[0] * (fs / nfft / 2.)
print(ppg_freq_1)
# ppg_fft_angle = torch.angle(ppg_fft)

# ppg_fft = ppg_fft[idx]
# ppg_fft_angle = ppg_fft_angle[idx]
# freqs = freqs[idx]
# plt.plot(freqs, ppg_fft)
# plt.plot(freqs, ppg_fft_angle)

'''
# %%
p = torch.angle(b)
print(a.shape)
print(p.shape)
# %%


pth = "/home/jitesh/data/SCAMPS/SCAMPS_Raw_500_72x72/P000001.mat_label0.npy"
data = np.load(pth)
print(data.shape)

# %%


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


fs = 25
rsp_nfft = _next_power_of_2(2000)
rsp_fft_freq = (60 * fs * torch.fft.rfftfreq(rsp_nfft))
rsp_freq_idx = torch.argwhere((rsp_fft_freq > 5) & (rsp_fft_freq < 33))

bvp_nfft = _next_power_of_2(500)
bvp_fft_freq = (60 * fs * torch.fft.rfftfreq(bvp_nfft))
bvp_freq_idx = torch.argwhere((bvp_fft_freq > 35) & (bvp_fft_freq < 185))

print(rsp_freq_idx.min(), rsp_freq_idx.max())
print(bvp_freq_idx.min(), bvp_freq_idx.max())

# %%
a = np.array([[1,2,3,4,5], [10, 11, 12, 13, 15]])
at = torch.FloatTensor(a)
print(torch.min(at, dim=1).values)
print(torch.mean(at, dim=1))
# %%
a = np.array([[1, 2, 3, 4, 5], [10, 11, 12, 13, 15]])
at = torch.FloatTensor(a)
print(at.shape)
bt = at.fliplr()
print(bt.shape)
print(at)
print(bt)
# %%

