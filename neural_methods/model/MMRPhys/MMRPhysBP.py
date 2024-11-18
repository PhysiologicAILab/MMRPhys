"""
MMRPhys: Remote Extraction of Multiple Physiological Signals using Label Guided Factorization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

nf_BVP = [8, 12, 16]
nf_RSP = [8, 16, 16]

class ConvBlock2D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dilation=[1, 1], bias=True, groups=1):
        super(ConvBlock2D, self).__init__()
        self.conv_block_2d = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=padding, bias=bias, dilation=dilation, groups=groups),
            nn.ReLU(),
            nn.InstanceNorm2d(out_channel),
        )

    def forward(self, x):
        return self.conv_block_2d(x)


class ConvBlock1D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dilation=1, bias=True, groups=1):
        super(ConvBlock1D, self).__init__()
        self.conv_block_1d = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding=padding, bias=bias, dilation=dilation, groups=groups),
            nn.ReLU(),
            nn.InstanceNorm1d(out_channel),
        )

    def forward(self, x):
        return self.conv_block_1d(x)


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


class BP_Estimation_Head(nn.Module):
    def __init__(self, md_config, device, dropout_rate=0.1, debug=False):
        super(BP_Estimation_Head, self).__init__()
        self.debug = debug
        self.device = device
        self.fs = md_config["FS"]

        self.time_scaling = 4

        self.win_len_bvp = 10 * self.fs
        self.win_bvp = torch.hann_window(self.win_len_bvp).to(device)
        self.hop_len_bvp = 4 * self.fs   # for freq_band matched number of windows and 500 signal length, this should be 10; len=1500, hop=150, gives 10 
        self.bvp_feat_res_x = 21 #10    # 200 hop and 425 win_les, total windows are 11
        self.min_hr = 35
        self.max_hr = 185
        self.bvp_nfft = _next_power_of_2(self.time_scaling * md_config["FRAME_NUM"])
        bvp_fft_freq = 60 * self.fs * torch.fft.rfftfreq(self.bvp_nfft)
        self.bvp_freq_idx = torch.argwhere((bvp_fft_freq > self.min_hr) & (bvp_fft_freq < self.max_hr))
        # self.bvp_freq_idx = self.bvp_freq_idx.permute(1, 0)
        self.bvp_min_freq_id = torch.min(self.bvp_freq_idx, dim=0).values.numpy()[0]   # 12 for fs = 25 and T = 500; 48 for fs=25, T=1500 
        self.bvp_max_freq_id = torch.max(self.bvp_freq_idx, dim=0).values.numpy()[0]    # 61 for fs = 25 and T = 500; 252 for fs=25, T=1500 
        self.bvp_feat_res_y = self.bvp_max_freq_id - self.bvp_min_freq_id

        if self.debug:
            print("self.bvp_nfft", self.bvp_nfft)
            print("[self.bvp_feat_res_x, self.bvp_feat_res_y]", [self.bvp_feat_res_x, self.bvp_feat_res_y])

        self.win_len_rsp = 40 * self.fs
        self.win_rsp = torch.hann_window(self.win_len_rsp).to(device)
        self.hop_len_rsp = 4 * self.fs  #  for freq_band matched number of windows and 1997 signal length, this should be 53
        self.rsp_feat_res_x = 21 # 10    # 200 hop and 10 * 25 win_les, total windows are 10
        self.min_rr = 5
        self.max_rr = 33
        # signal is to be 4 times concatenated - to increase the freq resolution in the desired band
        self.rsp_nfft = _next_power_of_2(self.time_scaling * md_config["FRAME_NUM"])
        rsp_fft_freq = (60 * self.fs * torch.fft.rfftfreq(self.rsp_nfft))
        self.rsp_freq_idx = torch.argwhere((rsp_fft_freq > self.min_rr) & (rsp_fft_freq < self.max_rr))
        self.rsp_min_freq_id = torch.min(self.rsp_freq_idx, dim=0).values.numpy()[0]    # 7 for fs = 25 and T = 4x500
        self.rsp_max_freq_id = torch.max(self.rsp_freq_idx, dim=0).values.numpy()[0]    # 45 for fs = 25 and T = 4x500
        self.rsp_feat_res_y = self.rsp_max_freq_id - self.rsp_min_freq_id

        if self.debug:
            print("self.rsp_nfft", self.rsp_nfft)
            print("[self.rsp_feat_res_x, self.rsp_feat_res_y]", [self.rsp_feat_res_x, self.rsp_feat_res_y])

        self.bvp_embeddings_phase_extractor_SBP = nn.Sequential(
            nn.InstanceNorm2d(1),                                                                  #B, 1, 13, 13
            ConvBlock2D(1, nf_BVP[2], kernel_size=[5, 5], stride=[1, 1], padding=[2, 2]),          #B, nf_BVP[2], 13, 13
            ConvBlock2D(nf_BVP[2], nf_BVP[2], kernel_size=[5, 5], stride=[1, 1], padding=[1, 1]),  #B, nf_BVP[2], 11, 11
            ConvBlock2D(nf_BVP[2], nf_BVP[2], kernel_size=[5, 5], stride=[1, 1], padding=[1, 1]),  #B, nf_BVP[2], 9, 9
            nn.Conv2d(nf_BVP[2], 1, kernel_size=[5, 5], stride=[1, 1], padding=[0, 0]),            #B, nf_BVP[2], 5, 5
        )

        self.bvp_embeddings_phase_extractor_DBP = nn.Sequential(
            nn.InstanceNorm2d(1),                                                                  #B, 1, 13, 13
            ConvBlock2D(1, nf_BVP[2], kernel_size=[5, 5], stride=[1, 1], padding=[2, 2]),          #B, nf_BVP[2], 13, 13
            ConvBlock2D(nf_BVP[2], nf_BVP[2], kernel_size=[5, 5], stride=[1, 1], padding=[1, 1]),  #B, nf_BVP[2], 11, 11
            ConvBlock2D(nf_BVP[2], nf_BVP[2], kernel_size=[5, 5], stride=[1, 1], padding=[1, 1]),  #B, nf_BVP[2], 9, 9
            nn.Conv2d(nf_BVP[2], 1, kernel_size=[5, 5], stride=[1, 1], padding=[0, 0]),            #B, nf_BVP[2], 5, 5
        )

        self.bvp_stft_feature_extractor_SBP = nn.Sequential(
            ConvBlock2D(1, nf_BVP[0], kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),          #B, nf_BVP[0], 202, 19
            ConvBlock2D(nf_BVP[0], nf_BVP[0], kernel_size=[3, 3], stride=[2, 1], padding=[0, 0]),  #B, nf_BVP[0], 100, 17
            ConvBlock2D(nf_BVP[0], nf_BVP[0], kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),  #B, nf_BVP[0], 98, 15
            nn.Dropout2d(p=0.2),

            ConvBlock2D(nf_BVP[0], nf_BVP[0], kernel_size=[3, 3], stride=[2, 1], padding=[0, 0]),  #B, nf_BVP[0], 48, 13
            ConvBlock2D(nf_BVP[0], nf_BVP[0], kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),  #B, nf_BVP[0], 46, 11
            ConvBlock2D(nf_BVP[0], nf_BVP[0], kernel_size=[3, 3], stride=[2, 1], padding=[0, 0]),  #B, nf_BVP[0], 22, 9
            nn.Dropout2d(p=0.2),

            ConvBlock2D(nf_BVP[0], nf_BVP[0], kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),  #B, nf_BVP[0], 20, 7
            ConvBlock2D(nf_BVP[0], nf_BVP[0], kernel_size=[3, 3], stride=[2, 1], padding=[0, 0]),  #B, nf_BVP[0], 9, 5
            nn.Conv2d(nf_BVP[0], 1, kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),            #B, 1, 7, 3
        )

        self.bvp_stft_feature_extractor_DBP = nn.Sequential(
            ConvBlock2D(1, nf_BVP[0], kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),          #B, nf_BVP[0], 202, 19
            ConvBlock2D(nf_BVP[0], nf_BVP[0], kernel_size=[3, 3], stride=[2, 1], padding=[0, 0]),  #B, nf_BVP[0], 100, 17
            ConvBlock2D(nf_BVP[0], nf_BVP[0], kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),  #B, nf_BVP[0], 98, 15
            nn.Dropout2d(p=0.2),

            ConvBlock2D(nf_BVP[0], nf_BVP[0], kernel_size=[3, 3], stride=[2, 1], padding=[0, 0]),  #B, nf_BVP[0], 48, 13
            ConvBlock2D(nf_BVP[0], nf_BVP[0], kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),  #B, nf_BVP[0], 46, 11
            ConvBlock2D(nf_BVP[0], nf_BVP[0], kernel_size=[3, 3], stride=[2, 1], padding=[0, 0]),  #B, nf_BVP[0], 22, 9
            nn.Dropout2d(p=0.2),

            ConvBlock2D(nf_BVP[0], nf_BVP[0], kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),  #B, nf_BVP[0], 20, 7
            ConvBlock2D(nf_BVP[0], nf_BVP[0], kernel_size=[3, 3], stride=[2, 1], padding=[0, 0]),  #B, nf_BVP[0], 9, 5
            nn.Conv2d(nf_BVP[0], 1, kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),            #B, 1, 7, 3
        )

        self.rsp_stft_feature_extractor_SBP = nn.Sequential(
            ConvBlock2D(1, nf_RSP[0], kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),          #B, nf_RSP[0], 36, 19
            ConvBlock2D(nf_RSP[0], nf_RSP[0], kernel_size=[3, 3], stride=[2, 2], padding=[0, 0]),  #B, nf_RSP[0], 17, 9
            nn.Dropout2d(p=0.2),

            ConvBlock2D(nf_RSP[0], nf_RSP[0], kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),  #B, nf_RSP[0], 15, 7
            ConvBlock2D(nf_RSP[0], nf_RSP[0], kernel_size=[3, 3], stride=[2, 1], padding=[0, 0]),  #B, nf_RSP[0], 7, 5
            nn.Conv2d(nf_RSP[0], 1, kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),            #B, nf_RSP[0], 5, 3
        )

        self.rsp_stft_feature_extractor_DBP = nn.Sequential(
            ConvBlock2D(1, nf_RSP[0], kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),          #B, nf_RSP[0], 36, 19
            ConvBlock2D(nf_RSP[0], nf_RSP[0], kernel_size=[3, 3], stride=[2, 2], padding=[0, 0]),  #B, nf_RSP[0], 17, 9
            nn.Dropout2d(p=0.2),

            ConvBlock2D(nf_RSP[0], nf_RSP[0], kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),  #B, nf_RSP[0], 15, 7
            ConvBlock2D(nf_RSP[0], nf_RSP[0], kernel_size=[3, 3], stride=[2, 1], padding=[0, 0]),  #B, nf_RSP[0], 7, 5
            nn.Conv2d(nf_RSP[0], 1, kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),            #B, nf_RSP[0], 5, 3
        )

        num_feats_bvp_phase_corr = 1 * 5 * 5
        num_feats_bvp_stft = 1 * 7 * 3
        num_feats_rsp_stft = 1 * 5 * 3
    
        total_feats = num_feats_bvp_stft + num_feats_rsp_stft + num_feats_bvp_phase_corr
        hidden_feats = total_feats

        self.final_dense_layer_SBP = nn.Sequential(
            nn.Linear(total_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, 1)
        )

        self.final_dense_layer_DBP = nn.Sequential(
            nn.Linear(total_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, 1)
        )


    def forward(self, bvp_embeddings, bvp_vec, rsp_vec):

        if self.debug:
            print(" BP Head")
            print(" rppg_embeddings.shape", bvp_embeddings.shape)
            print(" rBr.shape", rsp_vec.shape)

        with torch.no_grad():            

            # Normalize estimated BVP signals
            avg_bvp = torch.mean(bvp_vec, dim=1, keepdim=True)
            std_bvp = torch.std(bvp_vec, dim=1, keepdim=True)
            norm_bvp_vec = (bvp_vec - avg_bvp)/std_bvp

            bt, ch, t, h, w = bvp_embeddings.shape      # Expected dimensions: Batch, 16, 500, 11, 11
            bvp_matrix = norm_bvp_vec.unsqueeze(1).repeat(1, ch, 1).unsqueeze(3).unsqueeze(3).repeat(1, 1, 1, h, w)
            bvp_corr_matrix = F.cosine_similarity(bvp_embeddings, bvp_matrix, dim=2).abs()
            if self.debug:
                print("bvp_corr_matrix.shape:", bvp_corr_matrix.shape)
            ch_with_max_corr = torch.max(torch.mean(bvp_corr_matrix, dim=(2,3)), dim=1).indices
            bvp_corr_feats = torch.zeros((bt, 1, h, w)).to(self.device)
            for b_idx in range(bt):
                bvp_corr_feats[b_idx, 0, :, :] = bvp_corr_matrix[b_idx, ch_with_max_corr[b_idx], :, :]
            if self.debug:
                print("bvp_corr_feats.shape", bvp_corr_feats.shape)
    
            # bvp_corr_matrix_std_across_channels = torch.std(bvp_corr_matrix, dim=1, keepdim=True) #can be added to bvp_corr_matrix as additional channel


            # to increase the signal length for sufficient freqs in low-freq range
            # flipped_norm_bvp_vec = -1 * torch.fliplr(norm_bvp_vec)      
            # long_bvp_vec = torch.concat([norm_bvp_vec, flipped_norm_bvp_vec[:, 1:], norm_bvp_vec[:, 1:], flipped_norm_bvp_vec[:, 1:], norm_bvp_vec[:, 1:4]], dim=1)

            # STFT - magnitude for estimated BVP signal - to capture heart-rate and HRV
            long_norm_bvp_vec = torch.zeros((bt, t * self.time_scaling)).to(self.device)
            for b_idx in range(bt):
                temp_vec = norm_bvp_vec[b_idx, :]
                last_zero_crossing = torch.where(torch.diff(torch.sign(temp_vec)))[0][-1].cpu().numpy()
                trimmed_bvp_vec = temp_vec[:last_zero_crossing].unsqueeze(0)
                flipped_bvp_vec = -1 * torch.fliplr(trimmed_bvp_vec)
                temp_long_vec = torch.concat([trimmed_bvp_vec, flipped_bvp_vec[:, 1:], trimmed_bvp_vec[:, 1:],
                                             flipped_bvp_vec[:, 1:], trimmed_bvp_vec[:, 1:], flipped_bvp_vec[:, 1:],  trimmed_bvp_vec[:, 1:]], dim=1)
                temp_long_vec = temp_long_vec[0, :t*self.time_scaling]
                long_norm_bvp_vec[b_idx, :] = temp_long_vec
            
            feature_map_bvp_fft_magnitude = torch.zeros((bt, 1, self.bvp_feat_res_y, self.bvp_feat_res_x)).to(self.device)
            bvp_stft = torch.stft(long_norm_bvp_vec, n_fft=self.bvp_nfft, win_length=self.win_len_bvp,
                                window=self.win_bvp, hop_length=self.hop_len_bvp, return_complex=True)
            # bvp_stft = torch.stft(long_bvp_vec, n_fft=self.bvp_nfft, return_complex=True)

            # Normalized spectrogram, instance wise
            bvp_stft_mag = bvp_stft.real[:, self.bvp_min_freq_id:self.bvp_max_freq_id, :].abs()
            bvp_stft_mag_min = torch.min(bvp_stft_mag, dim=1, keepdim=True).values
            bvp_stft_mag_max = torch.max(bvp_stft_mag, dim=1, keepdim=True).values
            bvp_stft_mag = (bvp_stft_mag - bvp_stft_mag_min) / (bvp_stft_mag_max - bvp_stft_mag_min)
            feature_map_bvp_fft_magnitude[:, 0, :, :] = bvp_stft_mag

            # Normalize estimated RSP signals
            avg_rsp = torch.mean(rsp_vec, dim=1, keepdim=True)
            std_rsp = torch.std(rsp_vec, dim=1, keepdim=True)
            norm_rsp_vec = (rsp_vec - avg_rsp)/std_rsp

            # to increase the signal length for sufficient freqs in low-freq range
            # flipped_norm_rsp_vec = -1 * torch.fliplr(norm_rsp_vec)
            # long_rsp_vec = torch.concat([norm_rsp_vec, flipped_norm_rsp_vec[:, 1:], norm_rsp_vec[:, 1:], flipped_norm_rsp_vec[:, 1:], norm_rsp_vec[:, 1:4]], dim=1)
            
            # STFT - magnitude for estimated RSP signal - to capture Resp Rate and respiration variability
            long_norm_rsp_vec = torch.zeros((bt, t * self.time_scaling)).to(self.device)
            for b_idx in range(bt):
                temp_vec = norm_rsp_vec[b_idx, :]
                last_zero_crossing = torch.where(torch.diff(torch.sign(temp_vec)))[0][-1].cpu().numpy()
                trimmed_rsp_vec = temp_vec[:last_zero_crossing].unsqueeze(0)
                flipped_rsp_vec = -1 * torch.fliplr(trimmed_rsp_vec)
                temp_long_vec = torch.concat([trimmed_rsp_vec, flipped_rsp_vec[:, 1:], trimmed_rsp_vec[:, 1:],
                                             flipped_rsp_vec[:, 1:], trimmed_rsp_vec[:, 1:], flipped_rsp_vec[:, 1:], trimmed_rsp_vec[:, 1:]], dim=1)
                temp_long_vec = temp_long_vec[0, :t*self.time_scaling]
                long_norm_rsp_vec[b_idx, :] = temp_long_vec

            feature_map_rsp_fft_magnitude = torch.zeros((bt, 1, self.rsp_feat_res_y, self.rsp_feat_res_x)).to(self.device)
            rsp_stft = torch.stft(long_norm_rsp_vec, n_fft=self.rsp_nfft, win_length=self.win_len_rsp,
                                window=self.win_rsp, hop_length=self.hop_len_rsp, return_complex=True)
            # rsp_stft = torch.stft(long_norm_rsp_vec, n_fft=self.rsp_nfft, return_complex=True)
            rsp_stft_mag = rsp_stft.real[:, self.rsp_min_freq_id:self.rsp_max_freq_id, :].abs()

            # Normalized spectrogram, instance wise
            rsp_stft_mag_min = torch.min(rsp_stft_mag, dim=1, keepdim=True).values
            rsp_stft_mag_max = torch.max(rsp_stft_mag, dim=1, keepdim=True).values
            rsp_stft_mag = (rsp_stft_mag - rsp_stft_mag_min) / (rsp_stft_mag_max - rsp_stft_mag_min)
            feature_map_rsp_fft_magnitude[:, 0, :, :] = rsp_stft_mag

        if self.debug:
            print("feature_map_bvp_fft_magnitude.shape", feature_map_bvp_fft_magnitude.shape)
            print("feature_map_rsp_fft_magnitude.shape", feature_map_rsp_fft_magnitude.shape)

        # Convolutional blocks - separate feature extraction for BVP Embeddings (phase, magnitude), BVP (magnitude) and RSP (magnitude)
        bvp_corr_feats_SBP = self.bvp_embeddings_phase_extractor_SBP(bvp_corr_feats)
        bvp_STFT_feats_SBP = self.bvp_stft_feature_extractor_SBP(feature_map_bvp_fft_magnitude)
        rsp_STFT_feats_SBP = self.rsp_stft_feature_extractor_SBP(feature_map_rsp_fft_magnitude)

        bvp_corr_feats_DBP = self.bvp_embeddings_phase_extractor_DBP(bvp_corr_feats)
        bvp_STFT_feats_DBP = self.bvp_stft_feature_extractor_DBP(feature_map_bvp_fft_magnitude)
        rsp_STFT_feats_DBP = self.rsp_stft_feature_extractor_DBP(feature_map_rsp_fft_magnitude)

        if self.debug:
            print("bvp_corr_feats_SBP.shape", bvp_corr_feats_SBP.shape)
            print("bvp_STFT_feats_SBP.shape", bvp_STFT_feats_SBP.shape)
            print("rsp_STFT_feats_SBP.shape", rsp_STFT_feats_SBP.shape)

        # Fully connected layers
        bvp_corr_feats_SBP = bvp_corr_feats_SBP.view(bt, -1)
        bvp_STFT_feats_SBP = bvp_STFT_feats_SBP.view(bt, -1)
        rsp_STFT_feats_SBP = rsp_STFT_feats_SBP.view(bt, -1)

        bvp_corr_feats_DBP = bvp_corr_feats_DBP.view(bt, -1)
        bvp_STFT_feats_DBP = bvp_STFT_feats_DBP.view(bt, -1)
        rsp_STFT_feats_DBP = rsp_STFT_feats_DBP.view(bt, -1)

        all_feats_SBP = torch.concat([bvp_corr_feats_SBP, bvp_STFT_feats_SBP, rsp_STFT_feats_SBP], dim=1)
        all_feats_DBP = torch.concat([bvp_corr_feats_DBP, bvp_STFT_feats_DBP, rsp_STFT_feats_DBP], dim=1)

        SBP = self.final_dense_layer_SBP(all_feats_SBP)
        DBP = self.final_dense_layer_DBP(all_feats_DBP)
        
        rBP = torch.concat([SBP, DBP], dim=1)

        if self.debug:
            print(" rBP.shape", rBP.shape)

        return rBP