"""
MMRPhys: Remote Extraction of Multiple Physiological Signals using Label Guided Factorization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_methods.model.MMRPhys.FSAM import FeaturesFactorizationModule
from copy import deepcopy

nf_BVP = [8, 12, 16]
nf_RSP = [8, 16, 16]

model_config = {
    "TASKS": ["RSP"],
    "FS": 25,
    "MD_FSAM": False,
    "MD_TYPE": "SNMF_Label",
    "MD_R": 1,
    "MD_S": 1,
    "MD_STEPS": 4,
    "MD_INFERENCE": False,
    "MD_RESIDUAL": False,
    "INV_T": 1,
    "ETA": 0.9,
    "RAND_INIT": True,
    "in_channels": 3,
    "data_channels": 4,
    "align_channels": nf_BVP[2] // 2,
    "height": 72,
    "weight": 72,
    "batch_size": 4,
    "frames": 180,
    "debug": False,
    "assess_latency": False,
    "num_trials": 20,
    "visualize": False,
    "ckpt_path": "",
    "data_path": "",
    "label_path": ""
}


class ConvBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dilation=[1, 1, 1], bias=False, groups=1):
        super(ConvBlock3D, self).__init__()
        self.conv_block_3d = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding=padding, bias=bias, dilation=dilation, groups=groups),
            nn.Tanh(),
            nn.InstanceNorm3d(out_channel),
        )

    def forward(self, x):
        return self.conv_block_3d(x)


class ConvBlock2D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dilation=[1, 1], bias=True, groups=1):
        super(ConvBlock2D, self).__init__()
        self.conv_block_2d = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=padding, bias=bias, dilation=dilation, groups=groups),
            nn.Tanh(),
            nn.InstanceNorm2d(out_channel),
        )

    def forward(self, x):
        return self.conv_block_2d(x)


class ConvBlock1D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dilation=1, bias=True, groups=1):
        super(ConvBlock1D, self).__init__()
        self.conv_block_1d = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding=padding, bias=bias, dilation=dilation, groups=groups),
            nn.Tanh(),
            nn.InstanceNorm1d(out_channel),
        )

    def forward(self, x):
        return self.conv_block_1d(x)


class BVP_FeatureExtractor(nn.Module):
    def __init__(self, inCh, dropout_rate=0.1, debug=False):
        super(BVP_FeatureExtractor, self).__init__()
        # inCh, out_channel, kernel_size, stride, padding

        self.debug = debug
        #                                                        Input: #B, inCh, T, 72, 72
        self.bvp_feature_extractor = nn.Sequential(
            ConvBlock3D(inCh, nf_BVP[0], [3, 3, 3], [1, 2, 2], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_BVP[0], T, 35, 35
            ConvBlock3D(nf_BVP[0], nf_BVP[1], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_BVP[1], T, 33, 33
            ConvBlock3D(nf_BVP[1], nf_BVP[1], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_BVP[1], T, 31, 31
            nn.Dropout3d(p=dropout_rate),

            ConvBlock3D(nf_BVP[1], nf_BVP[2], [3, 3, 3], [1, 2, 2], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_BVP[2], T, 15, 15
            ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_BVP[2], T, 13, 13
            ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_BVP[2], T, 11, 11
            nn.Dropout3d(p=dropout_rate),
        )

    def forward(self, x):
        bvp_features = self.bvp_feature_extractor(x)
        if self.debug:
            print("BVP Feature Extractor")
            print("     bvp_features.shape", bvp_features.shape)
        return bvp_features


class BVP_Head(nn.Module):
    def __init__(self, md_config, device, dropout_rate=0.1, debug=False):
        super(BVP_Head, self).__init__()
        self.debug = debug

        # self.conv_layer = nn.Sequential(
        #     ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_BVP[2], T, 9, 9
        #     ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_BVP[2], T, 7, 7
        #     nn.Dropout3d(p=dropout_rate),
        # )

        self.use_fsam = md_config["MD_FSAM"]
        self.md_type = md_config["MD_TYPE"]
        self.md_infer = md_config["MD_INFERENCE"]
        self.md_res = md_config["MD_RESIDUAL"]

        if self.use_fsam:
            self.fsam = FeaturesFactorizationModule(nf_BVP[2], device, md_config, dim="3D", debug=debug)
            self.fsam_norm = nn.InstanceNorm3d(nf_BVP[2])
            self.bias1 = nn.Parameter(torch.tensor(1.0), requires_grad=True).to(device)

        self.final_layer = nn.Sequential(
            ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_BVP[2], T, 9, 9
            ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_BVP[2], T, 7, 7
            nn.Dropout3d(p=dropout_rate),

            ConvBlock3D(nf_BVP[2], nf_BVP[1], [3, 3, 3], [1, 1, 1], [1, 0, 0]),                         #B, nf_BVP[1], T, 5, 5
            ConvBlock3D(nf_BVP[1], nf_BVP[0], [3, 3, 3], [1, 1, 1], [1, 0, 0]),                   #B, nf_BVP[0], T, 3, 3
            nn.Conv3d(nf_BVP[0], 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),  #B, 1, T, 1, 1
        )

    def forward(self, length, bvp_embeddings=None, label_bvp=None):

        # bvp_embeddings = self.conv_layer(bvp_embeddings)

        if self.debug:
            print("BVP Head")
            print("     bvp_embeddings.shape", bvp_embeddings.shape)

        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            if "NMF" in self.md_type:
                att_mask, appx_error = self.fsam(bvp_embeddings - bvp_embeddings.min(), label_bvp) # to make it positive (>= 0)
            else:
                att_mask, appx_error = self.fsam(bvp_embeddings, label_bvp)

            if self.debug:
                print("att_mask.shape", att_mask.shape)

            # Multiplication with Residual connection
            x = torch.mul(bvp_embeddings - bvp_embeddings.min() + self.bias1, att_mask - att_mask.min() + self.bias1)
            factorized_embeddings = self.fsam_norm(x)
            factorized_embeddings = bvp_embeddings + factorized_embeddings          

            x = self.final_layer(factorized_embeddings)

        else:
            appx_error = 0
            x = self.final_layer(bvp_embeddings)

        if self.debug:
            print("bvp_embeddings.shape", bvp_embeddings.shape)
            print("x.shape", x.shape)

        rPPG = x.view(-1, length)

        if self.debug:
            print("     rPPG.shape", rPPG.shape)
        
        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            return rPPG, factorized_embeddings, appx_error
        else:
            return rPPG, bvp_embeddings, appx_error


class RSP_FeatureExtractor(nn.Module):
    def __init__(self, inCh=1, dropout_rate=0.1, debug=False):
        super(RSP_FeatureExtractor, self).__init__()
        # inCh, out_channel, kernel_size, stride, padding

        self.debug = debug
        #                                                                                     Input: #B, inCh, T, 72, 72
        self.rsp_feature_extractor = nn.Sequential(
            ConvBlock3D(inCh, nf_RSP[0], [3, 3, 3], [2, 1, 1], [1, 1, 1], dilation=[1, 1, 1]),       #B, nf_RSP[0], T//2, 72, 72
            ConvBlock3D(nf_RSP[0], nf_RSP[1], [3, 3, 3], [2, 2, 2], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[0], T//4, 35, 35
            ConvBlock3D(nf_RSP[1], nf_RSP[1], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[1], T//4, 33, 33
            nn.Dropout3d(p=dropout_rate),

            ConvBlock3D(nf_RSP[1], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[1], T//4, 31, 31
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 2, 2], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[2], T//4, 15, 15
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[2], T//4, 13, 13
            nn.Dropout3d(p=dropout_rate),
        )

    def forward(self, x):
        thermal_rsp_features = self.rsp_feature_extractor(x)
        if self.debug:
            print("Thermal Feature Extractor")
            print("     thermal_rsp_features.shape", thermal_rsp_features.shape)
        return thermal_rsp_features


class RSP_Head(nn.Module):
    def __init__(self, md_config, device, dropout_rate=0.1, debug=False):
        super(RSP_Head, self).__init__()
        self.debug = debug

        self.temporal_scale_factor = 4

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(self.temporal_scale_factor, 1, 1)),       #B, nf_RSP[2], T, 11, 11
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[2], T, 11, 11
        )

        self.use_fsam = md_config["MD_FSAM"]
        self.md_type = md_config["MD_TYPE"]
        self.md_infer = md_config["MD_INFERENCE"]
        self.md_res = md_config["MD_RESIDUAL"]

        # md_config = deepcopy(md_config)
        # md_config["MD_R"] = 4
        # md_config["MD_S"] = 1
        # md_config["MD_STEPS"] = 6

        if self.use_fsam:
            self.fsam = FeaturesFactorizationModule(nf_RSP[2], device, md_config, dim="3D", debug=debug)
            self.fsam_norm = nn.InstanceNorm3d(nf_RSP[2])
            self.bias1 = nn.Parameter(torch.tensor(1.0), requires_grad=True).to(device)

        self.final_layer = nn.Sequential(
            # ConvBlock3D(nf_RSP[2], nf_RSP[2], [5, 3, 3], [1, 3, 3], [4, 0, 0], dilation=[2, 1, 1]),   #B, nf_RSP[2], T, 7, 7
            # ConvBlock3D(nf_RSP[2], nf_RSP[1], [3, 3, 3], [1, 2, 2], [2, 0, 0], dilation=[2, 1, 1]),   #B, nf_RSP[1], T, 3, 3
            # nn.Conv3d(nf_RSP[1], 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),      #B, 1, T, 1, 1

            # ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [2, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),     #B, nf_RSP[2], T//2, 9, 9
            # ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [2, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),     #B, nf_RSP[2], T//4, 7, 7
            # ConvBlock3D(nf_RSP[2], nf_RSP[1], [3, 3, 3], [1, 2, 2], [1, 0, 0], dilation=[1, 1, 1]),     #B, nf_RSP[1], T//4, 3, 3
            # nn.Upsample(scale_factor=(self.temporal_scale_factor, 1, 1)),                               #B, nf_RSP[1], T, 3, 3
            # nn.Conv3d(nf_RSP[1], 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),        #B, 1, T, 1, 1

            # # ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),     #B, nf_RSP[2], T, 7, 7
            # # ConvBlock3D(nf_RSP[2], nf_RSP[1], [3, 3, 3], [1, 2, 2], [1, 0, 0], dilation=[1, 1, 1]),     #B, nf_RSP[1], T, 3, 3
            # # nn.Conv3d(nf_RSP[1], 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),        #B, 1, T, 1, 1

            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),     #B, nf_RSP[2], T, 9, 9
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),     #B, nf_RSP[2], T, 7, 7
            ConvBlock3D(nf_RSP[2], nf_RSP[1], [3, 3, 3], [1, 2, 2], [1, 0, 0], dilation=[1, 1, 1]),     #B, nf_RSP[1], T, 3, 3
            nn.Conv3d(nf_RSP[1], 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),        #B, 1, T, 1, 1
        )


    def forward(self, length, rsp_embeddings=None, label_rsp=None):

        voxel_embeddings = self.upsample(rsp_embeddings)

        if self.debug:
            print("RSP Head")
            print("     voxel_embeddings.shape", voxel_embeddings.shape)

        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            if "NMF" in self.md_type:
                att_mask, appx_error = self.fsam(voxel_embeddings - voxel_embeddings.min(), label_rsp) # to make it positive (>= 0)
            else:
                att_mask, appx_error = self.fsam(voxel_embeddings, label_rsp)

            if self.debug:
                print("att_mask.shape", att_mask.shape)

            # Multiplication with Residual connection
            x = torch.mul(voxel_embeddings - voxel_embeddings.min() + self.bias1, att_mask - att_mask.min() + self.bias1)
            factorized_embeddings = self.fsam_norm(x)
            factorized_embeddings = voxel_embeddings + factorized_embeddings           

            x = self.final_layer(factorized_embeddings)
        
        else:
            appx_error = 0
            # voxel_embeddings = self.upsample(voxel_embeddings)                
            x = self.final_layer(voxel_embeddings)

        if self.debug:
            print("voxel_embeddings.shape", voxel_embeddings.shape)
            print("x.shape", x.shape)

        rBr = x.view(-1, length)

        if self.debug:
            print("     rBr.shape", rBr.shape)
        
        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            return rBr, factorized_embeddings, appx_error
        else:
            return rBr, voxel_embeddings, appx_error


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


class BP_Estimation_Head(nn.Module):
    def __init__(self, md_config, device, dropout_rate=0.1, debug=False):
        super(BP_Estimation_Head, self).__init__()
        self.debug = debug

        self.device = device
        self.fs = md_config["FS"]
        self.sig_length = md_config["FRAME_NUM"]

        self.min_hr = 35
        self.max_hr = 185
        bvp_nfft = _next_power_of_2(self.sig_length)
        bvp_fft_freq = (60 * self.fs * torch.fft.rfftfreq(bvp_nfft))
        bvp_freq_idx = torch.argwhere((bvp_fft_freq > self.min_hr) & (bvp_fft_freq < self.max_hr))
        self.bvp_min_freq_id = bvp_freq_idx.min()    # 7 for fs = 25 and T = 4x500
        self.bvp_max_freq_id = bvp_freq_idx.max()    # 45 for fs = 25 and T = 4x500

        self.min_rr = 5
        self.max_rr = 33
        rsp_nfft = _next_power_of_2(self.sig_length)
        rsp_fft_freq = (60 * self.fs * torch.fft.rfftfreq(rsp_nfft))
        rsp_freq_idx = torch.argwhere((rsp_fft_freq > self.min_rr) & (rsp_fft_freq < self.max_rr))
        self.rsp_min_freq_id = rsp_freq_idx.min()    # 7 for fs = 25 and T = 4x500
        self.rsp_max_freq_id = rsp_freq_idx.max()    # 45 for fs = 25 and T = 4x500

        # Convolution groups are used to separately process magnitude and phase features of STFT
        self.bvp_embeddings_phase_extractor = nn.Sequential(
            ConvBlock2D(nf_BVP[2], nf_BVP[2], kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),  #B, nf_BVP[2], 11, 11
            ConvBlock2D(nf_BVP[2], nf_BVP[2], kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),  #B, nf_BVP[2], 11, 11
            nn.Dropout2d(p=0.2),

            ConvBlock2D(nf_BVP[2], nf_BVP[2], kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),  #B, nf_BVP[2], 9, 9
            ConvBlock2D(nf_BVP[2], nf_BVP[2], kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),  #B, nf_BVP[2], 9, 9
            nn.Dropout2d(p=0.2),

            ConvBlock2D(nf_BVP[2], nf_BVP[2], kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),  #B, nf_BVP[2], 7, 7
            ConvBlock2D(nf_BVP[2], nf_BVP[2], kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),  #B, nf_BVP[2], 7, 7
            nn.Conv2d(nf_BVP[2], 1, kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),            #B, 1, 5, 5
        )

        self.bvp_rate_feature_extractor = nn.Sequential(
            ConvBlock1D(1, 2, kernel_size=5, stride=1, padding=0),  # B, 2, 47
            ConvBlock1D(2, 4, kernel_size=5, stride=1, padding=0),  # B, 4, 43
            ConvBlock1D(4, 4, kernel_size=5, stride=2, padding=0),  # B, 4, 20
            nn.Conv1d(4, 1, kernel_size=5, stride=1, padding=0),  # B, 1, 16
        )

        self.rsp_rate_feature_extractor = nn.Sequential(
            ConvBlock1D(1, 4, kernel_size=3, stride=1, padding=1),  # B, 4, 9
            ConvBlock1D(4, 4, kernel_size=3, stride=1, padding=0),  # B, 4, 7
            nn.Conv1d(4, 1, kernel_size=3, stride=1, padding=0),  # B, 1, 5
        )

        num_feats_bvp_phase = 1 * 5 * 5        
        num_feats_bvp_rate = 1 * 16 * 1
        num_feats_rsp_rate = 1 * 5 * 1

        self.bvp_phase_dense_layer = nn.Sequential(
            nn.Linear(num_feats_bvp_phase, 2)
        )

        self.final_dense_layer = nn.Sequential(
            nn.Linear(num_feats_bvp_phase + num_feats_bvp_rate + num_feats_rsp_rate, 2),
        )

    def forward(self, bvp_embeddings, bvp_vec, rsp_vec):

        if self.debug:
            print(" BP Head")
            print(" bvp_embeddings.shape", bvp_embeddings.shape)
            print(" bvp_vec.shape:", bvp_vec.shape)
            print(" rsp_vec.shape", rsp_vec.shape)

        # TODO: # Imp Note: Normalize the fft mag for BVP and RSP; Normalize Phase angle.
                # Normalize all inputs to BP estimation head.

        bt, ch, t, h, w = bvp_embeddings.shape      # Expected dimensions: Batch, 16, 500, 31, 31

        bvp_matrix = bvp_vec.unsqueeze(1).repeat(1, ch, 1).unsqueeze(3).unsqueeze(3).repeat(1, 1, 1, h, w)
        bvp_corr_matrix = F.cosine_similarity(bvp_embeddings, bvp_matrix, dim=2).abs()

        if self.debug:
            print("bvp_corr_matrix.shape:", bvp_corr_matrix.shape)

        bvp_emb_feats = self.bvp_embeddings_phase_extractor(bvp_corr_matrix)

        # FFT - magnitude for estimated BVP signal - to capture Heart Rate
        avg_bvp = torch.mean(bvp_vec, dim=1, keepdim=True)
        std_bvp = torch.std(bvp_vec, dim=1, keepdim=True)
        norm_bvp_vec = (bvp_vec - avg_bvp)/std_bvp
        bvp_fft = torch.fft.rfft(norm_bvp_vec).real
        bvp_fft = bvp_fft[:, self.bvp_min_freq_id: self.bvp_max_freq_id]
        bvp_fft_avg = torch.mean(bvp_fft, dim=1, keepdim=True)
        bvp_fft_std = torch.std(bvp_fft, dim=1, keepdim=True)
        bvp_fft_norm = (bvp_fft - bvp_fft_avg) / bvp_fft_std
        bvp_fft_norm = bvp_fft_norm.unsqueeze(1)

        # FFT - magnitude for estimated RSP signal - to capture Resp Rate
        avg_rsp = torch.mean(rsp_vec, dim=1, keepdim=True)
        std_rsp = torch.std(rsp_vec, dim=1, keepdim=True)
        norm_rsp_vec = (rsp_vec - avg_rsp)/std_rsp
        rsp_fft = torch.fft.rfft(norm_rsp_vec).real
        rsp_fft = rsp_fft[:, self.rsp_min_freq_id: self.rsp_max_freq_id]
        rsp_fft_avg = torch.mean(rsp_fft, dim=1, keepdim=True)
        rsp_fft_std = torch.std(rsp_fft, dim=1, keepdim=True)
        rsp_fft_norm = (rsp_fft - rsp_fft_avg) / rsp_fft_std
        rsp_fft_norm = rsp_fft_norm.unsqueeze(1)

        if self.debug:
            print("bvp_fft_norm.shape:", bvp_fft_norm.shape)
            print("rsp_fft_norm.shape:", rsp_fft_norm.shape)

        bvp_fft_feats = self.bvp_rate_feature_extractor(bvp_fft_norm)
        rsp_fft_feats = self.rsp_rate_feature_extractor(rsp_fft_norm)

        if self.debug:
            print("bvp_emb_feats.shape", bvp_emb_feats.shape)
            print("bvp_fft_feats.shape", bvp_fft_feats.shape)
            print("rsp_fft_feats.shape", rsp_fft_feats.shape)

        bvp_emb_feats = bvp_emb_feats.view(bt, -1)
        bvp_fft_feats = bvp_fft_feats.view(bt, -1)
        rsp_fft_feats = rsp_fft_feats.view(bt, -1)

        res_bvp_phase = self.bvp_phase_dense_layer(bvp_emb_feats)
        rBP = res_bvp_phase + self.final_dense_layer(torch.concat([bvp_emb_feats, bvp_fft_feats, rsp_fft_feats], dim=1))

        if self.debug:
            print(" rBP.shape", rBP.shape)

        return rBP


class MMRPhysLNF(nn.Module):
    def __init__(self, frames, md_config, in_channels=4, dropout=0.2, device=torch.device("cpu"), debug=False):
        super(MMRPhysLNF, self).__init__()
        self.debug = debug
        self.in_channels = in_channels

        if self.in_channels == 4:
            self.rgb_norm = nn.InstanceNorm3d(3)
            self.thermal_norm = nn.InstanceNorm3d(1)
        else:
            print("Unsupported input channels")

        for key in model_config:
            if key not in md_config:
                md_config[key] = model_config[key]

        self.use_fsam = False
        self.md_infer = False

        if md_config["MD_FSAM"]:
            self.use_fsam = True
            self.md_infer = md_config["MD_INFERENCE"]
        else:
            pass
        
        self.tasks = md_config["TASKS"]
        if "BP" in self.tasks:
            self.wait_epochs_BP = md_config["Wait_Epochs"]

        if self.debug:
            print("nf_BVP:", nf_BVP)
            print("nf_RSP:", nf_RSP)

        self.bvp_feature_extractor = BVP_FeatureExtractor(inCh=3, dropout_rate=dropout, debug=debug)
        self.rsp_feature_extractor = RSP_FeatureExtractor(inCh=1, dropout_rate=dropout, debug=debug)
        
        self.rppg_head = BVP_Head(md_config=md_config, device=device, dropout_rate=dropout, debug=debug)
        self.rBr_head = RSP_Head(md_config=md_config, device=device, dropout_rate=dropout, debug=debug)

        if "BP" in self.tasks:
            self.rBP_head = BP_Estimation_Head(md_config=md_config, device=device, dropout_rate=dropout, debug=debug)


    def forward(self, x, label_bvp=None, label_rsp=None, epoch_count=-1): # [batch, Features=3, Temp=frames, Width=72, Height=72]
        
        [batch, channel, length, width, height] = x.shape        

        if self.debug:
            print("Input.shape", x.shape)

        x = torch.diff(x, dim=2)
        if self.debug:
            print("Diff Normalized shape", x.shape)

        rgb_x = self.rgb_norm(x[:, :3, :, :, :])
        thermal_x = self.thermal_norm(x[:, -1:, :, :, :])

        bvp_voxel_embeddings = self.bvp_feature_extractor(rgb_x)
        rsp_voxel_embeddings = self.rsp_feature_extractor(thermal_x)

        rPPG, factorized_embeddings_bvp, appx_error_bvp = self.rppg_head(length-1, bvp_embeddings=bvp_voxel_embeddings, label_bvp=label_bvp)
        rBr, factorized_embeddings_rsp, appx_error_rsp = self.rBr_head(length-1, rsp_embeddings=rsp_voxel_embeddings, label_rsp=label_rsp)

        if "BP" in self.tasks:
            if (self.training and (epoch_count == -1 or epoch_count > self.wait_epochs_BP)):        # -1 will make BP head learn from very beginning of the training
                rBP = self.rBP_head(bvp_embeddings=factorized_embeddings_bvp.detach(), bvp_vec=rPPG.detach(), rsp_vec=rBr.detach())
            elif not self.training:
                rBP = self.rBP_head(bvp_embeddings=factorized_embeddings_bvp.detach(), bvp_vec=rPPG.detach(), rsp_vec=rBr.detach())
            else:
                rBP = None
        else:
            rBP = None

        # if self.debug:
        #     print("rppg_feats.shape", rppg_feats.shape)

        # rPPG = rppg_feats.view(-1, length-1)

        if (self.training or self.debug or self.md_infer) and self.use_fsam:
            return_list = [rPPG, rBr, rBP, bvp_voxel_embeddings, rsp_voxel_embeddings, factorized_embeddings_bvp, appx_error_bvp, factorized_embeddings_rsp, appx_error_rsp]
        else:
            return_list = [rPPG, rBr, rBP, bvp_voxel_embeddings, rsp_voxel_embeddings]

        return return_list