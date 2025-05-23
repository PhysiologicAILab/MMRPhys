"""
Efficient and Robust Multidimensional Attention in Remote Physiological Sensing through Target Signal Constrained Factorization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_methods.model.MMRPhys.TSFM import FeaturesFactorizationModule
from neural_methods.model.MMRPhys.MMRPhysBP import BP_Estimation_Head
# from neural_methods.model.TNM import TNM
from copy import deepcopy

nf_BVP = [8, 12, 16]
nf_RSP = [16, 16, 16]

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


class ConvTransposeBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dilation=[1, 1, 1], bias=False, groups=1):
        super(ConvTransposeBlock3D, self).__init__()
        self.conv_trans_block_3d = nn.Sequential(
            nn.ConvTranspose3d(in_channel, out_channel, kernel_size, stride, padding=padding, bias=bias, dilation=dilation, groups=groups),
            nn.Tanh(),
            nn.InstanceNorm3d(out_channel),
        )

    def forward(self, x):
        return self.conv_trans_block_3d(x)



class RGB_BVP_FeatureExtractor(nn.Module):
    def __init__(self, inCh=3, dropout_rate=0.1, debug=False):
        super(RGB_BVP_FeatureExtractor, self).__init__()
        # inCh, out_channel, kernel_size, stride, padding

        self.debug = debug
        #                                                        Input: #B, inCh, T, 72, 72
        self.rgb_bvp_feature_extractor = nn.Sequential(
            ConvBlock3D(inCh, nf_BVP[0], [3, 3, 3], [1, 2, 2], [1, 0, 0]),      #B, nf_BVP[0], T, 35, 35
            ConvBlock3D(nf_BVP[0], nf_BVP[1], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf_BVP[1], T, 33, 33
            nn.Dropout3d(p=dropout_rate),

            ConvBlock3D(nf_BVP[1], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf_BVP[1], T, 31, 31
        )

    def forward(self, x):
        rgb_bvp_features = self.rgb_bvp_feature_extractor(x)
        if self.debug:
            print("RGB Feature Extractor")
            print("     rgb_bvp_features.shape", rgb_bvp_features.shape)
        return rgb_bvp_features



class Thermal_BVP_FeatureExtractor(nn.Module):
    def __init__(self, inCh=1, dropout_rate=0.1, debug=False):
        super(Thermal_BVP_FeatureExtractor, self).__init__()
        # inCh, out_channel, kernel_size, stride, padding

        self.debug = debug
        #                                                        Input: #B, inCh, T, 72, 72
        self.thermal_bvp_feature_extractor = nn.Sequential(
            ConvBlock3D(inCh, nf_BVP[0], [3, 3, 3], [1, 2, 2], [1, 0, 0]),      #B, nf_BVP[0], T, 35, 35
            ConvBlock3D(nf_BVP[0], nf_BVP[1], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf_BVP[1], T, 33, 33
            nn.Dropout3d(p=dropout_rate),

            ConvBlock3D(nf_BVP[1], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf_BVP[1], T, 31, 31
        )

    def forward(self, x):
        thermal_bvp_features = self.thermal_bvp_feature_extractor(x)
        if self.debug:
            print("Thermal Feature Extractor")
            print("     thermal_bvp_features.shape", thermal_bvp_features.shape)
        return thermal_bvp_features



class BVP_Head(nn.Module):
    def __init__(self, md_config, device, dropout_rate=0.1, debug=False):
        super(BVP_Head, self).__init__()
        self.debug = debug

        self.conv_layer_rgb = nn.Sequential(
            ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 2, 2], [1, 0, 0]),     #B, nf_BVP[2], T, 15, 15
            ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]),     #B, nf_BVP[2], T, 13, 13
            ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]),     #B, nf_BVP[2], T, 11, 11
            nn.Dropout3d(p=dropout_rate),
        )

        self.conv_layer_thermal = nn.Sequential(
            ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 2, 2], [1, 0, 0]),     #B, nf_BVP[2], T, 15, 15
            ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]),     #B, nf_BVP[2], T, 13, 13
            ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]),     #B, nf_BVP[2], T, 11, 11
            nn.Dropout3d(p=dropout_rate),
        )

        self.use_fsam = md_config["MD_FSAM"]
        self.md_type = md_config["MD_TYPE"]
        self.md_infer = md_config["MD_INFERENCE"]
        self.md_res = md_config["MD_RESIDUAL"]

        inC = 2 * nf_BVP[2]
        if self.use_fsam:
            self.fsam = FeaturesFactorizationModule(inC, device, md_config, dim="3D", debug=debug)
            self.fsam_norm = nn.InstanceNorm3d(inC)
            self.bias1 = nn.Parameter(torch.tensor(1.0), requires_grad=True).to(device)

        self.final_layer = nn.Sequential(
            ConvBlock3D(inC, nf_BVP[2], [3, 3, 3], [1, 2, 2], [1, 0, 0]),                         #B, nf_BVP[2], T, 5, 5
            ConvBlock3D(nf_BVP[2], nf_BVP[0], [3, 3, 3], [1, 1, 1], [1, 0, 0]),                   #B, nf_BVP[0], T, 3, 3
            nn.Conv3d(nf_BVP[0], 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),  #B, 1, T, 1, 1
        )

    def forward(self, length, rgb_embeddings=None, thermal_embeddings=None, label_bvp=None):

        rgb_embeddings = self.conv_layer_rgb(rgb_embeddings)
        thermal_embeddings = self.conv_layer_thermal(thermal_embeddings)
        voxel_embeddings = torch.concat([rgb_embeddings, thermal_embeddings], dim=1)

        if self.debug:
            print("BVP Head")
            print("     voxel_embeddings.shape", voxel_embeddings.shape)

        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            if "NMF" in self.md_type:
                att_mask, appx_error = self.fsam(voxel_embeddings - voxel_embeddings.min(), label_bvp) # to make it positive (>= 0)
            else:
                att_mask, appx_error = self.fsam(voxel_embeddings, label_bvp)

            if self.debug:
                print("att_mask.shape", att_mask.shape)

            # Multiplication with Residual connection
            x = torch.mul(voxel_embeddings - voxel_embeddings.min() + self.bias1, att_mask - att_mask.min() + self.bias1)
            factorized_embeddings = self.fsam_norm(x)
            factorized_embeddings = voxel_embeddings + factorized_embeddings

            x = self.final_layer(factorized_embeddings)

        else:
            appx_error = 0
            factorized_embeddings = None
            x = self.final_layer(voxel_embeddings)

        if self.debug:
            print("voxel_embeddings.shape", voxel_embeddings.shape)
            print("x.shape", x.shape)

        rPPG = x.view(-1, length)

        if self.debug:
            print("     rPPG.shape", rPPG.shape)
        
        return rPPG, factorized_embeddings, appx_error


class RGB_RSP_FeatureExtractor(nn.Module):
    def __init__(self, inCh=3, dropout_rate=0.1, debug=False):
        super(RGB_RSP_FeatureExtractor, self).__init__()
        # inCh, out_channel, kernel_size, stride, padding

        self.debug = debug
        #                                                                                     Input: #B, inCh, T//1,      72, 72
        self.rgb_rsp_feature_extractor = nn.Sequential(
            ConvBlock3D(inCh, nf_RSP[0], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),       #B, nf_RSP[0], T//1, 70, 70
            ConvBlock3D(nf_RSP[0], nf_RSP[1], [3, 3, 3], [2, 2, 2], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[1], T//2, 34, 34
            ConvBlock3D(nf_RSP[1], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[2], T//2, 32, 32
            nn.Dropout3d(p=dropout_rate),
        )

    def forward(self, x):
        rgb_rsp_features = self.rgb_rsp_feature_extractor(x)
        if self.debug:
            print("RGB RSP Feature Extractor")
            print("     rgb_rsp_features.shape", rgb_rsp_features.shape)
        return rgb_rsp_features


class Thermal_RSP_FeatureExtractor(nn.Module):
    def __init__(self, inCh=1, dropout_rate=0.1, debug=False):
        super(Thermal_RSP_FeatureExtractor, self).__init__()
        # inCh, out_channel, kernel_size, stride, padding

        self.debug = debug
        #                                                                                     Input: #B, inCh, T//1,      72, 72
        self.thermal_rsp_feature_extractor = nn.Sequential(
            ConvBlock3D(inCh, nf_RSP[0], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),       #B, nf_RSP[0], T//1, 70, 70
            ConvBlock3D(nf_RSP[0], nf_RSP[1], [3, 3, 3], [2, 2, 2], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[1], T//2, 34, 34
            ConvBlock3D(nf_RSP[1], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[2], T//2, 32, 32
            nn.Dropout3d(p=dropout_rate),
        )

    def forward(self, x):
        thermal_rsp_features = self.thermal_rsp_feature_extractor(x)
        if self.debug:
            print("Thermal Feature Extractor")
            print("     thermal_rsp_features.shape", thermal_rsp_features.shape)
        return thermal_rsp_features


class RSP_Head(nn.Module):
    def __init__(self, md_config, device, dropout_rate=0.1, debug=False):
        super(RSP_Head, self).__init__()
        self.debug = debug
        self.temporal_scale_factor = 4

        self.conv_block_rgb = nn.Sequential(
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[2], T//2, 30, 30
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [2, 2, 2], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[2], T//4, 14, 14
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[2], T//4, 12, 12
            nn.Dropout3d(p=dropout_rate),
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[2], T//4, 10, 10
        )

        self.conv_block_thermal = nn.Sequential(
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[2], T//2, 30, 30
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [2, 2, 2], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[2], T//4, 14, 14
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[2], T//4, 12, 12
            nn.Dropout3d(p=dropout_rate),
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[2], T//4, 10, 10
        )

        self.use_fsam = md_config["MD_FSAM"]
        self.md_type = md_config["MD_TYPE"]
        self.md_infer = md_config["MD_INFERENCE"]
        self.md_res = md_config["MD_RESIDUAL"]

        md_config = deepcopy(md_config)
        md_config["MD_R"] = 4
        # md_config["MD_S"] = 1
        # md_config["MD_STEPS"] = 3
        md_config["align_channels"] = nf_RSP[2] // 2

        inC = 2 * nf_RSP[2]
        if self.use_fsam:
            self.fsam = FeaturesFactorizationModule(inC, device, md_config, dim="3D", debug=debug)
            self.fsam_norm = nn.InstanceNorm3d(inC)
            self.bias1 = nn.Parameter(torch.tensor(1.0), requires_grad=True).to(device)

        self.final_layer = nn.Sequential(
            ConvBlock3D(inC, nf_RSP[2], [3, 3, 3], [1, 2, 2], [1, 0, 0], dilation=[1, 1, 1]),       #B, nf_RSP[2], T//4, 4, 4
            nn.Upsample(scale_factor=(self.temporal_scale_factor, 1, 1)),                           #B, nf_RSP[2], T//1, 4, 4
            nn.Conv3d(nf_RSP[2], 1, (3, 4, 4), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),    #B, 1, T//1, 1, 1
        )

    def forward(self, length, rgb_embeddings=None, thermal_embeddings=None, label_rsp=None):

        rgb_embeddings = self.conv_block_rgb(rgb_embeddings)
        thermal_embeddings = self.conv_block_thermal(thermal_embeddings)
        voxel_embeddings = torch.concat([rgb_embeddings, thermal_embeddings], dim=1)

        if self.debug:
            print("RSP Head")
            print("     voxel_embeddings.shape", voxel_embeddings.shape)

        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            label_rsp_down_sampled = F.avg_pool1d(label_rsp, kernel_size=5, stride=4, padding=2)
            if "NMF" in self.md_type:
                att_mask, appx_error = self.fsam(voxel_embeddings - voxel_embeddings.min(), label_rsp_down_sampled) # to make it positive (>= 0)
            else:
                att_mask, appx_error = self.fsam(voxel_embeddings, label_rsp_down_sampled)

            if self.debug:
                print("att_mask.shape", att_mask.shape)

            # Multiplication with Residual connection
            factorized_embeddings = torch.mul(voxel_embeddings - voxel_embeddings.min() + self.bias1, att_mask - att_mask.min() + self.bias1)
            factorized_embeddings = self.fsam_norm(factorized_embeddings)
            factorized_embeddings = voxel_embeddings + factorized_embeddings

            x = self.final_layer(factorized_embeddings)
        
        else:
            appx_error = 0
            factorized_embeddings = None
            x = self.final_layer(voxel_embeddings)

        if self.debug:
            print("voxel_embeddings.shape", voxel_embeddings.shape)
            print("x.shape", x.shape)

        rBr = x.view(-1, length)

        if self.debug:
            print("     rBr.shape", rBr.shape)
        
        return rBr, factorized_embeddings, appx_error


class MMRPhysLLF(nn.Module):
    def __init__(self, frames, md_config, in_channels=4, dropout=0.2, device=torch.device("cpu"), debug=False):
        super(MMRPhysLLF, self).__init__()
        self.debug = debug
        self.in_channels = in_channels

        if self.in_channels == 4:
            self.rgb_norm = nn.InstanceNorm3d(3)
            self.thermal_norm = nn.InstanceNorm3d(1)
        else:
            print("Unsupported input channels")
            exit()

        # No significant gains were observed when using TNM instead of InstanceNorm3D for rPPG estimation, excent high SNR.
        # rRSP estimation was impacted - as trend-removal may remove actual signal - which is slow-varying signal
        # if self.in_channels == 4:
        #     self.rgb_norm = TNM()
        #     self.thermal_norm = TNM()
        # else:
        #     print("Unsupported input channels")
        #     exit()

        for key in model_config:
            if key not in md_config:
                md_config[key] = model_config[key]

        if md_config["MD_FSAM"]:
            self.use_fsam = True
            self.md_infer = md_config["MD_INFERENCE"]
        else:
            self.use_fsam = False
            self.md_infer = False
        
        self.tasks = md_config["TASKS"]

        if self.debug:
            print("nf_BVP:", nf_BVP)
            print("nf_RSP:", nf_RSP)

        self.thermal_bvp_feature_extractor = Thermal_BVP_FeatureExtractor(1, dropout_rate=dropout, debug=self.debug)
        self.rgb_bvp_feature_extractor = RGB_BVP_FeatureExtractor(3, dropout_rate=dropout, debug=self.debug)
        self.rppg_head = BVP_Head(md_config=md_config, device=device, dropout_rate=dropout, debug=self.debug)
        
        self.thermal_rsp_feature_extractor = Thermal_RSP_FeatureExtractor(1, dropout_rate=dropout, debug=self.debug)
        self.rgb_rsp_feature_extractor = RGB_RSP_FeatureExtractor(3, dropout_rate=dropout, debug=self.debug)
        self.rBr_head = RSP_Head(md_config=md_config, device=device, dropout_rate=dropout, debug=self.debug)

        # if "BP" in self.tasks:
        self.rBP_head = BP_Estimation_Head(md_config=md_config, device=device, dropout_rate=dropout, debug=self.debug)

    def forward(self, x, label_bvp=None, label_rsp=None): # [batch, Features=3, Temp=frames, Width=72, Height=72]

        [batch, channel, length, width, height] = x.shape        

        if self.debug:
            print("Input.shape", x.shape)

        x = torch.diff(x, dim=2)
        if self.debug:
            print("Diff Normalized shape", x.shape)

        rgb_x = self.rgb_norm(x[:, :3, :, :, :])
        thermal_x = self.thermal_norm(x[:, -1:, :, :, :])

        if "BVP" in self.tasks:
            rgb_bvp_voxel_embeddings = self.rgb_bvp_feature_extractor(rgb_x)
            thermal_bvp_voxel_embeddings = self.thermal_bvp_feature_extractor(thermal_x)
        elif "BP" in self.tasks:
            with torch.no_grad():
                rgb_bvp_voxel_embeddings = self.rgb_bvp_feature_extractor(rgb_x)
                thermal_bvp_voxel_embeddings = self.thermal_bvp_feature_extractor(thermal_x)
        else:
            rgb_bvp_voxel_embeddings = thermal_bvp_voxel_embeddings = None

        if "RSP" in self.tasks:
            rgb_rsp_voxel_embeddings = self.rgb_rsp_feature_extractor(rgb_x)
            thermal_rsp_voxel_embeddings = self.thermal_rsp_feature_extractor(thermal_x)
        elif "BP" in self.tasks:
            with torch.no_grad():
                rgb_rsp_voxel_embeddings = self.rgb_rsp_feature_extractor(rgb_x)
                thermal_rsp_voxel_embeddings = self.thermal_rsp_feature_extractor(thermal_x)
        else:
            rgb_rsp_voxel_embeddings = thermal_rsp_voxel_embeddings = None

        if "BVP" in self.tasks:
            rPPG, factorized_embeddings_bvp, appx_error_bvp = self.rppg_head(length-1, rgb_embeddings=rgb_bvp_voxel_embeddings, thermal_embeddings=thermal_bvp_voxel_embeddings, label_bvp=label_bvp)
        elif "BP" in self.tasks:
            with torch.no_grad():
                rPPG, factorized_embeddings_bvp, appx_error_bvp = self.rppg_head(length-1, rgb_embeddings=rgb_bvp_voxel_embeddings, thermal_embeddings=thermal_bvp_voxel_embeddings, label_bvp=label_bvp)
        else:
            rPPG = factorized_embeddings_bvp = appx_error_bvp = None

        if "RSP" in self.tasks:
            rBr, factorized_embeddings_rsp, appx_error_rsp = self.rBr_head(length-1, rgb_embeddings=rgb_rsp_voxel_embeddings, thermal_embeddings=thermal_rsp_voxel_embeddings, label_rsp=label_rsp)
        elif "BP" in self.tasks:
            with torch.no_grad():
                rBr, factorized_embeddings_rsp, appx_error_rsp = self.rBr_head(length-1, rgb_embeddings=rgb_rsp_voxel_embeddings, thermal_embeddings=thermal_rsp_voxel_embeddings, label_rsp=label_rsp)
        else:
            rBr = factorized_embeddings_rsp = appx_error_rsp = None

        if "BP" in self.tasks:
            rBP = self.rBP_head(bvp_embeddings=rgb_bvp_voxel_embeddings.detach(), bvp_vec=rPPG.detach(), rsp_vec=rBr.detach())
        else:
            rBP = None

        return_list = [rPPG, rBr, rBP, rgb_bvp_voxel_embeddings, thermal_rsp_voxel_embeddings, factorized_embeddings_bvp, appx_error_bvp, factorized_embeddings_rsp, appx_error_rsp]

        return return_list