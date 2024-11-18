"""
MMRPhys: Remote Extraction of Multiple Physiological Signals using Label Guided Factorization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_methods.model.MMRPhys.FSAM import FeaturesFactorizationModule
from neural_methods.model.MMRPhys.MMRPhysBP import BP_Estimation_Head
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


class BVP_FeatureExtractor(nn.Module):
    def __init__(self, inCh, dropout_rate=0.1, debug=False):
        super(BVP_FeatureExtractor, self).__init__()
        # inCh, out_channel, kernel_size, stride, padding

        self.debug = debug
        #                                                        Input: #B, inCh, T, 72, 72
        self.bvp_feature_extractor = nn.Sequential(
            ConvBlock3D(inCh, nf_BVP[0], [3, 3, 3], [1, 2, 2], [1, 1, 1]),      #B, nf_BVP[0], T, 36, 36
            ConvBlock3D(nf_BVP[0], nf_BVP[1], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf_BVP[1], T, 34, 34
            nn.Dropout3d(p=dropout_rate),

            ConvBlock3D(nf_BVP[1], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf_BVP[1], T, 32, 32
            ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 2, 2], [1, 0, 0]), #B, nf_BVP[2], T, 15, 15
            ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf_BVP[2], T, 13, 13
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

        self.conv_layer = nn.Sequential(
            nn.Dropout3d(p=dropout_rate),
            ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]),     #B, nf_BVP[2], T, 11, 11
        )

        self.use_fsam = md_config["MD_FSAM"]
        self.md_type = md_config["MD_TYPE"]
        self.md_infer = md_config["MD_INFERENCE"]
        self.md_res = md_config["MD_RESIDUAL"]

        # if self.use_fsam:
        self.fsam = FeaturesFactorizationModule(nf_BVP[2], device, md_config, dim="3D", debug=debug)
        self.fsam_norm = nn.InstanceNorm3d(nf_BVP[2])
        self.bias1 = nn.Parameter(torch.tensor(1.0), requires_grad=True).to(device)

        self.final_layer = nn.Sequential(
            ConvBlock3D(nf_BVP[2], nf_BVP[1], [3, 3, 3], [1, 2, 2], [1, 0, 0]),     #B, nf_BVP[1], T, 5, 5
            ConvBlock3D(nf_BVP[1], nf_BVP[0], [3, 3, 3], [1, 1, 1], [1, 0, 0]),     #B, nf_BVP[0], T, 3, 3
            nn.Conv3d(nf_BVP[0], 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),  #B, 1, T, 1, 1
        )

    def forward(self, length, bvp_embeddings=None, label_bvp=None):

        bvp_embeddings = self.conv_layer(bvp_embeddings)

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
            factorized_embeddings = None
            x = self.final_layer(bvp_embeddings)

        if self.debug:
            print("bvp_embeddings.shape", bvp_embeddings.shape)
            print("x.shape", x.shape)

        rPPG = x.view(-1, length)

        if self.debug:
            print("     rPPG.shape", rPPG.shape)
        
        return rPPG, factorized_embeddings, appx_error


class RSP_FeatureExtractor(nn.Module):
    def __init__(self, inCh=1, dropout_rate=0.1, debug=False):
        super(RSP_FeatureExtractor, self).__init__()
        # inCh, out_channel, kernel_size, stride, padding

        self.debug = debug
        #                                                                                     Input: #B, inCh, T, 72, 72
        self.rsp_feature_extractor = nn.Sequential(
            ConvBlock3D(inCh, nf_RSP[0], [3, 3, 3], [2, 1, 1], [1, 1, 1], dilation=[1, 1, 1]),       #B, nf_RSP[0], T//2, 72, 72
            ConvBlock3D(nf_RSP[0], nf_RSP[1], [3, 3, 3], [2, 2, 2], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[1], T//4, 36, 36
            ConvBlock3D(nf_RSP[1], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[2], T//4, 34, 34
            nn.Dropout3d(p=dropout_rate),

            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[2], T//4, 32, 32
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 2, 2], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[2], T//4, 15, 15
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
        self.temporal_scale_factor = 2

        self.conv_block = nn.Sequential(
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]),     #B, nf_RSP[2], T//4, 13, 13
            nn.Upsample(scale_factor=(self.temporal_scale_factor, 1, 1)),           #B, nf_RSP[2], T//2, 13, 13
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]),     #B, nf_RSP[2], T//2, 11, 11
            nn.Upsample(scale_factor=(self.temporal_scale_factor, 1, 1)),           #B, nf_RSP[2], T, 11, 11
        )

        self.use_fsam = md_config["MD_FSAM"]
        self.md_type = md_config["MD_TYPE"]
        self.md_infer = md_config["MD_INFERENCE"]
        self.md_res = md_config["MD_RESIDUAL"]

        # md_config = deepcopy(md_config)
        # md_config["MD_R"] = 16
        # md_config["MD_S"] = 1
        # md_config["MD_STEPS"] = 8

        # if self.use_fsam:
        self.fsam = FeaturesFactorizationModule(nf_RSP[2], device, md_config, dim="3D", debug=debug)
        self.fsam_norm = nn.InstanceNorm3d(nf_RSP[2])
        self.bias1 = nn.Parameter(torch.tensor(1.0), requires_grad=True).to(device)

        self.final_layer = nn.Sequential(
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 2, 2], [1, 0, 0], dilation=[1, 1, 1]),     #B, nf_RSP[2], T, 5, 5
            ConvBlock3D(nf_RSP[2], nf_RSP[0], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),     #B, nf_RSP[2], T, 3, 3
            nn.Conv3d(nf_RSP[0], 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),        #B, 1, T, 1, 1
        )

    def forward(self, length, rsp_embeddings=None, label_rsp=None):

        voxel_embeddings = self.conv_block(rsp_embeddings)

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
            factorized_embeddings = None
            x = self.final_layer(voxel_embeddings)

        if self.debug:
            print("voxel_embeddings.shape", voxel_embeddings.shape)
            print("x.shape", x.shape)

        rBr = x.view(-1, length)

        if self.debug:
            print("     rBr.shape", rBr.shape)
        
        return rBr, factorized_embeddings, appx_error


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
            exit()

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

        self.bvp_feature_extractor = BVP_FeatureExtractor(inCh=3, dropout_rate=dropout, debug=self.debug)
        self.rsp_feature_extractor = RSP_FeatureExtractor(inCh=1, dropout_rate=dropout, debug=self.debug)
        
        self.rppg_head = BVP_Head(md_config=md_config, device=device, dropout_rate=dropout, debug=self.debug)
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
            bvp_voxel_embeddings = self.bvp_feature_extractor(rgb_x)
        elif "BP" in self.tasks:
            with torch.no_grad():
                bvp_voxel_embeddings = self.bvp_feature_extractor(rgb_x)
        else:
            bvp_voxel_embeddings = None

        if "RSP" in self.tasks:
            rsp_voxel_embeddings = self.rsp_feature_extractor(thermal_x)
        elif "BP" in self.tasks:
            with torch.no_grad():
                rsp_voxel_embeddings = self.rsp_feature_extractor(thermal_x)
        else:
            rsp_voxel_embeddings = None

        if "BVP" in self.tasks:
            rPPG, factorized_embeddings_bvp, appx_error_bvp = self.rppg_head(length-1, bvp_embeddings=bvp_voxel_embeddings, label_bvp=label_bvp)
        elif "BP" in self.tasks:
            with torch.no_grad():
                rPPG, factorized_embeddings_bvp, appx_error_bvp = self.rppg_head(length-1, bvp_embeddings=bvp_voxel_embeddings, label_bvp=label_bvp)
        else:
            rPPG = factorized_embeddings_bvp = appx_error_bvp = None

        if "RSP" in self.tasks:
            rBr, factorized_embeddings_rsp, appx_error_rsp = self.rBr_head(length-1, rsp_embeddings=rsp_voxel_embeddings, label_rsp=label_rsp)
        elif "BP" in self.tasks:
            with torch.no_grad():
                rBr, factorized_embeddings_rsp, appx_error_rsp = self.rBr_head(length-1, rsp_embeddings=rsp_voxel_embeddings, label_rsp=label_rsp)
        else:
            rBr = factorized_embeddings_rsp = appx_error_rsp = None

        if "BP" in self.tasks:
            rBP = self.rBP_head(bvp_embeddings=bvp_voxel_embeddings.detach(), bvp_vec=rPPG.detach(), rsp_vec=rBr.detach())
        else:
            rBP = None

        return_list = [rPPG, rBr, rBP, bvp_voxel_embeddings, rsp_voxel_embeddings, factorized_embeddings_bvp, appx_error_bvp, factorized_embeddings_rsp, appx_error_rsp]

        return return_list