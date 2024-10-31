"""
MMRPhys: Remote Extraction of Multiple Physiological Signals using Label Guided Factorization
"""

import torch
import torch.nn as nn
from neural_methods.model.MMRPhys.FSAM import FeaturesFactorizationModule
from copy import deepcopy

nf_BVP = [8, 12, 16]
nf_RSP = [8, 12, 16]

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
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dilation=[1, 1, 1], bias=False):
        super(ConvBlock3D, self).__init__()
        self.conv_block_3d = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding=padding, bias=bias, dilation=dilation),
            nn.Tanh(),
            nn.InstanceNorm3d(out_channel),
        )

    def forward(self, x):
        return self.conv_block_3d(x)


class RGB_FeatureExtractor(nn.Module):
    def __init__(self, inCh=3, dropout_rate=0.1, debug=False):
        super(RGB_FeatureExtractor, self).__init__()
        # inCh, out_channel, kernel_size, stride, padding

        self.debug = debug
        #                                                        Input: #B, inCh, T, 72, 72
        self.rgb_FeatureExtractor = nn.Sequential(
            ConvBlock3D(inCh, nf_BVP[0], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_BVP[0], T, 70, 70
            ConvBlock3D(nf_BVP[0], nf_BVP[1], [3, 3, 3], [1, 2, 2], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_BVP[1], T, 34, 34
            ConvBlock3D(nf_BVP[1], nf_BVP[1], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_BVP[1], T, 32, 32
            nn.Dropout3d(p=dropout_rate),
        )

    def forward(self, x):
        rgb_features = self.rgb_FeatureExtractor(x)
        if self.debug:
            print("RGB Feature Extractor")
            print("     rgb_features.shape", rgb_features.shape)
        return rgb_features


class rPPG_FeatureExtractor(nn.Module):
    def __init__(self, dropout_rate=0.1, debug=False):
        super(rPPG_FeatureExtractor, self).__init__()
        # nf_BVP[1], out_channel, kernel_size, stride, padding

        self.debug = debug
        #                                                        Input: #B, nf_BVP[1], T, 72, 72
        self.FeatureExtractor = nn.Sequential(
            ConvBlock3D(nf_BVP[1], nf_BVP[1], [3, 3, 3], [1, 2, 2], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_BVP[1], T, 15, 15
            ConvBlock3D(nf_BVP[1], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_BVP[2], T, 13, 13
            ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_BVP[2], T, 11, 11
            nn.Dropout3d(p=dropout_rate),

            ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_BVP[2], T, 9, 9
            ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_BVP[2], T, 7, 7
            nn.Dropout3d(p=dropout_rate),
        )

    def forward(self, x):
        voxel_embeddings = self.FeatureExtractor(x)
        if self.debug:
            print("rPPG Feature Extractor")
            print("     voxel_embeddings.shape", voxel_embeddings.shape)
        return voxel_embeddings


class BVP_Head(nn.Module):
    def __init__(self, md_config, device, dropout_rate=0.1, debug=False):
        super(BVP_Head, self).__init__()
        self.debug = debug

        self.use_fsam = md_config["MD_FSAM"]
        self.md_type = md_config["MD_TYPE"]
        self.md_infer = md_config["MD_INFERENCE"]
        self.md_res = md_config["MD_RESIDUAL"]

        if self.use_fsam:
            inC = nf_BVP[2]
            self.fsam = FeaturesFactorizationModule(inC, device, md_config, dim="3D", debug=debug)
            self.fsam_norm = nn.InstanceNorm3d(inC)
            self.bias1 = nn.Parameter(torch.tensor(1.0), requires_grad=True).to(device)
        else:
            inC = nf_BVP[2]

        self.final_layer = nn.Sequential(
            ConvBlock3D(inC, nf_BVP[1], [3, 3, 3], [1, 1, 1], [1, 0, 0]),                         #B, nf_BVP[1], T, 5, 5
            ConvBlock3D(nf_BVP[1], nf_BVP[0], [3, 3, 3], [1, 1, 1], [1, 0, 0]),                   #B, nf_BVP[0], T, 3, 3
            nn.Conv3d(nf_BVP[0], 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),  #B, 1, T, 1, 1
        )


    def forward(self, voxel_embeddings, batch, length, label_bvp=None):

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

            if self.md_res:
                # Multiplication with Residual connection
                x = torch.mul(voxel_embeddings - voxel_embeddings.min() + self.bias1, att_mask - att_mask.min() + self.bias1)
                factorized_embeddings = self.fsam_norm(x)
                factorized_embeddings = voxel_embeddings + factorized_embeddings
            else:
                # Multiplication
                x = torch.mul(voxel_embeddings - voxel_embeddings.min() + self.bias1, att_mask - att_mask.min() + self.bias1)
                factorized_embeddings = self.fsam_norm(x)            

            x = self.final_layer(factorized_embeddings)

        else:
            x = self.final_layer(voxel_embeddings)

        if self.debug:
            print("voxel_embeddings.shape", voxel_embeddings.shape)
            print("x.shape", x.shape)

        rPPG = x.view(-1, length)

        if self.debug:
            print("     rPPG.shape", rPPG.shape)
        
        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            return rPPG, factorized_embeddings, appx_error
        else:
            return rPPG


class Thermal_FeatureExtractor(nn.Module):
    def __init__(self, inCh=1, dropout_rate=0.1, debug=False):
        super(Thermal_FeatureExtractor, self).__init__()
        # inCh, out_channel, kernel_size, stride, padding

        self.debug = debug
        #                                                        Input: #B, inCh, T, 72, 72
        self.thermal_FeatureExtractor = nn.Sequential(
            ConvBlock3D(inCh, nf_RSP[0], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[0], T, 70, 70
            ConvBlock3D(nf_RSP[0], nf_RSP[1], [3, 3, 3], [1, 2, 2], [2, 0, 0], dilation=[2, 1, 1]), #B, nf_RSP[1], T, 34, 34
            ConvBlock3D(nf_RSP[1], nf_RSP[1], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_RSP[1], T, 32, 32
            nn.Dropout3d(p=dropout_rate),
        )

    def forward(self, x):
        thermal_features = self.thermal_FeatureExtractor(x)
        if self.debug:
            print("Thermal Feature Extractor")
            print("     thermal_features.shape", thermal_features.shape)
        return thermal_features


class rBr_FeatureExtractor(nn.Module):
    def __init__(self, dropout_rate=0.1, debug=False):
        super(rBr_FeatureExtractor, self).__init__()
        # nf_RSP[1], out_channel, kernel_size, stride, padding

        self.debug = debug
        #                                                        Input: #B, nf_RSP[1], T, 72, 72
        self.FeatureExtractor = nn.Sequential(
            ConvBlock3D(nf_RSP[1], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_RSP[2], T, 30, 30
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_RSP[2], T, 28, 28
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_RSP[2], T, 26, 26
            nn.Dropout3d(p=dropout_rate),

            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_RSP[2], T, 24, 24
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_RSP[2], T, 22, 22
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_RSP[2], T, 20, 20
            nn.Dropout3d(p=dropout_rate),

            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 2, 2], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_RSP[2], T, 9, 9
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_RSP[2], T, 7, 7
            nn.Dropout3d(p=dropout_rate),
        )

    def forward(self, x):
        voxel_embeddings = self.FeatureExtractor(x)
        if self.debug:
            print("rBr Feature Extractor")
            print("     voxel_embeddings.shape", voxel_embeddings.shape)
        return voxel_embeddings


class RSP_Head(nn.Module):
    def __init__(self, md_config, device, dropout_rate=0.1, debug=False):
        super(RSP_Head, self).__init__()
        self.debug = debug

        self.use_fsam = md_config["MD_FSAM"]
        self.md_type = md_config["MD_TYPE"]
        self.md_infer = md_config["MD_INFERENCE"]
        self.md_res = md_config["MD_RESIDUAL"]

        # md_config = deepcopy(md_config)
        # md_config["MD_R"] = 4
        # md_config["MD_S"] = 1
        # md_config["MD_STEPS"] = 6

        if self.use_fsam:
            inC = nf_RSP[2]
            self.fsam = FeaturesFactorizationModule(inC, device, md_config, dim="3D", debug=debug)
            self.fsam_norm = nn.InstanceNorm3d(inC)
            self.bias1 = nn.Parameter(torch.tensor(1.0), requires_grad=True).to(device)
        else:
            inC = nf_RSP[2]

        # self.upsample = nn.Upsample(scale_factor=(6, 1, 1))

        self.final_layer = nn.Sequential(
            ConvBlock3D(inC, nf_RSP[1], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),        #B, nf_RSP[1], T, 5, 5
            ConvBlock3D(nf_RSP[1], nf_RSP[0], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[0], T, 3, 3
            nn.Conv3d(nf_RSP[0], 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),     #B, 1, T, 1, 1
        )

    def forward(self, voxel_embeddings, batch, length, label_rsp=None):

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

            # # directly use att_mask   ---> difficult to converge without Residual connection. Needs high rank
            # factorized_embeddings = self.fsam_norm(att_mask)

            # # Residual connection: 
            # factorized_embeddings = voxel_embeddings + self.fsam_norm(att_mask)

            if self.md_res:
                # Multiplication with Residual connection
                x = torch.mul(voxel_embeddings - voxel_embeddings.min() + self.bias1, att_mask - att_mask.min() + self.bias1)
                factorized_embeddings = self.fsam_norm(x)
                factorized_embeddings = voxel_embeddings + factorized_embeddings
            else:
                # Multiplication
                x = torch.mul(voxel_embeddings - voxel_embeddings.min() + self.bias1, att_mask - att_mask.min() + self.bias1)
                factorized_embeddings = self.fsam_norm(x)            

            # # Concatenate
            # factorized_embeddings = torch.cat([voxel_embeddings, self.fsam_norm(x)], dim=1)
            # factorized_embeddings = self.upsample(factorized_embeddings)
            x = self.final_layer(factorized_embeddings)
        
        else:
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
            return rBr


class BP_Head(nn.Module):
    def __init__(self, dropout_rate=0.1, debug=False):
        super(BP_Head, self).__init__()
        self.debug = debug

        self.conv_block = nn.Sequential(
            ConvBlock3D(nf_BVP[2] + nf_RSP[2], nf_BVP[2], [1, 3, 3], [2, 1, 1], [0, 0, 0], dilation=[1, 1, 1], bias=True), #B, nf_BVP[2], T//2, 5, 5
            ConvBlock3D(nf_BVP[2], nf_BVP[2], [1, 3, 3], [2, 1, 1], [0, 0, 0], dilation=[1, 1, 1], bias=True), #B, nf_BVP[2], T//4, 3, 3
            ConvBlock3D(nf_BVP[2], nf_BVP[2], [1, 3, 3], [5, 1, 1], [0, 0, 0], dilation=[1, 1, 1], bias=True), #B, nf_BVP[2], T//20, 1, 1
        )

        self.final_dense_layer = nn.Sequential(
            nn.Linear(nf_BVP[2]*25, nf_BVP[2]),
            nn.Dropout(0.2),
            nn.Linear(nf_BVP[2], 2)
        )

    def forward(self, rppg_embeddings, rbr_embeddings, batch, length):

        if self.debug:
            print(" BP Head")
            print(" rppg_embeddings.shape", rppg_embeddings.shape)
            print(" rbr_embeddings.shape", rbr_embeddings.shape)

        x = torch.concat([rppg_embeddings, rbr_embeddings], dim=1)
        x = self.conv_block(x)
        
        if self.debug:
            print(" x.shape", x.shape)
        
        x = x.view(x.size(0), -1)
        
        if self.debug:
            print(" Flattened x.shape", x.shape)

        rBP = self.final_dense_layer(x)

        if self.debug:
            print(" rBP.shape", rBP.shape)

        return rBP


class BP_Head_Phase(nn.Module):
    def __init__(self, dropout_rate=0.1, debug=False):
        super(BP_Head_Phase, self).__init__()
        self.debug = debug
        self.inCh = nf_BVP[2] + nf_RSP[2]
        self.outCh = (nf_BVP[2] + nf_RSP[2]) // 2
        self.conv_block = nn.Sequential(
            ConvBlock3D(self.inCh, self.inCh, [1, 7, 7], [1, 1, 1], [0, 0, 0], dilation=[1, 1, 1], bias=True), #B, nf_BVP[2], T, 1, 1
        )

        self.final_conv_block = nn.Sequential(
            nn.Conv1d(self.inCh, self.outCh, 1),
            nn.Conv1d(self.outCh, 1, 1),
        )

        self.final_dense_layer = nn.Sequential(
            nn.Linear(500, 16),
            nn.Dropout(0.2),
            nn.Linear(16, 2)
        )

    def forward(self, rppg_embeddings, rbr_embeddings, batch, length):

        if self.debug:
            print(" BP Head")
            print(" rppg_embeddings.shape", rppg_embeddings.shape)
            print(" rbr_embeddings.shape", rbr_embeddings.shape)

        x = torch.concat([rppg_embeddings, rbr_embeddings], dim=1)
        x = self.conv_block(x)
        
        bt, ch, t, h, w = x.shape
        
        x = x.view(bt, ch, t)   #h = w = 1 is expected here

        x_fft = torch.zeros_like(x).to(x.device)
        for bn in range(bt):
            for cn in range(ch):
                x_fft[bn, cn, :] = torch.angle(torch.fft.fft(x[bn, cn, :]))

        x = self.final_conv_block(x)
        
        if self.debug:
            print(" x.shape", x.shape)
        
        x = x.view(x.size(0), -1)
        
        if self.debug:
            print(" Flattened x.shape", x.shape)

        rBP = self.final_dense_layer(x)

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

        if self.debug:
            print("nf_BVP:", nf_BVP)
            print("nf_RSP:", nf_RSP)

        self.rgb_feature_extractor = RGB_FeatureExtractor(3, dropout_rate=dropout, debug=debug)
        self.thermal_feature_extractor = Thermal_FeatureExtractor(1, dropout_rate=dropout, debug=debug)

        self.rppg_feature_extractor = rPPG_FeatureExtractor(dropout_rate=dropout, debug=debug)
        self.rppg_head = BVP_Head(md_config, device=device, dropout_rate=dropout, debug=debug)

        self.rbr_feature_extractor = rBr_FeatureExtractor(dropout_rate=dropout, debug=debug)        
        self.rBr_head = RSP_Head(md_config, device=device, dropout_rate=dropout, debug=debug)

        if "BP" in self.tasks:
            # self.rBP_head = BP_Head(dropout_rate=dropout, debug=debug)
            self.rBP_head = BP_Head_Phase(dropout_rate=dropout, debug=debug)


    def forward(self, x, label_bvp=None, label_rsp=None): # [batch, Features=3, Temp=frames, Width=72, Height=72]
        
        [batch, channel, length, width, height] = x.shape        

        if self.debug:
            print("Input.shape", x.shape)

        if self.in_channels == 1:
            x = torch.diff(x, dim=2)
            x = self.norm(x[:, -1:, :, :, :])
            # x = self.norm(x[:, -1:, :-1, :, :])   #if no diff used, then discard the last added frame
        elif self.in_channels == 3:
            x = torch.diff(x, dim=2)
            x = self.norm(x[:, :3, :, :, :])
        elif self.in_channels == 4:
            x = torch.diff(x, dim=2)
            rgb_x = self.rgb_norm(x[:, :3, :, :, :])
            thermal_x = self.thermal_norm(x[:, -1:, :, :, :])
            # x = torch.concat([rgb_x, thermal_x], dim = 1)
        else:
            try:
                print("Specified input channels:", self.in_channels)
                print("Data channels", channel)
                assert self.in_channels <= channel
            except:
                print("Incorrectly preprocessed data provided as input. Number of channels exceed the specified or default channels")
                print("Default or specified channels:", self.in_channels)
                print("Data channels [B, C, N, W, H]", x.shape)
                print("Exiting")
                exit()

        if self.debug:
            print("Diff Normalized shape", x.shape)

        rgb_x = self.rgb_feature_extractor(rgb_x)
        rppg_voxel_embeddings = self.rppg_feature_extractor(rgb_x)

        thermal_x = self.thermal_feature_extractor(thermal_x)
        rbr_voxel_embeddings = self.rbr_feature_extractor(thermal_x)

        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            rPPG, factorized_embeddings_ppg, appx_error_ppg = self.rppg_head(rppg_voxel_embeddings, batch, length-1, label_bvp)
            rBr, factorized_embeddings_br, appx_error_br = self.rBr_head(rbr_voxel_embeddings, batch, length-1, label_rsp)
        else:
            rPPG = self.rppg_head(rppg_voxel_embeddings, batch, length-1)
            rBr = self.rBr_head(rbr_voxel_embeddings, batch, length-1)

        if "BP" in self.tasks:
            rBP = self.rBP_head(rppg_voxel_embeddings.detach(), rbr_voxel_embeddings.detach(), batch, length-1)
        else:
            rBP = None

        # if self.debug:
        #     print("rppg_feats.shape", rppg_feats.shape)

        # rPPG = rppg_feats.view(-1, length-1)

        if (self.training or self.debug or self.md_infer) and self.use_fsam:
            return_list = [rPPG, rBr, rBP, rppg_voxel_embeddings, rbr_voxel_embeddings, factorized_embeddings_ppg, appx_error_ppg, factorized_embeddings_br, appx_error_br]
        else:
            return_list = [rPPG, rBr, rBP, rppg_voxel_embeddings, rbr_voxel_embeddings]

        return return_list