"""
MMRPhys: Remote Extraction of Multiple Physiological Signals using Label Guided Factorization
"""

import torch
import torch.nn as nn
from neural_methods.model.MMRPhys.FSAM import FeaturesFactorizationModule

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
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dilation=[1, 1, 1], bias=False):
        super(ConvBlock3D, self).__init__()
        self.conv_block_3d = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding=padding, bias=bias, dilation=dilation),
            nn.Tanh(),
            nn.InstanceNorm3d(out_channel),
        )

    def forward(self, x):
        return self.conv_block_3d(x)


class BVP_FeatureExtractor(nn.Module):
    def __init__(self, inCh, dropout_rate=0.1, debug=False):
        super(BVP_FeatureExtractor, self).__init__()
        # inCh, out_channel, kernel_size, stride, padding

        self.debug = debug
        #                                                        Input: #B, inCh, T, 72, 72
        self.bvp_feature_extractor = nn.Sequential(
            ConvBlock3D(inCh, nf_BVP[0], [3, 3, 3], [1, 2, 2], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_BVP[0], T, 35, 35
            ConvBlock3D(nf_BVP[0], nf_BVP[1], [3, 3, 3], [1, 1, 1], [1, 1, 1], dilation=[1, 1, 1]), #B, nf_BVP[1], T, 35, 35
            nn.Dropout3d(p=dropout_rate),

            ConvBlock3D(nf_BVP[1], nf_BVP[1], [3, 3, 3], [1, 1, 1], [1, 1, 1], dilation=[1, 1, 1]), #B, nf_BVP[1], T, 35, 35
            ConvBlock3D(nf_BVP[1], nf_BVP[2], [3, 3, 3], [1, 2, 2], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_BVP[2], T, 17, 17
            ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_BVP[2], T, 15, 15
            nn.Dropout3d(p=dropout_rate),

            ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_BVP[2], T, 13, 13
            ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_BVP[2], T, 11, 11
            ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_BVP[2], T, 9, 9
            nn.Dropout3d(p=dropout_rate),

            ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_BVP[2], T, 7, 7
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

    def forward(self, length, voxel_embeddings=None, label_bvp=None):

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
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[2], T//4, 29, 29
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[2], T//4, 27, 27
            nn.Dropout3d(p=dropout_rate),

            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[2], T//4, 25, 25
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[2], T//4, 23, 23
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]),  #B, nf_RSP[2], T//4, 21, 21
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
        self.time_scale_factor = 4

        # self.downsample = nn.AvgPool1d(kernel_size=self.time_scale_factor)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(self.time_scale_factor, 1, 1)),
        )

        self.use_fsam = md_config["MD_FSAM"]
        self.md_type = md_config["MD_TYPE"]
        self.md_infer = md_config["MD_INFERENCE"]
        # self.md_res = md_config["MD_RESIDUAL"]  # retained only this default option = True

        # md_config = deepcopy(md_config)
        # md_config["MD_R"] = 4
        # md_config["MD_S"] = 1
        # md_config["MD_STEPS"] = 6

        if self.use_fsam:
            inC = nf_RSP[2]
            self.fsam = FeaturesFactorizationModule(inC, device, md_config, dim="3D", debug=debug)
            self.fsam_norm = nn.InstanceNorm3d(inC)
            self.bias1 = nn.Parameter(torch.tensor(1.0), requires_grad=True).to(device)

        self.final_layer = nn.Sequential(
            ConvBlock3D(nf_RSP[2], nf_RSP[1], [5, 3, 3], [1, 3, 3], [4, 0, 0], dilation=[2, 1, 1]),         #B, nf_RSP[2], T, 7, 7
            ConvBlock3D(nf_RSP[1], nf_RSP[0], [3, 3, 3], [1, 2, 2], [2, 0, 0], dilation=[2, 1, 1]),         #B, nf_RSP[1], T, 3, 3
            nn.Conv3d(nf_RSP[0], 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0), dilation=(1, 1, 1), bias=False),     #B, 1, T, 1, 1
        )

        # self.upsample = nn.Upsample(scale_factor=self.time_scale_factor)         #B, 1, T


    def forward(self, length, voxel_embeddings=None, label_rsp=None):

        voxel_embeddings = self.upsample(voxel_embeddings)
        bt, ch, t, h, w = voxel_embeddings.shape

        if self.debug:
            print("RSP Head")
            print("     voxel_embeddings.shape", voxel_embeddings.shape)

        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            # down_sampled_rsp = self.downsample(label_rsp)
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

            # factorized_embeddings = self.upsample(factorized_embeddings)
            x = self.final_layer(factorized_embeddings)
        else:

            # voxel_embeddings = self.upsample(voxel_embeddings)
            x = self.final_layer(voxel_embeddings)

        if self.debug:
            print("voxel_embeddings.shape", voxel_embeddings.shape)
            print("x.shape", x.shape)

        # rBr = x.view(bt, 1, length//4)
        # rBr = self.upsample(rBr)
        rBr = x.view(bt, length)

        if self.debug:
            print("     rBr.shape", rBr.shape)
        
        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            return rBr, factorized_embeddings, appx_error
        else:
            return rBr


class BP_Head_Phase(nn.Module):
    def __init__(self, dropout_rate=0.1, debug=False):
        super(BP_Head_Phase, self).__init__()
        self.debug = debug
        self.inCh = nf_BVP[2] + nf_RSP[2]
        self.outCh = (nf_BVP[2] + nf_RSP[2]) // 2

        self.spatial_pool = nn.AvgPool3d(kernel_size=[1, 7, 7], stride=[1, 1, 1], padding=[0, 0, 0])

        # self.final_layer = nn.Sequential(
        #     nn.Conv1d(self.inCh, self.outCh, 1),
        #     nn.Conv1d(self.outCh, 1, 1),
        # )

        self.final_dense_layer = nn.Sequential(
            nn.Linear(500, 16),
            nn.Dropout(0.2),
            nn.Linear(16, 2)
        )

    def forward(self, rppg_embeddings=None, rbr_embeddings=None):

        if self.debug:
            print(" BP Head")
            print(" rppg_embeddings.shape", rppg_embeddings.shape)
            print(" rbr_embeddings.shape", rbr_embeddings.shape)

        x = torch.concat([rppg_embeddings, rbr_embeddings], dim=1)
        x = self.spatial_pool(x)
        
        bt, ch, t, h, w = x.shape
        
        x = x.view(bt, ch, t)   #h = w = 1 is expected here

        x_fft = torch.zeros_like(x).to(x.device)
        for bn in range(bt):
            for cn in range(ch):
                x_fft[bn, cn, :] = torch.angle(torch.fft.fft(x[bn, cn, :]))

        x = self.final_layer(x)
        
        if self.debug:
            print(" x.shape", x.shape)
        
        x = x.view(x.size(0), -1)
        
        if self.debug:
            print(" Flattened x.shape", x.shape)

        rBP = self.final_dense_layer(x)

        if self.debug:
            print(" rBP.shape", rBP.shape)

        return rBP


class MMRPhysLEF(nn.Module):
    def __init__(self, frames, md_config, in_channels=4, dropout=0.2, device=torch.device("cpu"), debug=False):
        super(MMRPhysLEF, self).__init__()
        self.debug = debug
        self.in_channels = in_channels

        if self.in_channels == 4:
            self.rgb_norm = nn.InstanceNorm3d(3)
            self.thermal_norm = nn.InstanceNorm3d(1)
        elif self.in_channels == 3:
            self.rgb_norm = nn.InstanceNorm3d(3)
        elif self.in_channels == 1:
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

        if "BVP" in self.tasks or "BP" in self.tasks:
            self.bvp_feature_extractor = BVP_FeatureExtractor(inCh=self.in_channels, dropout_rate=dropout, debug=debug)
        if "RSP" in self.tasks or "BP" in self.tasks:
            self.rsp_feature_extractor = RSP_FeatureExtractor(inCh=self.in_channels, dropout_rate=dropout, debug=debug)
        
        if "BVP" in self.tasks:
            self.rppg_head = BVP_Head(md_config=md_config, device=device, dropout_rate=dropout, debug=debug)

        if "RSP" in self.tasks:
            self.rBr_head = RSP_Head(md_config=md_config, device=device, dropout_rate=dropout, debug=debug)

        if "BP" in self.tasks:
            self.rBP_head = BP_Head_Phase(dropout_rate=dropout, debug=debug)

    def forward(self, x, label_bvp=None, label_rsp=None, epoch_count=-1):  # [batch, Features=3, Temp=frames, Width=72, Height=72]

        [batch, channel, length, width, height] = x.shape        

        if self.debug:
            print("Input.shape", x.shape)

        x = torch.diff(x, dim=2)
        if self.in_channels == 4:
            rgb_x = self.rgb_norm(x[:, :3, :, :, :])
            thermal_x = self.thermal_norm(x[:, -1:, :, :, :])
            x = torch.concat([rgb_x, thermal_x], dim = 1)
        elif self.in_channels == 3:
            x = self.rgb_norm(x[:, :3, :, :, :])
        elif self.in_channels == 1:
            x = self.thermal_norm(x[:, -1:, :, :, :])
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

        if "BVP" in self.tasks or "BP" in self.tasks:
            bvp_voxel_embeddings = self.bvp_feature_extractor(x)
        else:
            rPPG = bvp_voxel_embeddings = factorized_embeddings_ppg = appx_error_ppg = None

        if "RSP" in self.tasks or "BP" in self.tasks:
            rsp_voxel_embeddings = self.rsp_feature_extractor(x)
        else:
            rBr = rsp_voxel_embeddings = factorized_embeddings_br = appx_error_br = None

        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            if "BVP" in self.tasks or "BP" in self.tasks:
                rPPG, factorized_embeddings_ppg, appx_error_ppg = self.rppg_head(length-1, voxel_embeddings=bvp_voxel_embeddings, label_bvp=label_bvp)

            if "RSP" in self.tasks or "BP" in self.tasks:
                rBr, factorized_embeddings_br, appx_error_br = self.rBr_head(length-1, voxel_embeddings=rsp_voxel_embeddings, label_rsp=label_rsp)
    
        else:
            if "BVP" in self.tasks or "BP" in self.tasks:
                rPPG = self.rppg_head(length-1, voxel_embeddings=bvp_voxel_embeddings)
    
            if "RSP" in self.tasks or "BP" in self.tasks:
                rBr = self.rBr_head(length-1, voxel_embeddings=rsp_voxel_embeddings)

        if "BP" in self.tasks:
            try:
                assert self.in_channels == 4
            except:
                print("For BP estimation, both RGB and thermal channels are required to compute BVP and RSP")
                print("Specified channels: ", self.in_channels)
                print("Exiting")
                exit()
            rBP = self.rBP_head(bvp_voxel_embeddings.detach(), rsp_voxel_embeddings.detach())
        else:
            rBP = None

        # if self.debug:
        #     print("rppg_feats.shape", rppg_feats.shape)

        # rPPG = rppg_feats.view(-1, length-1)

        if (self.training or self.debug or self.md_infer) and self.use_fsam:
            return_list = [rPPG, rBr, rBP, bvp_voxel_embeddings, rsp_voxel_embeddings, factorized_embeddings_ppg, appx_error_ppg, factorized_embeddings_br, appx_error_br]
        else:
            return_list = [rPPG, rBr, rBP, bvp_voxel_embeddings, rsp_voxel_embeddings]

        return return_list