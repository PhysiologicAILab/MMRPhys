"""
MMRPhys: Remote Extraction of Multiple Physiological Signals using Label Guided Factorization
"""

import torch
import torch.nn as nn
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
            ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf_BVP[2], T, 7, 7
        )

        self.use_fsam = md_config["MD_FSAM"]
        self.md_type = md_config["MD_TYPE"]
        self.md_infer = md_config["MD_INFERENCE"]
        self.md_res = md_config["MD_RESIDUAL"]

        if self.use_fsam:           
            self.fsam = FeaturesFactorizationModule(nf_BVP[2], device, md_config, dim="3D", debug=debug)
            self.fsam_norm = nn.InstanceNorm3d(nf_BVP[2])
            self.bias1 = nn.Parameter(torch.tensor(1.0), requires_grad=True).to(device)

        self.final_layer = nn.Sequential(
            ConvBlock3D(nf_BVP[2], nf_BVP[1], [3, 3, 3], [1, 1, 1], [1, 0, 0]),                         #B, nf_BVP[1], T, 5, 5
            ConvBlock3D(nf_BVP[1], nf_BVP[0], [3, 3, 3], [1, 1, 1], [1, 0, 0]),                   #B, nf_BVP[0], T, 3, 3
            nn.Conv3d(nf_BVP[0], 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),  #B, 1, T, 1, 1
        )

    def forward(self, length, bvp_embeddings=None, label_bvp=None):

        voxel_embeddings = self.conv_layer(bvp_embeddings)

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
            return rPPG, voxel_embeddings, appx_error


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

        self.temporal_scale_factor = 4

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(self.temporal_scale_factor, 1, 1)),       #B, nf_RSP[2], T, 21, 21
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
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [5, 3, 3], [1, 3, 3], [4, 0, 0], dilation=[2, 1, 1]),           #B, nf_RSP[2], T, 7, 7
            ConvBlock3D(nf_RSP[2], nf_RSP[1], [3, 3, 3], [1, 2, 2], [2, 0, 0], dilation=[2, 1, 1]),     #B, nf_RSP[1], T, 3, 3
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


class BP_Head_Phase(nn.Module):
    def __init__(self, dropout_rate=0.1, debug=False):
        super(BP_Head_Phase, self).__init__()
        self.debug = debug

        self.spatial_pool_bvp = nn.AvgPool3d(kernel_size=[1, 7, 7], stride=[1, 1, 1], padding=[0, 0, 0])

        self.final_dense_layer = nn.Sequential(
            nn.Linear(500, 16),
            nn.Dropout(0.2),
            nn.Linear(16, 2)
        )

    def forward(self, rppg_embeddings=None, rBr=None):

        if self.debug:
            print(" BP Head")
            print(" rppg_embeddings.shape", rppg_embeddings.shape)
            print(" rBr.shape", rBr.shape)

        x = self.spatial_pool_bvp(rppg_embeddings)
        
        bt, ch, t, h, w = x.shape
        nfft = t
        x = x.view(bt, ch, t)   #h = w = 1 is expected here

        x_fft = torch.zeros_like(x).to(x.device)
        for bn in range(bt):
            for cn in range(ch):

                ppg_fft = torch.fft.rfft(x[bn, cn, :])
                ppg_fft_angle = torch.angle(ppg_fft)
                ppg_fft_freq = 60 * 25 * torch.fft.rfftfreq(nfft)
                # ppg_fft_freq
                ppg_foi = torch.argwhere((ppg_fft_freq > 40) & (ppg_fft_freq < 200))


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

        self.bvp_feature_extractor = BVP_FeatureExtractor(inCh=3, dropout_rate=dropout, debug=debug)
        self.rsp_feature_extractor = RSP_FeatureExtractor(inCh=1, dropout_rate=dropout, debug=debug)
        
        self.rppg_head = BVP_Head(md_config=md_config, device=device, dropout_rate=dropout, debug=debug)
        self.rBr_head = RSP_Head(md_config=md_config, device=device, dropout_rate=dropout, debug=debug)

        if "BP" in self.tasks:
            self.rBP_head = BP_Head_Phase(dropout_rate=dropout, debug=debug)


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
            if (self.training and (epoch_count == -1 or epoch_count > 4)):        # -1 will make BP head learn from very beginning of the training
                rBP = self.rBP_head(factorized_embeddings_bvp.detach(), rBr.detach())
            elif not self.training:
                rBP = self.rBP_head(factorized_embeddings_bvp.detach(), rBr.detach())
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