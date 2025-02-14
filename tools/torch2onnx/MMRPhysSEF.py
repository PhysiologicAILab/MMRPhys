"""
MMRPhys: Remote Extraction of Multiple Physiological Signals
"""

import torch
import torch.nn as nn

nf_BVP = [8, 12, 16]
nf_RSP = [16, 16, 16]

model_config = {
    "TASKS": ["BVP", "RSP"],
    "FS": 30,
    "in_channels": 3,
    "data_channels": 4,
    "height": 9,
    "weight": 9,
    "batch_size": 1,
    "frames": 300,
    "debug": False,
}


class ConvBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dilation=[1, 1, 1], bias=False, groups=1):
        super(ConvBlock3D, self).__init__()
        # Create individual layers instead of Sequential
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size, stride,
                              padding=padding, bias=bias, dilation=dilation, groups=groups)
        self.tanh = nn.Tanh()
        self.norm = nn.InstanceNorm3d(
            out_channel, track_running_stats=False, affine=True)

        # Initialize in eval mode
        self.train(False)
        self.norm.train(False)

    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)
        with torch.no_grad():
            x = self.norm(x)
        return x

class BVP_FeatureExtractor(nn.Module):
    def __init__(self, inCh, dropout_rate=0.1, debug=False):
        super(BVP_FeatureExtractor, self).__init__()
        # inCh, out_channel, kernel_size, stride, padding

        self.debug = debug
        #                                                        Input: #B, inCh, T, 9, 9
        self.bvp_feature_extractor = nn.Sequential(
            ConvBlock3D(inCh, nf_BVP[0], [3, 3, 3], [1, 1, 1], [1, 1, 1]),      #B, nf_BVP[0], T, 9, 9
            ConvBlock3D(nf_BVP[0], nf_BVP[1], [3, 3, 3], [1, 1, 1], [1, 1, 1]), #B, nf_BVP[1], T, 9, 9
            nn.Dropout3d(p=dropout_rate),

            ConvBlock3D(nf_BVP[1], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 1, 1]), #B, nf_BVP[1], T, 9, 9
        )

    def forward(self, x):
        bvp_features = self.bvp_feature_extractor(x)
        if self.debug:
            print("BVP Feature Extractor")
            print("     bvp_features.shape", bvp_features.shape)
        return bvp_features


class BVP_Head(nn.Module):
    def __init__(self, dropout_rate=0.1, debug=False):
        super(BVP_Head, self).__init__()
        self.debug = debug

        self.conv_layer = nn.Sequential(
            ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 1, 1]),     #B, nf_BVP[2], T, 9, 9
            ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 1, 1]),     #B, nf_BVP[2], T, 9, 9
            ConvBlock3D(nf_BVP[2], nf_BVP[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]),     #B, nf_BVP[2], T, 7, 7
            nn.Dropout3d(p=dropout_rate),
        )
        inC = nf_BVP[2]

        self.final_layer = nn.Sequential(
            ConvBlock3D(inC, nf_BVP[1], [3, 3, 3], [1, 1, 1], [1, 0, 0]),                         #B, nf_BVP[1], T, 5, 5
            ConvBlock3D(nf_BVP[1], nf_BVP[0], [3, 3, 3], [1, 1, 1], [1, 0, 0]),                   #B, nf_BVP[0], T, 3, 3
            nn.Conv3d(nf_BVP[0], 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),  #B, 1, T, 1, 1
        )

    def forward(self, length, bvp_embeddings=None):

        bvp_embeddings = self.conv_layer(bvp_embeddings)
        x = self.final_layer(bvp_embeddings)

        rPPG = x.view(-1, length)

        return rPPG


class RSP_FeatureExtractor(nn.Module):
    def __init__(self, inCh=1, dropout_rate=0.1, debug=False):
        super(RSP_FeatureExtractor, self).__init__()
        # inCh, out_channel, kernel_size, stride, padding

        self.debug = debug
        #                                                                                     Input: #B, inCh,      T//1, 9, 9
        self.rsp_feature_extractor = nn.Sequential(
            ConvBlock3D(inCh, nf_RSP[0], [3, 3, 3], [1, 1, 1], [1, 1, 1], dilation=[1, 1, 1]),       #B, nf_RSP[0], T//1, 9, 9
            ConvBlock3D(nf_RSP[0], nf_RSP[1], [3, 3, 3], [2, 1, 1], [1, 1, 1], dilation=[1, 1, 1]),  #B, nf_RSP[1], T//2, 9, 9
            ConvBlock3D(nf_RSP[1], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 1, 1], dilation=[1, 1, 1]),  #B, nf_RSP[2], T//2, 9, 9
            nn.Dropout3d(p=dropout_rate),
        )

    def forward(self, x):
        thermal_rsp_features = self.rsp_feature_extractor(x)
        return thermal_rsp_features


class RSP_Head(nn.Module):
    def __init__(self, dropout_rate=0.1, debug=False):
        super(RSP_Head, self).__init__()
        self.debug = debug
        self.temporal_scale_factor = 4

        self.conv_block = nn.Sequential(
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 1, 1], dilation=[1, 1, 1]),  #B, nf_RSP[2], T//2, 9, 9
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [2, 1, 1], [1, 1, 1], dilation=[1, 1, 1]),  #B, nf_RSP[2], T//4, 9, 9
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 1, 1], dilation=[1, 1, 1]),  #B, nf_RSP[2], T//4, 9, 9
            nn.Dropout3d(p=dropout_rate),
            ConvBlock3D(nf_RSP[2], nf_RSP[2], [3, 3, 3], [1, 1, 1], [1, 1, 1], dilation=[1, 1, 1]),  #B, nf_RSP[2], T//4, 9, 9
        )

        inC = nf_RSP[2]

        self.final_layer = nn.Sequential(
            ConvBlock3D(inC, nf_RSP[1], [3, 3, 3], [1, 2, 2], [1, 0, 0], dilation=[1, 1, 1]),       #B, nf_RSP[1], T//4, 4, 4
            nn.Upsample(scale_factor=(self.temporal_scale_factor, 1, 1)),                           #B, nf_RSP[2], T//1, 4, 4
            nn.Conv3d(nf_RSP[0], 1, (3, 4, 4), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),    #B, 1, T//1, 1, 1
        )

    def forward(self, length, rsp_embeddings=None):

        voxel_embeddings = self.conv_block(rsp_embeddings)
        x = self.final_layer(voxel_embeddings)
        rBr = x.view(-1, length)
        
        return rBr


class MMRPhysSEF(nn.Module):
    def __init__(self, frames, md_config, in_channels=4, dropout=0.2, device=torch.device("cpu"), debug=False):
        super(MMRPhysSEF, self).__init__()
        self.debug = debug
        self.in_channels = in_channels
        self.num_frames = frames

        self.rgb_norm = nn.InstanceNorm3d(3)    
        self.tasks = md_config["TASKS"]

        if "BVP" in self.tasks or "BP" in self.tasks:
            self.bvp_feature_extractor = BVP_FeatureExtractor(inCh=self.in_channels, dropout_rate=dropout, debug=debug)
        if "RSP" in self.tasks or "BP" in self.tasks:
            self.rsp_feature_extractor = RSP_FeatureExtractor(inCh=self.in_channels, dropout_rate=dropout, debug=debug)
        
        if "BVP" in self.tasks:
            self.rppg_head = BVP_Head(dropout_rate=dropout, debug=debug)

        if "RSP" in self.tasks:
            self.rBr_head = RSP_Head(dropout_rate=dropout, debug=debug)


    def forward(self, x): # [batch, Features=3, Temp=frames, Width=9, Height=9]

        # [batch, channel, length, width, height] = x.shape

        x = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        # x = torch.diff(x, dim=2)    # Removes any aperiod variations, and also removes spatial facial features - which are not required to learn by the model

        x = self.rgb_norm(x)

        if "BVP" in self.tasks:
            bvp_voxel_embeddings = self.bvp_feature_extractor(x)
        else:
            bvp_voxel_embeddings = None

        if "RSP" in self.tasks:
            rsp_voxel_embeddings = self.rsp_feature_extractor(x)
        else:
            rsp_voxel_embeddings = None

        if "BVP" in self.tasks:
            rPPG = self.rppg_head(self.num_frames-1, bvp_embeddings=bvp_voxel_embeddings)
        else:
            rPPG = None

        if "RSP" in self.tasks:
            rBr = self.rBr_head(self.num_frames-1, rsp_embeddings=rsp_voxel_embeddings)
        else:
            rBr = None

        return_list = [rPPG, rBr]

        return return_list