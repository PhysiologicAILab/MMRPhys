"""
MMRPhys: Remote Extraction of Multiple Physiological Signals using Label Guided Factorization
"""

import torch
import torch.nn as nn
from neural_methods.model.MMRPhys.FSAM import FeaturesFactorizationModule
# from copy import deepcopy

nf = [8, 16, 16]

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
    "align_channels": nf[2] // 2,
    "height": 36,
    "weight": 36,
    "batch_size": 4,
    "frames": 500,
    "debug": False,
    "assess_latency": False,
    "num_trials": 20,
    "visualize": False,
    "ckpt_path": "",
    "data_path": "",
    "label_path": ""
}


class ConvBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dilation=[1,1,1], bias=False):
        super(ConvBlock3D, self).__init__()
        self.conv_block_3d = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding=padding, bias=bias, dilation=dilation),
            nn.Tanh(),
            nn.InstanceNorm3d(out_channel),
        )

    def forward(self, x):
        return self.conv_block_3d(x)



class Base_FeatureExtractor(nn.Module):
    def __init__(self, inCh, dropout_rate=0.1, debug=False):
        super(Base_FeatureExtractor, self).__init__()
        # inCh, outCh, kernel_size, stride, padding

        self.debug = debug
        #                                                        Input: #B, inCh, T, 36, 36
        self.FeatureExtractor = nn.Sequential(
            ConvBlock3D(inCh, nf[0], [3, 3, 3], [1, 1, 1], [1, 1, 1], dilation=[1, 1, 1]),  #B, nf[0], T, 36, 36
            ConvBlock3D(nf[0], nf[1], [3, 3, 3], [1, 1, 1], [1, 1, 1], dilation=[1, 1, 1]), #B, nf[1], T, 36, 36
            ConvBlock3D(nf[1], nf[1], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf[1], T, 34, 34
            nn.Dropout3d(p=dropout_rate),
        )

    def forward(self, x):
        voxel_embeddings = self.FeatureExtractor(x)
        if self.debug:
            print("Base Feature Extractor")
            print("     voxel_embeddings.shape", voxel_embeddings.shape)
        return voxel_embeddings


class RGBTFeatureFusion_Fast(nn.Module):
    def __init__(self, dropout_rate=0.1, debug=False):
        super(RGBTFeatureFusion_Fast, self).__init__()

        self.debug = debug
        #                                                        Input: #B, 2*nf[1], T, 34, 34
        self.FeatureFusion = nn.Sequential(
            ConvBlock3D(2*nf[1], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf[2], T, 32, 32
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 2, 2], [1, 0, 0], dilation=[1, 1, 1]), #B, nf[2], T, 15, 15
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf[2], T, 13, 13
            nn.Dropout3d(p=dropout_rate),

        )
        #                                                        Input: #B, nf[1], T, 34, 34
        self.ConvBlock = nn.Sequential(
            ConvBlock3D(nf[1], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf[2], T, 32, 32
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 2, 2], [1, 0, 0], dilation=[1, 1, 1]), #B, nf[2], T, 15, 15
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf[2], T, 13, 13
            nn.Dropout3d(p=dropout_rate),
        )

    def forward(self, rgb=None, thermal=None):
        if rgb != None and thermal != None:
            x = torch.concat([rgb, thermal], dim=1)
            fused_embeddings = self.FeatureFusion(x)
        else:
            if rgb != None:
                x = rgb
            else:
                x = thermal
            fused_embeddings = self.ConvBlock(x)
        if self.debug:
            print("Feature Fusion Fast")
            print("     fused_embeddings.shape", fused_embeddings.shape)
        return fused_embeddings


class RGBTFeatureFusion_Slow(nn.Module):
    def __init__(self, dropout_rate=0.1, debug=False):
        super(RGBTFeatureFusion_Slow, self).__init__()

        self.debug = debug
        #                                                        Input: #B, 2*nf[1], T, 34, 34
        self.FeatureFusion = nn.Sequential(
            ConvBlock3D(2*nf[1], nf[2], [3, 3, 3], [1, 1, 1], [2, 0, 0], dilation=[2, 1, 1]), #B, nf[2], T, 32, 32
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 2, 2], [2, 0, 0], dilation=[2, 1, 1]), #B, nf[2], T, 15, 15
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf[2], T, 13, 13
            nn.Dropout3d(p=dropout_rate),
        )
        #                                                        Input: #B, nf[1], T, 34, 34
        self.ConvBlock = nn.Sequential(
            ConvBlock3D(nf[1], nf[2], [3, 3, 3], [1, 1, 1], [2, 0, 0], dilation=[2, 1, 1]), #B, nf[2], T, 32, 32
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 2, 2], [2, 0, 0], dilation=[2, 1, 1]), #B, nf[2], T, 15, 15
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf[2], T, 13, 13
            nn.Dropout3d(p=dropout_rate),
        )

    def forward(self, rgb=None, thermal=None):
        if rgb != None and thermal != None:
            x = torch.concat([rgb, thermal], dim=1)
            fused_embeddings = self.FeatureFusion(x)
        else:
            if rgb != None:
                x = rgb
            else:
                x = thermal
            fused_embeddings = self.ConvBlock(x)
        if self.debug:
            print("Feature Fusion Fast")
            print("     fused_embeddings.shape", fused_embeddings.shape)
        return fused_embeddings


class BVP_Head(nn.Module):
    def __init__(self, md_config, device, dropout_rate=0.1, debug=False):
        super(BVP_Head, self).__init__()
        self.debug = debug

        self.use_fsam = md_config["MD_FSAM"]
        self.md_type = md_config["MD_TYPE"]
        self.md_infer = md_config["MD_INFERENCE"]
        self.md_res = md_config["MD_RESIDUAL"]

        self.conv_block = nn.Sequential(
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf[2], T, 11, 11
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf[2], T, 9, 9
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf[2], T, 7, 7
            nn.Dropout3d(p=dropout_rate),
        )

        if self.use_fsam:
            inC = nf[2]
            self.fsam = FeaturesFactorizationModule(inC, device, md_config, dim="3D", sig_type="BVP", debug=debug)
            self.fsam_norm = nn.InstanceNorm3d(inC)
            self.bias1 = nn.Parameter(torch.tensor(1.0), requires_grad=True).to(device)
        else:
            inC = nf[2]

        self.final_layer = nn.Sequential(
            ConvBlock3D(inC, nf[1], [3, 3, 3], [1, 1, 1], [1, 0, 0]),                         #B, nf[1], T, 5, 5
            ConvBlock3D(nf[1], nf[0], [3, 3, 3], [1, 1, 1], [1, 0, 0]),                       #B, nf[0], T, 3, 3
            nn.Conv3d(nf[0], 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),  #B, 1, T, 1, 1
        )


    def forward(self, voxel_embeddings, batch, length, label_bvp=None):

        if self.debug:
            print("BVP Head")
            print("     voxel_embeddings.shape", voxel_embeddings.shape)

        voxel_embeddings = self.conv_block(voxel_embeddings)

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

        rPPG = x.view(-1, length)

        if self.debug:
            print("     rPPG.shape", rPPG.shape)
        
        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            return rPPG, factorized_embeddings, appx_error
        else:
            return rPPG



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

        self.conv_block = nn.Sequential(
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [2, 0, 0], dilation=[2, 1, 1]), #B, nf[2], T, 11, 11
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf[2], T, 9, 9
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf[2], T, 7, 7
            nn.Dropout3d(p=dropout_rate),
        )

        if self.use_fsam:
            inC = nf[2]
            self.fsam = FeaturesFactorizationModule(inC, device, md_config, dim="3D", sig_type="RSP", debug=debug)
            self.fsam_norm = nn.InstanceNorm3d(inC)
            self.bias1 = nn.Parameter(torch.tensor(1.0), requires_grad=True).to(device)
        else:
            inC = nf[2]

        # self.upsample = nn.Upsample(scale_factor=(6, 1, 1))

        self.final_layer = nn.Sequential(
            ConvBlock3D(inC, nf[1], [3, 3, 3], [1, 1, 1], [1, 0, 0]),                         #B, nf[1], T, 5, 5
            ConvBlock3D(nf[1], nf[0], [3, 3, 3], [1, 1, 1], [1, 0, 0]),                       #B, nf[0], T, 3, 3
            nn.Conv3d(nf[0], 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),  #B, 1, T, 1, 1
        )

    def forward(self, voxel_embeddings, batch, length, label_rsp=None):

        if self.debug:
            print("RSP Head")
            print("     voxel_embeddings.shape", voxel_embeddings.shape)

        voxel_embeddings = self.conv_block(voxel_embeddings)

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
            if self.debug:
                print("voxel_embeddings.shape", voxel_embeddings.shape)
            # voxel_embeddings = self.upsample(voxel_embeddings)
            if self.debug:
                print("voxel_embeddings.shape", voxel_embeddings.shape)
            x = self.final_layer(voxel_embeddings)
            if self.debug:
                print("x.shape", x.shape)

        rBr = x.view(-1, length)

        if self.debug:
            print("     rBr.shape", rBr.shape)
        
        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            return rBr, factorized_embeddings, appx_error
        else:
            return rBr



class BP_Head(nn.Module):
    def __init__(self, md_config, device, dropout_rate=0.1, debug=False):
        super(BP_Head, self).__init__()
        self.debug = debug

        self.use_fsam = md_config["MD_FSAM"]
        self.md_type = md_config["MD_TYPE"]
        self.md_infer = md_config["MD_INFERENCE"]
        self.md_res = md_config["MD_RESIDUAL"]

        self.conv_block = nn.Sequential(
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1], bias=True), #B, nf[2], T, 11, 11
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1], bias=True), #B, nf[2], T, 9, 9
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1], bias=True), #B, nf[2], T, 7, 7
            nn.Dropout3d(p=dropout_rate),
        )

        if self.use_fsam:
            inC = nf[2]
            self.fsam = FeaturesFactorizationModule(inC, device, md_config, dim="3D", sig_type="BVP", debug=debug)
            self.fsam_norm = nn.InstanceNorm3d(inC)
            self.bias1 = nn.Parameter(torch.tensor(1.0), requires_grad=True).to(device)
        else:
            inC = nf[2]

        self.final_layer = nn.Sequential(
            ConvBlock3D(inC, nf[1], [3, 3, 3], [1, 1, 1], [1, 0, 0], bias=True),                         #B, nf[1], T, 5, 5
            ConvBlock3D(nf[1], nf[0], [3, 3, 3], [1, 1, 1], [1, 0, 0], bias=True),                       #B, nf[0], T, 3, 3
            nn.Conv3d(nf[0], 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0), bias=True),  #B, 1, T, 1, 1
        )


    def forward(self, voxel_embeddings, batch, length, label_bp=None):

        if self.debug:
            print("BP Head")
            print("     voxel_embeddings.shape", voxel_embeddings.shape)

        voxel_embeddings = self.conv_block(voxel_embeddings)

        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            if "NMF" in self.md_type:
                att_mask, appx_error = self.fsam(voxel_embeddings - voxel_embeddings.min(), label_bp) # to make it positive (>= 0)
            else:
                att_mask, appx_error = self.fsam(voxel_embeddings, label_bp)

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

        rBP = x.view(-1, length)

        if self.debug:
            print("     rBP.shape", rBP.shape)
        
        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            return rBP, factorized_embeddings, appx_error
        else:
            return rBP



class EDA_Head(nn.Module):
    def __init__(self, md_config, device, dropout_rate=0.1, debug=False):
        super(EDA_Head, self).__init__()
        self.debug = debug

        self.use_fsam = md_config["MD_FSAM"]
        self.md_type = md_config["MD_TYPE"]
        self.md_infer = md_config["MD_INFERENCE"]
        self.md_res = md_config["MD_RESIDUAL"]

        # md_config = deepcopy(md_config)
        # md_config["MD_R"] = 4
        # md_config["MD_S"] = 1
        # md_config["MD_STEPS"] = 6

        self.conv_block = nn.Sequential(
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [2, 0, 0], dilation=[2, 1, 1]), #B, nf[2], T, 11, 11
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf[2], T, 9, 9
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], dilation=[1, 1, 1]), #B, nf[2], T, 7, 7
            nn.Dropout3d(p=dropout_rate),
        )

        if self.use_fsam:
            inC = nf[2]
            self.fsam = FeaturesFactorizationModule(inC, device, md_config, dim="3D", sig_type="RSP", debug=debug)
            self.fsam_norm = nn.InstanceNorm3d(inC)
            self.bias1 = nn.Parameter(torch.tensor(1.0), requires_grad=True).to(device)
        else:
            inC = nf[2]

        # self.upsample = nn.Upsample(scale_factor=(6, 1, 1))

        self.final_layer = nn.Sequential(
            ConvBlock3D(inC, nf[1], [3, 3, 3], [1, 1, 1], [1, 0, 0], bias=True),                         #B, nf[1], T, 5, 5
            ConvBlock3D(nf[1], nf[0], [3, 3, 3], [1, 1, 1], [1, 0, 0], bias=True),                       #B, nf[0], T, 3, 3
            nn.Conv3d(nf[0], 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0), bias=True),  #B, 1, T, 1, 1
        )

    def forward(self, voxel_embeddings, batch, length, label_eda=None):

        if self.debug:
            print("EDA Head")
            print("     voxel_embeddings.shape", voxel_embeddings.shape)

        voxel_embeddings = self.conv_block(voxel_embeddings)

        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            if "NMF" in self.md_type:
                att_mask, appx_error = self.fsam(voxel_embeddings - voxel_embeddings.min(), label_eda) # to make it positive (>= 0)
            else:
                att_mask, appx_error = self.fsam(voxel_embeddings, label_eda)

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
            if self.debug:
                print("voxel_embeddings.shape", voxel_embeddings.shape)
            # voxel_embeddings = self.upsample(voxel_embeddings)
            if self.debug:
                print("voxel_embeddings.shape", voxel_embeddings.shape)
            x = self.final_layer(voxel_embeddings)
            if self.debug:
                print("x.shape", x.shape)

        rEDA = x.view(-1, length)

        if self.debug:
            print("     rEDA.shape", rEDA.shape)
        
        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            return rEDA, factorized_embeddings, appx_error
        else:
            return rEDA


class MMRPhysFuseM(nn.Module):
    def __init__(self, frames, md_config, in_channels=3, dropout=0.2, device=torch.device("cpu"), debug=False):
        super(MMRPhysFuseM, self).__init__()
        self.debug = debug

        self.in_channels = in_channels
        if self.in_channels == 1 or self.in_channels == 3:
            self.norm = nn.InstanceNorm3d(self.in_channels)
        elif self.in_channels == 4:
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
            print("nf:", nf)

        self.base_feature_extractor_rgb = Base_FeatureExtractor(3, dropout_rate=dropout, debug=debug)
        self.base_feature_extractor_t = Base_FeatureExtractor(1, dropout_rate=dropout, debug=debug)

        if "BVP" in self.tasks:
            self.rppg_feature_fusion = RGBTFeatureFusion_Fast(dropout_rate=dropout, debug=debug)
            self.rppg_head = BVP_Head(md_config, device=device, dropout_rate=dropout, debug=debug)

        if "RSP" in self.tasks:
            self.rbr_feature_fusion = RGBTFeatureFusion_Slow(dropout_rate=dropout, debug=debug)
            self.rBr_head = RSP_Head(md_config, device=device, dropout_rate=dropout, debug=debug)

        if "BP" in self.tasks:
            self.rbp_feature_fusion = RGBTFeatureFusion_Fast(dropout_rate=dropout, debug=debug)
            self.rbp_head = BP_Head(md_config, device=device, dropout_rate=dropout, debug=debug)


        if "EDA" in self.tasks:
            self.rEDA_feature_fusion = RGBTFeatureFusion_Slow(dropout_rate=dropout, debug=debug)        
            self.rEDA_head = EDA_Head(md_config, device=device, dropout_rate=dropout, debug=debug)

    def forward(self, x, label_bvp=None, label_rsp=None, label_bp=None, label_eda=None):  # [batch, Features=4, Temp=frames, Width=36, Height=36]

        [batch, channel, length, width, height] = x.shape        

        if self.debug:
            print("Input.shape", x.shape)

        if self.in_channels == 1:
            thermal_x = torch.diff(x, dim=2)
            thermal_x = self.norm(thermal_x[:, -1:, :, :, :])
            # thermal_x = self.norm(thermal_x[:, -1:, :-1, :, :])   #if no diff used, then discard the last added frame
        elif self.in_channels == 3:
            rgb_x = torch.diff(x, dim=2)
            rgb_x = self.norm(rgb_x[:, :3, :, :, :])
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

        if self.in_channels == 3 or self.in_channels == 4:
            base_embeddings_rgb = self.base_feature_extractor_rgb(rgb_x)
        
        if self.in_channels == 1 or self.in_channels == 4:
            base_embeddings_t = self.base_feature_extractor_t(thermal_x)

        if "BVP" in self.tasks:
            if self.in_channels == 4:
                rppg_voxel_embeddings = self.rppg_feature_fusion(rgb=base_embeddings_rgb, thermal=base_embeddings_t)
            elif self.in_channels == 3:
                rppg_voxel_embeddings = self.rppg_feature_fusion(rgb=base_embeddings_rgb)
            elif self.in_channels == 1:
                rppg_voxel_embeddings = self.rppg_feature_fusion(thermal=base_embeddings_t)

        if "RSP" in self.tasks:
            if self.in_channels == 4:
                rbr_voxel_embeddings = self.rbr_feature_fusion(rgb=base_embeddings_rgb, thermal=base_embeddings_t)
            elif self.in_channels == 3:
                rbr_voxel_embeddings = self.rbr_feature_fusion(rgb=base_embeddings_rgb)
            elif self.in_channels == 1:
                rbr_voxel_embeddings = self.rbr_feature_fusion(thermal=base_embeddings_t)

        if "BP" in self.tasks:
            if self.in_channels == 4:
                rbp_voxel_embeddings = self.rbp_feature_fusion(rgb=base_embeddings_rgb, thermal=base_embeddings_t)
            elif self.in_channels == 3:
                rbp_voxel_embeddings = self.rbp_feature_fusion(rgb=base_embeddings_rgb)
            elif self.in_channels == 1:
                rbp_voxel_embeddings = self.rbp_feature_fusion(thermal=base_embeddings_t)

        if "EDA" in self.tasks:
            if self.in_channels == 4:
                rEDA_voxel_embeddings = self.rEDA_feature_fusion(rgb=base_embeddings_rgb, thermal=base_embeddings_t)
            elif self.in_channels == 3:
                rEDA_voxel_embeddings = self.rEDA_feature_fusion(rgb=base_embeddings_rgb)
            elif self.in_channels == 1:
                rEDA_voxel_embeddings = self.rEDA_feature_fusion(thermal=base_embeddings_t)

        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            if "BVP" in self.tasks:
                rPPG, factorized_embeddings_ppg, appx_error_ppg = self.rppg_head(rppg_voxel_embeddings, batch, length-1, label_bvp)
            else:
                rPPG = None
                rppg_voxel_embeddings = None
                factorized_embeddings_ppg = None
                appx_error_ppg = None

            if "BP" in self.tasks:
                rBP, factorized_embeddings_bp, appx_error_bp = self.rbp_head(rbp_voxel_embeddings, batch, length-1, label_bp)
            else:
                rBP = None
                rbp_voxel_embeddings = None
                factorized_embeddings_bp = None
                appx_error_bp = None

            if "RSP" in self.tasks:
                rBr, factorized_embeddings_br, appx_error_br = self.rBr_head(rbr_voxel_embeddings, batch, length-1, label_rsp)
            else:
                rBr = None
                rbr_voxel_embeddings = None
                factorized_embeddings_br = None
                appx_error_br = None

            if "EDA" in self.tasks:
                rEDA, factorized_embeddings_eda, appx_error_eda = self.rEDA_head(rEDA_voxel_embeddings, batch, length-1, label_eda)
            else:
                rEDA = None
                rEDA_voxel_embeddings = None
                factorized_embeddings_eda = None
                appx_error_eda = None

        else:
            if "BVP" in self.tasks:
                rPPG = self.rppg_head(rppg_voxel_embeddings, batch, length-1)
            else:
                rPPG = None
                rppg_voxel_embeddings = None

            if "BP" in self.tasks:
                rBP = self.rbp_head(rbp_voxel_embeddings, batch, length-1)
            else:
                rBP = None
                rbp_voxel_embeddings = None

            if "RSP" in self.tasks:
                rBr = self.rBr_head(rbr_voxel_embeddings, batch, length-1)
            else:
                rBr = None
                rbr_voxel_embeddings = None

            if "EDA" in self.tasks:
                rEDA = self.rEDA_head(rEDA_voxel_embeddings, batch, length-1)
            else:
                rEDA = None
                rEDA_voxel_embeddings = None

        # if self.debug:
        #     print("rppg_feats.shape", rppg_feats.shape)

        # rPPG = rppg_feats.view(-1, length-1)

        # if self.debug:
        #     if "BVP" in self.tasks:
        #         print("rPPG.shape", rPPG.shape)
        #     if "BP" in self.tasks:
        #         print("rBP.shape", rBP.shape)
        #     if "RSP" in self.tasks:
        #         print("rBr.shape", rBr.shape)
        #     if "EDA" in self.tasks:
        #         print("rEDA.shape", rEDA.shape)

        if (self.training or self.debug or self.md_infer) and self.use_fsam:
            return_list = [rPPG, rBr, rBP, rEDA, rppg_voxel_embeddings,
                           rbr_voxel_embeddings, rbp_voxel_embeddings, rEDA_voxel_embeddings,
                           factorized_embeddings_ppg, appx_error_ppg, factorized_embeddings_br, appx_error_br,
                           factorized_embeddings_bp, appx_error_bp, factorized_embeddings_eda, appx_error_eda]
        else:
            return_list = [rPPG, rBr, rBP, rEDA, rppg_voxel_embeddings,
                           rbr_voxel_embeddings, rbp_voxel_embeddings, rEDA_voxel_embeddings]
        
        return return_list
