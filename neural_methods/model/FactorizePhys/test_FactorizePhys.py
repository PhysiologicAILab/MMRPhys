"""
FactorizePhys: Effective Spatial-Temporal Attention in Remote Photo-plethysmography through Factorization of Voxel Embeddings
"""

import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import resample
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from neural_methods.model.FactorizePhys.FactorizePhys import FactorizePhys
# from ptflops import get_model_complexity_info
# from torch.utils.tensorboard import SummaryWriter

model_config = {
    "MD_FSAM": False,
    "MD_TYPE": "SNMF_LABEL",
    "MD_TRANSFORM": "T_KAB",
    "MD_R": 1,
    "MD_S": 1,
    "MD_STEPS": 4,
    "MD_INFERENCE": True,
    "MD_RESIDUAL": True,
    "in_channels": 3,
    "data_channels": 4,
    "height": 72,
    "weight": 72,
    "batch_size": 2,
    "frames": 160,
    "debug": True,
    "assess_latency": False,
    "num_trials": 20,
    "visualize": True,
    "ckpt_path": "./runs/exp/SCAMPS_Raw_160_72x72/PreTrainedModels/SCAMPS_SCAMPS_FactorizePhys_FSAM_Label_Epoch4.pth",
    "data_path": "data/SCAMPS/SCAMPS_Raw_160_72x72",
    "label_path": "data/SCAMPS/SCAMPS_Raw_160_72x72",
    "data_list": "data/SCAMPS/DataFileLists/SCAMPS_Raw_160_72x72_0.8_1.0.csv",
}

# ./runs/exp/SCAMPS_Raw_160_72x72/PreTrainedModels/SCAMPS_SCAMPS_FactorizePhys_FSAM_Label_Epoch4.pth
# ./runs/exp/SCAMPS_Raw_160_72x72/PreTrainedModels/SCAMPS_SCAMPS_FactorizePhys_SFSAM_Label_Epoch4.pth

class TestFactorizePhysBig(object):
    def __init__(self) -> None:
        self.ckpt_path = Path(model_config["ckpt_path"])
        self.data_path = Path(model_config["data_path"])
        self.label_path = Path(model_config["label_path"])

        self.use_fsam = model_config["MD_FSAM"]
        self.use_label = True if "label" in model_config["MD_TYPE"].lower() else False
        self.md_infer = model_config["MD_INFERENCE"]

        self.batch_size = model_config["batch_size"]
        self.frames = model_config["frames"]
        self.in_channels = model_config["in_channels"]
        self.data_channels = model_config["data_channels"]
        self.height = model_config["height"]
        self.width = model_config["weight"]
        self.debug = bool(model_config["debug"])
        self.assess_latency = bool(model_config["assess_latency"])
        self.visualize = model_config["visualize"]

        if self.visualize:
            # self.data_files = list(sorted(self.data_path.rglob("*input*.npy")))
            # self.label_files = list(sorted(self.data_path.rglob("*label*.npy")))

            self.data_files = pd.read_csv(model_config["data_list"])["input_files"].to_list()
            self.label_files = [d.replace("input", "label") for d in self.data_files]
            self.num_trials = len(self.data_files)

            self.plot_dir = Path.cwd().joinpath("plots").joinpath("inference")
            self.plot_dir.mkdir(parents=True, exist_ok=True)

            self.attention_map_dir = self.plot_dir.joinpath("attention_maps").joinpath(self.data_path.name).joinpath(self.ckpt_path.name)
            self.attention_map_dir.mkdir(parents=True, exist_ok=True)

        else:
            if self.assess_latency:
                self.num_trials = model_config["num_trials"]
            else:
                self.num_trials = 1

        if torch.cuda.is_available():
            self.device = torch.device(0)
        else:
            self.device = torch.device("cpu")

        md_config = {}
        md_config["FRAME_NUM"] = model_config["frames"]
        md_config["MD_S"] = model_config["MD_S"]
        md_config["MD_R"] = model_config["MD_R"]
        md_config["MD_STEPS"] = model_config["MD_STEPS"]
        md_config["MD_FSAM"] = model_config["MD_FSAM"]
        md_config["MD_TYPE"] = model_config["MD_TYPE"]
        md_config["MD_TRANSFORM"] = model_config["MD_TRANSFORM"]
        md_config["MD_INFERENCE"] = model_config["MD_INFERENCE"]
        md_config["MD_RESIDUAL"] = model_config["MD_RESIDUAL"]

        if self.visualize:
            self.net = nn.DataParallel(FactorizePhys(frames=self.frames, md_config=md_config,
                                device=self.device, in_channels=self.in_channels, debug=self.debug), device_ids=[0]).to(self.device)
            pretrained_model_path = str(self.ckpt_path)
            model_weights = torch.load(pretrained_model_path, map_location=self.device, weights_only=True)
            if self.use_fsam:
                self.net.load_state_dict(model_weights, strict=True)
            else:
                self.net.load_state_dict(model_weights, strict=False)
        else:
            self.net = FactorizePhys(frames=self.frames, md_config=md_config,
                                device=self.device, in_channels=self.in_channels, debug=self.debug).to(self.device)

        self.net.eval()
        if self.assess_latency:
            self.time_vec = []

        if self.debug:
            self.appx_error_list = []


    def load_data(self, num_trial):

        if self.visualize:
            self.np_data = np.load(str(self.data_files[num_trial]))
            self.np_label = np.load(str(self.label_files[num_trial]))[:, 0]
            self.np_label = np.expand_dims(self.np_label, 0)
            self.np_label = torch.tensor(self.np_label)

            # print("Chunk data shape", self.np_data.shape)
            # print("Chunk label shape", self.np_label.shape)
            # print("Min Max of input data:", np.min(self.np_data), np.max(self.np_data))
            # exit()

            self.test_data = np.transpose(self.np_data, (3, 0, 1, 2))
            self.test_data = torch.from_numpy(self.test_data)
            self.test_data = self.test_data.unsqueeze(0)

            last_frame = torch.unsqueeze(self.test_data[:, :, -1, :, :], 2).repeat(1, 1, 1, 1, 1)
            self.test_data = torch.cat((self.test_data, last_frame), 2)
            self.test_data = self.test_data.to(torch.float32).to(self.device)
        else:
            self.test_data = torch.rand(self.batch_size, self.data_channels, self.frames + 1, self.height, self.width)
            self.test_data = self.test_data.to(torch.float32).to(self.device)
            self.np_label = torch.rand(self.batch_size, self.frames).to(torch.float32).to(self.device)

    def run_inference(self, num_trial):

        if self.visualize:
            print("Processing:", self.data_files[num_trial])
        if self.assess_latency:
            t0 = time.time()

        if (self.md_infer or self.net.training or self.debug) and self.use_fsam:
            self.pred, self.vox_embed, self.factorized_embed, self.appx_error = self.net(self.test_data, self.np_label)
        else:
            self.pred, self.vox_embed = self.net(self.test_data)

        # macs, params = get_model_complexity_info(self.net(self.test_data), (self.batch_size, self.data_channels, self.frames + 1, self.height, self.width), as_strings=False,
        #                                         print_per_layer_stat=True, verbose=True)
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        if self.assess_latency:
            t1 = time.time()
            self.time_vec.append(t1-t0)

        if self.debug:
            print("pred.shape", self.pred.shape)
            if (self.md_infer or self.net.training or self.debug) and self.use_fsam:
                self.appx_error_list.append(self.appx_error.item())

        if self.visualize:
            self.save_attention_maps(num_trial)


    def save_attention_maps(self, num_trial):
        b, channels, enc_frames, enc_height, enc_width = self.vox_embed.shape
        label_matrix = self.np_label.unsqueeze(0).repeat(1, channels, 1).unsqueeze(2).unsqueeze(2).permute(0, 1, 4, 3, 2).repeat(1, 1, 1, enc_height, enc_width)
        label_matrix = label_matrix.to(device=self.device)
        corr_matrix = F.cosine_similarity(self.vox_embed, label_matrix, dim=2).abs()

        # avg_emb = torch.mean(self.vox_embed, dim=1)
        # b, enc_frames, enc_height, enc_width = avg_emb.shape
        # label_matrix = np_label.unsqueeze(0).unsqueeze(2).permute(0, 3, 2, 1).repeat(1, 1, enc_height, enc_width)
        # label_matrix = label_matrix.to(device=device)
        # corr_matrix = F.cosine_similarity(avg_emb, label_matrix, dim=1)

        if self.debug:
            print("corr_matrix.shape", corr_matrix.shape)
            print("self.test_data.shape:", self.test_data.shape)
            print("self.vox_embed.shape:", self.vox_embed.shape)

        self.test_data = self.test_data.detach().cpu().numpy()
        self.vox_embed = self.vox_embed.detach().cpu().numpy()
        corr_matrix = corr_matrix.detach().cpu().numpy()

        fig, ax = plt.subplots(4, 4, figsize=[16, 16])
        fig.tight_layout()

        ax[0, 0].imshow(self.np_data[0, ...].astype(np.uint8))
        ax[0, 0].axis('off')
        ax[0, 0].set_title("T: " + str(0))

        ax[0, 1].imshow(self.np_data[enc_frames//4, ...].astype(np.uint8))
        ax[0, 1].axis('off')
        ax[0, 1].set_title("T: " + str(enc_frames//4))

        ax[0, 2].imshow(self.np_data[enc_frames//2, ...].astype(np.uint8))
        ax[0, 2].axis('off')
        ax[0, 2].set_title("T: " + str(enc_frames//2))

        ax[0, 3].imshow(self.np_data[enc_frames-1, ...].astype(np.uint8))
        ax[0, 3].axis('off')
        ax[0, 3].set_title("T: " + str(enc_frames-1))

        cmap = "coolwarm"

        ch = 0
        ax[1, 0].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
        ax[1, 0].axis('off')
        ax[1, 0].set_title("Ch: " + str(ch))

        ch += 1
        ax[1, 1].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
        ax[1, 1].axis('off')
        ax[1, 1].set_title("Ch: " + str(ch))
 
        ch += 1
        ax[1, 2].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
        ax[1, 2].axis('off')
        ax[1, 2].set_title("Ch: " + str(ch))

        ch += 1
        ax[1, 3].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
        ax[1, 3].axis('off')
        ax[1, 3].set_title("Ch: " + str(ch))

        ch += 1
        ax[2, 0].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
        ax[2, 0].axis('off')
        ax[2, 0].set_title("Ch: " + str(ch))

        ch += 1
        ax[2, 1].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
        ax[2, 1].axis('off')
        ax[2, 1].set_title("Ch: " + str(ch))

        ch += 1
        ax[2, 2].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
        ax[2, 2].axis('off')
        ax[2, 2].set_title("Ch: " + str(ch))

        ch += 1
        ax[2, 3].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
        ax[2, 3].axis('off')
        ax[2, 3].set_title("Ch: " + str(ch))

        ch += 1
        ax[3, 0].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
        ax[3, 0].axis('off')
        ax[3, 0].set_title("Ch: " + str(ch))

        ch += 1
        ax[3, 1].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
        ax[3, 1].axis('off')
        ax[3, 1].set_title("Ch: " + str(ch))

        ch += 1
        ax[3, 2].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
        ax[3, 2].axis('off')
        ax[3, 2].set_title("Ch: " + str(ch))

        ch += 1
        ax[3, 3].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
        ax[3, 3].axis('off')
        ax[3, 3].set_title("Ch: " + str(ch))

        # plt.show()
        save_path = str(self.attention_map_dir.joinpath(str(Path(self.data_files[num_trial]).name.replace(".npy", "_attention_map.jpg"))))
        print("Saving:", save_path)
        plt.savefig(save_path)
        plt.close(fig)


    def output_summary_results(self):
        if self.assess_latency:
            print("Median time: ", np.median(self.time_vec))
            plt.plot(self.time_vec)
            plt.savefig(str(self.plot_dir.joinpath("Latency.jpg")))

        if self.debug:
            if (self.md_infer or self.net.training or self.debug) and self.use_fsam:
                print("Median error:", np.median(self.appx_error_list))

        pytorch_total_params = sum(p.numel() for p in self.net.parameters())
        print("Total parameters = ", pytorch_total_params)

        pytorch_trainable_params = sum(p.numel()
                                    for p in self.net.parameters() if p.requires_grad)
        print("Trainable parameters = ", pytorch_trainable_params)


if __name__ == "__main__":

    testObj = TestFactorizePhysBig()

    print("testObj.num_trials:", testObj.num_trials)
    for trial_num in range(testObj.num_trials):
        testObj.load_data(trial_num)
        testObj.run_inference(trial_num)

    testObj.output_summary_results()

    # writer.add_graph(net, test_data)
    # writer.close()