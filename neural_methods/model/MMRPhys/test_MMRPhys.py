"""
MMRPhys: Remote Extraction of Multiple Physiological Signals using Label Guided Factorization.
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
# from neural_methods.model.MMRPhys.MMRPhysLEF import MMRPhysLEF as MMRPhys
# from neural_methods.model.MMRPhys.MMRPhysLNF import MMRPhysLNF as MMRPhys
from neural_methods.model.MMRPhys.MMRPhysLLF import MMRPhysLLF as MMRPhys

model_config = {
    "TASKS": ["BVP", "BP", "RSP"],
    # "TASKS": ["BP"],
    "BP_USE_RSP": True,
    "FS": 25,
    "MD_FSAM": True,
    "MD_TYPE": "SNMF_Label",
    "MD_R": 1,
    "MD_S": 1,
    "MD_STEPS": 5,
    "MD_INFERENCE": False,
    "MD_RESIDUAL": True,
    "in_channels": 4,
    "data_channels": 4,
    "height": 72,
    "weight": 72,
    "batch_size": 2,
    "frames": 500,
    "debug": True,
    "assess_latency": False,
    "num_trials": 20,
    "visualize": False,
    "ckpt_path": "./runs/exp/BP4D_RGBT_500_72x72/PreTrainedModels/BP4D_MMRPhysLNF_BVP_RSP_RGBTx72_SFSAM_Label_Fold1_Epoch29.pth",
    "data_path": "/home/jitesh/data/BP4D/BP4D_RGBT_500_72x72",
}


class TestMMRPhys(object):
    def __init__(self) -> None:
        self.ckpt_path = Path(model_config["ckpt_path"])
        self.data_path = Path(model_config["data_path"])

        self.tasks = model_config["TASKS"]
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
            self.data_files = list(sorted(self.data_path.rglob("*input*.npy")))
            self.label_files = list(sorted(self.data_path.rglob("*label*.npy")))
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
        md_config["FS"] = model_config["FS"]
        md_config["MD_S"] = model_config["MD_S"]
        md_config["MD_R"] = model_config["MD_R"]
        md_config["MD_STEPS"] = model_config["MD_STEPS"]
        md_config["MD_FSAM"] = model_config["MD_FSAM"]
        md_config["MD_TYPE"] = model_config["MD_TYPE"]
        md_config["MD_INFERENCE"] = model_config["MD_INFERENCE"]
        md_config["MD_RESIDUAL"] = model_config["MD_RESIDUAL"]
        md_config["TASKS"] = model_config["TASKS"]
        md_config["BP_USE_RSP"] = model_config["BP_USE_RSP"]

        if self.visualize:
            self.net = nn.DataParallel(MMRPhys(frames=self.frames, md_config=md_config,
                                device=self.device, in_channels=self.in_channels, debug=self.debug), device_ids=[0]).to(self.device)
            pretrained_model_path = str(self.ckpt_path)
            model_weights = torch.load(pretrained_model_path, map_location=self.device, weights_only=True)
            if "BP" not in self.tasks:
                weights_trimmed = {k:v for k, v in model_weights.items() if not k.startswith('module.rBP_head')}
                self.net.load_state_dict(weights_trimmed, strict=False)
            else:
                self.net.load_state_dict(model_weights)

        else:
            self.net = MMRPhys(frames=self.frames, md_config=md_config,
                                device=self.device, in_channels=self.in_channels, debug=self.debug).to(self.device)

        self.net.eval()
        if self.assess_latency:
            self.time_vec = []

        if self.debug:
            self.appx_error_list = []


    def load_data(self, num_trial):

        if self.visualize:
            self.np_data = np.load(str(self.data_files[num_trial]))
            self.np_label = np.load(str(self.label_files[num_trial]))
            
            self.bvp_label = np.expand_dims(self.np_label[:, 0], 0)
            self.bvp_label = torch.tensor(self.bvp_label)

            self.rsp_label = np.expand_dims(self.np_label[:, 1], 0)
            self.rsp_label = torch.tensor(self.rsp_label)

            # bvp_label
            # resp_label
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
            self.bvp_label = torch.rand(self.batch_size, self.frames).to(torch.float32).to(self.device)
            self.rsp_label = torch.rand(self.batch_size, self.frames).to(torch.float32).to(self.device)

    def run_inference(self, num_trial):

        if self.visualize:
            print("Processing:", self.data_files[num_trial].name)
        if self.assess_latency:
            t0 = time.time()

        out = self.net(self.test_data, label_bvp=self.bvp_label, label_rsp=self.rsp_label)
        self.pred_bvp = out[0]
        self.pred_rsp = out[1]
        self.pred_rBP = out[2]
        self.vox_embed_bvp = out[3]
        self.vox_embed_rsp = out[4]

        if (self.md_infer or self.net.training or self.debug) and self.use_fsam:
            self.factorized_embed_ppg = out[5]
            self.appx_error_ppg = out[6]

        if self.assess_latency:
            t1 = time.time()
            self.time_vec.append(t1-t0)

        if self.debug:
            if "BVP" in self.tasks:
                print("pred_bvp.shape", self.pred_bvp.shape)
            if "RSP" in self.tasks:
                print("pred_rsp.shape", self.pred_rsp.shape)
            if "BP" in self.tasks:
                print("pred_rBP.shape", self.pred_rBP.shape)

            if (self.md_infer or self.net.training or self.debug) and self.use_fsam:
                if "BVP" in self.tasks:
                    self.appx_error_list.append(self.appx_error_ppg.item())

        if self.visualize:
            self.save_attention_maps(num_trial)


    def save_attention_maps(self, num_trial):
        if "BVP" in self.tasks:
            b, channels, enc_frames, enc_height, enc_width = self.vox_embed_bvp.shape
            
            label_matrix = self.bvp_label.unsqueeze(0).repeat(1, channels, 1).unsqueeze(
                2).unsqueeze(2).permute(0, 1, 4, 3, 2).repeat(1, 1, 1, enc_height, enc_width)

            print("b, channels, enc_frames, enc_height, enc_width: ", [b, channels, enc_frames, enc_height, enc_width])
            print("self.bvp_label.shape", self.bvp_label.shape)
            print("label_matrix.shape", label_matrix.shape)

            label_matrix = label_matrix.to(device=self.device)
            corr_matrix = F.cosine_similarity(self.vox_embed_bvp, label_matrix, dim=2).abs()

            # avg_emb = torch.mean(self.vox_embed_bvp, dim=1)
            # b, enc_frames, enc_height, enc_width = avg_emb.shape
            # label_matrix = bvp_label.unsqueeze(0).unsqueeze(2).permute(0, 3, 2, 1).repeat(1, 1, enc_height, enc_width)
            # label_matrix = label_matrix.to(device=device)
            # corr_matrix = F.cosine_similarity(avg_emb, label_matrix, dim=1)

            if self.debug:
                print("corr_matrix.shape", corr_matrix.shape)
                print("self.test_data.shape:", self.test_data.shape)
                print("self.vox_embed_bvp.shape:", self.vox_embed_bvp.shape)

            self.test_data = self.test_data.detach().cpu().numpy()
            self.vox_embed_bvp = self.vox_embed_bvp.detach().cpu().numpy()
            corr_matrix = corr_matrix.detach().cpu().numpy()

            fig, ax = plt.subplots(5, 4, figsize=[12, 16])
            fig.tight_layout()
            cmap = "coolwarm"

            n_row = 0
            ax[n_row, 0].imshow(self.np_data[0, :, :, 0:3])
            ax[n_row, 0].axis('off')

            ax[n_row, 1].imshow(self.np_data[enc_frames//3, :, :, 0:3])
            ax[n_row, 1].axis('off')

            ax[n_row, 2].imshow(self.np_data[2 * enc_frames//3, :, :, 0:3])
            ax[n_row, 2].axis('off')

            ax[n_row, 3].imshow(self.np_data[enc_frames-1, :, :, 0:3])
            ax[n_row, 3].axis('off')

            n_row = 1
            ch = 0
            ax[n_row, 0].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 0].axis('off')

            ch = 1
            ax[n_row, 1].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 1].axis('off')

            ch = 2
            ax[n_row, 2].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 2].axis('off')

            ch = 3
            ax[n_row, 3].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 3].axis('off')     

            n_row = 2
            ch = 4
            ax[n_row, 0].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 0].axis('off')

            ch = 5
            ax[n_row, 1].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 1].axis('off')

            ch = 6
            ax[n_row, 2].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 2].axis('off')

            ch = 7
            ax[n_row, 3].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 3].axis('off')

            n_row = 3
            ch = 8
            ax[n_row, 0].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 0].axis('off')

            ch = 9
            ax[n_row, 1].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 1].axis('off')

            ch = 10
            ax[n_row, 2].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 2].axis('off')

            ch = 11
            ax[n_row, 3].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 3].axis('off')

            n_row = 4
            ch = 12
            ax[n_row, 0].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 0].axis('off')

            ch = 13
            ax[n_row, 1].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 1].axis('off')

            ch = 14
            ax[n_row, 2].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 2].axis('off')

            ch = 15
            ax[n_row, 3].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 3].axis('off')


            # plt.show()
            plt.savefig(str(self.attention_map_dir.joinpath(str(self.data_files[num_trial].name.replace(".npy", "_BVP_attention_map.jpg")))))
            plt.close(fig)

        if "RSP" in self.tasks:
            b, channels, enc_frames, enc_height, enc_width = self.vox_embed_rsp.shape

            label_matrix = self.rsp_label.to(device=self.device)
            label_matrix = F.avg_pool1d(label_matrix, kernel_size=5, stride=4, padding=2)   #downsampling 4 times, aligning with the temporal dimensions of the embeddings

            label_matrix = label_matrix.unsqueeze(0).repeat(1, channels, 1).unsqueeze(
                2).unsqueeze(2).permute(0, 1, 4, 3, 2).repeat(1, 1, 1, enc_height, enc_width)

            print("b, channels, enc_frames, enc_height, enc_width: ", [b, channels, enc_frames, enc_height, enc_width])
            print("self.rsp_label.shape", self.rsp_label.shape)
            print("label_matrix.shape", label_matrix.shape)

            corr_matrix = F.cosine_similarity(self.vox_embed_rsp, label_matrix, dim=2).abs()

            # avg_emb = torch.mean(self.vox_embed_rsp, dim=1)
            # b, enc_frames, enc_height, enc_width = avg_emb.shape
            # label_matrix = rsp_label.unsqueeze(0).unsqueeze(2).permute(0, 3, 2, 1).repeat(1, 1, enc_height, enc_width)
            # label_matrix = label_matrix.to(device=device)
            # corr_matrix = F.cosine_similarity(avg_emb, label_matrix, dim=1)

            if self.debug:
                print("corr_matrix.shape", corr_matrix.shape)
                print("self.vox_embed_rsp.shape:", self.vox_embed_rsp.shape)

            self.vox_embed_rsp = self.vox_embed_rsp.detach().cpu().numpy()
            corr_matrix = corr_matrix.detach().cpu().numpy()

            fig, ax = plt.subplots(5, 4, figsize=[12, 16])
            fig.tight_layout()
            cmap = "coolwarm"

            n_row = 0
            ax[n_row, 0].imshow(self.np_data[0, :, :, 0:3])
            ax[n_row, 0].axis('off')

            ax[n_row, 1].imshow(self.np_data[4 * enc_frames//3, :, :, 0:3])
            ax[n_row, 1].axis('off')

            ax[n_row, 2].imshow(self.np_data[4 * 2 * enc_frames//3, :, :, 0:3])
            ax[n_row, 2].axis('off')

            ax[n_row, 3].imshow(self.np_data[4 * enc_frames-1, :, :, 0:3])
            ax[n_row, 3].axis('off')

            n_row = 1
            ch = 0
            ax[n_row, 0].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 0].axis('off')

            ch = 1
            ax[n_row, 1].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 1].axis('off')

            ch = 2
            ax[n_row, 2].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 2].axis('off')

            ch = 3
            ax[n_row, 3].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 3].axis('off')     

            n_row = 2
            ch = 4
            ax[n_row, 0].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 0].axis('off')

            ch = 5
            ax[n_row, 1].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 1].axis('off')

            ch = 6
            ax[n_row, 2].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 2].axis('off')

            ch = 7
            ax[n_row, 3].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 3].axis('off')

            n_row = 3
            ch = 8
            ax[n_row, 0].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 0].axis('off')

            ch = 9
            ax[n_row, 1].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 1].axis('off')

            ch = 10
            ax[n_row, 2].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 2].axis('off')

            ch = 11
            ax[n_row, 3].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 3].axis('off')

            n_row = 4
            ch = 12
            ax[n_row, 0].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 0].axis('off')

            ch = 13
            ax[n_row, 1].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 1].axis('off')

            ch = 14
            ax[n_row, 2].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 2].axis('off')

            ch = 15
            ax[n_row, 3].imshow(corr_matrix[0, ch, :, :], cmap=cmap, vmin=0, vmax=1)
            ax[n_row, 3].axis('off')

            # plt.show()
            plt.savefig(str(self.attention_map_dir.joinpath(str(self.data_files[num_trial].name.replace(".npy", "_RSP_attention_map.jpg")))))
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

    testObj = TestMMRPhys()

    print("testObj.num_trials:", testObj.num_trials)
    for trial_num in range(testObj.num_trials):
        testObj.load_data(trial_num)
        testObj.run_inference(trial_num)


    testObj.output_summary_results()

    # writer.add_graph(net, test_data)
    # writer.close()