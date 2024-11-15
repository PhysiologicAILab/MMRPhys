"""Trainer for FactorizePhys."""
import os
import numpy as np
import torch
import torch.optim as optim
import neurokit2 as nk
from evaluation.metrics import calculate_metrics, calculate_rsp_metrics, calculate_bp_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.model.MMRPhys.MMRPhysLEF import MMRPhysLEF
from neural_methods.model.MMRPhys.MMRPhysLNF import MMRPhysLNF
from neural_methods.model.MMRPhys.MMRPhysLLF import MMRPhysLLF
from neural_methods.model.MMRPhys.MMRPhysBig import MMRPhysBig
from neural_methods.model.MMRPhys.MMRPhysMedium import MMRPhysMedium
from neural_methods.model.MMRPhys.MMRPhysFuseL import MMRPhysFuseL
from neural_methods.model.MMRPhys.MMRPhysFuseM import MMRPhysFuseM
from neural_methods.model.MMRPhys.MMRPhysFuseS import MMRPhysFuseS
from neural_methods.model.MMRPhys.MMRPhysSmall import MMRPhysSmall
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm


class MMRPhysTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.dropout_rate = config.MODEL.DROP_RATE
        self.base_len = self.num_of_gpu
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0

        if torch.cuda.is_available() and config.NUM_OF_GPU_TRAIN > 0:
            dev_list = [int(d) for d in config.DEVICE.replace("cuda:", "").split(",")]
            self.device = torch.device(dev_list[0])     #currently toolbox only supports 1 GPU
            self.num_of_gpu = 1     #config.NUM_OF_GPU_TRAIN  # set number of used GPUs
        else:
            self.device = torch.device("cpu")  # if no GPUs set device is CPU
            self.num_of_gpu = 0  # no GPUs used

        frames = self.config.MODEL.MMRPhys.FRAME_NUM
        in_channels = self.config.MODEL.MMRPhys.CHANNELS
        model_type = self.config.MODEL.MMRPhys.TYPE
        model_type = model_type.lower()

        md_config = {}
        md_config["FRAME_NUM"] = self.config.MODEL.MMRPhys.FRAME_NUM
        md_config["MD_TYPE"] = self.config.MODEL.MMRPhys.MD_TYPE
        md_config["MD_FSAM"] = self.config.MODEL.MMRPhys.MD_FSAM
        md_config["MD_S"] = self.config.MODEL.MMRPhys.MD_S
        md_config["MD_R"] = self.config.MODEL.MMRPhys.MD_R
        md_config["MD_STEPS"] = self.config.MODEL.MMRPhys.MD_STEPS
        md_config["MD_INFERENCE"] = self.config.MODEL.MMRPhys.MD_INFERENCE
        md_config["MD_RESIDUAL"] = self.config.MODEL.MMRPhys.MD_RESIDUAL        
        md_config["TASKS"] = self.config.MODEL.MMRPhys.TASKS
        if self.config.TOOLBOX_MODE == "train_and_test" or self.config.TOOLBOX_MODE == "only_train":
            md_config["FS"] = self.config.TRAIN.DATA.FS
        else:
            md_config["FS"] = self.config.TEST.DATA.FS

        self.md_infer = self.config.MODEL.MMRPhys.MD_INFERENCE
        self.use_fsam = self.config.MODEL.MMRPhys.MD_FSAM
        self.tasks = self.config.MODEL.MMRPhys.TASKS
        if "BVP" in self.tasks or "RSP" in self.tasks or "BP" in self.tasks:
            print("Tasks:", self.tasks)
        else:
            print("Unknown estimation task... BVP, RSP, and BP are supported. Exiting the code...")
            exit()

        if model_type == "lef":
            self.model = MMRPhysLEF(frames=frames, md_config=md_config, in_channels=in_channels, dropout=self.dropout_rate, device=self.device)  # [4, T, 72, 72]
        elif model_type == "lnf":
            self.model = MMRPhysLNF(frames=frames, md_config=md_config, in_channels=in_channels, dropout=self.dropout_rate, device=self.device)  # [4, T, 72, 72]
        elif model_type == "llf":
            self.model = MMRPhysLLF(frames=frames, md_config=md_config, in_channels=in_channels, dropout=self.dropout_rate, device=self.device)  # [4, T, 72, 72]
        elif model_type == "big":
            self.model = MMRPhysBig(frames=frames, md_config=md_config, in_channels=in_channels, dropout=self.dropout_rate, device=self.device)  # [4, T, 144, 144]
        elif model_type == "medium":
            self.model = MMRPhysMedium(frames=frames, md_config=md_config, in_channels=in_channels, dropout=self.dropout_rate, device=self.device)  # [4, T, 36, 36]
        elif model_type == "fusel":
            self.model = MMRPhysFuseL(frames=frames, md_config=md_config, in_channels=in_channels, dropout=self.dropout_rate, device=self.device)  # [4, T, 36, 36]
        elif model_type == "fusem":
            self.model = MMRPhysFuseM(frames=frames, md_config=md_config, in_channels=in_channels, dropout=self.dropout_rate, device=self.device)  # [4, T, 36, 36]
        elif model_type == "fuses":
            self.model = MMRPhysFuseS(frames=frames, md_config=md_config, in_channels=in_channels, dropout=self.dropout_rate, device=self.device)  # [4, T, 9, 9]
        elif model_type == "small":
            self.model = MMRPhysSmall(frames=frames, md_config=md_config, in_channels=in_channels, dropout=self.dropout_rate, device=self.device)  # [4, T, 9, 9]
        else:
            print("Unexpected model type specified. Should be standard or big, but specified:", model_type)
            exit()


        if torch.cuda.device_count() > 0 and self.num_of_gpu > 0:  # distribute model across GPUs
            self.model = torch.nn.DataParallel(self.model, device_ids=[self.device])  # data parallel model
        else:
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        if self.config.TOOLBOX_MODE == "train_and_test" or self.config.TOOLBOX_MODE == "only_train":
            self.num_train_batches = len(data_loader["train"])

            if "BVP" in self.tasks:
                self.criterion_bvp = Neg_Pearson() #BVP
            if "RSP" in self.tasks:
                self.criterion_rsp = Neg_Pearson() #RSP

            if "BP" in self.tasks:
                # self.criterion_sbp = torch.nn.MSELoss()  # SBP
                self.criterion_sbp = torch.nn.SmoothL1Loss()    # SBP
                # self.criterion_dbp = torch.nn.MSELoss()  # DBP
                self.criterion_dbp = torch.nn.SmoothL1Loss()  # DBP

            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.TRAIN.LR)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.config.TRAIN.LR, epochs=self.config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
            
            pretrained_model_path = self.config.MODEL.MMRPhys.PRETRAINED
            if pretrained_model_path == "" and "BP" in self.tasks:
                print("Pretrained model not specified, which is required for training the model for BP estimation... ")
                print("Exiting the code ...")
                exit()            
            else:
                if ("BVP" in self.tasks or "RSP" in self.tasks):            
                    print("Loading pretrained model:", pretrained_model_path)
                    self.model.load_state_dict(torch.load(pretrained_model_path, map_location=self.device, weights_only=True), strict=True)   # BVP and RSP will be trained first with SFSAM, and when training for BP, SFSAM is not needed.
                else:
                    model_weights = torch.load(pretrained_model_path, map_location=self.device, weights_only=True)
                    weights_trimmed = {k:v for k, v in model_weights.items() if not k.startswith('module.rBP_head')}
                    self.model.load_state_dict(weights_trimmed, strict=False)

        elif self.config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("MMRPhys trainer initialized in incorrect toolbox mode!")


    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        avg_train_loss_bvp = []
        avg_train_loss_rsp = []
        avg_train_loss_bp = []
        avg_valid_loss_bvp = []
        avg_valid_loss_rsp = []
        avg_valid_loss_bp = []

        lrs = []

        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss_bvp = 0.0
            running_loss_rsp = 0.0
            running_loss_bp = 0.0

            train_loss = []
            train_loss_bvp = []
            train_loss_rsp = []
            train_loss_bp = []

            appx_error_list_bvp = []
            appx_error_list_rsp = []
            self.model.train()

            tbar = tqdm(data_loader["train"], ncols=150)

            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                
                data = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                label_bvp = label_rsp = label_bp = None
                label_sysBP = label_diaBP = None

                if len(labels.shape) == 3:
                    if labels.shape[-1] > 4:       # BP4D dataset 
                        # bvp, rsp, eda, ecg, hr, rr, sysBP, avgBP, diaBP
                        label_bvp = labels[..., 0]
                        # label_bvp = labels[..., 11]
                        label_rsp = labels[..., 1]
                        label_sysBP = labels[..., 6]
                        label_avgBP = labels[..., 7]
                        label_diaBP = labels[..., 8]
                        # label_bp = labels[..., 9]
                        label_bp = labels[..., 10]
                        SBP = torch.median(label_sysBP, dim=1).values
                        DBP = torch.median(label_diaBP, dim=1).values

                    elif labels.shape[-1] >= 1:     #SCAMPS dataset
                        label_bvp = labels[..., 0]
                        label_rsp = labels[..., 1]
                    else:                           # All other rPPG datasets (UBFC-rPPG, PURE, iBVP)
                        label_bvp = labels[..., 0]
                elif "BVP" in self.tasks:
                    label_bvp = labels
                elif "RSP" in self.tasks:
                    label_rsp = labels

                if "BVP" in self.tasks:
                    mean_label_bvp = torch.mean(label_bvp, dim=1, keepdim=True)
                    std_label_bvp = torch.std(label_bvp, dim=1, keepdim=True)
                    label_bvp = (label_bvp - mean_label_bvp) / std_label_bvp  # normalize
                
                if "RSP" in self.tasks:
                    mean_label_rsp = torch.mean(label_rsp, dim=1, keepdim=True)
                    std_label_rsp = torch.std(label_rsp, dim=1, keepdim=True)
                    label_rsp = (label_rsp - mean_label_rsp) / std_label_rsp  # normalize
                
                last_frame = torch.unsqueeze(data[:, :, -1, :, :], 2).repeat(1, 1, max(self.num_of_gpu, 1), 1, 1)
                data = torch.cat((data, last_frame), 2)

                # last_sample = torch.unsqueeze(labels[-1, :], 0).repeat(max(self.num_of_gpu, 1), 1)
                # labels = torch.cat((labels, last_sample), 0)
                # labels = torch.diff(labels, dim=0)
                # labels = labels/ torch.std(labels)  # normalize
                # labels[torch.isnan(labels)] = 0

                self.optimizer.zero_grad()
                out = self.model(data, label_bvp=label_bvp, label_rsp=label_rsp)
                
                pred_bvp = out[0]
                pred_rsp = out[1]
                pred_bp = out[2]

                if self.model.training and self.use_fsam:
                    appx_error_bvp = out[6]
                    appx_error_rsp = out[8]

                loss = 0

                if "BVP" in self.tasks:
                    mean_pred_bvp = torch.mean(pred_bvp, dim=1, keepdim=True)
                    std_pred_bvp = torch.std(pred_bvp, dim=1, keepdim=True)
                    pred_bvp = (pred_bvp - mean_pred_bvp) / std_pred_bvp  # normalize
                    loss_bvp = self.criterion_bvp(pred_bvp, label_bvp)
                    loss = loss + loss_bvp

                if "RSP" in self.tasks:
                    mean_pred_rsp = torch.mean(pred_rsp, dim=1, keepdim=True)
                    std_pred_rsp = torch.std(pred_rsp, dim=1, keepdim=True)
                    pred_rsp = (pred_rsp - mean_pred_rsp) / std_pred_rsp  # normalize
                    loss_rsp = self.criterion_rsp(pred_rsp, label_rsp)
                    loss = loss + loss_rsp
                
                if "BP" in self.tasks:
                    loss_bp = self.criterion_sbp(pred_bp[:, 0], SBP) + self.criterion_dbp(pred_bp[:, 1], DBP) 
                    loss = loss + loss_bp

                loss.backward()

                if "BVP" in self.tasks:
                    running_loss_bvp += loss_bvp.item()
                if "RSP" in self.tasks:
                    running_loss_rsp += loss_rsp.item()
                if "BP" in self.tasks:
                    running_loss_bp += loss_bp.item()

                if idx % 100 == 99:  # print every 100 mini-batches
                    print(f'[{epoch}, {idx + 1: 5d}]')
                    if "BVP" in self.tasks:
                        print(f'loss_bvp: {running_loss_bvp / 100:.3f}')    
                        running_loss_bvp = 0.0
                    if "RSP" in self.tasks:
                        print(f'loss_rsp: {running_loss_rsp / 100:.3f}')    
                        running_loss_rsp = 0.0
                    if "BP" in self.tasks:
                        print(f'loss_bp: {running_loss_bp / 100:.3f}')
                        running_loss_bp = 0.0

                train_loss.append(loss.item())
                if "BVP" in self.tasks:
                    train_loss_bvp.append(loss_bvp.item())
                if "RSP" in self.tasks:
                    train_loss_rsp.append(loss_rsp.item())
                if "BP" in self.tasks:
                    train_loss_bp.append(loss_bp.item())

                if self.use_fsam:
                    if "BVP" in self.tasks:
                        appx_error_list_bvp.append(appx_error_bvp.item())
                    if "RSP" in self.tasks:
                        appx_error_list_rsp.append(appx_error_rsp.item())

                # Append the current learning rate to the list
                lrs.append(self.scheduler.get_last_lr())

                self.optimizer.step()
                self.scheduler.step()

                bar_dict = {}
                if "BVP" in self.tasks:
                    bar_dict["loss_bvp"] = round(loss_bvp.item(), 2)
                if "RSP" in self.tasks:
                    bar_dict["loss_rsp"] = round(loss_rsp.item(), 2)
                if "BP" in self.tasks:
                    bar_dict["loss_bp"] = round(loss_bp.item(), 2)

                tbar.set_postfix(bar_dict, loss=loss.item())

            # Append the mean training loss for the epoch
            if "BVP" in self.tasks:
                avg_train_loss_bvp.append(round(np.mean(train_loss_bvp), 2))
            if "RSP" in self.tasks:
                avg_train_loss_rsp.append(round(np.mean(train_loss_rsp), 2))
            if "BP" in self.tasks:
                avg_train_loss_bp.append(round(np.mean(train_loss_bp), 2))

            print("Avg train loss: {}".format(np.round(np.mean(train_loss), 2)))
            # TODO: It should ideally be possible to use FSAM selectively for each task. This is current limitation
            if self.use_fsam:
                if "BVP" in self.tasks:
                    print("Avg appx error BVP: {}".format(np.round(np.mean(appx_error_list_bvp), 2)))
                if "RSP" in self.tasks:
                    print("Avg appx error RSP: {}".format(np.round(np.mean(appx_error_list_rsp), 2)))

            self.save_model(epoch)

            if not self.config.TEST.USE_LAST_EPOCH:
                
                valid_loss_bvp, valid_loss_rsp, valid_loss_bp = self.valid(data_loader, epoch)
                total_valid_loss = 0
                
                if "BVP" in self.tasks:
                    print('Validation loss BVP: ', valid_loss_bvp)
                    avg_valid_loss_bvp.append(valid_loss_bvp)
                    total_valid_loss += valid_loss_bvp
                if "RSP" in self.tasks:
                    print('Validation loss RSP: ', valid_loss_rsp)
                    avg_valid_loss_rsp.append(valid_loss_rsp)
                    total_valid_loss += valid_loss_rsp
                if "BP" in self.tasks:
                    print('Validation loss BP: ', valid_loss_bp)
                    avg_valid_loss_bp.append(valid_loss_bp)
                    total_valid_loss += valid_loss_bp

                print('Total validation loss: ', total_valid_loss)

                if self.min_valid_loss is None:
                    self.min_valid_loss = total_valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (total_valid_loss < self.min_valid_loss):
                    self.min_valid_loss = total_valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))
        
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            if "BVP" in self.tasks:
                self.plot_losses_and_lrs(avg_train_loss_bvp, avg_valid_loss_bvp, lrs, self.config, suff="BVP")
            if "RSP" in self.tasks:
                self.plot_losses_and_lrs(avg_train_loss_rsp, avg_valid_loss_rsp, lrs, self.config, suff="RSP")
            if "BP" in self.tasks:
                self.plot_losses_and_lrs(avg_train_loss_bp, avg_valid_loss_bp, lrs, self.config, suff="BP")


    def valid(self, data_loader, epoch=-1):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validing===")
        valid_loss_bvp = []
        valid_loss_rsp = []
        valid_loss_bp = []
        avg_valid_loss_bvp = 0
        avg_valid_loss_rsp = 0
        avg_valid_loss_bp = 0

        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=150)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")

                label_bvp = label_rsp = label_bp = label_eda = None
                label_sysBP = label_avgBP = label_diaBP = None

                data, labels = valid_batch[0].to(self.device), valid_batch[1].to(self.device)

                if len(labels.shape) == 3:
                    if labels.shape[-1] > 4:       # BP4D dataset 
                        label_bvp = labels[..., 0]
                        # label_bvp = labels[..., 11]
                        label_rsp = labels[..., 1]
                        label_sysBP = labels[..., 6]
                        label_avgBP = labels[..., 7]
                        label_diaBP = labels[..., 8]
                        # label_bp = labels[..., 9]
                        label_bp = labels[..., 10]
                        SBP = torch.median(label_sysBP, dim=1).values
                        DBP = torch.median(label_diaBP, dim=1).values

                    elif labels.shape[-1] >= 1:     #SCAMPS dataset
                        label_bvp = labels[..., 0]
                        label_rsp = labels[..., 1]
                    else:                           # All other rPPG datasets (UBFC-rPPG, PURE, iBVP)
                        label_bvp = labels[..., 0]
                elif "BVP" in self.tasks:
                    label_bvp = labels
                elif "RSP" in self.tasks:
                    label_rsp = labels

                if "BVP" in self.tasks:
                    mean_label_bvp = torch.mean(label_bvp, dim=1, keepdim=True)
                    std_label_bvp = torch.std(label_bvp, dim=1, keepdim=True)
                    label_bvp = (label_bvp - mean_label_bvp) / std_label_bvp  # normalize
                if "RSP" in self.tasks:
                    mean_label_rsp = torch.mean(label_rsp, dim=1, keepdim=True)
                    std_label_rsp = torch.std(label_rsp, dim=1, keepdim=True)
                    label_rsp = (label_rsp - mean_label_rsp) / std_label_rsp  # normalize

                last_frame = torch.unsqueeze(data[:, :, -1, :, :], 2).repeat(1, 1, max(self.num_of_gpu, 1), 1, 1)
                data = torch.cat((data, last_frame), 2)

                # last_sample = torch.unsqueeze(labels[-1, :], 0).repeat(max(self.num_of_gpu, 1), 1)
                # labels = torch.cat((labels, last_sample), 0)
                # labels = torch.diff(labels, dim=0)
                # labels = labels/ torch.std(labels)  # normalize
                # labels[torch.isnan(labels)] = 0

                out = self.model(data)
                
                pred_bvp = out[0]
                pred_rsp = out[1]
                pred_bp = out[2]

                loss = 0

                if "BVP" in self.tasks:
                    mean_pred_bvp = torch.mean(pred_bvp, dim=1, keepdim=True)
                    std_pred_bvp = torch.std(pred_bvp, dim=1, keepdim=True)
                    pred_bvp = (pred_bvp - mean_pred_bvp) / std_pred_bvp  # normalize
                    loss_bvp = self.criterion_bvp(pred_bvp, label_bvp)
                    valid_loss_bvp.append(loss_bvp.item())
                    loss += loss_bvp

                if "RSP" in self.tasks:
                    mean_pred_rsp = torch.mean(pred_rsp, dim=1, keepdim=True)
                    std_pred_rsp = torch.std(pred_rsp, dim=1, keepdim=True)
                    pred_rsp = (pred_rsp - mean_pred_rsp) / std_pred_rsp  # normalize
                    loss_rsp = self.criterion_rsp(pred_rsp, label_rsp)
                    valid_loss_rsp.append(loss_rsp.item())
                    loss += loss_rsp
                
                if "BP" in self.tasks:
                    loss_bp = self.criterion_sbp(pred_bp[:, 0], SBP) + self.criterion_dbp(pred_bp[:, 1], DBP)
                    valid_loss_bp.append(loss_bp.item())
                    loss += loss_bp

                valid_step += 1

                bar_dict = {}
                if "BVP" in self.tasks:
                    bar_dict["loss_bvp"] = round(loss_bvp.item(), 2)
                if "RSP" in self.tasks:
                    bar_dict["loss_rsp"] = round(loss_rsp.item(), 2)
                if "BP" in self.tasks:
                    bar_dict["loss_bp"] = round(loss_bp.item(), 2)

                # vbar.set_postfix(loss=loss.item())
                vbar.set_postfix(bar_dict, loss=loss.item())

            if "BVP" in self.tasks:
                avg_valid_loss_bvp = np.mean(np.asarray(valid_loss_bvp))
            if "RSP" in self.tasks:
                avg_valid_loss_rsp = np.mean(np.asarray(valid_loss_rsp))
            if "BP" in self.tasks:
                avg_valid_loss_bp = np.mean(np.asarray(valid_loss_bp))

        return avg_valid_loss_bvp, avg_valid_loss_rsp, avg_valid_loss_bp


    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("===Testing===")
        pred_bvp_dict = dict()
        pred_rsp_dict = dict()
        pred_sbp_dict = dict()
        pred_dbp_dict = dict()

        label_bvp_dict = dict()
        label_rsp_dict = dict()
        label_sbp_dict = dict()
        label_dbp_dict = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH, map_location=self.device))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path, map_location=self.device))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        self.model = self.model.to(self.device)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=150)):
                batch_size = test_batch[0].shape[0]
                data, labels_test = test_batch[0].to(self.device), test_batch[1].to(self.device)

                if len(labels_test.shape) == 3:
                    if labels_test.shape[-1] > 4:       # BP4D dataset 
                        label_bvp_test = labels_test[..., 0]
                        # label_bvp_test = labels_test[..., 11]
                        label_rsp_test = labels_test[..., 1]
                        label_sysBP_test = labels_test[..., 6]
                        label_avgBP_test = labels_test[..., 7]
                        label_diaBP_test = labels_test[..., 8]
                        # label_bp_test = labels_test[..., 9]
                        label_bp_test = labels_test[..., 10]
                        SBP_test = torch.median(label_sysBP_test, dim=1).values
                        DBP_test = torch.median(label_diaBP_test, dim=1).values

                    elif labels_test.shape[-1] >= 1:     #SCAMPS dataset
                        label_bvp_test = labels_test[..., 0]
                        label_rsp_test = labels_test[..., 1]
                    else:                           # All other rPPG datasets (UBFC-rPPG, PURE, iBVP)
                        label_bvp_test = labels_test[..., 0]

                elif "BVP" in self.tasks:
                    label_bvp_test = labels_test
                elif "RSP" in self.tasks:
                    label_rsp_test = labels_test

                if "BVP" in self.tasks:
                    mean_label_bvp_test = torch.mean(label_bvp_test, dim=1, keepdim=True)
                    std_label_bvp_test = torch.std(label_bvp_test, dim=1, keepdim=True)
                    label_bvp_test = (label_bvp_test - mean_label_bvp_test) / std_label_bvp_test  # normalize
                if "RSP" in self.tasks:
                    mean_label_rsp_test = torch.mean(label_rsp_test, dim=1, keepdim=True)
                    std_label_rsp_test = torch.std(label_rsp_test, dim=1, keepdim=True)
                    label_rsp_test = (label_rsp_test - mean_label_rsp_test) / std_label_rsp_test  # normalize

                last_frame = torch.unsqueeze(data[:, :, -1, :, :], 2).repeat(1, 1, max(self.num_of_gpu, 1), 1, 1)
                data = torch.cat((data, last_frame), 2)

                # last_sample = torch.unsqueeze(labels_test[-1, :], 0).repeat(max(self.num_of_gpu, 1), 1)
                # labels_test = torch.cat((labels_test, last_sample), 0)
                # labels_test = torch.diff(labels_test, dim=0)
                # labels_test = labels_test/ torch.std(labels_test)  # normalize
                # labels_test[torch.isnan(labels_test)] = 0
                
                out = self.model(data)

                pred_bvp_test = out[0]
                pred_rsp_test = out[1]
                pred_bp_test = out[2]

                if "BVP" in self.tasks:
                    mean_label_bvp_test = torch.mean(label_bvp_test, dim=1, keepdim=True).cpu()
                    std_label_bvp_test = torch.std(label_bvp_test, dim=1, keepdim=True).cpu()
                    mean_pred_bvp_test = torch.mean(pred_bvp_test, dim=1, keepdim=True).cpu()
                    std_pred_bvp_test = torch.std(pred_bvp_test, dim=1, keepdim=True).cpu()
                    # pred_bvp_test = (pred_bvp_test - mean_pred_bvp_test) / std_pred_bvp_test  # normalize
                if "RSP" in self.tasks:
                    mean_label_rsp_test = torch.mean(label_rsp_test, dim=1, keepdim=True).cpu()
                    std_label_rsp_test = torch.std(label_rsp_test, dim=1, keepdim=True).cpu()
                    mean_pred_rsp_test = torch.mean(pred_rsp_test, dim=1, keepdim=True).cpu()
                    std_pred_rsp_test = torch.std(pred_rsp_test, dim=1, keepdim=True).cpu()
                    # pred_rsp_test = (pred_rsp_test - mean_pred_rsp_test) / std_pred_rsp_test  # normalize


                if self.config.TEST.OUTPUT_SAVE_DIR:
                    if "BVP" in self.tasks:
                        label_bvp_test = label_bvp_test.cpu()
                        pred_bvp_test = pred_bvp_test.cpu()
                    if "RSP" in self.tasks:
                        label_rsp_test = label_rsp_test.cpu()
                        pred_rsp_test = pred_rsp_test.cpu()
                    if "BP" in self.tasks:
                        SBP_test = SBP_test.cpu()
                        DBP_test = DBP_test.cpu()
                        pred_bp_test = pred_bp_test.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])

                    if subj_index not in pred_bvp_dict.keys():
                        if "BVP" in self.tasks:
                            pred_bvp_dict[subj_index] = dict()
                            label_bvp_dict[subj_index] = dict()

                        if "RSP" in self.tasks:
                            pred_rsp_dict[subj_index] = dict()
                            label_rsp_dict[subj_index] = dict()

                        if "BP" in self.tasks:
                            pred_sbp_dict[subj_index] = dict()
                            pred_dbp_dict[subj_index] = dict()
                            label_sbp_dict[subj_index] = dict()
                            label_dbp_dict[subj_index] = dict()


                    if "BVP" in self.tasks:
                        if std_label_bvp_test[idx] > 0.001:
                            label_bvp_dict[subj_index][sort_index] = (label_bvp_test[idx] - mean_label_bvp_test[idx]) / std_label_bvp_test[idx]   #label_bvp_test[idx]    # standardize
                        else:
                            label_bvp_dict[subj_index][sort_index] = label_bvp_test[idx]
                        
                        if std_pred_bvp_test[idx] > 0.001:
                            pred_bvp_dict[subj_index][sort_index] = (pred_bvp_test[idx] - mean_pred_bvp_test[idx]) / std_pred_bvp_test[idx]   #pred_bvp_test[idx]    # standardize
                        else:
                            pred_bvp_dict[subj_index][sort_index] = pred_bvp_test[idx]

                    if "RSP" in self.tasks:
                        if (std_label_rsp_test[idx]) > 0.001:
                            label_rsp_dict[subj_index][sort_index] = (label_rsp_test[idx] - mean_label_rsp_test[idx]) / (std_label_rsp_test[idx])   #label_rsp_test[idx]    # standardize
                        else:
                            label_rsp_dict[subj_index][sort_index] = label_rsp_test[idx]

                        if (std_pred_rsp_test[idx]) > 0.001:
                            pred_rsp_dict[subj_index][sort_index] = (pred_rsp_test[idx] - mean_pred_rsp_test[idx]) / (std_pred_rsp_test[idx])  #pred_rsp_test[idx]    # standardize
                        else:
                            pred_rsp_dict[subj_index][sort_index] = pred_rsp_test[idx]

                    if "BP" in self.tasks:
                        label_sbp_dict[subj_index][sort_index] = SBP_test[idx]
                        label_dbp_dict[subj_index][sort_index] = DBP_test[idx]
                        pred_sbp_dict[subj_index][sort_index] = pred_bp_test[idx][0]
                        pred_dbp_dict[subj_index][sort_index] = pred_bp_test[idx][1]



        print('')
        if "BVP" in self.tasks:
            calculate_metrics(pred_bvp_dict, label_bvp_dict, self.config)
        
        if "RSP" in self.tasks:
            calculate_rsp_metrics(pred_rsp_dict, label_rsp_dict, self.config)

        if "BP" in self.tasks:
            calculate_bp_metrics(pred_sbp_dict, label_sbp_dict, pred_dbp_dict, label_dbp_dict, self.config)


        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs 
            if "BVP" in self.tasks:
                self.save_test_outputs(pred_bvp_dict, label_bvp_dict, self.config, suff="_bvp")
            if "RSP" in self.tasks:
                self.save_test_outputs(pred_rsp_dict, label_rsp_dict, self.config, suff="_rsp")
            if "BP" in self.tasks:
                self.save_test_outputs(pred_sbp_dict, label_sbp_dict, self.config, suff="_SBP")
                self.save_test_outputs(pred_dbp_dict, label_dbp_dict, self.config, suff="_DBP")

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)
