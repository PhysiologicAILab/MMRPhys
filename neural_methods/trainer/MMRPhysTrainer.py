"""Trainer for FactorizePhys."""
import os
import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics, calculate_rsp_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.model.MMRPhys.MMRPhys import MMRPhys
from neural_methods.model.MMRPhys.MMRPhysBig import MMRPhysBig
from neural_methods.model.MMRPhys.MMRPhysMedium import MMRPhysMedium
from neural_methods.model.MMRPhys.MMRPhysFuseM import MMRPhysFuseM
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
        md_config["FS"] = self.config.TRAIN.DATA.FS
        md_config["MD_TYPE"] = self.config.MODEL.MMRPhys.MD_TYPE
        md_config["MD_FSAM"] = self.config.MODEL.MMRPhys.MD_FSAM
        md_config["MD_S"] = self.config.MODEL.MMRPhys.MD_S
        md_config["MD_R"] = self.config.MODEL.MMRPhys.MD_R
        md_config["MD_STEPS"] = self.config.MODEL.MMRPhys.MD_STEPS
        md_config["MD_INFERENCE"] = self.config.MODEL.MMRPhys.MD_INFERENCE
        md_config["MD_RESIDUAL"] = self.config.MODEL.MMRPhys.MD_RESIDUAL        
        md_config["TASKS"] = self.config.MODEL.MMRPhys.TASKS

        self.md_infer = self.config.MODEL.MMRPhys.MD_INFERENCE
        self.use_fsam = self.config.MODEL.MMRPhys.MD_FSAM
        self.tasks = self.config.MODEL.MMRPhys.TASKS
        if "BVP" in self.tasks or "RSP" in self.tasks:
            pass
        else:
            print("Unknown modality... Only BVP and RSP are supported. Exiting the code...")
            exit()

        md_type = self.config.MODEL.MMRPhys.MD_TYPE.lower()
        self.use_bvp_hr = 0
        self.use_rsp_rr = 0

        if "snmf" in md_type:
            if "BVP" in self.tasks:
                self.use_bvp_hr = 1 if "label" in md_type else 2
            if "RSP" in self.tasks:
                self.use_rsp_rr = 1 if "label" in md_type else 2

        self.model = MMRPhys(frames=frames, md_config=md_config, in_channels=in_channels,
                                dropout=self.dropout_rate, device=self.device)  # [4, T, 72, 72]

        if model_type == "standard":
            self.model = MMRPhys(frames=frames, md_config=md_config, in_channels=in_channels, dropout=self.dropout_rate, device=self.device)  # [4, T, 72, 72]
        elif model_type == "big":
            self.model = MMRPhysBig(frames=frames, md_config=md_config, in_channels=in_channels, dropout=self.dropout_rate, device=self.device)  # [4, T, 144, 144]
        elif model_type == "medium":
            self.model = MMRPhysMedium(frames=frames, md_config=md_config, in_channels=in_channels, dropout=self.dropout_rate, device=self.device)  # [4, T, 36, 36]
        elif model_type == "fusem":
            self.model = MMRPhysFuseM(frames=frames, md_config=md_config, in_channels=in_channels, dropout=self.dropout_rate, device=self.device)  # [4, T, 36, 36]
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
            self.criterion1 = Neg_Pearson()
            self.criterion2 = Neg_Pearson()
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.config.TRAIN.LR)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.config.TRAIN.LR, epochs=self.config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif self.config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("MMRPhys trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        mean_training_loss_bvp = []
        mean_training_loss_rsp = []
        mean_valid_loss_bvp = []
        mean_valid_loss_rsp = []
        mean_appx_error = []
        lrs = []
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss1 = 0.0
            running_loss2 = 0.0

            train_loss = []
            train_loss_bvp = []
            train_loss_rsp = []
            appx_error_list = []
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=120)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                
                data = batch[0].to(self.device)
                labels = batch[1].to(self.device)

                if len(labels.shape) == 3:
                    if labels.shape[-1] > 4:       # BP4D dataset 
                        # bvp, rsp, eda, ecg, hr, rr, sysBP, avgBP, diaBP
                        label_bvp = labels[..., 0]
                        label_rsp = labels[..., 1]
                        label_eda = labels[..., 2]
                        label_hr = labels[..., 4]
                        label_rr = labels[..., 5]
                        label_sysBP = labels[..., 6]
                        label_avgBP = labels[..., 7]
                        label_diaBP = labels[..., 8]

                    elif labels.shape[-1] >= 3:     #SCAMPS dataset
                        label_bvp = labels[..., 0]
                        label_rsp = labels[..., 1]
                        label_hr = labels[..., 2]
                        # label_rr = labels[..., 3] #TODO: once SCAMPS dataset is added with RSP values, after downsampling.. uncomment this
                    else:                           # All other rPPG datasets (UBFC-rPPG, PURE, iBVP)
                        label_bvp = labels[..., 0]
                        label_hr = labels[..., 1]
                elif "BVP" in self.tasks:
                    label_bvp = labels
                elif "RSP" in self.tasks:
                    label_rsp = labels

                # if "BVP" in self.tasks:
                #     mean_label_bvp = torch.mean(label_bvp, dim=1).unsqueeze(1)
                #     std_label_bvp = torch.std(label_bvp, dim=1).unsqueeze(1)
                #     label_bvp = (label_bvp - mean_label_bvp) / std_label_bvp  # normalize
                # if "RSP" in self.tasks:
                #     mean_label_rsp = torch.mean(label_rsp, dim=1).unsqueeze(1)
                #     std_label_rsp = torch.std(label_rsp, dim=1).unsqueeze(1)
                #     label_rsp = (label_rsp - mean_label_rsp) / std_label_rsp  # normalize
                
                last_frame = torch.unsqueeze(data[:, :, -1, :, :], 2).repeat(1, 1, max(self.num_of_gpu, 1), 1, 1)
                data = torch.cat((data, last_frame), 2)

                # last_sample = torch.unsqueeze(labels[-1, :], 0).repeat(max(self.num_of_gpu, 1), 1)
                # labels = torch.cat((labels, last_sample), 0)
                # labels = torch.diff(labels, dim=0)
                # labels = labels/ torch.std(labels)  # normalize
                # labels[torch.isnan(labels)] = 0

                self.optimizer.zero_grad()
                if self.model.training and self.use_fsam:
                    if "BVP" in self.tasks and "RSP" in self.tasks:
                        if self.use_bvp_hr == 0:
                            pred_bvp, pred_rsp, vox_embed, factorized_embed, appx_error, factorized_embed_br, appx_error_br = self.model(data)
                        elif self.use_bvp_hr == 1:
                            pred_bvp, pred_rsp, vox_embed, factorized_embed, appx_error, factorized_embed_br, appx_error_br = self.model(data, label_bvp=label_bvp, label_rsp=label_rsp)
                        elif self.use_bvp_hr == 2:
                            pred_bvp, pred_rsp, vox_embed, factorized_embed, appx_error, factorized_embed_br, appx_error_br = self.model(data, label_bvp=label_hr, label_rsp=label_rr)
                    elif "BVP" in self.tasks:
                        if self.use_bvp_hr == 0:
                            pred_bvp, vox_embed, factorized_embed, appx_error = self.model(data)
                        elif self.use_bvp_hr == 1:
                            pred_bvp, vox_embed, factorized_embed, appx_error = self.model(data, label_bvp=label_bvp)
                        elif self.use_bvp_hr == 2:
                            pred_bvp, vox_embed, factorized_embed, appx_error = self.model(data, label_bvp=label_hr)
                    elif "RSP" in self.tasks:
                        if self.use_rsp_rr == 0:
                            pred_rsp, vox_embed, factorized_embed, appx_error = self.model(data)
                        elif self.use_rsp_rr == 1:
                            pred_rsp, vox_embed, factorized_embed, appx_error = self.model(data, label_rsp=label_rsp)
                        elif self.use_rsp_rr == 2:
                            pred_rsp, vox_embed, factorized_embed, appx_error = self.model(data, label_rsp=label_rr)
                            
                    # else:
                    #     pred_rsp, vox_embed, factorized_embed_br, appx_error_br = self.model(data)
                else:
                    if "BVP" in self.tasks and "RSP" in self.tasks:
                        pred_bvp, pred_rsp, vox_embed = self.model(data)
                    elif "BVP" in self.tasks:
                        pred_bvp, vox_embed = self.model(data)
                    elif "RSP" in self.tasks:
                        pred_rsp, vox_embed = self.model(data)
                    # else:
                    #     pred_rsp, vox_embed = self.model(data)
                
                if "BVP" in self.tasks:
                    # mean_pred_bvp = torch.mean(pred_bvp, dim=1).unsqueeze(1)
                    # std_pred_bvp = torch.std(pred_bvp, dim=1).unsqueeze(1)
                    # pred_bvp = (pred_bvp - mean_pred_bvp) / std_pred_bvp  # normalize
                    loss_bvp = self.criterion1(pred_bvp, label_bvp)

                if "RSP" in self.tasks:
                    # mean_pred_rsp = torch.mean(pred_rsp, dim=1).unsqueeze(1)
                    # std_pred_rsp = torch.std(pred_rsp, dim=1).unsqueeze(1)
                    # pred_rsp = (pred_rsp - mean_pred_rsp) / std_pred_rsp  # normalize
                    loss_rsp = self.criterion2(pred_rsp, label_rsp)
                
                if "BVP" in self.tasks and "RSP" in self.tasks:
                    loss = loss_bvp + loss_rsp
                elif "BVP" in self.tasks:
                    loss = loss_bvp
                else:
                    loss = loss_rsp
                
                loss.backward()
                if "BVP" in self.tasks:
                    running_loss1 += loss_bvp.item()
                if "RSP" in self.tasks:
                    running_loss2 += loss_rsp.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    if "BVP" in self.tasks and "RSP" in self.tasks:
                        print(f'[{epoch}, {idx + 1:5d}] loss_bvp: {running_loss1 / 100:.3f} loss_rsp: {running_loss2 / 100:.3f}')    
                        running_loss1 = 0.0
                        running_loss2 = 0.0
                    elif "BVP" in self.tasks:
                        print(f'[{epoch}, {idx + 1:5d}] loss_bvp: {running_loss1 / 100:.3f}')    
                        running_loss1 = 0.0
                    elif "RSP" in self.tasks:
                        print(f'[{epoch}, {idx + 1:5d}] loss_rsp: {running_loss2 / 100:.3f}')    
                        running_loss2 = 0.0

                train_loss.append(loss.item())
                if "BVP" in self.tasks:
                    train_loss_bvp.append(loss_bvp.item())
                if "RSP" in self.tasks:
                    train_loss_rsp.append(loss_rsp.item())
                if self.use_fsam:
                    appx_error_list.append(appx_error.item())

                # Append the current learning rate to the list
                lrs.append(self.scheduler.get_last_lr())

                self.optimizer.step()
                self.scheduler.step()
                
                if self.use_fsam:
                    if "BVP" in self.tasks and "RSP" in self.tasks:
                        tbar.set_postfix({"appx_error": appx_error.item(), "loss_bvp":loss_bvp.item(), "loss_rsp": loss_rsp.item()}, loss=loss.item())
                    elif "BVP" in self.tasks:
                        tbar.set_postfix({"appx_error": appx_error.item()}, loss=loss.item())
                    elif "RSP" in self.tasks:
                        tbar.set_postfix({"appx_error": appx_error.item()}, loss=loss.item())
                else:
                    if "BVP" in self.tasks and "RSP" in self.tasks:
                        tbar.set_postfix({"loss_bvp":loss_bvp.item(), "loss_rsp": loss_rsp.item()}, loss=loss.item())
                    else:
                        tbar.set_postfix(loss=loss.item())

            # Append the mean training loss for the epoch
            if "BVP" in self.tasks:
                mean_training_loss_bvp.append(np.mean(train_loss_bvp))
            if "RSP" in self.tasks:
                mean_training_loss_rsp.append(np.mean(train_loss_rsp))
            if self.use_fsam:
                mean_appx_error.append(np.mean(appx_error_list))
                print("Mean train loss: {}, Mean appx error: {}".format(np.round(np.mean(train_loss), 3), np.round(np.mean(appx_error_list), 3)))
            else:
                print("Mean train loss: {}".format(np.round(np.mean(train_loss), 3)))

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH:
                if "BVP" in self.tasks and "RSP" in self.tasks:
                    valid_loss_bvp, valid_loss_rsp = self.valid(data_loader)
                    print('validation losses: ', valid_loss_bvp, valid_loss_rsp)
                    total_valid_loss = valid_loss_bvp + valid_loss_rsp
                elif "BVP" in self.tasks:
                    valid_loss_bvp = self.valid(data_loader)
                    print('validation loss: ', valid_loss_bvp)
                    total_valid_loss = valid_loss_bvp
                elif "RSP" in self.tasks:
                    valid_loss_rsp = self.valid(data_loader)
                    print('validation loss: ', valid_loss_rsp)
                    total_valid_loss = valid_loss_rsp

                if "BVP" in self.tasks:
                    mean_valid_loss_bvp.append(valid_loss_bvp)
                if "RSP" in self.tasks:
                    mean_valid_loss_rsp.append(valid_loss_rsp)

                if self.min_valid_loss is None:
                    self.min_valid_loss = total_valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (total_valid_loss < self.min_valid_loss):
                    self.min_valid_loss = total_valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            if "BVP" in self.tasks and "RSP" in self.tasks:
                self.plot_losses_and_lrs(mean_training_loss_bvp, mean_valid_loss_bvp, lrs, self.config, mean_training_loss_rsp, mean_valid_loss_rsp)
            elif "BVP" in self.tasks:
                self.plot_losses_and_lrs(mean_training_loss_bvp, mean_valid_loss_bvp, lrs, self.config)
            elif "RSP" in self.tasks:
                self.plot_losses_and_lrs(mean_training_loss_rsp, mean_valid_loss_rsp, lrs, self.config)

    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validing===")
        valid_loss_bvp = []
        valid_loss_rsp = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=120)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")

                data, labels = valid_batch[0].to(self.device), valid_batch[1].to(self.device)
                if len(labels.shape) == 3:
                    if labels.shape[-1] > 4:       # BP4D dataset 
                        label_bvp = labels[..., 0]
                        label_rsp = labels[..., 1]
                        label_eda = labels[..., 2]
                        label_hr = labels[..., 4]
                        label_rr = labels[..., 5]
                        label_sysBP = labels[..., 6]
                        label_avgBP = labels[..., 7]
                        label_diaBP = labels[..., 8]

                    elif labels.shape[-1] >= 3:     #SCAMPS dataset
                        label_bvp = labels[..., 0]
                        label_rsp = labels[..., 1]
                        label_hr = labels[..., 2]
                        # label_rr = labels[..., 3] #TODO: once SCAMPS dataset is added with RSP values, after downsampling.. uncomment this
                    else:                           # All other rPPG datasets (UBFC-rPPG, PURE, iBVP)
                        label_bvp = labels[..., 0]
                        label_hr = labels[..., 1]
                elif "BVP" in self.tasks:
                    label_bvp = labels
                elif "RSP" in self.tasks:
                    label_rsp = labels

                # if "BVP" in self.tasks:
                #     mean_label_bvp = torch.mean(label_bvp, dim=1).unsqueeze(1)
                #     std_label_bvp = torch.std(label_bvp, dim=1).unsqueeze(1)
                #     label_bvp = (label_bvp - mean_label_bvp) / std_label_bvp  # normalize
                # if "RSP" in self.tasks:
                #     mean_label_rsp = torch.mean(label_rsp, dim=1).unsqueeze(1)
                #     std_label_rsp = torch.std(label_rsp, dim=1).unsqueeze(1)
                #     label_rsp = (label_rsp - mean_label_rsp) / std_label_rsp  # normalize

                last_frame = torch.unsqueeze(data[:, :, -1, :, :], 2).repeat(1, 1, max(self.num_of_gpu, 1), 1, 1)
                data = torch.cat((data, last_frame), 2)

                # last_sample = torch.unsqueeze(labels[-1, :], 0).repeat(max(self.num_of_gpu, 1), 1)
                # labels = torch.cat((labels, last_sample), 0)
                # labels = torch.diff(labels, dim=0)
                # labels = labels/ torch.std(labels)  # normalize
                # labels[torch.isnan(labels)] = 0

                if self.md_infer and self.use_fsam:
                    if "BVP" in self.tasks and "RSP" in self.tasks:
                        pred_bvp, pred_rsp, vox_embed, factorized_embed, appx_error, factorized_embed_br, appx_error_br = self.model(data)
                    elif "BVP" in self.tasks:
                        pred_bvp, vox_embed, factorized_embed, appx_error = self.model(data)
                    else:
                        pred_rsp, vox_embed, factorized_embed_br, appx_error_br = self.model(data)
                else:
                    if "BVP" in self.tasks and "RSP" in self.tasks:
                        pred_bvp, pred_rsp, vox_embed = self.model(data)
                    elif "BVP" in self.tasks:
                        pred_bvp, vox_embed = self.model(data)
                    else:
                        pred_rsp, vox_embed = self.model(data)

                if "BVP" in self.tasks:
                    # mean_pred_bvp = torch.mean(pred_bvp, dim=1).unsqueeze(1)
                    # std_pred_bvp = torch.std(pred_bvp, dim=1).unsqueeze(1)
                    # pred_bvp = (pred_bvp - mean_pred_bvp) / std_pred_bvp  # normalize
                    loss_bvp = self.criterion1(pred_bvp, label_bvp)
                    valid_loss_bvp.append(loss_bvp.item())

                if "RSP" in self.tasks:
                    # mean_pred_rsp = torch.mean(pred_rsp, dim=1).unsqueeze(1)
                    # std_pred_rsp = torch.std(pred_rsp, dim=1).unsqueeze(1)
                    # pred_rsp = (pred_rsp - mean_pred_rsp) / std_pred_rsp  # normalize
                    loss_rsp = self.criterion2(pred_rsp, label_rsp)
                    valid_loss_rsp.append(loss_rsp.item())
                
                if "BVP" in self.tasks and "RSP" in self.tasks:
                    loss = loss_bvp + loss_rsp
                elif "BVP" in self.tasks:
                    loss = loss_bvp
                elif "RSP" in self.tasks:
                    loss = loss_rsp

                valid_step += 1
                # vbar.set_postfix(loss=loss.item())
                if self.md_infer and self.use_fsam:
                    if "BVP" in self.tasks and "RSP" in self.tasks:
                        vbar.set_postfix({"appx_error": appx_error.item(), "loss_bvp":loss_bvp.item(), "loss_rsp": loss_rsp.item()}, loss=loss.item())
                    else:
                        vbar.set_postfix(loss=loss.item())
                else:
                    if "BVP" in self.tasks and "RSP" in self.tasks:
                        vbar.set_postfix({"loss_bvp":loss_bvp.item(), "loss_rsp": loss_rsp.item()}, loss=loss.item())
                    else:
                        vbar.set_postfix(loss=loss.item())

            if "BVP" in self.tasks:
                valid_loss_bvp = np.asarray(valid_loss_bvp)
            if "RSP" in self.tasks:
                valid_loss_rsp = np.asarray(valid_loss_rsp)
        
        if "BVP" in self.tasks and "RSP" in self.tasks:
            return np.mean(valid_loss_bvp), np.mean(valid_loss_rsp)
        elif "BVP" in self.tasks:
            return np.mean(valid_loss_bvp)
        elif "RSP" in self.tasks:
            return np.mean(valid_loss_rsp)

    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("===Testing===")
        pred_bvp_dict = dict()
        label_bvp_dict = dict()
        pred_rsp_dict = dict()
        label_rsp_dict = dict()

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
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data, labels_test = test_batch[0].to(self.device), test_batch[1].to(self.device)

                if len(labels_test.shape) == 3:
                    if labels_test.shape[-1] > 4:       # BP4D dataset 
                        label_bvp_test = labels_test[..., 0]
                        label_rsp_test = labels_test[..., 1]
                        label_eda_test = labels_test[..., 2]
                        label_hr_test = labels_test[..., 4]
                        label_rr_test = labels_test[..., 5]
                        label_sysBP_test = labels_test[..., 6]
                        label_avgBP_test = labels_test[..., 7]
                        label_diaBP_test = labels_test[..., 8]

                    elif labels_test.shape[-1] >= 3:     #SCAMPS dataset
                        label_bvp_test = labels_test[..., 0]
                        label_rsp_test = labels_test[..., 1]
                        label_hr_test = labels_test[..., 2]
                        # label_rr = labels_test[..., 3] #TODO: once SCAMPS dataset is added with RSP values, after downsampling.. uncomment this
                    else:                           # All other rPPG datasets (UBFC-rPPG, PURE, iBVP)
                        label_bvp_test = labels_test[..., 0]
                        label_hr_test = labels_test[..., 1]

                elif "BVP" in self.tasks:
                    label_bvp_test = labels_test
                elif "RSP" in self.tasks:
                    label_rsp_test = labels_test

                # if "BVP" in self.tasks:
                #     mean_label_bvp_test = torch.mean(label_bvp_test, dim=1).unsqueeze(1)
                #     std_label_bvp_test = torch.std(label_bvp_test, dim=1).unsqueeze(1)
                #     label_bvp_test = (label_bvp_test - mean_label_bvp_test) / std_label_bvp_test  # normalize
                # if "RSP" in self.tasks:
                #     mean_label_rsp_test = torch.mean(label_rsp_test, dim=1).unsqueeze(1)
                #     std_label_rsp_test = torch.std(label_rsp_test, dim=1).unsqueeze(1)
                #     label_rsp_test = (label_rsp_test - mean_label_rsp_test) / std_label_rsp_test  # normalize

                last_frame = torch.unsqueeze(data[:, :, -1, :, :], 2).repeat(1, 1, max(self.num_of_gpu, 1), 1, 1)
                data = torch.cat((data, last_frame), 2)

                # last_sample = torch.unsqueeze(labels_test[-1, :], 0).repeat(max(self.num_of_gpu, 1), 1)
                # labels_test = torch.cat((labels_test, last_sample), 0)
                # labels_test = torch.diff(labels_test, dim=0)
                # labels_test = labels_test/ torch.std(labels_test)  # normalize
                # labels_test[torch.isnan(labels_test)] = 0

                if self.md_infer and self.use_fsam:
                    if "BVP" in self.tasks and "RSP" in self.tasks:
                        pred_bvp_test, pred_rsp_test, vox_embed, factorized_embed, appx_error, factorized_embed_br, appx_error_br = self.model(data)
                    elif "BVP" in self.tasks:
                        pred_bvp_test, vox_embed, factorized_embed, appx_error = self.model(data)
                    else:
                        pred_rsp_test, vox_embed, factorized_embed_br, appx_error_br = self.model(data)
                else:
                    if "BVP" in self.tasks and "RSP" in self.tasks:
                        pred_bvp_test, pred_rsp_test, vox_embed = self.model(data)
                    elif "BVP" in self.tasks:
                        pred_bvp_test, vox_embed = self.model(data)
                    else:
                        pred_rsp_test, vox_embed = self.model(data)

                if "BVP" in self.tasks:
                    mean_label_bvp_test = torch.mean(label_bvp_test, dim=1).unsqueeze(1).cpu()
                    std_label_bvp_test = torch.std(label_bvp_test, dim=1).unsqueeze(1).cpu()
                    mean_pred_bvp_test = torch.mean(pred_bvp_test, dim=1).unsqueeze(1).cpu()
                    std_pred_bvp_test = torch.std(pred_bvp_test, dim=1).unsqueeze(1).cpu()
                    # pred_bvp_test = (pred_bvp_test - mean_pred_bvp_test) / std_pred_bvp_test  # normalize
                if "RSP" in self.tasks:
                    mean_label_rsp_test = torch.mean(label_rsp_test, dim=1).unsqueeze(1).cpu()
                    std_label_rsp_test = torch.std(label_rsp_test, dim=1).unsqueeze(1).cpu()
                    mean_pred_rsp_test = torch.mean(pred_rsp_test, dim=1).unsqueeze(1).cpu()
                    std_pred_rsp_test = torch.std(pred_rsp_test, dim=1).unsqueeze(1).cpu()
                    # pred_rsp_test = (pred_rsp_test - mean_pred_rsp_test) / std_pred_rsp_test  # normalize

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    if "BVP" in self.tasks:
                        label_bvp_test = label_bvp_test.cpu()
                        pred_bvp_test = pred_bvp_test.cpu()
                    if "RSP" in self.tasks:
                        label_rsp_test = label_rsp_test.cpu()
                        pred_rsp_test = pred_rsp_test.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in pred_bvp_dict.keys():
                        pred_bvp_dict[subj_index] = dict()
                        label_bvp_dict[subj_index] = dict()
                        pred_rsp_dict[subj_index] = dict()
                        label_rsp_dict[subj_index] = dict()

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
                        if std_label_rsp_test[idx] > 0.001:
                            label_rsp_dict[subj_index][sort_index] = (label_rsp_test[idx] - mean_label_rsp_test[idx]) / std_label_rsp_test[idx]   #label_rsp_test[idx]    # standardize
                        else:
                            label_rsp_dict[subj_index][sort_index] = label_rsp_test[idx]

                        if std_pred_rsp_test[idx] > 0.001:
                            pred_rsp_dict[subj_index][sort_index] = (pred_rsp_test[idx] - mean_pred_rsp_test[idx]) / std_pred_rsp_test[idx]   #pred_rsp_test[idx]    # standardize
                        else:
                            pred_rsp_dict[subj_index][sort_index] = pred_rsp_test[idx]

        print('')
        if "BVP" in self.tasks:
            calculate_metrics(pred_bvp_dict, label_bvp_dict, self.config)
        if "RSP" in self.tasks:
            calculate_rsp_metrics(pred_rsp_dict, label_rsp_dict, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs 
            if "BVP" in self.tasks:
                self.save_test_outputs(pred_bvp_dict, label_bvp_dict, self.config, suff="_bvp")
            if "RSP" in self.tasks:
                self.save_test_outputs(pred_rsp_dict, label_rsp_dict, self.config, suff="_rsp")

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)
