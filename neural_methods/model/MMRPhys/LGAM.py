import torch
import torch.nn as nn
from torch.nn import functional as F

class LGAM_org(nn.Module):
    def __init__(self, device, debug=False):
        self.cs_threshold = 0.3
        self.device = device
        self.debug = debug
        super().__init__()

    def forward(self, voxel_embeddings, label):
        with torch.no_grad():
            if self.debug:
                print("voxel_embeddings.shape", voxel_embeddings.shape) #[B, 16, 160, 7, 7])
            batch, channels, enc_frames, enc_height, enc_width = voxel_embeddings.shape
            if self.debug:
                print("label.shape", label.shape)
            
            # Generate a matrix of the dimension of voxel embeddings, with each temporal vector as ground-truth signal
            label_matrix = label.unsqueeze(1).repeat(1, channels, 1)
            label_matrix = label_matrix.unsqueeze(-1).unsqueeze(-1)
            label_matrix = label_matrix.repeat(1, 1, 1, enc_height, enc_width)
            label_matrix = label_matrix.to(device=self.device)
            if self.debug:
                print("label_matrix.shape", label_matrix.shape)
            
            # Generate cosine similarity map
            corr_matrix = F.cosine_similarity(voxel_embeddings, label_matrix, dim=2).abs()
            
            # Generate attention mask by picking the voxels highly correlating with with the ground-truth.
            # threshold = torch.mean(corr_matrix).item()  #dynamic threshold
            threshold = self.cs_threshold
            # corr_matrix[corr_matrix >= threshold] = 1
            # corr_matrix[corr_matrix < threshold] = 0
            att_mask = corr_matrix.unsqueeze(2).repeat(1, 1, enc_frames, 1, 1)
            
            if self.debug:
                print("att_mask.shape", att_mask.shape)
            # att_mask = voxel_embeddings * att_mask
            # if self.debug:
            #     print("att_mask.shape", att_mask.shape)
            return att_mask
        


class LGAM(nn.Module):
    def __init__(self, device, debug=False):
        self.cs_threshold = 0.3
        self.device = device
        self.debug = debug
        super().__init__()

    def forward(self, voxel_embeddings, label=None):
        with torch.no_grad():
            if self.debug:
                print("voxel_embeddings.shape", voxel_embeddings.shape) #[B, 16, 160, 7, 7])
            batch, channels, enc_frames, enc_height, enc_width = voxel_embeddings.shape
            
            vec_embeddings = voxel_embeddings.view(batch, -1, enc_frames)
            mean_vec = torch.mean(vec_embeddings, dim=1)
            vec_embeddings = mean_vec.unsqueeze(
                1).unsqueeze(-1).unsqueeze(-1).repeat(1, channels, 1, enc_height, enc_width)
            if self.debug:
                print("vec_embeddings.shape", vec_embeddings.shape)

            # Generate cosine similarity map
            corr_matrix = F.cosine_similarity(voxel_embeddings, vec_embeddings, dim=2).abs()
            
            # Generate attention mask by picking the voxels highly correlating with with the mean-vector.
            # threshold = torch.mean(corr_matrix).item()  #dynamic threshold
            threshold = self.cs_threshold
            # corr_matrix[corr_matrix >= threshold] = 1
            corr_matrix[corr_matrix < threshold] = 0
            att_mask = corr_matrix.unsqueeze(2).repeat(1, 1, enc_frames, 1, 1)
            
            if self.debug:
                print("att_mask.shape", att_mask.shape)

            return att_mask
