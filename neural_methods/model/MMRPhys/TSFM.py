"""
Efficient and Robust Multidimensional Attention in Remote Physiological Sensing through Target Signal Constrained Factorization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np


class _MatrixDecompositionBase(nn.Module):
    def __init__(self, device, md_config, debug=False, dim="3D"):
        super().__init__()

        self.dim = dim
        self.md_type = md_config["MD_TYPE"]
        self.S = md_config["MD_S"]
        self.R = md_config["MD_R"]
        self.debug = debug

        self.train_steps = md_config["MD_STEPS"]
        self.eval_steps = md_config["MD_STEPS"]

        self.inv_t = md_config["INV_T"]
        self.eta = md_config["ETA"]

        self.rand_init = md_config["RAND_INIT"]
        self.device = device

        # print('Dimension:', self.dim)
        # print('S', self.S)
        # print('D', self.D)
        # print('R', self.R)
        # print('train_steps', self.train_steps)
        # print('eval_steps', self.eval_steps)
        # print('inv_t', self.inv_t)
        # print('eta', self.eta)
        # print('rand_init', self.rand_init)

    def get_epsilon(self):
        """Return appropriate epsilon based on precision"""
        # Check if torch.amp.autocast is enabled
        if hasattr(torch, 'is_autocast_enabled') and torch.is_autocast_enabled():
            return 1e-3  # Use larger epsilon for half precision
        
        # Try to check parameter dtype
        try:
            param = next(self.parameters())
            is_half = param.dtype == torch.float16
            return 1e-3 if is_half else 1e-6
        except StopIteration:
            # No parameters available, default to fp32 epsilon
            return 1e-6

    def _build_bases(self, B, S, D, R):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, y=None, return_bases=False):

        if self.debug:
            print("Org x.shape", x.shape)

        if self.dim == "3D":        # (B, C, T, H, W) -> (B * S, D, N)
            B, C, T, H, W = x.shape

            # # dimension of vector of our interest is T (rPPG signal as T dimension), so forming this as vector
            # # From spatial and channel dimension, which are features, only 2-4 shall be enough to generate the approximated attention matrix
            D = T // self.S
            N = C * H * W 

            # # smoothening the temporal dimension
            # x = x.view(B * self.S, N, D)
            # # print("Intermediate-1 x", x.shape)

            # sample_1 = x[:, :, 0].unsqueeze(2)
            # sample_2 = x[:, :, -1].unsqueeze(2)
            # x = torch.cat([sample_1, x, sample_2], dim=2)
            # gaussian_kernel = [1.0, 1.0, 1.0]
            # kernels = torch.FloatTensor([[gaussian_kernel]]).repeat(N, N, 1).to(self.device)
            # bias = torch.FloatTensor(torch.zeros(N)).to(self.device)
            # x = F.conv1d(x, kernels, bias=bias, padding="valid")
            # x = (x - x.min()) / (x.max() - x.min())

            # x = x.permute(0, 2, 1)
            # # print("Intermediate-2 x", x.shape)

            x = x.view(B * self.S, D, N)

        elif self.dim == "2D":      # (B, C, H, W) -> (B * S, D, N)
            B, C, H, W = x.shape
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)

        elif self.dim == "2D_TSM":  # (B*frame_depth, C, H, W) -> (B, D, N)
            B, C, H, W = x.shape
            BN = B
            B = B // self.S
            D = self.S
            N = C * H * W
            x = x.view(B, D, N)
            self.S = 1  # re-setting this for local inference

        elif self.dim == "1D":                       # (B, C, L) -> (B * S, D, N)
            B, C, L = x.shape
            D = L // self.S
            N = C
            x = x.view(B * self.S, D, N)

        else:
            print("Dimension not supported")
            exit()

        if self.debug:
            print("MD_Type", self.md_type)
            print("MD_S", self.S)
            print("MD_D", D)
            print("MD_N", N)
            print("MD_R", self.R)
            print("MD_TRAIN_STEPS", self.train_steps)
            print("MD_EVAL_STEPS", self.eval_steps)
            print("x.view(B * self.S, D, N)", x.shape)

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R)
        else:
            bases = self.bases.repeat(B, 1, 1).to(self.device)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))


        if self.dim == "3D":

            apply_smoothening = False
            if apply_smoothening:
                # smoothening the temporal dimension
                x = x.view(B, D * self.S, N)    #Joining temporal dimension for contiguous smoothening
                # print("Intermediate-0 x", x.shape)            
                x = x.permute(0, 2, 1)
                # print("Intermediate-1 x", x.shape)

                sample_1 = x[:, :, 0].unsqueeze(2)
                # sample_2 = x[:, :, 0].unsqueeze(2)
                sample_3 = x[:, :, -1].unsqueeze(2)
                # sample_4 = x[:, :, -1].unsqueeze(2)
                x = torch.cat([sample_1, x, sample_3], dim=2)
                # x = torch.cat([sample_1, sample_2, x, sample_3, sample_4], dim=2)
                # gaussian_kernel = [0.25, 0.50, 0.75, 0.50, 0.25]
                # gaussian_kernel = [0.33, 0.66, 1.00, 0.66, 0.33]
                # gaussian_kernel = [0.3, 0.7, 1.0, 0.7, 0.3]
                # gaussian_kernel = [0.3, 1.0, 1.0, 1.0, 0.3]
                # gaussian_kernel = [0.20, 0.80, 1.00, 0.80, 0.20]
                # gaussian_kernel = [1.0, 1.0, 1.0]
                gaussian_kernel = [0.8, 1.0, 0.8]
                kernels = torch.FloatTensor([[gaussian_kernel]]).repeat(N, N, 1).to(self.device)
                bias = torch.FloatTensor(torch.zeros(N)).to(self.device)
                x = F.conv1d(x, kernels, bias=bias, padding="valid")
                # x = (x - x.min()) / (x.max() - x.min())
                # x = (x - x.mean()) / (x.std())
                # x = x - x.min()
                x = (x - x.min())/(x.std())

                # print("Intermediate-2 x", x.shape)

            # (B * S, D, N) -> (B, C, T, H, W)
            x = x.view(B, C, T, H, W)
        elif self.dim == "2D":
            # (B * S, D, N) -> (B, C, H, W)
            x = x.view(B, C, H, W)

        elif self.dim == "2D_TSM":
            # (B, D, N) -> (B, C, H, W)
            x = x.view(BN, C, H, W)

        else:
            # (B * S, D, N) -> (B, C, L)
            x = x.view(B, C, L)

        # (B * L, D, R) -> (B, L, N, D)
        bases = bases.view(B, self.S, D, self.R)

        if not self.rand_init and not self.training and not return_bases:
            self.online_update(bases)

        # if not self.rand_init or return_bases:
        #     return x, bases
        # else:
        return x

    @torch.no_grad()
    def online_update(self, bases):
        # (B, S, D, R) -> (S, D, R)
        update = bases.mean(dim=0)
        self.bases += self.eta * (update - self.bases)
        self.bases = F.normalize(self.bases, dim=1)


class NMF(_MatrixDecompositionBase):
    def __init__(self, device, md_config, debug=False, dim="3D"):
        super().__init__(device, md_config, debug=debug, dim=dim)
        self.device = device
        self.inv_t = 1

    def _build_bases(self, B, S, D, R):
        # bases = torch.rand((B * S, D, R)).to(self.device)
        bases = torch.ones((B * S, D, R)).to(self.device)
        bases = F.normalize(bases, dim=1)

        return bases

    @torch.no_grad()
    def local_step(self, x, bases, coef):
        eps = self.get_epsilon()
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Clip denominator to prevent extreme values
        denominator = torch.clamp(denominator, min=eps)
        # Multiplicative Update
        coef = coef * (numerator / denominator)
        coef = torch.nan_to_num(coef, nan=0.0, posinf=1.0, neginf=0.0)  # Handle any NaNs

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Clip denominator
        denominator = torch.clamp(denominator, min=eps)
        # Multiplicative Update
        bases = bases * (numerator / denominator)
        bases = torch.nan_to_num(bases, nan=0.0, posinf=1.0, neginf=0.0)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        eps = self.get_epsilon()
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Clip denominator
        denominator = torch.clamp(denominator, min=eps)
        # multiplication update
        coef = coef * (numerator / denominator)
        coef = torch.nan_to_num(coef, nan=0.0, posinf=1.0, neginf=0.0)  # Handle any NaNs

        return coef


class _SmoothMatrixDecompositionBase(nn.Module):
    def __init__(self, device, md_config, debug=False, dim="3D"):
        super().__init__()

        self.dim = dim
        self.md_type = md_config["MD_TYPE"]
        self.S = md_config["MD_S"]
        self.R = md_config["MD_R"]
        self.debug = debug

        self.train_steps = md_config["MD_STEPS"]
        self.eval_steps = md_config["MD_STEPS"]

        self.inv_t = md_config["INV_T"]
        self.eta = md_config["ETA"]

        self.rand_init = md_config["RAND_INIT"]
        self.device = device

        # print('Dimension:', self.dim)
        # print('S', self.S)
        # print('D', self.D)
        # print('R', self.R)
        # print('train_steps', self.train_steps)
        # print('eval_steps', self.eval_steps)
        # print('inv_t', self.inv_t)
        # print('eta', self.eta)
        # print('rand_init', self.rand_init)

    def get_epsilon(self):
        """Return appropriate epsilon based on precision"""
        # Check if torch.amp.autocast is enabled
        if hasattr(torch, 'is_autocast_enabled') and torch.is_autocast_enabled():
            return 1e-3  # Use larger epsilon for half precision
        
        # Try to check parameter dtype
        try:
            param = next(self.parameters())
            is_half = param.dtype == torch.float16
            return 1e-3 if is_half else 1e-6
        except StopIteration:
            # No parameters available, default to fp32 epsilon
            return 1e-6

    def _build_bases(self, B, S, D, R):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    @torch.no_grad()
    def local_inference(self, x, SNMF_estimators, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), torch.bmm(SNMF_estimators, bases))
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, SNMF_estimators, bases, coef)

        return bases, coef

    def compute_coef(self, x, SNMF_estimators, bases, coef):
        raise NotImplementedError

    def forward(self, x, y=None, return_bases=False):

        if self.debug:
            print("\n*****************")
            print("Org x.shape", x.shape)
            print("y.shape:", y.shape)
            print("MD_Type:", self.md_type)
            print("*****************")

        if self.dim == "3D":        # (B, C, T, H, W) -> (B * S, D, N)
            B, C, T, H, W = x.shape

            # # dimension of vector of our interest is T (rPPG signal as T dimension), so forming this as vector
            # # From spatial and channel dimension, which are features, only 2-4 shall be enough to generate the approximated attention matrix
            D = T // self.S
            N = C * H * W
            x = x.view(B * self.S, D, N)

        elif self.dim == "2D":      # (B, C, H, W) -> (B * S, D, N)
            B, C, H, W = x.shape
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)

        elif self.dim == "1D":                       # (B, C, L) -> (B * S, D, N)
            B, C, L = x.shape
            D = L // self.S
            N = C
            x = x.view(B * self.S, D, N)

        else:
            print("Dimension not supported")
            exit()

        P = D

        if "rbf" in self.md_type.lower():
            sig0 = torch.tensor(6.0)
            sig1 = torch.tensor(8.0)
            sig2 = torch.tensor(12.0)
            sig3 = torch.tensor(15.0)
            sig4 = torch.tensor(18.0)
            sig5 = torch.tensor(24.0)

            # dt = torch.tensor((P - 1) / (D - 1))
            tt = torch.arange(0, D).unsqueeze(1)
            nn = torch.arange(0, P).unsqueeze(0)
            # print(tt.shape, nn.shape)
            dd = torch.pow(tt - nn, 2)
            rbf0 = torch.exp((-0.5 * dd)/(2 * torch.pow(sig0, 2)))
            rbf1 = torch.exp((-0.5 * dd)/(2 * torch.pow(sig1, 2)))
            rbf2 = torch.exp((-0.5 * dd)/(2 * torch.pow(sig2, 2)))
            rbf3 = torch.exp((-0.5 * dd)/(2 * torch.pow(sig3, 2)))
            rbf4 = torch.exp((-0.5 * dd)/(2 * torch.pow(sig4, 2)))
            rbf5 = torch.exp((-0.5 * dd)/(2 * torch.pow(sig5, 2)))
            rbfN = torch.ones(P, 1)

            SNMF_estimators = torch.cat([
                rbf0[:, torch.arange(0, P, 1)],
                rbf1[:, torch.arange(0, P, 1)],
                rbf2[:, torch.arange(0, P, 2)],
                rbf3[:, torch.arange(0, P, 2)],
                rbf4[:, torch.arange(0, P, 3)],
                rbf5[:, torch.arange(0, P, 3)],
                rbfN,
            ], dim=1)

            SNMF_estimators = SNMF_estimators.repeat(B * self.S, 1, 1).to(self.device)
            SNMF_est_shape2 = SNMF_estimators.shape[2]

        elif "label" in self.md_type.lower():
            SNMF_estimators = torch.zeros((B, P, 1)).to(self.device)    #only label as estimator
            for bt in range(B):
                sig = y[bt, :]
                mn = torch.min(sig)
                mx = torch.max(sig)
                sig = 1 + ((sig - mn)/(mx - mn))
                SNMF_estimators[bt, :, 0] = sig

            SNMF_est_shape2 = SNMF_estimators.shape[2]

        else:
            print("Invalid SNMF option specified... Exiting...")
            exit()

        if self.debug:
            print("MD_Type", self.md_type)
            print("MD_S", self.S)
            print("MD_D", D)
            print("MD_N", N)
            print("MD_R", self.R)
            print("MD_TRAIN_STEPS", self.train_steps)
            print("MD_EVAL_STEPS", self.eval_steps)
            print("x.view(B * self.S, D, N)", x.shape)
            print("SNMF_estimators.shape", SNMF_estimators.shape)

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, SNMF_est_shape2, self.R)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, SNMF_est_shape2, self.R)
        else:
            bases = self.bases.repeat(B, 1, 1).to(self.device)

        bases, coef = self.local_inference(x, SNMF_estimators, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, SNMF_estimators, bases, coef)

        bases_prod = torch.bmm(SNMF_estimators, bases)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases_prod, coef.transpose(1, 2))


        if self.dim == "3D":
            # (B * S, D, N) -> (B, C, T, H, W)
            x = x.view(B, C, T, H, W)
        elif self.dim == "2D":
            # (B * S, D, N) -> (B, C, H, W)
            x = x.view(B, C, H, W)
        else:
            # (B * S, D, N) -> (B, C, L)
            x = x.view(B, C, L)

        # (B * L, D, R) -> (B, L, N, D)
        # bases_prod_updated = torch.bmm(SNMF_estimators, bases)
        bases = torch.bmm(SNMF_estimators.transpose(1, 2), bases_prod)
        bases = bases.view(B, self.S, SNMF_est_shape2, self.R)

        if not self.rand_init and not self.training and not return_bases:
            print("it comes here")
            self.online_update(bases)

        # if not self.rand_init or return_bases:
        #     return x, bases
        # else:
        return x

    @torch.no_grad()
    def online_update(self, bases):
        # (B, S, D, R) -> (S, D, R)
        update = bases.mean(dim=0)
        self.bases += self.eta * (update - self.bases)
        self.bases = F.normalize(self.bases, dim=1)


class SNMF(_SmoothMatrixDecompositionBase):
    def __init__(self, device, md_config, debug=False, dim="3D"):
        super().__init__(device, md_config, debug=debug, dim=dim)
        self.device = device
        self.inv_t = 1

    def _build_bases(self, B, S, D, R):
        bases = torch.rand((B * S, D, R)).to(self.device)
        # bases = torch.ones((B * S, D, R)).to(self.device)
        bases = F.normalize(bases, dim=1)

        return bases

    @torch.no_grad()
    def local_step(self, x, SNMF_estimators, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        eps = self.get_epsilon()
        bases_prod = torch.bmm(SNMF_estimators, bases)

        numerator = torch.bmm(x.transpose(1, 2), bases_prod)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases_prod.transpose(1, 2).bmm(bases_prod))
        # Clip denominator to prevent extreme values
        denominator = torch.clamp(denominator, min=eps)
        # Multiplicative Update
        coef = coef * (numerator / denominator)
        coef = torch.nan_to_num(coef, nan=0.0, posinf=1.0, neginf=0.0)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases_prod.bmm(coef.transpose(1, 2).bmm(coef))

        # Clip denominator
        denominator = torch.clamp(denominator, min=eps)
        # Multiplicative Update
        # bases = (bases_prod * numerator / (denominator + 1e-6))
        bases = torch.bmm(SNMF_estimators.transpose(1, 2), (bases_prod * (numerator / denominator)))
        bases = torch.nan_to_num(bases, nan=0.0, posinf=1.0, neginf=0.0)
        # print(bases.shape)
        # exit()

        return bases, coef

    def compute_coef(self, x, SNMF_estimators, bases, coef):
        eps = self.get_epsilon()
        bases_prod = torch.bmm(SNMF_estimators, bases)

        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases_prod)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases_prod.transpose(1, 2).bmm(bases_prod))
        # Clip denominator
        denominator = torch.clamp(denominator, min=eps)
        # multiplication update
        coef = coef * (numerator / denominator)
        coef = torch.nan_to_num(coef, nan=0.0, posinf=1.0, neginf=0.0)  # Handle any NaNs

        return coef


class VQ(_MatrixDecompositionBase):
    def __init__(self, device, md_config, debug=False, dim="3D"):
        super().__init__(device, md_config, debug=debug, dim=dim)
        self.device = device

    def _build_bases(self, B, S, D, R):
        # bases = torch.randn((B * S, D, R)).to(self.device)
        bases = torch.ones((B * S, D, R)).to(self.device)
        bases = F.normalize(bases, dim=1)
        return bases

    @torch.no_grad()
    def local_step(self, x, bases, _):
        eps = self.get_epsilon()
        # (B * S, D, N), normalize x along D (for cosine similarity)
        std_x = F.normalize(x, dim=1)

        # (B * S, D, R), normalize bases along D (for cosine similarity)
        std_bases = F.normalize(bases, dim=1, eps=eps)

        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(std_x.transpose(1, 2), std_bases)

        # softmax along R
        coef = F.softmax(self.inv_t * coef, dim=-1)

        # normalize along N
        coef = coef / (eps + coef.sum(dim=1, keepdim=True))
        coef = torch.nan_to_num(coef, nan=0.0, posinf=1.0, neginf=0.0)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        bases = torch.bmm(x, coef)

        return bases, coef


    def compute_coef(self, x, bases, _):
        eps = self.get_epsilon()
        with torch.no_grad():
            # (B * S, D, N) -> (B * S, 1, N)
            x_norm = x.norm(dim=1, keepdim=True)

        # Clip x_norm to prevent division by very small values
        x_norm = torch.clamp(x_norm, min=eps)
        # (B * S, D, N) / (B * S, 1, N) -> (B * S, D, N)
        std_x = x / x_norm

        # (B * S, D, R), normalize bases along D (for cosine similarity)
        std_bases = F.normalize(bases, dim=1, eps=eps)

        # (B * S, N, D)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(std_x.transpose(1, 2), std_bases)

        # softmax along R
        coef = F.softmax(self.inv_t * coef, dim=-1)

        return coef


class ConvBNReLU(nn.Module):
    @classmethod
    def _same_paddings(cls, kernel_size, dim):
        if dim == "3D":
            if kernel_size == (1, 1, 1):
                return (0, 0, 0)
            elif kernel_size == (3, 3, 3):
                return (1, 1, 1)
        elif dim == "2D" or dim == "2D_TSM":
            if kernel_size == (1, 1):
                return (0, 0)
            elif kernel_size == (3, 3):
                return (1, 1)
        else:
            if kernel_size == 1:
                return 0
            elif kernel_size == 3:
                return 1

    def __init__(self, in_c, out_c, dim,
                 kernel_size=1, stride=1, padding='same',
                 dilation=1, groups=1, act='relu', apply_bn=False, apply_act=True):
        super().__init__()

        self.apply_bn = apply_bn
        self.apply_act = apply_act
        self.dim = dim
        if dilation == 1:
            if self.dim == "3D":
                dilation = (1, 1, 1)
            elif self.dim == "2D" or dim == "2D_TSM":
                dilation = (1, 1)
            else:
                dilation = 1

        if kernel_size == 1:
            if self.dim == "3D":
                kernel_size = (1, 1, 1)
            elif self.dim == "2D" or dim == "2D_TSM":
                kernel_size = (1, 1)
            else:
                kernel_size = 1

        if stride == 1:
            if self.dim == "3D":
                stride = (1, 1, 1)
            elif self.dim == "2D" or dim == "2D_TSM":
                stride = (1, 1)
            else:
                stride = 1

        if padding == 'same':
            padding = self._same_paddings(kernel_size, dim)

        if self.dim == "3D":
            self.conv = nn.Conv3d(in_c, out_c,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation,
                                  groups=groups,
                                  bias=False)
        elif self.dim == "2D" or dim == "2D_TSM":
            self.conv = nn.Conv2d(in_c, out_c,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation,
                                  groups=groups,
                                  bias=False)
        else:
            self.conv = nn.Conv1d(in_c, out_c,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation,
                                  groups=groups,
                                  bias=False)

        if act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = nn.ReLU(inplace=True)

        if self.apply_bn:
            if self.dim == "3D":
                self.bn = nn.InstanceNorm3d(out_c)
            elif self.dim == "2D" or dim == "2D_TSM":
                self.bn = nn.InstanceNorm2d(out_c)
            else:
                self.bn = nn.InstanceNorm1d(out_c)

    def forward(self, x):
        x = self.conv(x)
        if self.apply_act:
            x = self.act(x)
        if self.apply_bn:
            x = self.bn(x)
        return x


class FeaturesFactorizationModule(nn.Module):
    def __init__(self, inC, device, md_config, dim="3D", debug=False):
        super().__init__()

        self.device = device
        self.dim = dim
        md_type = md_config["MD_TYPE"]
        align_C = md_config["align_channels"]  # inC // 2  # // 2 #// 8

        if self.dim == "3D":
            if "nmf" in md_type.lower():
                self.pre_conv_block = nn.Sequential(
                    nn.Conv3d(inC, align_C, (1, 1, 1)), 
                    nn.ReLU(inplace=True))
            else:
                self.pre_conv_block = nn.Conv3d(inC, align_C, (1, 1, 1))
        elif self.dim == "2D" or self.dim == "2D_TSM":
            if "nmf" in md_type.lower():
                self.pre_conv_block = nn.Sequential(
                    nn.Conv2d(inC, align_C, (1, 1)),
                    nn.ReLU(inplace=True)
                    )
            else:
                self.pre_conv_block = nn.Conv2d(inC, align_C, (1, 1))
        elif self.dim == "1D":
            if "nmf" in md_type.lower():
                self.pre_conv_block = nn.Sequential(
                    nn.Conv1d(inC, align_C, 1),
                    nn.ReLU(inplace=True)
                    )
            else:
                self.pre_conv_block = nn.Conv1d(inC, align_C, 1)
        else:
            print("Dimension not supported")

        if "snmf" in md_type.lower():
            self.md_block = SNMF(self.device, md_config, dim=self.dim, debug=debug)
        elif "nmf" in md_type.lower():
            self.md_block = NMF(self.device, md_config, dim=self.dim, debug=debug)
        elif "vq" in md_type.lower():
            self.md_block = VQ(self.device, md_config, dim=self.dim, debug=debug)
        else:
            print("Unknown type specified for MD_TYPE:", md_type)
            exit()

        if self.dim == "3D":
            if "nmf" in md_type.lower():
                self.post_conv_block = nn.Sequential(
                    ConvBNReLU(align_C, align_C, dim=self.dim, kernel_size=1),
                    nn.Conv3d(align_C, inC, 1, bias=False)
                    )
            else:
                self.post_conv_block = nn.Sequential(
                    ConvBNReLU(align_C, align_C, dim=self.dim, kernel_size=1, apply_act=False), 
                    nn.Conv3d(align_C, inC, 1, bias=False)
                    )
        elif self.dim == "2D" or self.dim == "2D_TSM":
            if "nmf" in md_type.lower():
                self.post_conv_block = nn.Sequential(
                    ConvBNReLU(align_C, align_C, dim=self.dim, kernel_size=1), 
                    nn.Conv2d(align_C, inC, 1, bias=False)
                    )
            else:
                self.post_conv_block = nn.Sequential(
                    ConvBNReLU(align_C, align_C, dim=self.dim, kernel_size=1, apply_act=False),
                    nn.Conv2d(align_C, inC, 1, bias=False)
                    )
        else:
            if "nmf" in md_type.lower():
                self.post_conv_block = nn.Sequential(
                    ConvBNReLU(align_C, align_C, dim=self.dim, kernel_size=1),
                    nn.Conv1d(align_C, inC, 1, bias=False)
                    )
            else:
                self.post_conv_block = nn.Sequential(
                    ConvBNReLU(align_C, align_C, dim=self.dim, kernel_size=1, apply_act=False),
                    nn.Conv1d(align_C, inC, 1, bias=False)
                    )

        self._init_weight()


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                N = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / N))
            elif isinstance(m, nn.Conv2d):
                N = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / N))
            elif isinstance(m, nn.Conv1d):
                N = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / N))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y=None):
        x = self.pre_conv_block(x)
        att = self.md_block(x, y)
        dist = torch.dist(x, att)
        att = self.post_conv_block(att)

        return att, dist

    def online_update(self, bases):
        if hasattr(self.md_block, 'online_update'):
            self.md_block.online_update(bases)

