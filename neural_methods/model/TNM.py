import torch
import torch.nn as nn

class TNM(nn.Module):
    def __init__(self, enabled=True, frames=0, axis=2, eps=1e-6):
        super(TNM, self).__init__()
        self.enabled = enabled
        self.frames = frames
        self.axis = axis
        self.eps = eps

        def norm(x):
            if self.frames == 0:
                self.frames = x.shape[axis]
            dtype = x.dtype
            x_ = x.to(torch.float32)
            x_ = x_.reshape((*x.shape[:self.axis], -1, self.frames, *x.shape[self.axis+1:]))

            mean = x_.mean(dim=self.axis + 1, keepdim=True)
            tshape = [1] * len(x_.shape)
            tshape[self.axis + 1] = self.frames
            t = torch.linspace(0, 1, self.frames).reshape(tshape).to(x.device)
            
            n = ((t - 0.5) * (x_ - mean)).sum(dim=self.axis + 1, keepdim=True)
            d = ((t - 0.5) ** 2).sum(dim=self.axis + 1, keepdim=True)
            i = mean - n / d * 0.5
            trend = n / d * t + i
            x_ = x_ - trend
            std = ((x_ ** 2).mean(dim=self.axis + 1, keepdim=True) + self.eps).sqrt()
            x_ = x_ / std
            
            x_ = x_.reshape(x.shape)
            return x_.to(dtype)
        
        self.norm = norm

    def forward(self, x):
        if self.enabled:
            return self.norm(x)
        else:
            return x