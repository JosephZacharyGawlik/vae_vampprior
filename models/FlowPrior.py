"""
Real Non-Volume Preserving (Real NVP) Normalizing Flow Implementation.

This module provides a Flow-based Prior using a sequence of Affine Coupling layers, 
allowing for flexible prior modeling in variational inference frameworks. The 
architecture is adapted from the Real NVP tutorial by xqding, which transforms 
samples from a simple base distribution into complex distributions while 
maintaining a tractable Jacobian determinant.

Sources:
    - xqding RealNVP Tutorial: 
      https://github.com/xqding/RealNVP/blob/master/Real%20NVP%20Tutorial.ipynb
    - FlowPrior: Learning Expressive Priors for Latent Variable Sentence Models 
      (Li et al., 2019): https://aclanthology.org/2021.naacl-main.259/ 
"""

import torch
import torch.nn as nn
import torch.nn.init as init

class Affine_Coupling(nn.Module):

    def __init__(self, mask, hidden_dim):
        super(Affine_Coupling, self).__init__()
        self.input_dim = len(mask)
        self.hidden_dim = hidden_dim

        ## mask to seperate positions that do not change and positions that change.
        ## mask[i] = 1 means the ith position does not change.
        self.mask = nn.Parameter(mask, requires_grad = False)

        ## layers used to compute scale in affine transformation
        self.scale_fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.scale_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.scale_fc3 = nn.Linear(self.hidden_dim, self.input_dim)
        self.scale = nn.Parameter(torch.Tensor(self.input_dim))
        init.normal_(self.scale)

        ## layers used to compute translation in affine transformation 
        self.translation_fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.translation_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.translation_fc3 = nn.Linear(self.hidden_dim, self.input_dim)

    def _compute_scale(self, x):
        ## compute scaling factor using unchanged part of x with a neural network
        s = torch.relu(self.scale_fc1(x*self.mask))
        s = torch.relu(self.scale_fc2(s))
        s = torch.relu(self.scale_fc3(s)) * self.scale        
        return s

    def _compute_translation(self, x):
        ## compute translation using unchanged part of x with a neural network        
        t = torch.relu(self.translation_fc1(x*self.mask))
        t = torch.relu(self.translation_fc2(t))
        t = self.translation_fc3(t)        
        return t
    
    def forward(self, x):
        ## convert latent space variable to observed variable
        s = self._compute_scale(x)
        t = self._compute_translation(x)
        
        y = self.mask*x + (1-self.mask)*(x*torch.exp(s) + t)        
        logdet = torch.sum((1 - self.mask)*s, -1)
        
        return y, logdet

    def inverse(self, y):
        ## convert observed varible to latent space variable
        s = self._compute_scale(y)
        t = self._compute_translation(y)
                
        x = self.mask*y + (1-self.mask)*((y - t)*torch.exp(-s))
        logdet = torch.sum((1 - self.mask)*(-s), -1)
        
        return x, logdet

class FlowPrior(nn.Module):

    def __init__(self, dim, hidden_dim, n_layers):
        super(FlowPrior, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Create alternating binary masks for coupling layers
        masks = []
        for i in range(n_layers):
            if i % 2 == 0:
                mask = torch.tensor([1 if j % 2 == 0 else 0 for j in range(dim)])
            else:
                mask = torch.tensor([0 if j % 2 == 0 else 1 for j in range(dim)])
            masks.append(mask)

        # Create a sequence of Affine Coupling layers
        self.coupling_layers = nn.ModuleList([
            Affine_Coupling(mask, hidden_dim) for mask in masks
        ])

    def forward(self, z):
        log_det_jacobian = 0
        for layer in self.coupling_layers:
            z, logdet = layer.forward(z)
            log_det_jacobian += logdet
        return z, log_det_jacobian

    def inverse(self, z):
        log_det_jacobian = 0
        for layer in reversed(self.coupling_layers):
            z, logdet = layer.inverse(z)
            log_det_jacobian += logdet
        return z, log_det_jacobian
