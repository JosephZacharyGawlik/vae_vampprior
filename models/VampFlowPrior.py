import torch
import torch.nn as nn
import math
from models.FlowPrior import FlowPrior
from utils.distributions import log_Normal_diag

class VampFlowPrior(nn.Module):
    def __init__(self, encoder, pseudo_inputs, latent_dim, flow_layers, flow_hidden_dim, cuda=False, weighted=False):
        super().__init__()
        self.encoder = encoder  # your VAE encoder
        self.pseudo_inputs = nn.Parameter(pseudo_inputs)  # shape [C, input_dim]
        self.n_components = pseudo_inputs.size(0)
        self.latent_dim = latent_dim
        self.cuda = cuda
        self.weighted = weighted

        # One FlowPrior per pseudo-input
        self.flows = nn.ModuleList([
            FlowPrior(latent_dim, flow_hidden_dim, flow_layers)
            for _ in range(self.n_components)
        ])

        # Learnable mixture weights if desired
        if self.weighted:
            self.logits = nn.Parameter(torch.zeros(self.n_components))
        else:
            self.logits = None

    def log_prob(self, z):
        """
        Compute log p(z) under mixture of flows VampPrior
        z: [MB, latent_dim]
        returns: [MB] log-probabilities
        """
        MB = z.size(0)
        device = z.device

        if self.weighted:
            weights = torch.softmax(self.logits, dim=0)
        else:
            weights = torch.ones(self.n_components, device=device) / self.n_components

        # 1. Vectorized Encoder: Get all component params at once [C, latent_dim]
        q_mean, q_logvar = self.encoder(self.pseudo_inputs)

        # 2. Compute all flows and stack: eps [C, MB, latent_dim], logdet [C, MB]
        results = [self.flows[i].inverse(z) for i in range(self.n_components)]
        eps_all = torch.stack([r[0] for r in results])
        logdet_all = torch.stack([r[1] for r in results])

        # 3. Vectorized Base Log Prob with broadcasting
        # Reshape for [C, 1, latent_dim] vs [C, MB, latent_dim]
        q_mean = q_mean.unsqueeze(1)
        q_logvar = q_logvar.unsqueeze(1)
        
        diff = (eps_all - q_mean) ** 2
        var = torch.exp(q_logvar)
        log_base = -0.5 * (diff / var + q_logvar + math.log(2 * math.pi)).sum(-1)

        # 4. Combine with log_weights
        if self.weighted:
            # self.logits is a Parameter
            log_weights = torch.log_softmax(self.logits, dim=0)
        else:
            # self.logits is None, so we create a uniform log-weight tensor on the fly
            log_weights = torch.full((self.n_components,), -math.log(self.n_components), device=device)
        
        # Now this will always work
        log_probs = log_base + logdet_all + log_weights.view(-1, 1)

        # 5. LogSumExp across mixture components (dim 0 is the C components)
        return torch.logsumexp(log_probs, dim=0)
