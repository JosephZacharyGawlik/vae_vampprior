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
        log_probs = []

        if self.weighted:
            weights = torch.softmax(self.logits, dim=0)
        else:
            weights = torch.ones(self.n_components, device=device) / self.n_components

        # iterate over pseudo-inputs
        for i in range(self.n_components):
            # 1. Encoder output for pseudo-input
            pseudo_input = self.pseudo_inputs[i:i+1]  # 1 x input_dim
            q_mean, q_logvar = self.encoder(pseudo_input)  # 1 x latent_dim

            # 2. Inverse flow: z -> epsilon
            eps, logdet = self.flows[i].inverse(z)  # MB x latent_dim, MB

            # 3. Base Gaussian log prob
            log_base = -0.5 * ((eps - q_mean) ** 2 / torch.exp(q_logvar) + q_logvar + math.log(2*math.pi)).sum(-1)  # MB

            # 4. Add log-det Jacobian
            log_probs.append(log_base + logdet + math.log(weights[i]))

        # 5. LogSumExp across mixture components
        log_probs = torch.stack(log_probs, dim=0)  # C x MB
        max_log, _ = torch.max(log_probs, dim=0)
        log_prob = max_log + torch.log(torch.sum(torch.exp(log_probs - max_log), dim=0))  # MB

        return log_prob
