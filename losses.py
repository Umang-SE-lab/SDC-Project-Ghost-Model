import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridFocalFedProxLoss(nn.Module):

    def __init__(self, alpha=0.80, gamma=2.0, mu=0.001):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.mu = mu

    def forward(self, outputs, targets, local_model, global_params):

        ce_loss = F.cross_entropy(
            outputs,
            targets,
            reduction="none",
        )

        pt = torch.exp(-ce_loss)

        focal_loss = (
            self.alpha
            * (1 - pt) ** self.gamma
            * ce_loss
        ).mean()

        proximal_term = 0.0

        for local_param, global_param in zip(
            local_model.parameters(),
            global_params,
        ):
            proximal_term += torch.sum(
                (local_param - global_param) ** 2
            )

        return focal_loss + (self.mu / 2.0) * proximal_term