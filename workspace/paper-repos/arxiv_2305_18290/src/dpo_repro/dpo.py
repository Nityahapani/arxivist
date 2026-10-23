"""Direct Preference Optimization (Rafailov et al., 2023), Section 4, Eq. 7.

L_DPO = -E[ log sigma( beta * log(pi(y_w|x)/pi_ref(y_w|x)) - beta * log(pi(y_l|x)/pi_ref(y_l|x)) ) ]

The implicit reward is r_hat(x, y) = beta * log(pi(y|x) / pi_ref(y|x)) (Section 5).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_logps(logits: torch.Tensor, ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Sum of log p(token_t | tokens_<t) over response positions.

    logits: [B, T, V] for inputs ids [B, T]; response_mask: [B, T] True where ids[t] is a response token.
    """
    logp = F.log_softmax(logits[:, :-1].float(), -1)
    tok = torch.gather(logp, 2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (tok * response_mask[:, 1:].float()).sum(-1)


def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta: float = 0.1, label_smoothing: float = 0.0):
    """Inputs are per-example sequence log-probabilities. Returns (loss, chosen_reward, rejected_reward).

    label_smoothing > 0 gives the conservative variant (not in the paper; kept at 0 by default).
    """
    logits = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    losses = -(1 - label_smoothing) * F.logsigmoid(logits) - label_smoothing * F.logsigmoid(-logits)
    return losses.mean(), (beta * (pi_w - ref_w)).detach(), (beta * (pi_l - ref_l)).detach()


def dpo_gradient_weight(pi_w, pi_l, ref_w, ref_l, beta: float = 0.1):
    """sigma(r_hat(x, y_l) - r_hat(x, y_w)): the per-example weight in the DPO gradient (Section 4)."""
    return torch.sigmoid(beta * ((pi_l - ref_l) - (pi_w - ref_w)))
