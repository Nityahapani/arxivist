"""SFT -> preference sampling -> DPO pipeline (Sections 4 and 6), plus reward/KL evaluation (Fig. 2).

1. SFT the policy on unlabeled responses (here: uniform random tokens);
2. sample two responses per prompt from pi_SFT and label them with the Bradley-Terry ground truth;
3. initialise pi_theta = pi_ref = pi_SFT and minimise L_DPO (Eq. 7) with RMSprop and linear warmup;
4. report mean ground-truth reward and sequence-level KL(pi_theta || pi_ref) of fresh samples.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import build_pairs, random_prompts, response_mask, true_reward
from .dpo import dpo_loss, sequence_logps
from .model import TinyGPT, sample
from .utils import RunLogger, count_params, get_device, set_seed


@torch.no_grad()
def evaluate(policy, ref, prompts, n_new, positive, g):
    seqs = sample(policy, prompts, n_new, g=g)
    mask = response_mask(seqs, prompts.size(1))
    kl = (sequence_logps(policy(seqs), seqs, mask) - sequence_logps(ref(seqs), seqs, mask)).mean().item()
    return true_reward(seqs[:, prompts.size(1):], positive).mean().item(), kl


def run(cfg: dict, output_root: Path, smoke: bool = False) -> dict:
    seed = cfg.get("seed", 0)
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    device = get_device(cfg.get("device"))
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    V, P, R = d["vocab"], d["prompt_len"], d["response_len"]
    positive = torch.arange(d["n_positive"])
    policy = TinyGPT(V, m["d_model"], m["n_layers"], m["n_heads"], m["d_ff"], P + R).to(device)
    logger = RunLogger(Path(output_root) / cfg.get("run_name", "dpo"), cfg)

    # 1. SFT on unlabeled data
    opt = torch.optim.AdamW(policy.parameters(), lr=t["sft_lr"])
    for step in range(t["sft_steps"]):
        x = torch.randint(0, V, (t["batch_size"], P + R), generator=g).to(device)
        loss = F.cross_entropy(policy(x)[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    # 2. Offline preference dataset sampled from pi_SFT
    prompts = random_prompts(d["n_pairs"], P, V, g).to(device)
    y1 = sample(ref, prompts, R, g=g)[:, P:]
    y2 = sample(ref, prompts, R, g=g)[:, P:]
    chosen, rejected = build_pairs(prompts, y1, y2, positive, d["reward_scale"], g)
    mask = response_mask(chosen, P)
    eval_prompts = random_prompts(d["n_eval"], P, V, g).to(device)
    r0, _ = evaluate(ref, ref, eval_prompts, R, positive, g)

    # 3. DPO
    opt = torch.optim.RMSprop(policy.parameters(), lr=t["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, t["warmup_steps"])))
    policy.train()
    for step in range(1, t["steps"] + 1):
        idx = torch.randint(0, len(chosen), (t["batch_size"],), generator=g)
        cw, cl, mk = chosen[idx], rejected[idx], mask[idx]
        both = torch.cat([cw, cl])
        pi = sequence_logps(policy(both), both, torch.cat([mk, mk]))
        with torch.no_grad():
            rf = sequence_logps(ref(both), both, torch.cat([mk, mk]))
        B = len(idx)
        loss, rw, rl = dpo_loss(pi[:B], pi[B:], rf[:B], rf[B:], t["beta"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if step % t["log_every"] == 0 or step == t["steps"]:
            logger.log({"step": step, "dpo_loss": loss.item(), "reward_margin": (rw - rl).mean().item(),
                        "reward_accuracy": (rw > rl).float().mean().item()})
    policy.eval()
    r1, kl = evaluate(policy, ref, eval_prompts, R, positive, g)
    return logger.summary({"params": count_params(policy), "sft_true_reward": r0, "dpo_true_reward": r1,
                           "kl_to_ref": kl, "final_dpo_loss": loss.item(), "is_paper_result": False if smoke else None})
