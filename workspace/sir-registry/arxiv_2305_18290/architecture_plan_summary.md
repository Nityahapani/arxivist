# Architecture / Reproduction Plan — DPO (arXiv:2305.18290)

**Paper:** [Rafailov et al. (2023), *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*](https://arxiv.org/abs/2305.18290)
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_2305_18290/README.md)
**Evidence source:** the arXiv paper. This repository is not the authors' official code.

## Pipeline

```text
π_SFT ──sample pairs──▶ BT labels ──▶ DPO(π_θ ← π_SFT, π_ref = π_SFT) ──▶ reward / KL evaluation
```

## Reproduction assumptions and limits

- No numeric targets are hard-coded; read targets from Figure 2 (reward/KL frontier), Figure 3 (TL;DR and HH win rates) and Table 2 of the paper.
- A synthetic controlled-generation task with a known reward stands in for IMDb + sentiment classifier.
- Smoke SFT uses uniform random tokens.

These choices are marked as assumptions in `sir.json` and in the README. Results must not be described as exact recovery of unpublished author-run logs.
