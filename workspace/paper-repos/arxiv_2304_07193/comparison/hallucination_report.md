# Assumption and ambiguity report — arXiv:2304.07193

## Verified directly from the paper

- Combined DINO + iBOT objective with untied heads.
- Sinkhorn-Knopp centring and KoLeo regulariser.
- Multi-crop sizes, ViT/14 variants, LVD-142M size.

## Explicit assumptions in this reproduction

1. **Hyperparameters:** LR, warmup, prototype counts, head sizes and teacher momentum start in configs/paper.json are best-recalled values; verify against the paper's appendix. *Basis:* Not all values were confirmed.
2. **Masking:** Random (not block-wise) patch masking with ratio U[0.1, 0.5] on 50% of global crops. *Basis:* Simplification of iBOT block masking.
3. **Efficiency features:** FlashAttention/xFormers, sequence packing, FSDP and the high-resolution phase are not implemented. *Basis:* Implementation details that do not change the objective.
4. **Data:** Synthetic images replace LVD-142M; the curation pipeline is not implemented. *Basis:* LVD-142M is not public.

These assumptions are also recorded in the SIR (`workspace/sir-registry/arxiv_2304_07193/sir.json`) and in the README. None is presented as a recovered original run artifact.
