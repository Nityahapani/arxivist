# Assumption and ambiguity report — arXiv:2304.02643

## Verified directly from the paper

- Three-component architecture and decoder design.
- Loss weighting 20:1 focal:dice and IoU MSE.
- SA-1B size (11M images, 1.1B masks).

## Explicit assumptions in this reproduction

1. **Optimiser details:** AdamW lr 8e-4, warmup 250, 90k iterations, batch 256, weight decay 0.1, step decay, drop path 0.4, layer decay 0.8 (configs/paper.json). *Basis:* Best-recalled from Appendix A; must be checked.
2. **Multimask policy:** Multiple masks only on the first round when no box is given; otherwise the single-mask token. *Basis:* Paper predicts a single mask when more than one prompt is given.
3. **Encoder details:** No relative positional embeddings in windowed attention; random init instead of MAE pretraining. *Basis:* Simplification for a CPU scaffold.
4. **Text prompts:** Text-prompt (CLIP) experiments are not implemented. *Basis:* Exploratory in the paper.

These assumptions are also recorded in the SIR (`workspace/sir-registry/arxiv_2304_02643/sir.json`) and in the README. None is presented as a recovered original run artifact.
