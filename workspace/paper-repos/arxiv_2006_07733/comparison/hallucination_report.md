# Assumption and ambiguity report — arXiv:2006.07733

## Verified directly from the paper

- Eq. 1–2 and symmetrisation.
- Projector/predictor MLP sizes 4096/256 with BN.
- τ_base 0.996 with cosine increase to 1.
- LARS, LR 0.2×bs/256, wd 1.5e-6, batch 4096, 1000 epochs, 10-epoch warmup.
- Augmentation probabilities of Table 6.

## Explicit assumptions in this reproduction

1. **BN statistics in target:** BatchNorm running statistics of the target network are copied from the online network. *Basis:* Only parameters are averaged by Eq. 1; buffers are not discussed.
2. **Linear protocol:** Short SGD linear probe on frozen features in smoke runs. *Basis:* Paper protocol is 80 epochs with an LR grid.
3. **ImageNet loader:** Not included; synthetic data and CIFAR-10 only. *Basis:* Scope limit.

These assumptions are also recorded in the SIR (`workspace/sir-registry/arxiv_2006_07733/sir.json`) and in the README. None is presented as a recovered original run artifact.
