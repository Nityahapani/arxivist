# Assumption and ambiguity report — arXiv:2111.06377

## Verified directly from the paper

- 75% random masking, uniform without replacement.
- Encoder operates on visible patches only; decoder adds mask tokens.
- Default decoder 8 blocks × 512-d.
- Loss on masked patches; normalized pixels improve accuracy.
- AdamW β2=0.95, weight decay 0.05, base LR 1.5e-4, batch 4096, cosine, 40 warm-up epochs.

## Explicit assumptions in this reproduction

1. **Fine-tuning:** Pretraining and linear probe are implemented; full fine-tuning recipe (layer-wise LR decay, RandAug, mixup/cutmix, drop path) is not part of this scaffold. *Basis:* Scope limit; required for the fine-tuning numbers.
2. **Linear probe:** Probe on the mean of encoder patch tokens with a feature-standardisation step (stand-in for BatchNorm without affine). *Basis:* Paper uses [CLS] or average pooling with an extra BN layer.
3. **ImageNet loader:** An ImageNet data loader must be supplied by the contributor (`data.name=imagefolder`). *Basis:* Not available in the authoring environment.

These assumptions are also recorded in the SIR (`workspace/sir-registry/arxiv_2111_06377/sir.json`) and in the README. None is presented as a recovered original run artifact.
