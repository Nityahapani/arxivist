# Assumption and ambiguity report — arXiv:2201.03545

## Verified directly from the paper

- Stage depths (3,3,9,3) for T and (3,3,27,3) for S/B/L/XL.
- 4×4 stride-4 stem; 7×7 depthwise conv; inverted bottleneck ×4; GELU; LN; separate downsampling layers.
- Training: AdamW, lr 4e-3, 300 epochs, batch 4096, weight decay 0.05, Mixup/CutMix/RandAug/RE, label smoothing, layer scale 1e-6.

## Explicit assumptions in this reproduction

1. **Augmentations:** RandAugment and Random Erasing are not implemented in the scaffold; Mixup/CutMix/label smoothing are. *Basis:* Scope limit; the contributor should add them (e.g. timm) before reporting.
2. **EMA:** EMA of weights (decay 0.9999) is optional and evaluated separately. *Basis:* Paper notes EMA helps larger models.
3. **ImageNet loader:** An ImageNet loader must be supplied. *Basis:* Not available in the authoring environment.

These assumptions are also recorded in the SIR (`workspace/sir-registry/arxiv_2201_03545/sir.json`) and in the README. None is presented as a recovered original run artifact.
