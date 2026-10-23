# Assumption and ambiguity report — arXiv:2002.05709

## Verified directly from the paper

- NT-Xent with cosine similarity and temperature τ; 2(N−1) negatives per anchor.
- Projection head g is a 2-layer MLP; representation h (before g) is used downstream.
- Augmentations: random crop + resize, flip, colour distortion, Gaussian blur.
- LARS optimizer, LR = 0.3 × BatchSize/256, weight decay 1e-6, warmup + cosine.

## Explicit assumptions in this reproduction

1. **CIFAR protocol:** Run the CIFAR-10 variant (ResNet-50, CIFAR stem, no blur, batch 1024, tau 0.5, colour strength 0.5, ~1000 epochs) as the primary reproducible target. *Basis:* ImageNet-scale pretraining is beyond typical contributor hardware; Appendix B.9 describes the CIFAR-10 setting.
2. **Linear eval optimizer:** Linear evaluation uses Nesterov-momentum SGD on frozen, pre-extracted features without augmentation. *Basis:* Simplification of the paper's linear protocol (which trains with random crops/flips).
3. **Hue jitter:** Colour jitter is implemented in-tensor (brightness/contrast/saturation 0.8s, hue 0.2s, random order). *Basis:* Mirrors torchvision ColorJitter semantics used in Appendix A pseudo-code.
4. **LARS details:** LARS with trust coefficient 0.001, momentum 0.9; 1-D parameters (bias, BN) excluded from adaptation and weight decay. *Basis:* Matches common SimCLR implementations; paper states LARS + weight decay 1e-6.

These assumptions are also recorded in the SIR (`workspace/sir-registry/arxiv_2002_05709/sir.json`) and in the README. None is presented as a recovered original run artifact.
