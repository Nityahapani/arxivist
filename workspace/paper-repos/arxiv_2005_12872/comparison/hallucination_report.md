# Assumption and ambiguity report — arXiv:2005.12872

## Verified directly from the paper

- Architecture sizes and query count.
- Matching cost and loss weights.
- Optimiser settings, clipping, LR drop, batch 64.

## Explicit assumptions in this reproduction

1. **Backbone weights:** Backbone is randomly initialised here; paper uses ImageNet-pretrained ResNet. *Basis:* torchvision weights not available offline.
2. **Data:** Synthetic coloured rectangles in smoke runs; COCO loader and augmentations not included. *Basis:* Offline environment.
3. **Steps:** Epochs converted to iterations with ~1848 iterations/epoch at batch 64. *Basis:* COCO train has ~118k images.

These assumptions are also recorded in the SIR (`workspace/sir-registry/arxiv_2005_12872/sir.json`) and in the README. None is presented as a recovered original run artifact.
