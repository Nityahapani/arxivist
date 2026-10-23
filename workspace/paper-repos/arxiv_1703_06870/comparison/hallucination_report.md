# Assumption and ambiguity report — arXiv:1703.06870

## Verified directly from the paper

- Mask head architecture and 28×28 resolution.
- Per-class sigmoid mask loss on GT class.
- RoIAlign sampling (4 points per bin).
- Training schedule 160k iterations, LR 0.02, drop at 120k.

## Explicit assumptions in this reproduction

1. **RoIAlign convention:** Default aligned=True (half-pixel offset, later Detectron2 convention); aligned=False reproduces original Detectron. *Basis:* Paper does not specify pixel-centre convention.
2. **Backbone weights:** Random init in this repo; paper uses ImageNet-pretrained backbones. *Basis:* Offline environment.
3. **RPN details:** FPN-style RPN (anchor sizes 32–512 on P2–P6, 256 anchors/image, NMS 0.7, 2000/1000 proposals). *Basis:* Inherited from Faster R-CNN / FPN papers.
4. **Warmup:** Linear warmup over 500 iterations in paper config (Detectron convention, not in paper). *Basis:* Stabilises large-LR start.

These assumptions are also recorded in the SIR (`workspace/sir-registry/arxiv_1703_06870/sir.json`) and in the README. None is presented as a recovered original run artifact.
