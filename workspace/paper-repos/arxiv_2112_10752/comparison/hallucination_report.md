# Assumption and ambiguity report — arXiv:2112.10752

## Verified directly from the paper

- Two-stage design and objectives (Eq. 2–3).
- Cross-attention formulation (Eq. 4–5).
- KL / VQ regularisation options.

## Explicit assumptions in this reproduction

1. **Perceptual loss:** LPIPS is not implemented (needs pretrained VGG weights); first stage uses L1 + KL + PatchGAN. *Basis:* Offline environment.
2. **Noise schedule:** 'Scaled linear' β schedule 0.00085 → 0.012 with T = 1000 (as in the released LDM configs). *Basis:* Paper text does not list the schedule explicitly.
3. **Architecture sizes:** Channel multipliers, attention resolutions and iteration counts in configs/paper.json are best-recalled values. *Basis:* Check the paper's hyperparameter tables.
4. **Latent scaling:** Latents are rescaled by 1/std estimated on the training set before diffusion (KL models). *Basis:* As described for KL-regularised latents.

These assumptions are also recorded in the SIR (`workspace/sir-registry/arxiv_2112_10752/sir.json`) and in the README. None is presented as a recovered original run artifact.
