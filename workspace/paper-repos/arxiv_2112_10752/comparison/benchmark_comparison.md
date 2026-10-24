# Paper-to-reproduction comparison — arXiv:2112.10752

This file separates results **reported in the paper** from results **actually run in this checkout**. The protocol is implemented in `configs/paper.json`, but paper-scale runs have not been executed here.

| Claim / result | Paper report | Current reproduction status |
|---|---|---|
| CelebA-HQ FID (LDM-4) | 5.11 | Not run |
| ImageNet class-conditional FID (LDM-4-G) | 3.60 | Not run |
| MS-COCO text-to-image FID | 12.63 | Not run |
| DDIM with oracle recovers x₀ | — | Verified |

The only executed training run is the CPU smoke test (`configs/smoke.json`); it verifies the training/output path, not any numeric result in the paper. Do not copy smoke values into the paper column. The contributor running the paper-scale protocol should replace "not run" entries with measured values, the exact command, hardware, and seed(s).
