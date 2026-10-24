# Assumption and ambiguity report — arXiv:2305.14314

## Verified directly from the paper

- NF4 / DQ definitions and overheads.
- Eq. 5–6.
- Appendix B hyperparameters: r=64, α=16, LR 2e-4/1e-4, batch 16, max grad norm 0.3, constant LR.

## Explicit assumptions in this reproduction

1. **NF4 construction:** Asymmetric quantile construction with offset 0.9677083 (8 positive, 7 negative levels plus exact zero), matching the published NF4 table. *Basis:* Paper describes the construction; the offset follows the reference bitsandbytes code.
2. **DQ data type:** Second-level constants are 8-bit symmetric integers per 256-block (centred by the mean). *Basis:* Paper uses 8-bit floats; the overhead in bits is identical.
3. **Paged optimizers:** Not implemented (CPU reference); standard Adam is used. *Basis:* Paging only affects memory spikes, not the math.
4. **Base model:** A tiny GPT pretrained in-script stands in for LLaMA. *Basis:* CPU-testable pipeline.

These assumptions are also recorded in the SIR (`workspace/sir-registry/arxiv_2305_14314/sir.json`) and in the README. None is presented as a recovered original run artifact.
