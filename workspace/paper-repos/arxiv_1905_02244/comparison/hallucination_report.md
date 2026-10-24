# Assumption and ambiguity report — arXiv:1905.02244

## Verified directly from the paper

- bneck specifications of Tables 1 and 2.
- h-swish definition.
- Training: RMSProp momentum 0.9, LR 0.1, batch 4096, dropout, wd 1e-5, EMA 0.9999.

## Explicit assumptions in this reproduction

1. **LR decay:** Interpret 'learning rate decay rate of 0.01 every 3 epochs' as multiplying LR by 0.97 every 3 epochs. *Basis:* The literal wording is ambiguous; 0.97/3-epoch is a common reading in public reimplementations.
2. **Dropout:** Interpret 'dropout of 0.8' as keep-probability 0.8 (dropout rate 0.2) before the classifier. *Basis:* TensorFlow convention of keep_prob.
3. **RMSProp epsilon:** Use eps=0.0316 (≈ sqrt(1e-3)) to mimic TensorFlow's epsilon placement. *Basis:* Framework-difference adjustment.
4. **Epoch budget:** ~600 epochs (187,500 steps at batch 4096). *Basis:* Not stated in the paper.

These assumptions are also recorded in the SIR (`workspace/sir-registry/arxiv_1905_02244/sir.json`) and in the README. None is presented as a recovered original run artifact.
