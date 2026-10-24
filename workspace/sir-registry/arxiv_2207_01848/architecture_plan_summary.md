# Architecture / Reproduction Plan — TabPFN (arXiv:2207.01848)

**Paper:** [Hollmann et al. (2022), *TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second*](https://arxiv.org/abs/2207.01848)
**Implementation:** [Independent reimplementation](../../paper-repos/arxiv_2207_01848/README.md)
**Evidence source:** the arXiv paper. This repository is not the authors' official code.

## Prior fitting

```text
repeat: D ~ prior;  split D into (D_train, D_test) at random position
        loss = CE(q_θ(y_test | x_test, D_train))
```

## Reproduction assumptions and limits

- A simplified SCM/BNN mixture (random MLP structure, random activations, node noise, quantile thresholds) without feature warping, categorical features or missing values.
- Adam with warmup-cosine; steps, batch and LR are assumptions.
- Feature-permutation and class-relabelling ensemble; power transforms are omitted.
- Smoke evaluation on sklearn's breast cancer dataset instead of OpenML-CC18.

These choices are marked as assumptions in `sir.json` and in the README. Results must not be described as exact recovery of unpublished author-run logs.
