# Assumption and ambiguity report — arXiv:1901.08573

## Verified directly from the paper

- Objective: CE(f(X),Y) + max KL(f(X)||f(X'))/λ over an ε-ball.
- CIFAR-10: WRN-34-10, ε=0.031, step 0.007, 10 iterations for training.
- Evaluation: FGSM^k white-box with 20 iterations, step 0.003.
- Reported 84.92% natural / 56.61% robust at 1/λ=6.

## Explicit assumptions in this reproduction

1. **Epochs:** Train 76 epochs with LR 0.1 decayed x0.1 at epoch 75, batch 128, weight decay 2e-4. *Basis:* Taken from the authors' public CIFAR-10 script; the paper text does not fully specify the schedule.
2. **Evaluation attack:** PGD-20 on cross-entropy, step 0.003, random uniform start in the eps-ball. *Basis:* Paper reports FGSM^k with 20 iterations and step 0.003; random start is a common convention.
3. **Input range:** No normalisation layer; inputs in [0, 1] so eps=0.031 is in pixel units. *Basis:* Matches the threat model description.

These assumptions are also recorded in the SIR (`workspace/sir-registry/arxiv_1901_08573/sir.json`) and in the README. None is presented as a recovered original run artifact.
