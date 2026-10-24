# Assumption and ambiguity report — arXiv:2106.09685

## Verified directly from the paper

- ΔW = BA with A ∈ R^{r×k}, B ∈ R^{d×r}; W0 frozen; output scaled by α/r.
- A initialised with random Gaussian, B with zeros.
- Main experiments adapt only attention weights (W_q, W_v) and freeze MLP modules.
- RoBERTa-base LoRA uses r=8 and reports a GLUE average of 87.2 with 0.3M trainable parameters.

## Explicit assumptions in this reproduction

1. **A initialisation:** Initialise A from N(0, (1/r)^2) and B to zero; kaiming-uniform (loralib default) available via init='kaiming'. *Basis:* The paper specifies a random Gaussian for A and zero for B but not the Gaussian's variance.
2. **GLUE hyperparameters:** Use the RoBERTa-base per-task batch size / epochs / learning rate transcribed into configs/paper.json from the appendix table; contributors should verify each value against the paper. *Basis:* Values were transcribed manually and could contain transcription errors.
3. **Classifier head:** Train the task classifier head together with LoRA matrices; all other weights frozen. *Basis:* Standard in the released GLUE scripts; the paper counts LoRA parameters separately.
4. **Smoke task:** CPU smoke test uses a synthetic pretrain -> adapt task with a 2-layer, width-32 model; it validates mechanics only. *Basis:* No GPU or pretrained checkpoints in the authoring environment.

These assumptions are also recorded in the SIR (`workspace/sir-registry/arxiv_2106_09685/sir.json`) and in the README. None is presented as a recovered original run artifact.
