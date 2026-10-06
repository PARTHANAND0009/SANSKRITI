"""Stage 3a: tuned lens (Belrose et al. 2023). STUB.

Plan:
- One affine translator per decoder layer, A_l h + b_l, initialised to identity,
  trained so that lens_logits(model, translator(h_l), l) matches the model's
  final-layer distribution (KL(final || lens)), as in the paper.
- Training text: a held-out general corpus, NOT the SANSKRITI prompts, so the
  lens is not fit to the evaluation questions.
- Translators saved per model to lenses/{model_key}.pt and applied through
  crystal.lens.lens_logits(..., translator=...).
- Inputs: the resid_post activations from stage 2 (acts/{model}/...).
"""


def main():
    raise NotImplementedError("stage 3a (tuned lens) is not implemented yet")


if __name__ == "__main__":
    main()
