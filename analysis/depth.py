"""Stage 4a: crystallization depth per (model, question, readout). STUB.

Plan:
- Load activations (crystal.io.load_acts) and the analysis set
  (crystal.io.load_analysis_set; ambiguous_gold rows excluded, proxy models
  refused).
- For each layer: lens_logits -> restricted_probs over the model's option ids
  -> top-1 flag. l* = crystallization_layer(flags) (1-indexed), d = l*/L.
  Questions the model gets wrong at the last layer have l* = None and are
  reported separately, not dropped silently.
- Same for the permuted-option prompts (option-order control) and for the
  tuned lens.
- Output: results/depth_{model}.parquet with qid, readout, variant, l_star, d.
"""


def main():
    raise NotImplementedError("stage 4a is not implemented yet")


if __name__ == "__main__":
    main()
