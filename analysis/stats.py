"""Stage 4b: does depth depend on documentation? STUB.

Plan:
- Join depth (analysis/depth.py) with freq.parquet (entity log_freq, tier) and
  state-level scores.
- Primary test: mixed-effects regression of d on entity log_freq and state
  log_freq with random intercepts for attribute (and question_type), per model.
- Secondary: low vs high tier comparison within attribute (permutation test),
  and the same models on the option-permuted prompts as a control.
- All seeds from config/run.yaml; proxy models are never included.
"""


def main():
    raise NotImplementedError("stage 4b is not implemented yet")


if __name__ == "__main__":
    main()
