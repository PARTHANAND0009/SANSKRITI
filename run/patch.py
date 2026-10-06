"""Stage 3b: activation patching as a causal check on l*. STUB.

Plan:
- Clean prompt: the original question. Corrupted prompt: the same question with
  the entity span (entities.parquet span_start/span_end) replaced by a
  same-attribute entity from a different state (the swap pool from stage 1a).
- For each layer l, patch the clean resid_post at the final prompt token into
  the corrupted run (and the reverse), measure the restored gold restricted
  log-prob. Compare the layer where restoration saturates with l* from the lens.
- Uses the same hooks as run/forward.py (crystal.lens.decoder_layers).
"""


def main():
    raise NotImplementedError("stage 3b (activation patching) is not implemented yet")


if __name__ == "__main__":
    main()
