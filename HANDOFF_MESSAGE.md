Hey Arijit! The GPU part of the SANSKRITI layer-depth project is ready for you to run whenever you get a slot 🙂

Repo: https://github.com/PARTHANAND0009/SANSKRITI/tree/claude/research-codebase-setup-f0mhpj
RUN_FOR_ARIJIT.md in there has everything (HF token, setup, the one command, troubleshooting).

Please run `scripts/preflight.py` first. It takes about 5 min and ends with PASS or tells you exactly what's wrong. After a PASS, the full run is one command inside tmux. It takes roughly 6 to 8 hours on a 40 GB A100 (4 to 6 on 80 GB) and needs about 160 GB of disk. If it stops, just rerun the same command and it picks up where it left off.

When it's done it leaves a results_<date>.tar.gz in the repo folder, a few hundred MB. Send it over Drive and paste me the sha256 line from the end of run.log. No git or push access needed.

Quick update on where things are:
1. The full pipeline is validated on CPU with all three models on 2k questions, and the lens at the last layer reproduces the model output.
2. Early layers pick one answer letter for almost every question (B for Llama, A for Qwen), so we now run every question in all 4 cyclic option orders, which cancels that out.
3. In the pilot, answers settle at about 0.56 of depth in Llama, 0.71 in Qwen and 0.67 in Gemma, so it's quite model-specific.
4. Whether entity frequency changes that depth is still open. The pilot is mixed, and the plan is preregistered, so we wait for your run.

Thanks a lot, ping me if anything breaks!
