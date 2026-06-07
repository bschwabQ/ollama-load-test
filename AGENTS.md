# AGENTS.md

Operational guide for running a benchmark in this repo. Read this before adding
a new model's results. `README.md` documents the tool; this file is the playbook.

## What this is

A throughput benchmark for [Ollama](https://ollama.com) models. A run does a
warmup plus 10 iterations over a fixed prompt set and writes a per-GPU summary to
`results/<GPU>_<model>.txt`. The point is comparable tokens/sec (TPS), TTFT, and
VRAM numbers across models and machines.

## Running a benchmark

```bash
source venv/bin/activate                       # repo ships a venv; deps are just `requests`
OLLAMA_MODEL=<model:tag> python ollama-test.py --benchmark --no-think
OLLAMA_MODEL=<model:tag> python ollama-test.py --benchmark --think    # thinking-capable models only
```

`--benchmark` = warmup + 10 iterations, results auto-saved. The GPU is
auto-detected via `nvidia-smi`; pass `--gpu <name>` to override the filename
label. The model must already be pulled (`ollama pull <model:tag>`) — the script
runs a preflight check and exits with a clear message if the server is down or
the model is missing, so there's no need to guess.

## The one thing that will bite you: explicit think flags

For thinking-capable models (check `ollama show <model>` for a `thinking`
capability — gemma4, qwen3.x, etc.), **always pass an explicit `--think` or
`--no-think`.** Never run the bare default.

With no think flag the model still thinks, but the think phase folds into
time-to-first-token: TTFT balloons to tens of seconds, thinking tokens read as 0,
and the reported TPS inflates to physically impossible values (e.g. 327 t/s on a
12B Q4 whose real ceiling is ~105). The script now prints a `⚠ TTFT dominates`
warning when this happens — if you see it, the numbers are garbage; re-run with an
explicit flag. Non-thinking models can be run either way; `--no-think` is the
clean default for them.

For a thinking-capable model, benchmark **both** `--no-think` and `--think` so the
results dir has both variants (this matches how gemma4 / qwen3.x are already
recorded).

## Results

Auto-written to `results/<gpu>_<model-slug>[_think|_nothink].txt`, where the model
slug replaces `:` and `/` with `-`. Examples:

- `5090_gemma4-12b-it-q4_K_M_nothink.txt`
- `5090_gemma4-12b-it-q4_K_M_think.txt`

The header records GPU, driver, Ollama version, context/batch, and the sampling
params, so each file is self-describing. Keep sampling params at the script
defaults (temp 0.7, top_p 0.9, etc.) unless you have a reason to change them —
consistency is what makes cross-model comparison meaningful.

`ollama_overnight.csv` / `.jsonl` are gitignored scratch logs (every iteration
appended). If the script warns that the CSV uses an older column layout, delete
both files for a clean schema; they're disposable.

## After a run

1. **Sanity-check the numbers.** TPS should be steady (low variance), TTFT should
   be sub-second for short prompts, and there should be no `⚠` warning. Wild
   swings or a multi-second TTFT mean something is off (usually a missing think
   flag, or a cold/oversubscribed GPU).
2. **Update `ollama-tuning.md`** if there's a finding worth recording — a results
   table for the model, a tuning observation, a quirk. Mirror the existing
   per-model / per-GPU sections.
3. **Commit.** Convention in this repo is one commit for the results file(s) and a
   separate commit for any script/doc changes. End commit messages with:
   `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
   Only push when the user asks.

## Environment

- Runs on WSL2 and native Linux. CUDA on WSL2 is within ~10-13% of native.
- The Ollama server is expected to run with flash attention + q8_0 KV cache +
  `KEEP_ALIVE=-1` (the docker/systemd setup is in `README.md` and
  `ollama-tuning.md`). Those settings are assumed by the recorded results.
- `OLLAMA_HOST` defaults to `http://127.0.0.1:11434`. If you point it at a remote
  server, note that `nvidia-smi` still reads the *local* GPU — pass `--gpu` so the
  results file is labeled correctly (the script warns about this too).
- Different machines have their own gitignored scratch logs; clearing them on one
  box does not affect another.
