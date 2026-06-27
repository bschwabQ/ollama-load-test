# Modelfiles

Reusable [Ollama Modelfiles](https://docs.ollama.com/modelfile) for models we
benchmark in this repo. Each one pins a GGUF source, the model-card sampling
params, and a sane default context, so anyone can reproduce the exact model the
`results/` numbers were measured against.

## Available models

| Modelfile | Model | Arch | Default quant | On-GPU size | Notes |
|-----------|-------|------|---------------|-------------|-------|
| `ornith-9b.Modelfile`  | Ornith-1.0-9B  | Qwen3.5 dense | Q8_0   | ~9.5 GB  | Fits any ≥12 GB GPU |
| `ornith-35b.Modelfile` | Ornith-1.0-35B | Qwen3.5 MoE (~3B active) | Q5_K_M | ~24.7 GB | Fits a 32 GB GPU; MoE so fast despite size |

## Build

From the repo root:

```bash
ollama create ornith-9b  -f modelfiles/ornith-9b.Modelfile
ollama create ornith-35b -f modelfiles/ornith-35b.Modelfile
```

Ollama pulls the GGUF from Hugging Face on first `create` (no separate
`ollama pull` needed). To build everything:

```bash
for f in modelfiles/*.Modelfile; do
  ollama create "$(basename "${f%.Modelfile}")" -f "$f"
done
```

## Requirements

- **Recent Ollama** (≥ 0.30) — the Ornith GGUFs use the `qwen3.5` / `qwen3.5moe`
  architectures; older builds won't recognize them and fail at load.
- Server is expected to run with flash attention + `OLLAMA_KV_CACHE_TYPE=q8_0` +
  `OLLAMA_KEEP_ALIVE=-1` (see the top-level `README.md`).

## Customizing

- **Quant:** edit the `:TAG` on the `FROM` line (e.g. `:Q4_K_M` for a smaller
  card). Available tags are listed on the model's `*-GGUF` Hugging Face repo.
- **Context:** edit `PARAMETER num_ctx`. Native max is 262144; bigger contexts
  cost more KV-cache VRAM.
- **Sampling:** the baked-in `temperature`/`top_p`/`top_k` are the model-card
  recommendations and apply to `ollama run` / API use. The benchmark
  (`ollama-test.py`) passes its own `--temp`/`--top_p`, which override these.

## Reasoning mode

Ornith is a reasoning model — it emits a `<think>…</think>` chain-of-thought by
default. When benchmarking, always pass an explicit `--think` or `--no-think`
(see `AGENTS.md`). Roughly ~70% of generated tokens are reasoning, so `--think`
runs produce ~4× the tokens (and wall-clock) of `--no-think` on the same prompts.
