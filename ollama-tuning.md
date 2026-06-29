# Ollama Performance Tuning

Tuning notes for running Ollama on a single GPU (tested on RTX 5060 Ti 16GB VRAM).

## Current Docker Setup

```bash
docker run -d \
  --gpus=all \
  -v /mnt/f/ollamadocker:/root/.ollama \
  -p 11434:11434 \
  -e OLLAMA_HOST=0.0.0.0:11434 \
  -e OLLAMA_FLASH_ATTENTION=1 \
  -e OLLAMA_KV_CACHE_TYPE=q8_0 \
  -e OLLAMA_KEEP_ALIVE=-1 \
  --name ollama \
  --restart always \
  ollama/ollama
```

## Optimized Docker Setup

Tested adding `OLLAMA_NUM_PARALLEL=1` and `OLLAMA_GPU_OVERHEAD=0` — **no improvement**, slightly worse in practice. Stick with the current setup above.

## Environment Variables

### OLLAMA_FLASH_ATTENTION=1
Enables flash attention, reducing VRAM usage and improving throughput for supported architectures. Gemma 4 is on the allowlist in Ollama 0.20.0 so it should auto-enable, but setting explicitly guarantees it.

### OLLAMA_KV_CACHE_TYPE=q8_0
Quantizes the KV cache from f16 to q8_0, cutting KV cache VRAM by ~50% with negligible quality loss at 4-8K context. Requires flash attention to be active (silently ignored otherwise).

**Caution:** Gemma 3 had a bug (fixed in v0.12.5) where KV cache quantization caused 5-8x slowdowns due to CUDA kernel fallback to CPU. Gemma 4 is architecturally similar — if TPS drops dramatically, remove this setting.

| Cache Type | VRAM vs f16 | Speed Impact (8K ctx) | Speed Impact (64K ctx) |
|------------|-------------|----------------------|----------------------|
| f16        | 1x baseline | baseline             | baseline             |
| q8_0       | ~0.5x       | -3 to -5%            | -8 to -12%           |
| q4_0       | ~0.25x      | -3%                  | -35%                 |

### OLLAMA_KEEP_ALIVE=-1
Keeps models pinned in VRAM indefinitely. Eliminates cold-start penalties between requests.

### OLLAMA_NUM_PARALLEL=1
**Tested — no benefit.** Was expected to help by reserving only one KV cache slot, but made no meaningful difference on gemma4:e4b. The default auto-selection appears fine for this model on 16GB VRAM.

### OLLAMA_GPU_OVERHEAD=0
**Tested — no benefit.** Reclaims reserved VRAM but had no measurable impact on TPS. Not worth the OOM risk.

## Benchmark Parameters

### num_ctx (context window)
Default in benchmark: 8192. Reducing to 4096 can yield 5-15% TPS improvement for short prompts by reducing KV cache size and per-token attention computation. Only matters if your prompt + response fits within the smaller window.

### num_batch (prompt eval batch size)
Default in benchmark: 1024. Only affects prompt processing speed (TTFT), not token generation speed. No meaningful gain from going above 1024. Reducing to 512 has negligible effect on short prompts.

## NVIDIA GPU Tweaks

```bash
# Persistence mode — keeps GPU initialized, reduces latency after idle
sudo nvidia-smi -pm 1

# Exclusive process mode — prevents GPU contention from other CUDA processes
sudo nvidia-smi -c EXCLUSIVE_PROCESS
```

## WSL2 Notes

- CUDA performance on WSL2 is within 10-13% of native Linux, gap narrows for GPU-bound workloads like LLM inference.
- Keep model data inside WSL2's ext4 filesystem, not on `/mnt/c/` or `/mnt/f/` — the 9P bridge is 3-5x slower for I/O. This affects model loading, not inference.

## Benchmark Results (gemma4:e4b, RTX 5060 Ti, no-think, num_ctx=8192)

| Run | Config Changes | Avg TPS | Min | Max | TTFT |
|-----|---------------|---------|-----|-----|------|
| Baseline | q8_0 KV, flash attn, keep alive | 34.7 t/s | 32.0 | 38.8 | ~1.0s |
| + NUM_PARALLEL=1, GPU_OVERHEAD=0 | added to baseline | 32.7 t/s | 29.8 | 36.2 | ~1.1s |
| f16 KV + NUM_PARALLEL=1 | removed q8_0 | 30.6 t/s | 27.4 | 34.4 | ~1.1s |
| Back to baseline | same as first run | 32.4 t/s | 30.1 | 35.0 | ~1.1s |

### Findings

- **q8_0 KV cache helps** — removing it dropped TPS from 32.7 to 30.6 with same settings. No sign of the Gemma 3 slowdown bug on Gemma 4.
- **NUM_PARALLEL=1 and GPU_OVERHEAD=0 made no meaningful difference** — not worth adding.
- **Run-to-run variance is ~2-3 t/s** — the initial 34.7 baseline was likely a favorable run. Steady-state is ~32-33 t/s.
- **The current docker setup (flash attn + q8_0 KV + keep alive) is already near-optimal** for this model on 16GB VRAM.

## Benchmark Results (gemma4:12b-it-q4_K_M, RTX 5090)

Run on Ollama 0.30.6, driver 610.47, num_ctx=8192, num_batch=1024, q8_0 KV cache + flash attention. Model resident footprint: 8.4 GB VRAM (weights + 8K q8_0 KV), fits trivially in the 5090's 32 GB.

| Mode | Avg TPS | Min | Max | Avg TTFT | Notes |
|------|---------|-----|-----|----------|-------|
| `--no-think` | 102.3 t/s | 98.8 | 103.9 | 693 ms | pure response generation |
| `--think`    | 103.1 t/s | 101.8 | 104.9 | 736 ms | 6,550 thinking tokens across 10 iters |

### Findings

- **Thinking is "free" on throughput** — think and no-think generate at the same ~102–103 t/s. Thinking doesn't slow the per-token rate; it just emits more tokens (and so more wall-clock time per request). The `--think` run produced 15,204 total tokens vs 7,181 for `--no-think` over the same 10 prompts.
- **Very low run-to-run variance** — under 2 t/s spread in both modes, far tighter than the older gemma4:31b runs. The 5090 is comfortably bandwidth-fed for a 12B Q4 model.
- **TTFT is low and clean (~0.7 s) in both modes.** With explicit `--think`, the first thinking token carries an empty `response` field, so TTFT is measured at the first thinking token rather than absorbing the entire think phase.
- **q8_0 KV cache + flash attention showed no slowdown** on 0.30.6 — no sign of the old Gemma 3 KV-quant CUDA fallback bug. Keep the recommended docker settings.

### Use explicit `--think` / `--no-think` for thinking-capable Gemma models

Running these models with **no** think flag (the bare default) is unreliable for benchmarking. The model still thinks, but with no explicit `think:true` the think phase isn't streamed as separate tokens — it gets folded into time-to-first-token. TTFT balloons (30+ s for a single prompt), thinking tokens read as 0, and the streamed TPS inflates to physically impossible values (a 12B Q4 caps around 105 t/s on the 5090, yet the bare-default run reported 327 t/s). This is exactly what produced the messy `5090_gemma4-31b.txt` default run — 0 thinking tokens, ~20 s TTFT, and wild 38–458 t/s swings. Always pass an explicit `--think` or `--no-think`. The benchmark now warns when TTFT dominates total run time, which catches this case.

## Benchmark Results (gemma4:12b-it-q4_K_M, RTX 5060 Ti 16GB)

Run on Ollama 0.30.6, driver 610.47, num_ctx=8192, num_batch=1024, q8_0 KV cache + flash attention — same docker setup and same Ollama build as the 5090 run above, so the only variable is the GPU. Model resident footprint: 7.6 GB on disk; fits comfortably in 16 GB VRAM alongside the 8K q8_0 KV cache.

| Mode | Avg TPS | Min | Max | Avg TTFT | Notes |
|------|---------|-----|-----|----------|-------|
| `--no-think` | 37.7 t/s | 37.4 | 38.2 | 639 ms | 7,030 tokens over 10 iters |
| `--think`    | 37.7 t/s | 36.8 | 38.3 | 812 ms | 6,342 thinking tokens over 10 iters |

### Findings

- **Thinking is "free" on throughput here too** — identical ~37.7 t/s in both modes, matching the 5090 pattern. Think mode just emits more tokens (14,964 total vs 7,030 for no-think), so it costs wall-clock, not per-token rate.
- **Very low variance** (<1.5 t/s spread in both modes) and **clean sub-second TTFT** with no `⚠ TTFT dominates` warning — the explicit think flags worked as intended.
- **~37% of the 5090's throughput.** The 5060 Ti runs this model at 37.7 t/s vs the 5090's ~102–103 t/s — a ~2.7× gap that tracks the memory-bandwidth difference, as expected for a bandwidth-bound 12B Q4 dense model. Both GPUs are otherwise on identical software/settings.

## 12B variant shootout — best gemma4 12B for a 16GB Blackwell card (RTX 5060 Ti)

The 5060 Ti 16GB is the cheapest/slowest 50-series card, so "what's the best 12B you can actually run on it" is a useful question. All runs: Ollama 0.30.6, driver 610.47, num_ctx=8192, num_batch=1024, q8_0 KV cache + flash attention (the standard docker setup), `--no-think`, 10 iterations. Resident VRAM is from `ollama ps` with the model loaded.

| Variant | Disk | Resident VRAM | Avg TPS | vs Q4_K_M | Quality tier | Runs on Linux? |
|---------|------|---------------|---------|-----------|--------------|----------------|
| `12b-it-qat` | 7.2 GB | 7.7 GB | **39.5 t/s** | **+5%** | Q4 size, **QAT-recovered** quality | ✅ |
| `12b-it-q4_K_M` | 7.6 GB | 8.1 GB | 37.7 t/s | baseline | naive Q4 | ✅ |
| `12b-it-q8_0` | 12 GB | 13 GB | 25.7 t/s | −32% | near-bf16 | ✅ (100% GPU, no offload) |
| `12b-nvfp4` | 10 GB | — | — | — | Blackwell FP4 | ❌ `412: requires macOS` |
| `12b-mxfp8` | 12 GB | — | — | — | FP8 | ❌ `412: requires macOS` |

### Findings

- **`qat` is the winner — it dominates `q4_K_M` on all three axes at once.** It's *faster* (39.5 vs 37.7 t/s), *smaller* (7.7 vs 8.1 GB resident), and *higher quality* (quantization-aware training recovers most of the accuracy lost by naive Q4). There is no reason to prefer plain `q4_K_M` over `qat` on this hardware. Make `qat` the default 12B.
- **`q8_0` is the quality-max option and it does fit** — 13 GB resident stays 100% on GPU in 16 GB with the 8K q8_0 KV cache, no CPU offload. The cost is throughput: 25.7 t/s, ~65% of `qat`. Pick it only when output quality matters more than speed. (No headroom for much longer contexts, though — 13 GB + a larger KV would start to spill.)
- **`nvfp4` and `mxfp8` are macOS-gated for gemma4 too**, exactly as already documented for qwen3.6 — the pull fails outright with `412: this model requires macOS` (MLX kernels, not CUDA). So despite the 5060 Ti being native Blackwell, you cannot use Ollama's hardware-FP4 path on Linux. Real Blackwell FP4 still needs vLLM / TensorRT-LLM with an NVFP4 checkpoint, out of Ollama.
- **Throughput tracks weight size, as expected for a bandwidth-bound dense model** — 7.7 GB → 39.5 t/s, 13 GB → 25.7 t/s is almost exactly inverse-linear, confirming the 5060 Ti is memory-bandwidth limited here, not compute limited.
- **Think/no-think parity holds across variants** — `qat` runs 39.5 t/s no-think vs 39.3 t/s think. As with `q4_K_M`, thinking costs tokens (wall-clock), not per-token rate, so the no-think number is the right one for cross-variant comparison.

### Bottom line for a 16GB card

Default to **`gemma4:12b-it-qat`** (fastest *and* best quality at Q4 footprint). Step up to **`12b-it-q8_0`** only when you want maximum fidelity and can spend ~35% throughput. Skip the FP4/FP8 tags — they don't run on Linux.

## 12B variant shootout — RTX 5090 (32GB)

The same three variants on the 5090, for a cross-machine comparison. Identical software/settings (Ollama 0.30.6, driver 610.47, num_ctx=8192, num_batch=1024, q8_0 KV cache + flash attention, `--no-think`, 10 iterations). Resident VRAM from `ollama ps`. The FP4/FP8 tags are macOS-gated and skipped for the same reason as above.

| Variant | Resident VRAM | 5090 TPS | vs `qat` | 5060 Ti TPS | 5090 / 5060 Ti |
|---------|---------------|----------|----------|-------------|----------------|
| `12b-it-qat` | 8.0 GB | **105.9 t/s** | baseline | 39.5 | 2.68× |
| `12b-it-q4_K_M` | 8.4 GB | 102.3 t/s | −3% | 37.7 | 2.71× |
| `12b-it-q8_0` | 13.7 GB | 77.2 t/s | −27% | 25.7 | **3.00×** |

### Findings

- **`qat` wins here too** — 105.9 t/s, fastest and smallest, edging `q4_K_M` by ~3.5%. The 16GB recommendation carries over: `qat` is the default 12B on either card. Think parity holds (105.7 t/s with `--think`).
- **`q8_0` is much more attractive on the 5090 than on the 5060 Ti.** It costs only **−27%** vs `qat` here (77.2 vs 105.9), against **−35%** on the 5060 Ti — and resident VRAM is a non-issue at 13.7/32 GB. So on the 32GB card, `q8_0` is a reasonable quality-max default; on the 16GB card the steeper penalty plus tight headroom make `qat` the clearer pick.
- **The 5090 is not fully bandwidth-bound on a 12B.** If it were, going 8.0 → 13.7 GB of weights would drop throughput to ~62 t/s (inverse-linear); the actual 77.2 t/s is well above that. Contrast the 5060 Ti, where q8_0 lands almost exactly on the inverse-linear prediction — that card *is* bandwidth-limited. This also explains why the 5090's lead stretches to 3.0× on `q8_0` but sits at ~2.7× on the lighter Q4s: on a small model the 5090 has bandwidth to spare, so the cards are relatively closer.

### Bottom line

`qat` for throughput on both cards. On the 5090, `q8_0` is a sensible quality-max default (only −27%, VRAM is free); on the 5060 Ti, reserve `q8_0` for when fidelity genuinely outweighs the ~35% hit.

## Driver + Ollama upgrade re-run — 12B shootout on RTX 5090

Re-ran the full 12B variant shootout on the 5090 after upgrading the stack: Ollama **0.30.6 → 0.30.9** and NVIDIA driver **610.47 → 610.62** (CUDA UMD 13.3). Everything else was held constant — same RTX 5090, same docker config (flash attention + q8_0 KV cache + `KEEP_ALIVE=-1`, default `NUM_PARALLEL`), num_ctx=8192, num_batch=1024, seed=42, `--no-think`, 10 iterations, same prompt set. The container was recreated to the documented-standard env before the runs, so the only changed variables are the Ollama and driver versions. Each variant's workload matched its prior run to within a few tokens, so the TPS delta is the upgrade alone. Resident VRAM from `ollama ps`.

| Variant | Resident (0.30.9) | 0.30.6 TPS | 0.30.9 TPS | upgrade Δ | vs qat (0.30.9) |
|---------|-------------------|------------|------------|-----------|-----------------|
| `12b-it-qat`    | 7.7 GB | 105.9 | **113.4 t/s** | **+7.1%** | baseline |
| `12b-it-q4_K_M` | 8.1 GB | 102.3 | **109.9 t/s** | **+7.4%** | −3.1% |
| `12b-it-q8_0`   | 13 GB  | 77.2  | **81.2 t/s**  | **+5.2%** | −28% |

(`qat` think parity holds on the new stack too: 113.6 t/s with `--think` vs 113.4 no-think, TTFT flat at ~0.7–0.8 s in both modes.)

### Findings

- **The upgrade lifted every variant ~5–7%, ranking unchanged.** qat +7.1%, q4_K_M +7.4%, q8_0 +5.2%, all on byte-identical settings and matching token counts, with tight variance and no `⚠ TTFT dominates` warning. The 0.30.6→0.30.9 + driver bump is a real, free throughput win, not a measurement artifact.
- **`qat` is still the best 12B — fastest *and* smallest.** 113.4 t/s at 7.7 GB resident, edging `q4_K_M` by ~3% (same margin as on 0.30.6) on less VRAM and with QAT-recovered quality. The verdict is unchanged: default to `qat`.
- **`q8_0` gained the least (+5.2% vs ~+7%).** It's the most bandwidth-bound of the three — 13 GB of weights to stream per token — so a runtime/driver improvement that's partly compute/overhead shows up smaller. Its gap to qat is essentially unchanged (−28% vs the old −27%): still the quality-max pick when fidelity outweighs ~28% throughput, and VRAM-trivial at 13/32 GB.
- **Resident VRAM is leaner across the board on 0.30.9** — 7.7 / 8.1 / 13 GB vs the old 8.0 / 8.4 / 13.7 GB. A few hundred MB per model; consistent with the newer build, not workload-dependent.

### Bottom line (new stack)

Same ranking as before, just faster: `qat` for throughput on the 5090 (113.4 t/s, smallest footprint), `q8_0` as the quality-max default when you can spend the ~28% (VRAM is free on a 32 GB card). The driver + Ollama upgrade is worth taking — ~7% for free with no config change.

## Driver + Ollama upgrade re-run — 12B shootout on RTX 5060 Ti

Same upgrade re-run on the 16GB card, to see whether the ~5–7% the 5090 picked up also lands on the cheap/slow Blackwell part. Stack: Ollama **0.30.6 → 0.30.10** and NVIDIA driver **610.47 → 610.62** (CUDA UMD 13.3). The container was recreated from the documented-standard docker config (flash attention + q8_0 KV cache + `KEEP_ALIVE=-1`, default `NUM_PARALLEL`); everything else held constant — same RTX 5060 Ti, num_ctx=8192, num_batch=1024, seed=42, `--no-think`, 10 iterations, same prompt set. Each variant's workload matched its prior run to within a few tokens, so the delta is the upgrade alone. Resident VRAM from `ollama ps` at 8K context.

(One stack caveat: this card landed on Ollama **0.30.10** — a patch newer than the 5090's 0.30.9 — because "pull the latest" grabbed a same-day image. The 0.30.6→0.30.x runtime jump and the driver bump are shared with the 5090 run; the 0.30.9→0.30.10 patch gap is negligible and doesn't affect the conclusions below.)

| Variant | Resident (0.30.10) | 0.30.6 TPS | 0.30.10 TPS | upgrade Δ | vs qat (0.30.10) |
|---------|--------------------|------------|-------------|-----------|------------------|
| `12b-it-qat`    | 7.7 GB | 39.5 | **40.5 t/s** | **+2.5%** | baseline |
| `12b-it-q4_K_M` | 8.1 GB | 37.7 | **38.3 t/s** | **+1.6%** | −5.4% |
| `12b-it-q8_0`   | 13 GB  | 25.7 | **25.8 t/s** | **+0.4%** | −36% |

(`qat` think parity holds on the new stack here too: 40.3 t/s with `--think` vs 40.5 no-think, 6,048 thinking tokens over 10 iters, TTFT ~0.66–0.83 s in both modes, no `⚠ TTFT dominates` warning.)

### Findings

- **The 5060 Ti barely moved — and that's the headline.** Where the 5090 gained a uniform ~5–7% across all three variants, the 5060 Ti gained only +2.5% (qat), +1.6% (q4_K_M), and +0.4% (q8_0) on byte-identical settings and matching token counts. The same Ollama+driver upgrade is nearly free throughput on the big card and almost a no-op on the small one.
- **The gain shrinks as the weights grow — the bandwidth-bound fingerprint.** qat (7.7 GB, +2.5%) > q4_K_M (8.1 GB, +1.6%) > q8_0 (13 GB, +0.4%). A runtime/driver upgrade mostly buys back compute, kernel-launch, and overhead time; on a card that's already pinned against its memory-bandwidth ceiling there's almost nothing for that to recover, and the more bytes-per-token a variant streams, the less it sees. q8_0 — the most bandwidth-bound — is within noise of its old number. This is the same mechanism the doc already noted for q8_0 on the 5090, just applied to the whole card: the 5060 Ti is bandwidth-limited on a 12B, so it can't cash in a compute-side improvement.
- **Ranking and footprint are unchanged.** `qat` is still the best 12B — fastest *and* smallest at 7.7 GB resident, edging `q4_K_M` by ~5% and beating `q8_0` by 36% (was 35% on the old stack — essentially flat). Resident VRAM is identical to the old stack (7.7 / 8.1 / 13 GB), all 100% on GPU; the leaner-footprint effect the 5090 saw on 0.30.9 doesn't appear here because the 5060 Ti was already at these numbers on 0.30.6.
- **The upgrade widened the 5090's lead.** Because the 5090 captured the full ~5–7% and the 5060 Ti captured almost none, the cross-machine ratio stretched on the new stack:

| Variant | 5090 (0.30.9) | 5060 Ti (0.30.10) | 5090 / 5060 Ti (new) | old stack |
|---------|---------------|-------------------|----------------------|-----------|
| `12b-it-qat`    | 113.4 | 40.5 | **2.80×** | 2.68× |
| `12b-it-q4_K_M` | 109.9 | 38.3 | **2.87×** | 2.71× |
| `12b-it-q8_0`   | 81.2  | 25.8 | **3.15×** | 3.00× |

### Bottom line (new stack, 5060 Ti)

Take the upgrade — it's free and never regresses — but don't expect the 5090's ~7% on this card: a bandwidth-bound 16GB Blackwell part gets ≤2.5% and effectively nothing on q8_0. The model verdict is unchanged: default to **`gemma4:12b-it-qat`** (40.5 t/s, 7.7 GB, QAT quality), step up to **`12b-it-q8_0`** (25.8 t/s, 13 GB) only when fidelity outweighs the ~36% throughput hit. If you want the upgrade to actually pay off on a 12B, it pays off on the bigger card.

## MTP / speculative decoding on Linux+CUDA — gemma4 is not supported

Empirically established on Ollama 0.30.6 (CUDA, RTX 5060 Ti) by trying to wire up a draft model with `ollama create` + the Modelfile `DRAFT` directive. Three runtime facts, each from an actual error:

1. **`DRAFT` is MTP-specific, not generic speculative decoding.** Attaching a plain small model (`gemma4:e2b`) as a draft for a `gemma4:12b-it-qat` target fails at load with `context type MTP requested but model doesn't contain MTP layers` (the llama-server segfaults). You cannot use an arbitrary small model as a vanilla draft — the draft must itself contain trained MTP layers.

2. **The CUDA MTP path only accepts a qwen3.5 base.** Importing Google's official `google/gemma-4-E4B-it-assistant` MTP drafter (a 159 MB `model.safetensors`, `model_type: gemma4_assistant`) via `DRAFT` fails with `MTP draft safetensors require a qwen3.5 base model, got "gemma4"`.

3. **gemma4 MTP lives only in the MLX (macOS) backend** — see ollama/ollama PR #15980, titled *"mlx: Gemma4 MTP speculative decoding"*.

**Conclusion:** gemma4 MTP / speculative decoding is impossible on Linux + CUDA through Ollama, for any size (12B, E4B, …), regardless of drafter source. This is the same pattern as the macOS-gated `nvfp4`/`mxfp8` tags: gemma4's accelerated paths in Ollama are consistently Apple-only, and on Linux/CUDA you get standard autoregressive inference. To actually exercise MTP speculative decoding on this card you'd have to switch model families to **qwen3.5** (the only CUDA-supported MTP base), which is a separate experiment.

### `ollama create` + DRAFT mechanics (for reference)

For when a *supported* (qwen3.5) base is used:

```
# Modelfile
FROM <ollama-model-or-safetensors-dir>
DRAFT <path-to-draft-gguf-or-safetensors-dir>
```

- `DRAFT` takes a **filesystem path**, not an Ollama model name (a model name gets `stat`-ed as a relative path and fails).
- `--experimental` forces `FROM` to be parsed as a safetensors dir too, so don't combine it with a `FROM <ollama-model-name>`.
- `--draft-quantize <level>` quantizes the draft during create.
- If using Docker with a bind-mounted data dir, the draft files must live under that mount (or be `docker cp`-ed into the container) for the server to read them.

## DGX Spark (GB10, unified memory)

Setup notes for running qwen3.6:27b and qwen3.6:35b-a3b-q8_0 pinned in memory simultaneously on a DGX Spark. The GB10 shares ~122 GB of LPDDR5X between CPU and GPU (Ollama reports `memory.total` as `[N/A]` — the `gpu_info()` helper falls back to `/proc/meminfo`).

### systemd override

`/etc/systemd/system/ollama.service.d/override.conf`:

```ini
[Service]
Environment="OLLAMA_MAX_LOADED_MODELS=2"
Environment="OLLAMA_NUM_PARALLEL=2"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_KEEP_ALIVE=-1"
```

Memory budget with both models resident:

| Component | Size |
|---|---|
| qwen3.6:35b-a3b-q8_0 weights | 38 GB |
| qwen3.6:27b (Q4_K_M) weights | 17 GB |
| KV cache (8K ctx, 2 slots × 2 models, q8_0) | ~4 GB |
| **Total** | **~59 GB / 122 GB** |

`NUM_PARALLEL=2` rather than 4: each slot pre-allocates its own KV cache at load time whether or not it's used, so 2 is enough for occasional overlap without paying for 4× KV. Confirmed no single-stream regression.

### Benchmark results (GB10, num_ctx=8192, no-think)

| Model | Weights | Avg TPS | TTFT |
|---|---|---|---|
| qwen3.6:35b-a3b-q8_0 (MoE, 3B active) | 38 GB | 43.2 t/s | 273 ms |
| qwen3.6:27b (Q4_K_M, dense) | 17 GB | 10.8 t/s | 331 ms |
| qwen3.6:27b-q8_0 (dense) | 30 GB | 7.1 t/s | 384 ms |

Dense qwen3.6:27b is memory-bandwidth-bound — TPS scales inversely with weight size, and both variants are running at 65–80% of the LPDDR5X ceiling (~273 GB/s ÷ weights). The MoE variant crushes both because only ~3B params are active per token. For throughput on GB10, prefer MoE.

### NVFP4 / MXFP8 note

Ollama's `qwen3.6:27b-nvfp4` and `-mxfp8` tags are gated to macOS (`412: this model requires macOS`) — they ship MLX kernels, not Blackwell-native CUDA. Real Blackwell FP4 on Linux needs an out-of-Ollama runtime (vLLM / TensorRT-LLM with NVFP4 checkpoints).

## Benchmark Results (Ornith-1.0 9B vs 35B, RTX 5090)

[Ornith-1.0](https://huggingface.co/deepreinforce-ai) is DeepReinforce's agentic-coding family, post-trained on Qwen 3.5 with RL that learns its own task scaffolds. Two sizes fit a 32 GB card: the **9B dense** (`qwen35`) and the **35B MoE** (`qwen35moe`, 34.7B total / ~3B active per token). Both are reasoning models — they emit a `<think>…</think>` chain-of-thought by default. Run on Ollama 0.30.11, driver 610.62, num_ctx=32768, num_batch=1024, q8_0 KV cache + flash attention, seed=42, 10 iterations. Sampling is the model-card spec (temp=0.6, top_p=0.95, top_k=20) rather than the repo default — see the caveat below. Resident VRAM from `ollama ps`; both load 100% on GPU. Modelfiles are in `modelfiles/`. Figures are from a clean re-run on an otherwise-idle machine — see the re-run note below.

| Model | Arch | Quant | Resident | Mode | Avg TPS | Avg TTFT | Total tok | Think tok |
|-------|------|-------|----------|------|---------|----------|-----------|-----------|
| Ornith-9B  | qwen35 dense (9B)     | Q8_0   | 10 GB | `--no-think` | 112.2 t/s     | 639 ms | 6,608  | — |
| Ornith-9B  | qwen35 dense (9B)     | Q8_0   | 10 GB | `--think`    | 114.3 t/s     | 624 ms | 25,316 | 17,416 (69%) |
| Ornith-35B | qwen35moe (3B active) | Q5_K_M | 24 GB | `--no-think` | **178.5 t/s** | 648 ms | 8,318  | — |
| Ornith-35B | qwen35moe (3B active) | Q5_K_M | 24 GB | `--think`    | **179.3 t/s** | 653 ms | 28,814 | 21,176 (73%) |

### Findings

- **The 35B MoE is faster *and* bigger than the 9B dense — by a wide margin.** 179.3 vs 114.3 t/s (`--think`), a **~57%** throughput win, on a fuller-but-still-100%-GPU footprint (24 vs 10 GB). TTFT is a wash — ~620–650 ms either way, the 9B marginally lower as you'd expect from the smaller model (the 35B's apparent TTFT edge in the first pass was a download-contention artifact; see the re-run note). The mechanism is the one this doc keeps hitting (qwen3.6:35b-a3b, GB10): an MoE activates only ~3B of its 34.7B params per token, so it streams far fewer weight bytes per token than a 9B-dense-at-Q8, which reads all 9B every token. On a 32 GB 5090 there is no throughput reason to run the 9B — the 35B is the better model *and* the faster one. The 9B's only edge is the ~14 GB of VRAM it leaves free for co-locating other models.
- **Quant isn't the cause — active params are.** The 35B is Q5_K_M and the 9B is Q8_0, so this isn't a clean iso-quant comparison. But the gap is far too large for the quant delta to explain: a 9B at Q5 would still read ~9B params/token vs the MoE's ~3B. The MoE active-param advantage dominates; the quant difference is second-order.
- **Think/no-think parity holds on per-token rate, same as gemma4.** Each model generates at the same speed whether reasoning or not (9B: 114.3 vs 112.2; 35B: 179.3 vs 178.5). Thinking doesn't slow the per-token rate — it just emits far more tokens. ~69% (9B) and ~73% (35B) of all output is reasoning, so `--think` runs produce ~3.5–4× the tokens and wall-clock of `--no-think` on the same prompts (9B: 25.3K vs 6.6K tok; 35B: 28.8K vs 8.3K). The no-think TPS is the right number for cross-model rate comparison; the think token volume is the right number for sizing real agentic latency.
- **The 35B finishes more work in less time.** Despite generating *more* total tokens in think mode (28.8K vs the 9B's 25.3K), its think pass wall-clock is 160.7 s vs the 9B's 220.8 s — a bigger reasoning workload done ~27% faster.
- **Bandwidth framing is consistent.** The 35B at 24 GB resident reads fewer bytes/token than the 9B at 10 GB resident, because only the ~3B active experts plus shared weights move per token. That's the whole MoE story, and it lines up with the "throughput tracks bytes-streamed-per-token" thread running through the gemma4 and GB10 sections — except here the *bigger* model streams *less* per token.

### Harness note: Ornith streams `<think>` inline on `/api/generate`

Unlike qwen3.x — which streams reasoning in a separate `thinking` field that `ollama-test.py` already counts — the Ornith HF GGUFs emit their chain-of-thought **inline in the `response` text** on `/api/generate`, and the chat template **prefills the opening `<think>`**, so only the closing `</think>` reaches the stream. Net effect: the separate `thinking` field is empty and the think-token counter read **0** on the first runs, even though the model was clearly reasoning (and `--no-think` *did* shrink output ~4×, proving reasoning was on by default).

`ollama-test.py` now recovers the split: when no `thinking`-field tokens arrive but the output contains `</think>`, it attributes the text up to the first `</think>` as thinking and prorates the server's `eval_count` by character length. `/api/chat` separates the field natively for Ornith, but the benchmark stays on `/api/generate` for comparability with the existing results, so the inline-tag recovery is what makes the Think column correct.

### Caveat: sampling params differ from the repo default

These four runs use the Ornith model-card sampling (temp=0.6, top_p=0.95, top_k=20), not the repo standard (temp=0.7, top_p=0.9). Sampling choice doesn't affect decode rate, so the throughput comparisons above and against other models in this doc are valid. It does affect *which* tokens are produced and total output length, so the token counts aren't strictly iso-config with the gemma/qwen runs. Re-run at the repo defaults if you need strict cross-model parity on output volume.

### Note: numbers are from a clean re-run

The figures above are a re-run on a quiet machine, and they replaced a noisier first pass. In that first pass the 9B benchmarks happened to overlap the 35B GGUF still downloading in the background, and the I/O contention inflated the 9B's TTFT (844/868 ms) and shaved ~4% off its TPS — enough to make the 35B look like it had the lower TTFT, which on a clean box it doesn't. Re-running with nothing else on the machine, token counts came back byte-identical (seed=42 is deterministic), the 9B's TTFT dropped ~25% to ~625 ms, its TPS rose to 112–114 t/s, and the 35B — clean both times — barely moved (177.7 → 179.3). Lesson for this harness: a background `ollama pull` or any disk/network load skews TTFT and depresses TPS a few percent, so benchmark on an otherwise-idle machine and re-run anything measured while a download was in flight.

### Bottom line

On a 32 GB 5090, **`ornith-35b` (Q5_K_M) is the default** — bigger, smarter, fits 100% on GPU, and ~57% faster than the 9B thanks to MoE. Reach for `ornith-9b` (Q8_0) only when you need the VRAM headroom. Budget ~4× tokens/latency for `--think` on either; that's where the agentic-coding gains live, so keep it on for real work and reserve `--no-think` for quick chat.

## TODO

- [ ] Test with `num_ctx=4096` to see if shorter context improves TPS
- [ ] Compare think vs no-think with current config
- [ ] Try vLLM or TensorRT-LLM with NVFP4 qwen3.6 weights on GB10
