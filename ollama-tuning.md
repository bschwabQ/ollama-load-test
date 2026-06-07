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

## TODO

- [ ] Test with `num_ctx=4096` to see if shorter context improves TPS
- [ ] Compare think vs no-think with current config
- [ ] Try vLLM or TensorRT-LLM with NVFP4 qwen3.6 weights on GB10
