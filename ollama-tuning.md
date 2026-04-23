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
