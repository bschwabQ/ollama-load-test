# Ollama Load Test

A continuous benchmark tool for [Ollama](https://ollama.com) models with live streaming output, real-time throughput metrics, and automatic logging.

## Features

- **Live progress bar** with token count, thinking token tracking, and average TPS
- **Benchmark mode** — warmup + 10 iterations with GPU results saved to `results/`
- **GPU auto-detection** via nvidia-smi, results named by GPU and model (e.g. `5090_gemma4-31b.txt`)
- **Thinking token support** for reasoning models (qwen3.5, etc.)
- **TTFT (Time To First Token)** measurement per run
- **VRAM and model size** snapshots via the Ollama API
- **Dual logging** — JSONL and CSV for easy analysis
- **Runs indefinitely** by default (great for overnight soak tests), or for a fixed number of iterations
- **Configurable sampling parameters** — temperature, top_p, min_p, mirostat, repeat penalty, context size, and more

## Recommended Ollama Docker Setup

```bash
docker run -d \
  --gpus=all \
  -v /path/to/ollama/data:/root/.ollama \
  -p 11434:11434 \
  -e OLLAMA_HOST=0.0.0.0:11434 \
  -e OLLAMA_FLASH_ATTENTION=1 \
  -e OLLAMA_KV_CACHE_TYPE=q8_0 \
  -e OLLAMA_KEEP_ALIVE=-1 \
  --name ollama \
  --restart always \
  ollama/ollama
```

| Setting | Purpose |
|---------|---------|
| `OLLAMA_FLASH_ATTENTION=1` | Reduces VRAM usage, improves throughput at longer contexts |
| `OLLAMA_KV_CACHE_TYPE=q8_0` | Quantizes KV cache — fits larger models/contexts in VRAM |
| `OLLAMA_KEEP_ALIVE=-1` | Keeps models loaded in VRAM indefinitely (no cold-start penalty) |

## Quick Start

```bash
# Make sure Ollama is running and the model is pulled
ollama pull qwen3.5:9b

# Set up the virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run a standard benchmark (warmup + 10 iterations)
python ollama-test.py --benchmark
```

## Usage

```
python ollama-test.py [OPTIONS]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--iterations` | `0` | Number of iterations (0 = run forever) |
| `--min-output-tokens` | `256` | Minimum expected output tokens (used for progress bar scaling) |
| `--log-jsonl` | `ollama_overnight.jsonl` | JSONL log file path |
| `--log-csv` | `ollama_overnight.csv` | CSV log file path |
| `--temp` | `0.7` | Sampling temperature |
| `--top_p` | `0.9` | Top-p (nucleus) sampling |
| `--min_p` | `0.05` | Min-p sampling threshold |
| `--mirostat` | `0` | Mirostat mode (0 = off, 1 or 2 = on) |
| `--mirostat_eta` | `0.1` | Mirostat learning rate |
| `--mirostat_tau` | `5.0` | Mirostat target entropy |
| `--num_ctx` | `8192` | Context window size |
| `--num_batch` | `1024` | Batch size |
| `--repeat_last_n` | `64` | Repeat penalty lookback window |
| `--repeat_penalty` | `1.08` | Repeat penalty |
| `--seed` | `42` | Random seed |
| `--sleep` | `2.0` | Seconds to wait between runs |
| `--benchmark` | off | Standard benchmark: warmup + 10 iterations, saves to results dir |
| `--results-dir` | `results` | Directory for GPU result summaries |
| `--gpu` | auto-detect | Override GPU name for results filename |
| `--think` | off | Enable thinking/reasoning mode |
| `--no-think` | off | Disable thinking/reasoning mode |
| `--no-warmup` | off | Skip the warmup iteration |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen3.5:9b` | Model to benchmark |

### Examples

```bash
# Standard benchmark — warmup + 10 iterations, saves results to results/<GPU>_<model>.txt
python ollama-test.py --benchmark

# Run 10 iterations with a specific model
OLLAMA_MODEL=llama3:8b python ollama-test.py --iterations 10

# Overnight soak test with lower temperature
python ollama-test.py --temp 0.5 --sleep 5

# Benchmark with thinking disabled (lower TTFT for chat use cases)
OLLAMA_MODEL=gemma4:31b python ollama-test.py --benchmark --no-think

# Benchmark against a remote Ollama instance
OLLAMA_HOST=http://192.168.1.100:11434 python ollama-test.py --benchmark
```

## Output

Each run prints a summary line:

```
✓ 4815 tokens (+4303 thinking) | 51.4 t/s | 93.7s
```

Results are appended to both a JSONL file (one JSON object per line) and a CSV file for downstream analysis. When using `--benchmark`, a summary file is saved to `results/<GPU>_<model>.txt` (e.g. `5090_gemma4-31b_nothink.txt`).

## Prompts

The tool cycles through a built-in set of diverse prompts covering summarization, code generation, mathematical reasoning, long-form writing, and technical explanation to exercise different generation patterns.
