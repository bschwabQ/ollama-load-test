# Ollama Load Test

A continuous benchmark tool for [Ollama](https://ollama.com) models with live streaming output, real-time throughput metrics, and automatic logging.

## Features

- **Live progress bar** with token count, instantaneous/average/peak TPS
- **Global peak TPS tracking** across all iterations
- **TTFT (Time To First Token)** measurement per run
- **VRAM and model size** snapshots via the Ollama API
- **Dual logging** — JSONL and CSV for easy analysis
- **Runs indefinitely** by default (great for overnight soak tests), or for a fixed number of iterations
- **Configurable sampling parameters** — temperature, top_p, min_p, mirostat, repeat penalty, context size, and more

## Quick Start

```bash
# Make sure Ollama is running and the model is pulled
ollama pull qwen3.5:9b

# Set up the virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the benchmark
python ollama-test.py
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
| `--num_batch` | `512` | Batch size |
| `--repeat_last_n` | `64` | Repeat penalty lookback window |
| `--repeat_penalty` | `1.08` | Repeat penalty |
| `--seed` | `42` | Random seed |
| `--sleep` | `2.0` | Seconds to wait between runs |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen3.5:9b` | Model to benchmark |

### Examples

```bash
# Run 10 iterations with a specific model
OLLAMA_MODEL=llama3:8b python ollama-test.py --iterations 10

# Overnight soak test with lower temperature
python ollama-test.py --temp 0.5 --sleep 5

# Benchmark against a remote Ollama instance
OLLAMA_HOST=http://192.168.1.100:11434 python ollama-test.py
```

## Output

Each run prints a summary line:

```
✓ 312 tokens | 45.2 t/s (peak 52.1, global 53.8) | 6.9s
```

Results are appended to both a JSONL file (one JSON object per line) and a CSV file for downstream analysis.

## Prompts

The tool cycles through a built-in set of diverse prompts covering summarization, code generation, mathematical reasoning, long-form writing, and technical explanation to exercise different generation patterns.
