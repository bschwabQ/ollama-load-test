#!/usr/bin/env python3
"""
Overnight Ollama benchmark with live console streaming, status line,
peak TPS tracking, and JSONL/CSV logging.

Requirements:
  - Ollama running locally with the target model pulled
  - Python 3.9+
  - pip install -r requirements.txt  (requests)

Environment overrides:
  OLLAMA_HOST=http://127.0.0.1:11434
  OLLAMA_MODEL=qwen3.5:9b
"""

import argparse
import csv
import itertools
import json
import os
import sys
import time
from datetime import datetime

import requests

# Defaults can be overridden via env
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")

PROMPTS = [
  "Summarize this: In ~200 words, explain the tradeoffs of MoE routing in 20B-class models.",
  "Write Python to parse a CSV and compute a rolling 7d average, then explain complexity.",
  "Reason step-by-step: Prove that the sum of first n odd numbers equals n^2.",
  "Generate ~300 tokens on software architecture tradeoffs for LLM apps.",
  "Explain flash attention’s effect on KV cache bandwidth and throughput at long context.",
]

def stream_generate(prompt, options):
    """
    Streams JSON lines from Ollama /api/generate with partial tokens in 'response' and final metrics on 'done': true.
    """
    url = f"{OLLAMA_HOST}/api/generate"
    data = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
        "options": options
    }
    # Robust streaming setup with timeouts; rely on requests.iter_lines for incremental chunks.
    with requests.post(url, json=data, stream=True, timeout=600) as r:
        r.raise_for_status()
        # Ensure consistent decoding behavior; requests may leave encoding None for binary streams.
        if not r.encoding:
            r.encoding = "utf-8"
        for raw in r.iter_lines(decode_unicode=True, chunk_size=1):
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                # Defensive: ignore partial/invalid lines; continue streaming
                continue

def ollama_version():
    """
    Returns Ollama version string if available from /api/version.
    """
    try:
        url = f"{OLLAMA_HOST}/api/version"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json().get("version", "unknown")
    except Exception:
        return "unknown"

def vram_info():
    """
    Returns (vram_bytes, model_size_bytes) if available from /api/ps (not all builds provide VRAM fields).
    """
    try:
        url = f"{OLLAMA_HOST}/api/ps"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        ps = r.json()
        for m in ps.get("models", []):
            # name usually matches "<repo>:<tag>" like "gpt-oss:20b"
            if m.get("name") == MODEL:
                return m.get("size_vram", None), m.get("size", None)
    except Exception:
        return None, None
    return None, None

def main():
    ap = argparse.ArgumentParser(description="Overnight Ollama benchmark with live streaming, peak TPS, and logging")
    ap.add_argument("--iterations", type=int, default=0, help="0 = run indefinitely")
    ap.add_argument("--min-output-tokens", type=int, default=256)
    ap.add_argument("--log-jsonl", default="ollama_overnight.jsonl")
    ap.add_argument("--log-csv", default="ollama_overnight.csv")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--mirostat", type=int, default=0)
    ap.add_argument("--mirostat_eta", type=float, default=0.1)
    ap.add_argument("--mirostat_tau", type=float, default=5.0)
    ap.add_argument("--num_ctx", type=int, default=8192)
    ap.add_argument("--num_batch", type=int, default=512)
    ap.add_argument("--repeat_last_n", type=int, default=64)
    ap.add_argument("--repeat_penalty", type=float, default=1.08)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=2.0, help="Seconds between runs")
    ap.add_argument("--min_p", type=float, default=0.05, help="Avoid 0.0 which can hurt throughput")
    args = ap.parse_args()

    # Prepare CSV header
    new_csv = not os.path.exists(args.log_csv)
    csv_file = open(args.log_csv, "a", newline="", encoding="utf-8")
    csvw = csv.writer(csv_file)
    if new_csv:
        csvw.writerow([
            "ts","model","prompt_idx","ttft_ms","gen_ms","out_tokens",
            "tps","peak_tps","total_ms","vram_bytes","model_size_bytes",
            "temp","top_p","mirostat","mirostat_eta","mirostat_tau",
            "num_ctx","num_batch","repeat_last_n","repeat_penalty","seed","min_p",
            "error"
        ])

    options = {
        "temperature": args.temp,
        "top_p": args.top_p,
        "num_ctx": args.num_ctx,
        "num_batch": args.num_batch,
        "repeat_last_n": args.repeat_last_n,
        "repeat_penalty": args.repeat_penalty,
        "seed": args.seed,
        "min_p": args.min_p,
    }

    # Only include mirostat options if enabled (some Ollama versions don't support them)
    if args.mirostat > 0:
        options["mirostat"] = args.mirostat
        options["mirostat_eta"] = args.mirostat_eta
        options["mirostat_tau"] = args.mirostat_tau

    prompt_cycle = itertools.cycle(enumerate(PROMPTS))
    iteration = 0
    global_peak_tps = 0.0

    version = ollama_version()
    print(f"Ollama version: {version}", flush=True)
    print(f"Using host={OLLAMA_HOST} model={MODEL}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)

    try:
        while True:
            iteration += 1
            idx, prompt = next(prompt_cycle)
            ts = datetime.utcnow().isoformat()

            print("\n" + "=" * 80, flush=True)
            print(f"[{ts}] Iteration {iteration} | Prompt #{idx}", flush=True)
            print("-" * 80, flush=True)
            print(prompt, flush=True)
            print("-" * 80, flush=True)

            start = time.perf_counter()
            ttft = None
            tokens = 0
            accum_text = []
            error = None

            last_tick = time.perf_counter()
            last_tokens = 0
            peak_tps = 0.0
            estimated_total = args.min_output_tokens

            try:
                for msg in stream_generate(prompt, options):
                    # Handle errors surfaced by Ollama
                    if "error" in msg:
                        error = msg["error"]
                        break

                    # First-token timing
                    if ttft is None and ("response" in msg or "done" in msg):
                        ttft = (time.perf_counter() - start) * 1000.0
                        print(f"\nTTFT: {ttft:.2f} ms", flush=True)

                    # Accumulate streamed token chunks (don't print during generation)
                    if "response" in msg and msg["response"]:
                        chunk = msg["response"]
                        accum_text.append(chunk)
                        # Increment token count for each chunk (rough approximation during streaming)
                        tokens += 1

                    # Update token count if provided by server (will override approximation at end)
                    if "eval_count" in msg:
                        try:
                            tokens = int(msg["eval_count"])
                        except (TypeError, ValueError):
                            pass

                    # Live status line every ~0.5s
                    now = time.perf_counter()
                    if now - last_tick >= 0.5:
                        elapsed = now - start
                        gen_elapsed = elapsed - ((ttft or 0) / 1000.0)
                        inst_tps = max(0.0, (tokens - last_tokens) / max(1e-6, now - last_tick))
                        avg_tps = (tokens / max(1e-6, gen_elapsed)) if gen_elapsed > 0 else 0.0
                        if inst_tps > peak_tps:
                            peak_tps = inst_tps
                        if inst_tps > global_peak_tps:
                            global_peak_tps = inst_tps

                        # Create a compact progress bar that keeps growing
                        bar_width = 30
                        # Use a formula that ensures continuous growth without hitting 100%
                        # progress = tokens / (tokens + buffer)  - asymptotically approaches 1.0
                        buffer = args.min_output_tokens * 0.5  # 128 with default settings
                        progress = min(0.95, tokens / (tokens + buffer))
                        filled = int(bar_width * progress)
                        bar = "█" * filled + "░" * (bar_width - filled)

                        status = (
                            f"\r[{bar}] {tokens:4d}tok | "
                            f"{avg_tps:5.1f} t/s | peak {peak_tps:5.1f} | "
                            f"global {global_peak_tps:5.1f} | {elapsed:4.1f}s"
                        )
                        print(status, end="", flush=True)
                        last_tokens = tokens
                        last_tick = now

                    # End of stream
                    if msg.get("done", False):
                        break

            except KeyboardInterrupt:
                print("\nInterrupted by user.", flush=True)
                break
            except Exception as e:
                error = str(e)

            # Finalize run timing
            end = time.perf_counter()
            total_ms = (end - start) * 1000.0
            gen_ms = total_ms - (ttft if ttft is not None else 0.0)

            # If server didn't emit eval_count, estimate from text tokens
            if tokens == 0:
                out_text = "".join(accum_text)
                tokens = max(args.min_output_tokens, len(out_text.split()))

            # Throughput
            tps = (tokens / (gen_ms / 1000.0)) if gen_ms > 0 else 0.0

            # Clear progress bar and print the generated text
            print("\r" + " " * 100 + "\r", end="", flush=True)
            print("".join(accum_text), flush=True)
            print("", flush=True)

            # VRAM snapshot
            vram_bytes, model_size = vram_info()

            # Log record
            record = {
                "ts": ts,
                "model": MODEL,
                "prompt_idx": idx,
                "ttft_ms": round(ttft if ttft is not None else -1, 2),
                "gen_ms": round(gen_ms, 2),
                "out_tokens": tokens,
                "tps": round(tps, 3),
                "peak_tps": round(peak_tps, 3),
                "total_ms": round(total_ms, 2),
                "vram_bytes": vram_bytes,
                "model_size_bytes": model_size,
                "temp": args.temp,
                "top_p": args.top_p,
                "mirostat": args.mirostat,
                "mirostat_eta": args.mirostat_eta,
                "mirostat_tau": args.mirostat_tau,
                "num_ctx": args.num_ctx,
                "num_batch": args.num_batch,
                "repeat_last_n": args.repeat_last_n,
                "repeat_penalty": args.repeat_penalty,
                "seed": args.seed,
                "min_p": args.min_p,
                "error": error
            }

            # Write JSONL
            try:
                with open(args.log_jsonl, "a", encoding="utf-8") as jf:
                    jf.write(json.dumps(record) + "\n")
            except Exception as e:
                print(f"[warn] failed to write JSONL: {e}", file=sys.stderr, flush=True)

            # Write CSV
            try:
                csvw.writerow([
                    record["ts"], record["model"], record["prompt_idx"], record["ttft_ms"], record["gen_ms"],
                    record["out_tokens"], record["tps"], record["peak_tps"], record["total_ms"], record["vram_bytes"], record["model_size_bytes"],
                    record["temp"], record["top_p"], record["mirostat"], record["mirostat_eta"], record["mirostat_tau"],
                    record["num_ctx"], record["num_batch"], record["repeat_last_n"], record["repeat_penalty"], record["seed"], record["min_p"],
                    record["error"]
                ])
                csv_file.flush()
            except Exception as e:
                print(f"[warn] failed to write CSV: {e}", file=sys.stderr, flush=True)

            # End-of-run summary
            if error:
                print(f"✗ Error: {error}", flush=True)
            else:
                print(f"✓ {tokens} tokens | {tps:.1f} t/s (peak {peak_tps:.1f}, global {global_peak_tps:.1f}) | {gen_ms/1000:.1f}s", flush=True)

            if args.iterations > 0 and iteration >= args.iterations:
                break

            time.sleep(args.sleep)
    finally:
        try:
            csv_file.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()

