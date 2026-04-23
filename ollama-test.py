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
import subprocess
import sys
import time
from datetime import datetime, timezone

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

def stream_generate(prompt, options, think=None):
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
    if think is not None:
        data["think"] = think
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

def gpu_info():
    """
    Returns dict with GPU name, VRAM total, and driver version via nvidia-smi.
    """
    name, vram, driver = "Unknown GPU", 0, "unknown"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 3:
            name, driver = parts[0], parts[2]
            try:
                vram = int(parts[1])
            except ValueError:
                # Unified-memory parts (e.g. Grace Blackwell GB10) report [N/A].
                # Fall back to total system RAM, since GPU and CPU share it.
                vram = _system_ram_mib()
    except Exception:
        pass
    return {"name": name, "vram_total_mib": vram, "driver": driver}


def _system_ram_mib():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 0


def gpu_slug(name):
    """Turn 'NVIDIA GeForce RTX 5060 Ti' into '5060Ti'."""
    # Strip common prefixes
    for prefix in ("NVIDIA GeForce RTX ", "NVIDIA GeForce GTX ", "NVIDIA GeForce ",
                   "NVIDIA RTX ", "NVIDIA "):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace(" ", "")


def model_slug(name):
    """Turn 'qwen3.5:9b' into 'qwen3.5-9b' (filesystem-safe)."""
    return name.replace(":", "-").replace("/", "-")


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
    ap.add_argument("--num_batch", type=int, default=1024)
    ap.add_argument("--repeat_last_n", type=int, default=64)
    ap.add_argument("--repeat_penalty", type=float, default=1.08)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=2.0, help="Seconds between runs")
    ap.add_argument("--min_p", type=float, default=0.05, help="Avoid 0.0 which can hurt throughput")
    ap.add_argument("--results-dir", default="results", help="Directory for GPU result summaries")
    ap.add_argument("--gpu", default=None, help="Override GPU name for results filename (auto-detected if omitted)")
    ap.add_argument("--no-warmup", action="store_true", help="Skip the warmup iteration")
    ap.add_argument("--think", action="store_true", default=None, help="Enable thinking/reasoning mode")
    ap.add_argument("--no-think", action="store_true", help="Disable thinking/reasoning mode")
    ap.add_argument("--benchmark", action="store_true", help="Standard benchmark: warmup + 10 iterations, saves to results dir")
    args = ap.parse_args()

    # Resolve think flag: --think → True, --no-think → False, neither → None (model default)
    think = None
    if args.think:
        think = True
    elif args.no_think:
        think = False

    # Benchmark mode: warmup + 10 iterations
    if args.benchmark:
        args.iterations = 10
        args.no_warmup = False

    # Prepare CSV header
    new_csv = not os.path.exists(args.log_csv)
    csv_file = open(args.log_csv, "a", newline="", encoding="utf-8")
    csvw = csv.writer(csv_file)
    if new_csv:
        csvw.writerow([
            "ts","model","prompt_idx","ttft_ms","gen_ms","out_tokens",
            "tps","total_ms","vram_bytes","model_size_bytes",
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
    run_results = []

    version = ollama_version()
    gi = gpu_info()
    print(f"Ollama version: {version}", flush=True)
    print(f"GPU: {gi['name']} ({gi['vram_total_mib']} MiB, driver {gi['driver']})", flush=True)
    print(f"Using host={OLLAMA_HOST} model={MODEL}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)

    # Warmup: run one iteration to load the model into VRAM, then discard it
    if not args.no_warmup:
        warmup_prompt = PROMPTS[0]
        print("\n" + "=" * 80, flush=True)
        print("Warmup: loading model into VRAM...", flush=True)
        print("-" * 80, flush=True)
        try:
            warmup_start = time.perf_counter()
            warmup_tokens = 0
            warmup_thinking = 0
            last_tick = warmup_start
            for msg in stream_generate(warmup_prompt, options, think=think):
                if "error" in msg:
                    print(f"Warmup error: {msg['error']}", flush=True)
                    break
                if "thinking" in msg and msg["thinking"]:
                    warmup_thinking += 1
                if "response" in msg and msg["response"]:
                    warmup_tokens += 1
                now = time.perf_counter()
                if now - last_tick >= 0.5:
                    elapsed = now - warmup_start
                    think_str = f" think:{warmup_thinking}" if warmup_thinking > 0 else ""
                    print(f"\r  {warmup_tokens:4d}tok{think_str} | {elapsed:4.1f}s", end="", flush=True)
                    last_tick = now
                if msg.get("done", False):
                    break
            warmup_elapsed = time.perf_counter() - warmup_start
            think_str = f", {warmup_thinking} thinking" if warmup_thinking > 0 else ""
            print(f"\rWarmup done: {warmup_tokens} tokens{think_str} in {warmup_elapsed:.1f}s (discarded)", flush=True)
        except Exception as e:
            print(f"Warmup failed: {e} (continuing anyway)", flush=True)
        # Reset the prompt cycle so iteration 1 starts fresh from prompt #0
        prompt_cycle = itertools.cycle(enumerate(PROMPTS))
        time.sleep(args.sleep)

    try:
        while True:
            iteration += 1
            idx, prompt = next(prompt_cycle)
            ts = datetime.now(timezone.utc).isoformat()

            print("\n" + "=" * 80, flush=True)
            print(f"[{ts}] Iteration {iteration} | Prompt #{idx}", flush=True)
            print("-" * 80, flush=True)
            print(prompt, flush=True)
            print("-" * 80, flush=True)

            start = time.perf_counter()
            ttft = None
            tokens = 0
            thinking_tokens = 0
            accum_text = []
            error = None

            last_tick = time.perf_counter()
            last_tokens = 0
            estimated_total = args.min_output_tokens

            try:
                for msg in stream_generate(prompt, options, think=think):
                    # Handle errors surfaced by Ollama
                    if "error" in msg:
                        error = msg["error"]
                        break

                    # First-token timing
                    if ttft is None and ("response" in msg or "done" in msg):
                        ttft = (time.perf_counter() - start) * 1000.0
                        print(f"\nTTFT: {ttft:.2f} ms", flush=True)

                    # Count thinking tokens (streamed in separate field by some models)
                    if "thinking" in msg and msg["thinking"]:
                        thinking_tokens += 1

                    # Accumulate streamed token chunks (don't print during generation)
                    if "response" in msg and msg["response"]:
                        chunk = msg["response"]
                        accum_text.append(chunk)
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
                        avg_tps = (tokens / max(1e-6, gen_elapsed)) if gen_elapsed > 0 else 0.0

                        # Create a compact progress bar that keeps growing
                        bar_width = 30
                        total_tok = tokens + thinking_tokens
                        buffer = args.min_output_tokens * 0.5  # 128 with default settings
                        progress = min(0.95, total_tok / (total_tok + buffer))
                        filled = int(bar_width * progress)
                        bar = "█" * filled + "░" * (bar_width - filled)

                        think_str = f" think:{thinking_tokens}" if thinking_tokens > 0 else ""
                        status = (
                            f"\r[{bar}] {tokens:4d}tok{think_str} | "
                            f"{avg_tps:5.1f} t/s | {elapsed:4.1f}s"
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
                    record["out_tokens"], record["tps"], record["total_ms"], record["vram_bytes"], record["model_size_bytes"],
                    record["temp"], record["top_p"], record["mirostat"], record["mirostat_eta"], record["mirostat_tau"],
                    record["num_ctx"], record["num_batch"], record["repeat_last_n"], record["repeat_penalty"], record["seed"], record["min_p"],
                    record["error"]
                ])
                csv_file.flush()
            except Exception as e:
                print(f"[warn] failed to write CSV: {e}", file=sys.stderr, flush=True)

            # Collect run result
            if not error:
                run_results.append({
                    "tokens": tokens,
                    "thinking_tokens": thinking_tokens,
                    "tps": tps,
                    "ttft_ms": ttft if ttft is not None else -1,
                    "gen_ms": gen_ms,
                    "total_ms": total_ms,
                })

            # End-of-run summary
            if error:
                print(f"✗ Error: {error}", flush=True)
            else:
                think_str = f" (+{thinking_tokens} thinking)" if thinking_tokens > 0 else ""
                print(f"✓ {tokens} tokens{think_str} | {tps:.1f} t/s | {gen_ms/1000:.1f}s", flush=True)

            if args.iterations > 0 and iteration >= args.iterations:
                break

            time.sleep(args.sleep)
    finally:
        try:
            csv_file.close()
        except Exception:
            pass

        # Write GPU results summary
        if run_results:
            slug = args.gpu or gpu_slug(gi["name"])
            mslug = model_slug(MODEL)
            think_suffix = "_think" if think is True else "_nothink" if think is False else ""
            os.makedirs(args.results_dir, exist_ok=True)
            result_path = os.path.join(args.results_dir, f"{slug}_{mslug}{think_suffix}.txt")

            tps_vals = [r["tps"] for r in run_results]
            ttft_vals = [r["ttft_ms"] for r in run_results if r["ttft_ms"] > 0]
            tok_vals = [r["tokens"] for r in run_results]
            think_vals = [r["thinking_tokens"] for r in run_results]
            gen_vals = [r["gen_ms"] for r in run_results]

            avg_tps = sum(tps_vals) / len(tps_vals)
            min_tps = min(tps_vals)
            max_tps = max(tps_vals)
            avg_ttft = sum(ttft_vals) / len(ttft_vals) if ttft_vals else -1
            total_tokens = sum(tok_vals)
            total_thinking = sum(think_vals)
            total_time = sum(gen_vals) / 1000.0

            lines = [
                f"GPU Results: {gi['name']}",
                f"{'=' * 50}",
                f"",
                f"Hardware",
                f"  GPU:          {gi['name']}",
                f"  VRAM:         {gi['vram_total_mib']} MiB",
                f"  Driver:       {gi['driver']}",
                f"",
                f"Software",
                f"  Ollama:       {version}",
                f"  Model:        {MODEL}",
                f"  Context:      {args.num_ctx}",
                f"  Batch size:   {args.num_batch}",
                f"",
                f"Results ({len(run_results)} iterations{', warmup discarded' if not args.no_warmup else ''})",
                f"  Avg TPS:      {avg_tps:.1f} t/s",
                f"  Min TPS:      {min_tps:.1f} t/s",
                f"  Max TPS:      {max_tps:.1f} t/s",
                f"  Avg TTFT:     {avg_ttft:.0f} ms",
                f"  Total tokens: {total_tokens:,}",
            ]
            if total_thinking > 0:
                lines.append(f"  Think tokens: {total_thinking:,}")
            lines += [
                f"  Total time:   {total_time:.1f}s",
                f"",
                f"Per-iteration breakdown",
                f"  {'#':>3}  {'Tokens':>7}  {'Think':>7}  {'TPS':>7}  {'TTFT':>8}  {'Gen':>7}",
                f"  {'-'*3}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*7}",
            ]
            for i, r in enumerate(run_results, 1):
                lines.append(
                    f"  {i:3d}  {r['tokens']:7,}  {r['thinking_tokens']:7,}  "
                    f"{r['tps']:6.1f}  "
                    f"{r['ttft_ms']:7.0f}ms  {r['gen_ms']/1000:6.1f}s"
                )
            lines += [
                f"",
                f"Generated: {datetime.now(timezone.utc).isoformat()}",
            ]

            with open(result_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            print(f"\nResults saved to {result_path}", flush=True)

if __name__ == "__main__":
    main()

