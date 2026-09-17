"""R11 GPU benchmark: load Qwen2.5-7B 4-bit and measure VRAM + latency.

Loads the deliberator via TransformersDeliberator (the production path), runs a
real structured call on a sample observation, and records peak VRAM and
call latency. Writes a JSON report.

    python scripts/benchmark_deliberator.py --model models/qwen2.5-7b-instruct --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "vendor" / "ARC-AGI-3-Agents"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R11 deliberator GPU benchmark")
    parser.add_argument("--model", default=str(_ROOT / "models" / "qwen2.5-7b-instruct"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--out", type=Path, default=_ROOT / "artifacts" / "deliberator_benchmark.json")
    args = parser.parse_args(argv)

    import torch
    from agent.deliberator import DeliberatorConfig, DeliberatorInput
    from agent.deliberator_transformers import TransformersDeliberator

    torch.cuda.reset_peak_memory_stats()
    vram_before = torch.cuda.memory_allocated() / 1024**3

    cfg = DeliberatorConfig(
        model_name="qwen2.5-7b-instruct", model_version="world-model-v0",
        max_new_tokens=768, max_input_tokens=4096, max_wall_seconds=120.0,
    )
    print(f"[bench] loading {args.model} 4-bit on {args.device} ...", flush=True)
    t_load = time.time()
    delib = TransformersDeliberator(cfg, model_path=args.model, device=args.device, load_now=True)
    load_secs = time.time() - t_load
    vram_after_load = torch.cuda.memory_allocated() / 1024**3
    print(f"[bench] loaded in {load_secs:.1f}s; VRAM allocated {vram_after_load:.2f} GiB", flush=True)

    sample = DeliberatorInput(
        game_id="bench", level_id=0,
        recent_grids=(tuple(tuple((r + c) % 16 for c in range(16)) for r in range(16)),),
        recent_actions=((3, None, None), (4, None, None)),
        recent_outcomes=(False, False),
        active_belief_summaries=("a blue block moved right after action 3",),
        active_goal_summaries=("reach the green region on the right",),
        prediction_errors=(0.2, 0.15),
        real_actions_since_last_call=8,
    )

    latencies = []
    n_hyp = n_goal = 0
    repaired = 0
    for i in range(args.calls):
        t0 = time.time()
        out = delib.call(sample)
        dt = time.time() - t0
        latencies.append(dt)
        n_hyp = len(out.mechanic_hypotheses)
        n_goal = len(out.goal_signals)
        repaired += int(out.repair_applied)
        print(f"[bench] call {i+1}: {dt:.2f}s, {n_hyp} hyps, {n_goal} goals, "
              f"repair={out.repair_applied}, tokens~{out.raw_tokens_used}", flush=True)

    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    report = {
        "model": args.model,
        "device": args.device,
        "load_seconds": round(load_secs, 2),
        "vram_before_gib": round(vram_before, 3),
        "vram_after_load_gib": round(vram_after_load, 3),
        "peak_vram_gib": round(peak_vram, 3),
        "total_gpu_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "calls": args.calls,
        "latency_seconds": {
            "mean": round(sum(latencies) / len(latencies), 3),
            "min": round(min(latencies), 3),
            "max": round(max(latencies), 3),
        },
        "last_hypotheses": n_hyp,
        "last_goals": n_goal,
        "repairs": repaired,
        "vram_headroom_ok": peak_vram <= 0.8 * (torch.cuda.get_device_properties(0).total_memory / 1024**3),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n[bench] REPORT:")
    print(json.dumps(report, indent=2))
    print(f"[bench] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
