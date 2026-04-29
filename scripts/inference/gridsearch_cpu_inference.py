#!/usr/bin/env python3
"""Grid-search CPU inference over batch_size, num_threads, num_interop_threads.

Runs ``cropland_croptype_mapping_local.py`` once per configuration as a fresh
subprocess (PyTorch only allows ``set_num_interop_threads`` to be called once
per process), parses the ``[profile] total_time=...`` line emitted by
``--memory-logging``, and produces heatmap plots of wall-clock runtime.

Example
-------
    python scripts/inference/gridsearch_cpu_inference.py \
        --batch-sizes 256 512 1024 2048 \
        --num-threads 1 2 4 \
        --num-interop-threads 1 2 4 \
        --output-dir ./gridsearch_results
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "inference" / "cropland_croptype_mapping_local.py"

TOTAL_TIME_RE = re.compile(r"\[profile\]\s+total_time=([0-9.]+)s")

# Mapping from mode label -> extra runner CLI flags. Add new combos here.
MODE_FLAGS: Dict[str, List[str]] = {
    "baseline": [],
    "bf16": ["--optimize-bf16-autocast"],
    "int8": ["--optimize-dynamic-int8"],
    "jit": ["--optimize-jit-freeze"],
    "jit+bf16": ["--optimize-jit-freeze", "--optimize-bf16-autocast"],
    "jit+int8": ["--optimize-jit-freeze", "--optimize-dynamic-int8"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-sizes", type=int, nargs="+",
                   default=[2048])
    p.add_argument("--num-threads", type=int, nargs="+", default=[ 2])
    p.add_argument("--num-interop-threads", type=int, nargs="+",
                   default=[2])
    p.add_argument("--modes", nargs="+", choices=list(MODE_FLAGS.keys()),
                   default=["baseline"],
                   help="Optimization modes to compare (each adds extra runner flags)")
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "gridsearch_results")
    p.add_argument("--baseline-time", type=float, default=None,
                   help="Reference time (s) for the percent-change heatmap. "
                        "If omitted, uses the result for --baseline-config.")
    p.add_argument("--baseline-config", type=int, nargs=3,
                   metavar=("BATCH", "THREADS", "INTEROP"),
                   default=None,
                   help="Grid cell to use as baseline (default: max batch, "
                        "threads=1, interop=1)")
    p.add_argument("--repeats", type=int, default=1,
                   help="Repeat each configuration N times and keep the minimum")
    p.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                   help="Extra args forwarded to the runner (after --)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--skip-run", action="store_true",
                   help="Skip execution and just re-plot from results.json")
    return p.parse_args()


def run_one(batch: int, threads: int, interop: int, mode: str,
            output_dir: Path, extra_args: List[str],
            dry_run: bool) -> Tuple[Optional[float], str]:
    """Run a single config; return (total_time_seconds, log_text)."""
    safe_mode = mode.replace("+", "_")
    nc_out = output_dir / f"out_b{batch}_t{threads}_i{interop}_{safe_mode}.nc"
    cmd = [
        sys.executable, str(RUNNER),
        "--batch-size", str(batch),
        "--cpu-num-threads", str(threads),
        "--cpu-num-interop-threads", str(interop),
        "--memory-logging",
        "--output", str(nc_out),
    ] + list(MODE_FLAGS[mode]) + list(extra_args)

    print(f"\n>>> b={batch:>4} t={threads} i={interop} mode={mode:<10}  cmd: {' '.join(cmd)}")
    if dry_run:
        return None, ""

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    wall = time.perf_counter() - t0
    log = (proc.stdout or "") + "\n" + (proc.stderr or "")

    if proc.returncode != 0:
        print(f"    FAILED (rc={proc.returncode}); wall={wall:.2f}s")
        tail = "\n".join(log.splitlines()[-15:])
        print(tail)
        return None, log

    matches = TOTAL_TIME_RE.findall(log)
    if not matches:
        print(f"    no total_time found; wall={wall:.2f}s")
        return None, log
    total_s = float(matches[-1])
    print(f"    total_time={total_s:.3f}s  (subprocess wall={wall:.2f}s)")
    return total_s, log


def collect(args: argparse.Namespace) -> Dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    configs = list(product(args.batch_sizes, args.num_threads,
                           args.num_interop_threads, args.modes))
    print(f"Total configurations: {len(configs)} (repeats={args.repeats})")

    results: List[Dict] = []
    started = datetime.now().isoformat(timespec="seconds")
    for batch, threads, interop, mode in configs:
        per_run: List[float] = []
        for r in range(args.repeats):
            total_s, log = run_one(
                batch, threads, interop, mode, args.output_dir,
                args.extra_args, args.dry_run,
            )
            if log:
                safe_mode = mode.replace("+", "_")
                fname = f"b{batch}_t{threads}_i{interop}_{safe_mode}_r{r}.log"
                (log_dir / fname).write_text(log, encoding="utf-8")
            if total_s is not None:
                per_run.append(total_s)
        best = min(per_run) if per_run else None
        results.append({
            "batch_size": batch,
            "num_threads": threads,
            "num_interop_threads": interop,
            "mode": mode,
            "total_time_s": best,
            "all_runs_s": per_run,
        })

    payload = {
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "baseline_time_s": args.baseline_time,
        "baseline_config": args.baseline_config,
        "batch_sizes": args.batch_sizes,
        "num_threads": args.num_threads,
        "num_interop_threads": args.num_interop_threads,
        "modes": args.modes,
        "results": results,
    }
    out_json = args.output_dir / "results.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved results to {out_json}")
    return payload


def _grid(payload: Dict, interop: int, mode: str = "baseline") -> np.ndarray:
    bsz = payload["batch_sizes"]
    thr = payload["num_threads"]
    grid = np.full((len(bsz), len(thr)), np.nan, dtype=float)
    for r in payload["results"]:
        if (
            r["num_interop_threads"] != interop
            or r.get("mode", "baseline") != mode
            or r["total_time_s"] is None
        ):
            continue
        i = bsz.index(r["batch_size"])
        j = thr.index(r["num_threads"])
        grid[i, j] = r["total_time_s"]
    return grid


def _annotate(ax, grid: np.ndarray, fmt: str, color_thresh: float,
              compare: Optional[np.ndarray] = None) -> None:
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", color="black")
                continue
            ref = v if compare is None else compare[i, j]
            color = "white" if (not np.isnan(ref) and ref > color_thresh) else "black"
            ax.text(j, i, fmt.format(v), ha="center", va="center", color=color)


def plot_absolute(payload: Dict, out_path: Path, mode: str = "baseline") -> None:
    bsz = payload["batch_sizes"]
    thr = payload["num_threads"]
    interops = payload["num_interop_threads"]

    grids = [_grid(payload, i, mode) for i in interops]
    finite = np.concatenate([g[np.isfinite(g)].ravel() for g in grids]) \
        if any(np.isfinite(g).any() for g in grids) else np.array([0.0, 1.0])
    vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))

    fig, axes = plt.subplots(
        1, len(interops), figsize=(4.2 * len(interops), 4.2), sharey=True
    )
    if len(interops) == 1:
        axes = [axes]
    for ax, interop, grid in zip(axes, interops, grids):
        im = ax.imshow(grid, cmap="viridis", aspect="auto",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(thr)))
        ax.set_xticklabels(thr)
        ax.set_yticks(range(len(bsz)))
        ax.set_yticklabels(bsz)
        ax.set_xlabel("cpu_num_threads")
        ax.set_title(f"interop_threads = {interop}")
        _annotate(ax, grid, "{:.2f}s",
                  color_thresh=(vmin + vmax) / 2.0)
    axes[0].set_ylabel("batch_size")
    fig.colorbar(im, ax=axes, label="total_time (s)", shrink=0.8)
    fig.suptitle(f"CPU inference grid search — wall-clock time (mode={mode})")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"  wrote {out_path}")


def plot_pct_change(payload: Dict, baseline: float, baseline_label: str,
                    out_path: Path, mode: str = "baseline") -> None:
    bsz = payload["batch_sizes"]
    thr = payload["num_threads"]
    interops = payload["num_interop_threads"]

    grids = [(100.0 * (_grid(payload, i, mode) / baseline - 1.0)) for i in interops]
    finite = np.concatenate([g[np.isfinite(g)].ravel() for g in grids]) \
        if any(np.isfinite(g).any() for g in grids) else np.array([-1.0, 1.0])
    vabs = float(np.nanmax(np.abs(finite))) or 1.0

    fig, axes = plt.subplots(
        1, len(interops), figsize=(4.2 * len(interops), 4.2), sharey=True
    )
    if len(interops) == 1:
        axes = [axes]
    for ax, interop, grid in zip(axes, interops, grids):
        im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto",
                       vmin=-vabs, vmax=vabs)
        ax.set_xticks(range(len(thr)))
        ax.set_xticklabels(thr)
        ax.set_yticks(range(len(bsz)))
        ax.set_yticklabels(bsz)
        ax.set_xlabel("cpu_num_threads")
        ax.set_title(f"interop_threads = {interop}")
        _annotate(ax, grid, "{:+.1f}%", color_thresh=vabs * 0.6)
    axes[0].set_ylabel("batch_size")
    fig.colorbar(im, ax=axes,
                 label=f"% change vs baseline {baseline_label} ({baseline:.3f}s)",
                 shrink=0.8)
    fig.suptitle(f"CPU inference grid search — % change vs baseline (mode={mode})")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"  wrote {out_path}")


def _resolve_baseline(payload: Dict, mode: str = "baseline") -> Tuple[float, str]:
    """Pick the baseline runtime: explicit time, explicit cell, or fallback."""
    explicit = payload.get("baseline_time_s")
    cfg = payload.get("baseline_config")
    if cfg is None:
        cfg = [max(payload["batch_sizes"]), 1, 1]
    batch, threads, interop = cfg
    label = f"(batch={batch}, threads={threads}, interop={interop}, mode={mode})"

    if explicit is not None:
        return float(explicit), f"user-specified ({float(explicit):.3f}s)"

    for r in payload["results"]:
        if (
            r["batch_size"] == batch
            and r["num_threads"] == threads
            and r["num_interop_threads"] == interop
            and r.get("mode", "baseline") == mode
            and r["total_time_s"] is not None
        ):
            return float(r["total_time_s"]), label
    raise RuntimeError(
        f"Baseline cell {label} not found in results. "
        f"Pass --baseline-time or --baseline-config explicitly."
    )


def best_table(payload: Dict) -> str:
    rows = [r for r in payload["results"] if r["total_time_s"] is not None]
    if not rows:
        return "(no successful runs)"
    rows.sort(key=lambda r: r["total_time_s"])
    lines = [
        "Top 10 fastest configurations:",
        f"{'batch':>6} {'threads':>8} {'interop':>8} {'mode':<10} {'time_s':>9}",
    ]
    for r in rows[:10]:
        lines.append(
            f"{r['batch_size']:>6} {r['num_threads']:>8} "
            f"{r['num_interop_threads']:>8} {r.get('mode', 'baseline'):<10} "
            f"{r['total_time_s']:>9.3f}"
        )
    return "\n".join(lines)


def plot_modes(payload: Dict, out_path: Path) -> None:
    """Bar chart of total_time per mode, grouped by (batch, threads, interop)."""
    modes = payload.get("modes", ["baseline"])
    if len(modes) < 2:
        return
    # Group results by config tuple.
    configs: "OrderedDict[Tuple[int,int,int], Dict[str, Optional[float]]]" = (
        OrderedDict()
    )
    for r in payload["results"]:
        key = (r["batch_size"], r["num_threads"], r["num_interop_threads"])
        configs.setdefault(key, {})[r.get("mode", "baseline")] = r["total_time_s"]

    n_groups = len(configs)
    n_modes = len(modes)
    fig_w = max(6.0, 1.4 * n_groups * n_modes)
    fig, ax = plt.subplots(figsize=(fig_w, 4.6))
    width = 0.8 / n_modes
    x = np.arange(n_groups)
    cmap = plt.get_cmap("tab10")

    for m_idx, mode in enumerate(modes):
        vals = [configs[k].get(mode, None) for k in configs.keys()]
        plot_vals = [v if v is not None else 0.0 for v in vals]
        offsets = x + (m_idx - (n_modes - 1) / 2.0) * width
        bars = ax.bar(offsets, plot_vals, width=width,
                      label=mode, color=cmap(m_idx % 10))
        for rect, v in zip(bars, vals):
            if v is None:
                continue
            ax.text(rect.get_x() + rect.get_width() / 2.0,
                    rect.get_height(), f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"b{b}\nt{t} i{i}" for (b, t, i) in configs.keys()],
                       fontsize=9)
    ax.set_ylabel("total_time (s)")
    ax.set_title("CPU inference — optimization modes comparison")
    ax.legend(title="mode", loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"  wrote {out_path}")


def main() -> None:
    args = parse_args()

    if args.skip_run:
        payload = json.loads(
            (args.output_dir / "results.json").read_text(encoding="utf-8")
        )
    else:
        payload = collect(args)

    if args.dry_run:
        return

    print("\n" + best_table(payload))

    modes = payload.get("modes", ["baseline"])
    has_baseline = "baseline" in modes
    plot_mode = "baseline" if has_baseline else modes[0]

    try:
        baseline_s, baseline_label = _resolve_baseline(payload, plot_mode)
        print(f"\nBaseline for % change: {baseline_label} -> {baseline_s:.3f}s")
        plot_absolute(payload, args.output_dir / "heatmap_absolute.png",
                      mode=plot_mode)
        plot_pct_change(payload, baseline_s, baseline_label,
                        args.output_dir / "heatmap_pct_change.png",
                        mode=plot_mode)
    except RuntimeError as exc:
        print(f"Skipping heatmaps: {exc}")

    if len(modes) > 1:
        plot_modes(payload, args.output_dir / "modes_comparison.png")

    plt.show()


if __name__ == "__main__":
    main()
