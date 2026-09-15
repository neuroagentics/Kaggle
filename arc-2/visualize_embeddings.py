"""visualize_embeddings.py — 2D UMAP projection of hyperbolic HPM embeddings.

Loads the GlobalMemoryBank seed bank, projects all embeddings to 2D via UMAP,
and saves an annotated scatter plot (PNG) + a raw CSV of coordinates.

Usage:
    python visualize_embeddings.py
    python visualize_embeddings.py --seed-bank hyper_arc/seed_bank.json \\
        --output embeddings_umap.png

Requirements (not in requirements.txt — install separately):
    pip install umap-learn matplotlib

What to look for:
    GOOD: Semantically similar programs (e.g. all rotate-heavy programs)
          cluster together in tight groups.
    BAD:  Random static / uniform scatter → geoopt embedding logic is flawed,
          or the seed bank contains too few programs to show structure.
          Minimum ~50 programs needed for meaningful clustering.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

# ── Soft-import optional deps ─────────────────────────────────────────────

try:
    import umap
except ImportError:
    print(
        "[ERROR] umap-learn not installed.\n"
        "        Install with: pip install umap-learn",
        flush=True,
    )
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")   # headless — works on Kaggle
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError:
    print(
        "[ERROR] matplotlib not installed.\n"
        "        Install with: pip install matplotlib",
        flush=True,
    )
    sys.exit(1)

from hyper_arc.hpm import GlobalMemoryBank


# ── Label helpers ─────────────────────────────────────────────────────────

def _program_label(program: list) -> str:
    """Short label: dominant primitive in the program."""
    if not program:
        return "empty"
    names = [name for name, _ in program]
    most_common = Counter(names).most_common(1)[0][0]
    return most_common


def _program_signature(program: list) -> str:
    """Primitive sequence as a string for tooltip / CSV."""
    return "→".join(name for name, _ in program) if program else "(empty)"


# ── Cluster quality metric ────────────────────────────────────────────────

def _intra_inter_ratio(
    coords: "torch.Tensor",
    labels: list[str],
) -> float:
    """Rough cluster quality: mean intra-cluster dist / mean inter-cluster dist.

    Values < 1.0 indicate clustering (good).
    Values ≈ 1.0 indicate random scatter (bad).
    """
    import numpy as np
    unique_labels = list(set(labels))
    if len(unique_labels) < 2:
        return float("nan")

    label_arr = [labels.index(l) for l in labels]  # noqa: E741
    coords_np = coords.numpy() if hasattr(coords, "numpy") else coords

    intra_dists: list[float] = []
    inter_dists: list[float] = []

    for li, la in enumerate(unique_labels):
        mask = [l == la for l in labels]  # noqa: E741
        in_pts  = coords_np[[i for i, m in enumerate(mask) if m]]
        out_pts = coords_np[[i for i, m in enumerate(mask) if not m]]
        if len(in_pts) >= 2:
            for i in range(len(in_pts)):
                for j in range(i + 1, len(in_pts)):
                    intra_dists.append(float(np.linalg.norm(in_pts[i] - in_pts[j])))
        if len(in_pts) >= 1 and len(out_pts) >= 1:
            for p in in_pts:
                for q in out_pts:
                    inter_dists.append(float(np.linalg.norm(p - q)))

    if not intra_dists or not inter_dists:
        return float("nan")
    return float(sum(intra_dists) / len(intra_dists)) / \
           float(sum(inter_dists) / len(inter_dists))


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="2D UMAP projection of HPM hyperbolic embeddings."
    )
    parser.add_argument(
        "--seed-bank",
        default="hyper_arc/seed_bank.json",
        help="Path to seed_bank.json (default: hyper_arc/seed_bank.json).",
    )
    parser.add_argument(
        "--output",
        default="embeddings_umap.png",
        help="Output PNG path (default: embeddings_umap.png).",
    )
    parser.add_argument(
        "--csv",
        default="embeddings_umap.csv",
        help="Output CSV path for raw coordinates.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors (default: 15). Lower = finer local structure.",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist (default: 0.1).",
    )
    args = parser.parse_args()

    seed_path = Path(args.seed_bank)
    if not seed_path.exists():
        print(
            f"[ERROR] seed_bank.json not found at {seed_path}.\n"
            "        Run build_global_memory.py first.",
            flush=True,
        )
        return 1

    # ── Load seed bank ────────────────────────────────────────────────────
    gmb = GlobalMemoryBank(k=5)
    gmb.load_seed_bank(seed_path)
    n = len(gmb)
    print(f"[INFO] Loaded {n} programs from {seed_path}", flush=True)

    if n < 5:
        print(
            f"[WARN] Only {n} programs in seed bank. "
            "UMAP needs at least 5 for meaningful projection. "
            "Run build_global_memory.py with more tasks.",
            flush=True,
        )
        return 1

    if n < 50:
        print(
            f"[WARN] {n} programs is low for cluster validation. "
            "Recommend ≥ 50 for semantic clustering signal.",
            flush=True,
        )

    # ── Extract embeddings ────────────────────────────────────────────────
    embeddings = torch.stack([e.embedding for e in gmb._entries])  # (N, DIM)
    labels     = [_program_label(e.program)    for e in gmb._entries]
    sigs       = [_program_signature(e.program) for e in gmb._entries]

    print(f"[INFO] Embedding matrix: {tuple(embeddings.shape)}", flush=True)
    print(f"[INFO] Unique dominant primitives: {set(labels)}", flush=True)

    # ── UMAP projection ───────────────────────────────────────────────────
    n_neighbors = min(args.n_neighbors, n - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=args.min_dist,
        metric="euclidean",   # UMAP operates on the raw embedding vectors
        random_state=42,
    )
    print(
        f"[INFO] Running UMAP (n_neighbors={n_neighbors}, "
        f"min_dist={args.min_dist}) ...",
        flush=True,
    )
    coords_2d = reducer.fit_transform(embeddings.numpy())  # (N, 2)

    # ── Cluster quality check ─────────────────────────────────────────────
    import numpy as np
    ratio = _intra_inter_ratio(coords_2d, labels)
    if not isinstance(ratio, float) or ratio != ratio:  # NaN
        print("[INFO] Cluster quality: insufficient distinct groups to measure.", flush=True)
    elif ratio < 0.6:
        print(f"[GOOD] Cluster quality ratio = {ratio:.3f} (<0.6) — "
              "semantically similar programs cluster tightly. "
              "Embedding logic is working.", flush=True)
    elif ratio < 1.0:
        print(f"[OK]   Cluster quality ratio = {ratio:.3f} (0.6–1.0) — "
              "moderate clustering. Consider more seed programs.", flush=True)
    else:
        print(
            f"[WARN] Cluster quality ratio = {ratio:.3f} (≥1.0) — "
            "embeddings look like random scatter.\n"
            "       Possible causes:\n"
            "         1. Too few seed programs (run with --n-tasks 200+)\n"
            "         2. geoopt expmap0 projection is collapsing vectors\n"
            "         3. _encode_program produces near-identical vectors\n"
            "       Review hyper_arc/hpm.py _encode_program and _project.",
            flush=True,
        )

    # ── Plot ──────────────────────────────────────────────────────────────
    unique_labels = sorted(set(labels))
    color_map     = {la: cm.tab20(i / max(len(unique_labels), 1))
                     for i, la in enumerate(unique_labels)}
    colors = [color_map[la] for la in labels]

    fig, ax = plt.subplots(figsize=(12, 9))
    scatter = ax.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=colors,
        s=60,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.4,
    )

    # Add index labels for small seed banks
    if n <= 80:
        for i, (x, y) in enumerate(coords_2d):
            ax.annotate(
                str(i),
                (x, y),
                fontsize=6,
                alpha=0.6,
                textcoords="offset points",
                xytext=(4, 2),
            )

    # Legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map[la], markersize=8, label=la)
        for la in unique_labels
    ]
    ax.legend(handles=handles, title="Dominant Primitive",
              fontsize=8, title_fontsize=9, loc="best")

    ax.set_title(
        f"Hyperbolic Program Memory — UMAP Projection\n"
        f"{n} programs | cluster ratio={ratio:.3f if isinstance(ratio, float) and ratio == ratio else 'N/A'}",
        fontsize=12,
    )
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"[INFO] Plot saved to {args.output}", flush=True)

    # ── CSV export ────────────────────────────────────────────────────────
    csv_lines = ["index,x,y,dominant_primitive,program_signature"]
    for i, (x, y) in enumerate(coords_2d):
        csv_lines.append(
            f'{i},{x:.6f},{y:.6f},{labels[i]},"{sigs[i]}"'
        )
    Path(args.csv).write_text("\n".join(csv_lines))
    print(f"[INFO] Coordinates saved to {args.csv}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
