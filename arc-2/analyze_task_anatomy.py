"""Task anatomy + common-thread finder for ARC tasks.

Answers three things the reasoner needs but currently guesses:
  1. WHAT DOES THIS TASK ENTAIL? A plain-language structural summary of the
     input->output transformation, derived from deterministic perception
     (shape change, palette change, object counts, correspondences,
     translations, recolors, symmetry, separators) -- never the answer grid.
  2. WHAT IS THE COMMON THREAD? Tasks are grouped by a coarse "transformation
     signature" so tasks that behave alike cluster together. A shared signature
     is the unit a transferable lesson attaches to.
  3. TRANSFER HOOK: emits a signature key per task that a lesson/memory record
     can be filed under and recalled for structurally similar future tasks.

Usage:
  python analyze_task_anatomy.py --task-ids e9afcf9a 44f52bb0 ...
  python analyze_task_anatomy.py --all-validation           # whole fold
  python analyze_task_anatomy.py --unsolved                 # bakeoff-unsolved set
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any, Mapping

from hyper_arc.contender.perception import perceive_task
from hyper_arc.contender.experience_bank import task_fingerprint_v2
from hyper_arc.contender.data_protocol import load_verified_split


def _shape_relation(ish, osh) -> str:
    ih, iw = ish
    oh, ow = osh
    if (ih, iw) == (oh, ow):
        return "same-size"
    if (ih, iw) == (ow, oh):
        return "transposed-size"
    if oh == 1 and ow == 1:
        return "reduce-to-1x1"
    if ih and iw and oh % ih == 0 and ow % iw == 0:
        return f"upscale-{oh // ih}x{ow // iw}"
    if oh <= ih and ow <= iw:
        return "shrink"
    return "reshape"


def anatomy(task_id: str, task_data: Mapping[str, Any]) -> dict[str, Any]:
    """Structural, answer-free description of what the task does."""
    state = perceive_task(task_id, task_data)
    shape_rels = Counter()
    palette_changes = 0
    object_delta = 0
    recolors = 0
    translations = 0
    added = 0
    removed = 0
    for delta in state.deltas:
        shape_rels[_shape_relation(delta.input_shape, delta.output_shape)] += 1
        added += len(delta.added_object_ids)
        removed += len(delta.removed_object_ids)
        for corr in delta.correspondences:
            if corr.color_changed:
                recolors += 1
            if corr.translation != (0, 0):
                translations += 1
    # Symmetry gained/lost across the pair (first example, as a cue).
    in_syms = set(state.grids[0].symmetries) if state.grids else set()
    # find the matching output grid symmetry
    out_syms: set[str] = set()
    for g in state.grids:
        if g.grid_ref.endswith(":output") and g.grid_ref.startswith("perceive"):
            pass
    n = len(task_data.get("train", []))
    fp = task_fingerprint_v2(task_data)

    # A coarse behavior signature: the dominant shape relation + which kinds of
    # change dominate. This is the "common thread" key.
    dominant_shape = shape_rels.most_common(1)[0][0] if shape_rels else "unknown"
    change_kinds = []
    if recolors:
        change_kinds.append("recolor")
    if translations:
        change_kinds.append("move")
    if added:
        change_kinds.append("add-objects")
    if removed:
        change_kinds.append("remove-objects")
    if not change_kinds:
        change_kinds.append("holistic")  # no clean object correspondence
    signature = f"{dominant_shape}|{'+'.join(sorted(set(change_kinds)))}"

    return {
        "task_id": task_id,
        "train_pairs": n,
        "shape_relation": dict(shape_rels),
        "recolors": recolors,
        "translations": translations,
        "added_objects": added,
        "removed_objects": removed,
        "tags": list(fp.tags),
        "signature": signature,
    }


def _describe(a: dict[str, Any]) -> str:
    parts = [f"{a['train_pairs']} demos", a["signature"]]
    detail = []
    if a["recolors"]:
        detail.append(f"{a['recolors']} recolors")
    if a["translations"]:
        detail.append(f"{a['translations']} moves")
    if a["added_objects"]:
        detail.append(f"+{a['added_objects']} obj")
    if a["removed_objects"]:
        detail.append(f"-{a['removed_objects']} obj")
    if detail:
        parts.append("(" + ", ".join(detail) + ")")
    return "  ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-ids", nargs="*")
    parser.add_argument("--all-validation", action="store_true")
    parser.add_argument("--unsolved", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    development, _ = load_verified_split(
        "data/arc-agi-2/arc-agi_training_challenges.json",
        "config/arc2_split_v1.json",
    )
    validation_ids = list(development)[640:]

    # The tasks the bakeoff actually touched (from the funnel analysis).
    bakeoff_unsolved = [
        "0520fde7", "3aa6fb7a", "3ac3eb23", "44f52bb0", "4522001f", "59341089",
        "60a26a3e", "6d1d5c90", "72a961c9", "73ccf9c2", "88a62173", "a85d4709",
        "b7fb29bc", "dae9d2b5", "e6de6e8f", "ff72ca3e",
    ]

    if args.task_ids:
        ids = args.task_ids
    elif args.unsolved:
        ids = bakeoff_unsolved
    elif args.all_validation:
        ids = validation_ids
    else:
        ids = ["e9afcf9a"] + bakeoff_unsolved

    anatomies = []
    clusters: dict[str, list[str]] = defaultdict(list)
    for tid in ids:
        if tid not in development:
            print(f"[skip] {tid} not in development split")
            continue
        a = anatomy(tid, development[tid])
        anatomies.append(a)
        clusters[a["signature"]].append(tid)
        print(f"{tid}:  {_describe(a)}")

    print("\n=== COMMON THREADS (tasks grouped by transformation signature) ===")
    for sig, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        print(f"  [{len(members)}] {sig}")
        print(f"        {', '.join(members)}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(
                {"anatomies": anatomies, "clusters": clusters}, fh, indent=2
            )
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
