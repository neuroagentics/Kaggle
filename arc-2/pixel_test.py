"""pixel_test.py — 3x3 Pixel Audit Test.

Verifies spatial correctness of rotate, translate, extract_object, and overlay
on a known 3x3 grid with a single red pixel (color=2) at (row=0, col=0).

Expected results:
  rotate 90 CW  : pixel moves from (0,0) → (0,2)
  translate dx=1: pixel moves from (0,0) → (0,1)
  extract_object: only the red pixel survives, rest zeroed
  overlay       : red pixel composited over a blue background
"""
import sys
sys.path.insert(0, ".")

from hyper_arc.esb import ESB
from hyper_arc.spatial_dsl import SpatialDSL

DSL = SpatialDSL()
RED  = 2
BLUE = 1

PASS = "✓ PASS"
FAIL = "✗ FAIL"

failures = []

def check(label: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    print(f"  {status}  {label}" + (f"  [{detail}]" if detail else ""))
    if not condition:
        failures.append(label)

print("=" * 60)
print("3x3 Pixel Audit Test")
print("=" * 60)

# ── Setup ──────────────────────────────────────────────────────────────────
grid = [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]
esb = ESB.from_grid(grid)
esb.data[0, 0, 0] = RED   # red pixel at (row=0, col=0)

print(f"\nInitial grid (channel 0):\n{esb.data[0].tolist()}")

# ── Test 1: rotate 90° CW ──────────────────────────────────────────────────
# CW 90° maps (row, col) → (col, H-1-row)
# For H=3: (0,0) → (0, 2)
print("\n[1] rotate(degrees=90) — CW rotation")
r = DSL.rotate(esb, 90)
print(f"    Result:\n    {r.data[0].tolist()}")
check("pixel moves to (row=0, col=2)",
      r.data[0, 0, 2].item() == RED,
      f"got {r.data[0, 0, 2].item()}")
check("original position (0,0) is empty",
      r.data[0, 0, 0].item() == 0,
      f"got {r.data[0, 0, 0].item()}")
check("all other cells are 0",
      (r.data[0].sum().item()) == RED,
      f"sum={r.data[0].sum().item()}")

# ── Test 2: translate dx=1 dy=0 ───────────────────────────────────────────
# Pixel at (0,0) should move to (0,1)
print("\n[2] translate(dx=1, dy=0) — move right by 1")
t = DSL.translate(esb, dx=1, dy=0)
print(f"    Result:\n    {t.data[0].tolist()}")
check("pixel moves to (row=0, col=1)",
      t.data[0, 0, 1].item() == RED,
      f"got {t.data[0, 0, 1].item()}")
check("original position (0,0) vacated",
      t.data[0, 0, 0].item() == 0,
      f"got {t.data[0, 0, 0].item()}")
check("all other cells are 0",
      t.data[0].sum().item() == RED,
      f"sum={t.data[0].sum().item()}")

# ── Test 3: translate dx=0 dy=1 ───────────────────────────────────────────
# Pixel at (0,0) should move to (1,0)
print("\n[3] translate(dx=0, dy=1) — move down by 1")
d = DSL.translate(esb, dx=0, dy=1)
print(f"    Result:\n    {d.data[0].tolist()}")
check("pixel moves to (row=1, col=0)",
      d.data[0, 1, 0].item() == RED,
      f"got {d.data[0, 1, 0].item()}")
check("original position (0,0) vacated",
      d.data[0, 0, 0].item() == 0,
      f"got {d.data[0, 0, 0].item()}")

# ── Test 4: extract_object ─────────────────────────────────────────────────
# Only connected component of RED at (x=0, y=0) survives
# Put a second non-connected red pixel to confirm isolation
grid2 = [[2, 0, 0],
         [0, 0, 0],
         [0, 0, 2]]   # two isolated red pixels
esb2 = ESB.from_grid(grid2)
print("\n[4] extract_object(x=0, y=0) — isolate top-left pixel")
print(f"    Input:\n    {esb2.data[0].tolist()}")
ex = DSL.extract_object(esb2, x=0, y=0)
print(f"    Result:\n    {ex.data[0].tolist()}")
check("seed pixel (0,0) retained",
      ex.data[0, 0, 0].item() == RED,
      f"got {ex.data[0, 0, 0].item()}")
check("unconnected pixel (2,2) zeroed",
      ex.data[0, 2, 2].item() == 0,
      f"got {ex.data[0, 2, 2].item()}")
check("only one pixel survives",
      ex.data[0].sum().item() == RED,
      f"sum={ex.data[0].sum().item()}")

# ── Test 5: overlay ────────────────────────────────────────────────────────
# Blue background 3x3; overlay red pixel at (0,0) → (0,0) should be RED
bg_grid = [[BLUE]*3 for _ in range(3)]
bg = ESB.from_grid(bg_grid)
fg = ESB.from_grid([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
fg.data[0, 0, 0] = RED

print("\n[5] overlay(blend_mode='overwrite') — red pixel over blue background")
print(f"    Background:\n    {bg.data[0].tolist()}")
print(f"    Foreground:\n    {fg.data[0].tolist()}")
ov = DSL.overlay(bg, fg, blend_mode="overwrite")
print(f"    Result:\n    {ov.data[0].tolist()}")
check("(0,0) is RED (fg overwrites bg)",
      ov.data[0, 0, 0].item() == RED,
      f"got {ov.data[0, 0, 0].item()}")
check("(0,1) stays BLUE (fg was 0)",
      ov.data[0, 0, 1].item() == BLUE,
      f"got {ov.data[0, 0, 1].item()}")
check("(1,0) stays BLUE",
      ov.data[0, 1, 0].item() == BLUE,
      f"got {ov.data[0, 1, 0].item()}")

# ── Test 6: rotate 180° round-trip ────────────────────────────────────────
print("\n[6] rotate(180) — pixel at (0,0) must move to (2,2)")
r180 = DSL.rotate(esb, 180)
print(f"    Result:\n    {r180.data[0].tolist()}")
check("pixel moves to (row=2, col=2)",
      r180.data[0, 2, 2].item() == RED,
      f"got {r180.data[0, 2, 2].item()}")

# ── Test 7: four 90° rotations = identity ─────────────────────────────────
print("\n[7] rotate(90) × 4 — must return to original")
import torch
result = esb
for _ in range(4):
    result = DSL.rotate(result, 90)
check("full round-trip is identity",
      torch.all(result.data == esb.data).item())

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if failures:
    print(f"RESULT: FAILED — {len(failures)} check(s) failed:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("RESULT: ALL CHECKS PASSED — DSL spatial correctness confirmed.")
    sys.exit(0)
