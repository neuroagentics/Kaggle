# Tech Stack

## Language
- Python 3.10+
- Type annotations throughout (`from __future__ import annotations`)

## Core Dependencies (pinned in `requirements.txt`)
| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.3.* | All tensor ops; ESB is `torch.int64 (C,H,W)` |
| `geoopt` | 0.5.* | Poincaré Ball manifold, `expmap0`, geodesic distance |
| `numpy` | 1.26.* | Supporting numerics |
| `kaggle` | 1.6.* | CLI-based dataset download |
| `pytest` | 8.2.* | Unit tests |
| `hypothesis` | 6.* | Property-based tests |

## No Build System
Plain Python project. No Makefile, no `pyproject.toml`, no `setup.py`.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Download ARC-AGI-2 dataset (requires ~/.kaggle/kaggle.json)
python download_data.py --dest ./data/arc-agi-2

# Build the HPM seed bank from training data (offline, pre-inference)
python build_global_memory.py --data-dir ./data/arc-agi-2 --n-tasks 50 --output hyper_arc/seed_bank.json

# Run the solver (auto-detects Kaggle vs. local paths)
python main.py
python main.py --data-dir ./data/arc-agi-2 --output ./submission.json

# Run all tests
pytest tests/ -v

# Run property-based tests only
pytest tests/ -v -k "property"
```

## Testing Approach
- **Unit tests** (`pytest`) — one test per DSL primitive, ESB operation, and cost function
- **Property-based tests** (`hypothesis`) — 23 named properties covering shape invariants, round-trips, and algorithm correctness
- All tests live in `tests/` and are run with `pytest tests/`
