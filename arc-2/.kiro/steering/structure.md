# Project Structure

## Root Layout

```
arc-2/
├── main.py                    # Pipeline entry point — orchestrates full solve loop
├── download_data.py           # Standalone Kaggle CLI dataset downloader
├── build_global_memory.py     # Offline seed bank generator (run before inference)
├── requirements.txt           # Pinned Python dependencies
├── hyper_arc/                 # Core solver package
│   ├── __init__.py            # Public re-exports for the package
│   ├── esb.py                 # EideticSpatialBuffer dataclass
│   ├── spatial_dsl.py         # SpatialDSL (12 primitives) + DSLProgram type
│   ├── hpm.py                 # GlobalMemoryBank + LocalTaskBuffer
│   ├── mcts.py                # MCTSEngine + MCTSNode + enumerate_actions
│   ├── cost.py                # exact_match + partial_reward
│   ├── utils.py               # Grid I/O, JSON helpers
│   └── seed_bank.json         # Pre-built HPM seed programs (binary artifact)
├── tests/
│   ├── __init__.py
│   ├── test_esb.py
│   ├── test_spatial_dsl.py
│   ├── test_cost.py
│   └── test_hpm.py
└── .kiro/specs/hyper-arc-solver/
    ├── requirements.md
    ├── design.md
    └── tasks.md
```

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `esb.py` | `ESB` dataclass wrapping `torch.int64 (C,H,W)` tensors; `pad`, `crop`, `mask`, `shift`, `from_grid`, `to_grid` |
| `spatial_dsl.py` | Stateless `SpatialDSL` class with 12 pure-tensor primitives; `DSLProgram = list[PrimitiveCall]`; `apply_program` |
| `hpm.py` | `GlobalMemoryBank` (read-only seed bank, loaded once at startup) and `LocalTaskBuffer` (per-task, updated on success); both use Poincaré Ball embeddings via `geoopt` |
| `mcts.py` | `MCTSNode` dataclass; `MCTSEngine` with `_select`, `_expand`, `_rollout`, `_backprop`, `solve`; `enumerate_actions` |
| `cost.py` | `exact_match(candidate, target) -> bool`; `partial_reward(candidate, target) -> float` |
| `main.py` | CLI args; Kaggle env detection; checkpoint resume; task loop; per-task error handling; `submission.json` output |

## Key Architectural Rules

- `download_data.py` has **no imports from `hyper_arc/`** — it is fully self-contained
- `SpatialDSL` methods are all `@staticmethod` — zero instance state, zero learned weights
- ESB operations always return **new ESB instances** (immutable/functional style)
- `GlobalMemoryBank` is **never mutated during inference** — only `LocalTaskBuffer` is updated when programs succeed
- `main.py` instantiates a **fresh `LocalTaskBuffer` per task**
- All `print()` calls in `main.py` use `flush=True` for live Kaggle Notebook output

## Data Flow

```
ARC JSON → ESB.from_grid() → MCTSEngine.solve() → DSLProgram → apply_program() → submission.json
                                    ↑
                          GlobalMemoryBank (seed bank)
                          LocalTaskBuffer  (task-local successes)
```

## Checkpoint File (for Kaggle session resume)

Saved via `torch.save` after every task:
```python
{
    "submission":    dict[str, list],   # task_id → predicted grids
    "completed_ids": set[str],           # fully processed task IDs
    "global_memory": list[{...}],        # GlobalMemoryBank entries
}
```
