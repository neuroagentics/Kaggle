# Architecture Rules

## Dependency Versioning

Never hardcode strict, outdated version pins in `requirements.txt`.
Use `>=` for all major libraries to ensure compatibility with modern Python environments.

**Bad:**
```
torch==2.3.*
numpy==1.26.*
```

**Good:**
```
torch>=2.6.0
numpy>=1.26.0
```

Rationale: strict pins break installs on Python 3.11+ and newer CUDA toolchains.
Minimum versions should reflect the lowest version the code is actually tested against.
If a specific upper bound is needed (e.g., a known breaking API change), document why inline.

## Python Version

Target Python 3.10+. Do not use syntax or stdlib features unavailable before 3.10.
`match/case`, `X | Y` union types in annotations, and `tomllib` are all fine.

## Tensor dtype

ARC grid tensors use `torch.int8`. Do not silently upcast to `int32`/`int64` in DSL
operations without an explicit cast and a comment explaining why.

## No Neural Approximations in DSL

All `SpatialDSL` primitives must be pure deterministic tensor ops.
No `nn.Module`, no learned weights, no `requires_grad=True` on grid data.
