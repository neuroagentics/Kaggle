# Architecture Rules

## Dependency Versioning

Use tested bounded ranges for development dependencies. Deployment must freeze
and hash the exact offline artifacts and record the tested Kaggle base image.
Never replace a release lock with unbounded minimum versions. README.md is the
current release contract; these rules do not authorize a release by themselves.

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

Target the tested Kaggle Python 3.12 runtime and local Python 3.13 test runtime.
`tomllib` requires Python 3.11+, not 3.10. Compatibility claims require tests.

## Tensor dtype

ARC grid tensors use `torch.int8`. Do not silently upcast to `int32`/`int64` in DSL
operations without an explicit cast and a comment explaining why.

## No Neural Approximations in DSL

All `SpatialDSL` primitives must be pure deterministic tensor ops.
No `nn.Module`, no learned weights, no `requires_grad=True` on grid data.
