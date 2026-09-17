# Upstream provenance

The initial competition plumbing was imported on 2026-08-30 from:

- `arcprize/ARC-AGI-3-Kaggle-Starter`
- upstream commit `eeb1535404f321d280a8f9194bbc1d7aca5f05fc`

The compatible framework was reviewed from:

- `arcprize/ARC-AGI-3-Agents`
- upstream commit `4743e7d0aaae0ded0d98a89a7e282e63564cd58b`

The generated `notebooks/submission.ipynb` is produced from
`agent/my_agent.py` by `scripts/build_notebook.py`. Edit the agent source and
regenerate the notebook; do not hand-edit the generated notebook.

The upstream starter currently requires a project-local Kaggle access token for
its command-line publishing workflow. This repository must never track that
token. Browser upload remains an alternative for the first plumbing run.
