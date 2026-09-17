"""Build `notebooks/submission.ipynb` from the `agent/` package.

The submission entrypoint is WorldModelAgent (the full internal-world loop with
an explicit baseline fallback when no trained simulator checkpoint is bundled).

The notebook follows the exact pattern used by Kaggle's official sample
("ARC3 Sample Submission - Stochastic Goose"):

  Cell 1: install pinned ARC runtime wheels into an isolated target directory.
  Cell 2: reconstruct the `agent` package under /tmp from embedded module bodies.
  Cell 3: if running inside the Kaggle competition rerun, wait for the
          gateway sidecar, copy the framework into /tmp/, register
          MyAgent, and run `python main.py --agent myagent`.
  Cell 4: otherwise (during commit / save-and-run-all), write a dummy
          submission.parquet so Kaggle accepts the commit.
  Cell 5: validate the output schema and reject dummy output during a rerun.

You don't normally need to call this directly — `make submit` runs it for you.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE THIS ONE LINE TO PICK YOUR KAGGLE ACCELERATOR
# Options:
#   "cpu"      — no GPU. Good for the random starter or any non-ML agent.
#   "t4"       — Nvidia T4 ×2 (default; matches Kaggle's sample submission).
#   "p100"     — Nvidia P100 (single big-memory GPU).
#   "rtx6000"  — Nvidia RTX 6000 (g4-standard-48). ARC-AGI-3 exclusive,
#                burns GPU quota faster — use only when you're confident.
# ─────────────────────────────────────────────────────────────────────────────
ACCELERATOR = "t4"

# Internal mapping; don't edit unless Kaggle adds new options.
_ACCELERATORS = {
    "cpu":     {"name": "none",            "gpu": False},
    "t4":      {"name": "nvidiaTeslaT4",   "gpu": True},
    "p100":    {"name": "nvidiaTeslaP100", "gpu": True},
    "rtx6000": {"name": "nvidiaRtx6000",   "gpu": True},
}

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "agent"
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"

# The submission entrypoint is WorldModelAgent: the full internal-world loop with
# an explicit baseline fallback when no trained simulator checkpoint is bundled.
# It imports from the whole agent/ package, so we bundle every runtime module
# (not just my_agent.py). Training-only and dev-only modules are excluded to keep
# the notebook lean and free of torch-training imports on the critical path.
AGENT_ENTRYPOINT_MODULE = "world_model_agent"
AGENT_ENTRYPOINT_CLASS = "WorldModelAgent"

# Runtime agent modules to bundle into the notebook (order irrelevant; Python
# resolves imports lazily). simulator.py is included but only imported by
# world_model_agent when a checkpoint is present, so torch stays optional.
_RUNTIME_AGENT_MODULES = (
    "internal_world.py",
    "memory.py",
    "model.py",
    "device.py",
    "deliberator.py",
    "deliberator_transformers.py",
    "simulator.py",
    "controller.py",
    "my_agent.py",
    "world_model_agent.py",
    "online_replay.py",
)


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _collect_agent_modules() -> dict[str, str]:
    """Read every runtime agent module. Raises if any is missing."""
    modules: dict[str, str] = {}
    for name in _RUNTIME_AGENT_MODULES:
        path = AGENT_DIR / name
        if not path.exists():
            raise SystemExit(f"Could not find required agent module: {path}")
        modules[name] = path.read_text(encoding="utf-8")
    for name in ("__init__.py", "adapter.py", "train_simulator.py"):
        modules["../training/" + name] = (ROOT / "training" / name).read_text(encoding="utf-8")
    return modules


def build() -> dict:
    agent_modules = _collect_agent_modules()

    install_cell = code_cell(
        dedent(
            """\
            import os
            from pathlib import Path
            import shutil
            import subprocess
            import sys

            # Keep ARC's Pillow 12 requirement out of Kaggle's global Python
            # environment, where preinstalled Gradio requires Pillow < 12.
            # The agent subprocess receives this directory through PYTHONPATH;
            # Kaggle's notebook renderer continues using its original packages.
            RUNTIME_DIR = '/tmp/arc3_runtime'
            WHEEL_DIR = ('/kaggle/input/competitions/'
                         'arc-prize-2026-arc-agi-3/arc_agi_3_wheels')
            wheel_dir = Path(WHEEL_DIR)
            required_wheels = {
                'arc-agi 0.9.8': 'arc_agi-0.9.8-*.whl',
                'arcengine 0.9.3': 'arcengine-0.9.3-*.whl',
                'pillow 12.2.0': 'pillow-12.2.0-*.whl',
            }
            missing = [
                label
                for label, pattern in required_wheels.items()
                if not any(wheel_dir.glob(pattern))
            ]
            if missing:
                raise FileNotFoundError(
                    f'Missing required offline wheels in {WHEEL_DIR}: {missing}'
                )

            # A notebook cell can be rerun in the same worker. Rebuild the
            # isolated target so stale package files can never survive.
            runtime_dir = Path(RUNTIME_DIR)
            if runtime_dir.exists():
                shutil.rmtree(runtime_dir)
            runtime_dir.mkdir(parents=True)
            subprocess.run(
                [
                    sys.executable, '-m', 'pip', 'install',
                    '--disable-pip-version-check',
                    '--quiet',
                    '--no-index', '--find-links', WHEEL_DIR,
                    '--target', RUNTIME_DIR, '--no-deps',
                    'arc-agi==0.9.8',
                    'arcengine==0.9.3',
                    'pillow==12.2.0',
                ],
                check=True,
                timeout=300,
            )

            runtime_env = os.environ.copy()
            runtime_env['PYTHONPATH'] = (
                RUNTIME_DIR + os.pathsep + runtime_env.get('PYTHONPATH', '')
            )
            runtime_env['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
            runtime_env['PYTHONDONTWRITEBYTECODE'] = '1'
            runtime_env['PYTHONHASHSEED'] = '0'
            probe = subprocess.run(
                [
                    sys.executable,
                    '-X', 'frozen_modules=off',
                    '-c',
                    ('from importlib.metadata import version; '
                     'import arc_agi, arcengine, PIL, dotenv; '
                     'print("ARC runtime OK:", '
                     'version("arc-agi"), version("arcengine"), '
                     'version("pillow"), version("python-dotenv")); '
                     'print("ARC runtime path:", arc_agi.__file__)'),
                ],
                env=runtime_env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if probe.returncode != 0:
                raise RuntimeError(
                    'ARC runtime probe failed:\\n'
                    + probe.stdout
                    + probe.stderr
                )
            print(probe.stdout, end='')
            """
        )
    )

    # We write the agent package to /tmp/ (not /kaggle/working/) so it does NOT
    # appear as a notebook output. Otherwise the "Submit to Competition" UI would
    # offer source files as candidate submissions alongside submission.parquet.
    #
    # The full internal-world agent (WorldModelAgent) imports the whole `agent`
    # package, so we bundle every runtime module and reconstruct the package on
    # disk. The modules are embedded as a JSON blob; a small loop writes them out
    # plus a generated __init__.py so `import agent.world_model_agent` resolves.
    modules_blob = json.dumps(agent_modules, indent=0)
    write_agent_cell = code_cell(
        "import json, os\n"
        "from pathlib import Path\n"
        "\n"
        "# Reconstruct the `agent` package under /tmp so the framework subprocess\n"
        "# (which gets /tmp/arc3_agent_pkg on PYTHONPATH) can import it.\n"
        "_PKG_ROOT = Path('/tmp/arc3_agent_pkg')\n"
        "_AGENT_DIR = _PKG_ROOT / 'agent'\n"
        "_AGENT_DIR.mkdir(parents=True, exist_ok=True)\n"
        "(_AGENT_DIR / '__init__.py').write_text('', encoding='utf-8')\n"
        "_MODULES = json.loads(r'''" + modules_blob + "''')\n"
        "for _name, _body in _MODULES.items():\n"
        "    (_AGENT_DIR / _name).parent.mkdir(parents=True, exist_ok=True)\n"
        "    (_AGENT_DIR / _name).write_text(_body, encoding='utf-8')\n"
        "print('Wrote agent package:', sorted(p.name for p in _AGENT_DIR.glob('*.py')))\n"
    )

    run_cell_source = dedent(
        """\
        import os
        from pathlib import Path
        import shutil
        import subprocess
        import sys

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            print('RUN_MODE=competition_rerun: starting the scored ARC agent')
            # Wait for the gateway sidecar to be ready.
            subprocess.run(
                [
                    'curl', '--fail', '--retry', '999', '--retry-all-errors',
                    '--retry-delay', '5', '--retry-max-time', '600',
                    'http://gateway:8001/api/games',
                ],
                check=True,
            )

            # Copy the framework into a writable location.
            framework_source = Path(
                '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/'
                'ARC-AGI-3-Agents'
            )
            framework_dir = Path('/tmp/ARC-AGI-3-Agents')
            if framework_dir.exists():
                shutil.rmtree(framework_dir)
            shutil.copytree(framework_source, framework_dir)

            # Drop a thin template that imports WorldModelAgent from the bundled
            # `agent` package (reconstructed under /tmp/arc3_agent_pkg earlier).
            # The template is a framework-visible module named `my_agent` for
            # registry compatibility; the real logic lives in the agent package.
            (framework_dir / 'agents/templates/my_agent.py').write_text(
                'from agent.world_model_agent import WorldModelAgent as MyAgent\\n'
            )

            # Register the agent in the framework's registry. We rewrite
            # __init__.py because the upstream version eagerly imports
            # templates with deps we don't ship (langgraph, smolagents, etc.).
            with open(framework_dir / 'agents/__init__.py', 'w') as f:
                f.write(\"\"\"from typing import Type
        from dotenv import load_dotenv
        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.random_agent import Random
        from .templates.my_agent import MyAgent

        load_dotenv()

        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
            'random': Random,
            'myagent': MyAgent,
        }
        \"\"\")

            # Point the framework at the gateway sidecar.
            with open(framework_dir / '.env', 'w') as f:
                f.write(\"\"\"SCHEME=http
        HOST=gateway
        PORT=8001
        ARC_API_KEY=test-key-123
        ARC_BASE_URL=http://gateway:8001/
        OPERATION_MODE=online
        ENVIRONMENTS_DIR=
        RECORDINGS_DIR=/tmp/arc3_recordings
        \"\"\")

            # Run only the agent process against the isolated ARC package set.
            # The gateway records every action and emits submission.parquet.
            agent_env = os.environ.copy()
            agent_env['PYTHONPATH'] = (
                '/tmp/arc3_runtime'
                + os.pathsep
                + '/tmp/arc3_agent_pkg'
                + os.pathsep
                + agent_env.get('PYTHONPATH', '')
            )
            agent_env['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
            agent_env['MPLBACKEND'] = 'agg'
            agent_env['PYTHONDONTWRITEBYTECODE'] = '1'
            agent_env['PYTHONHASHSEED'] = '0'
            agent_env['PYTHONUNBUFFERED'] = '1'
            # Never score an accidental explorer/stub fallback. Required model
            # paths must be configured to attached offline inputs before release.
            agent_env['ARC3_REQUIRE_LEARNED'] = '1'
            agent_env['HF_HUB_OFFLINE'] = '1'
            agent_env['TRANSFORMERS_OFFLINE'] = '1'
            agent_env['ARC3_DEVICE'] = 'cuda:0'
            # Point the required model paths at the attached Kaggle inputs. These
            # slugs MUST match the dataset/model attached to the notebook (see
            # kernel-metadata.json dataset_sources / model_sources). The offline
            # bundle convention: a Kaggle Dataset named 'arc3-world-model-v0'
            # containing the trained simulator + the Qwen2.5-7B-Instruct weights.
            _WM_INPUT = '/kaggle/input/arc3-world-model-v0'
            agent_env.setdefault('ARC3_SIMULATOR_CHECKPOINT',
                                 _WM_INPUT + '/simulator/simulator.pt')
            agent_env.setdefault('ARC3_DELIBERATOR_PATH',
                                 _WM_INPUT + '/qwen2.5-7b-instruct')
            for required_model_path in ('ARC3_SIMULATOR_CHECKPOINT', 'ARC3_DELIBERATOR_PATH'):
                if not agent_env.get(required_model_path) or not Path(agent_env[required_model_path]).exists():
                    raise RuntimeError(
                        f'Missing offline model input: {required_model_path}='
                        f'{agent_env.get(required_model_path)!r}. Attach the '
                        f'arc3-world-model-v0 dataset to the notebook (see '
                        f'kernel-metadata.json dataset_sources).')
            subprocess.run(
                [
                    sys.executable, '-X', 'frozen_modules=off',
                    'main.py', '--agent', 'myagent',
                ],
                cwd=framework_dir,
                env=agent_env,
                check=True,
                timeout=8 * 3600,
            )
        else:
            print(
                'RUN_MODE=commit_smoke_test: the private gateway is unavailable; '
                'the scored agent run starts only after Submit to Competition.'
            )
        """
    )
    run_cell = code_cell(run_cell_source)

    dummy_submission_cell = code_cell(
        dedent(
            """\
            import os
            if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                # Save-and-run-all (commit) mode: emit a dummy submission so the
                # commit succeeds. The real submission.parquet is produced by the
                # gateway during competition rerun.
                import pandas as pd
                submission = pd.DataFrame(
                    data=[['1_0', '1', True, 1]],
                    columns=['row_id', 'game_id', 'end_of_game', 'score'])
                output_path = '/kaggle/working/submission.parquet'
                submission.to_parquet(output_path, index=False)
                print(
                    'COMMIT_ARTIFACT=dummy: wrote schema-valid, unscored '
                    f'{output_path}; rows={len(submission)}'
                )
                display(submission.head())
            """
        )
    )

    validation_cell = code_cell(
        dedent(
            """\
            import os
            from pathlib import Path
            import pandas as pd

            submission_path = Path('/kaggle/working/submission.parquet')
            if not submission_path.is_file():
                raise FileNotFoundError(
                    f'Expected submission artifact was not created: {submission_path}'
                )

            checked_submission = pd.read_parquet(submission_path)
            expected_columns = ['row_id', 'game_id', 'end_of_game', 'score']
            if list(checked_submission.columns) != expected_columns:
                raise RuntimeError(
                    'Invalid submission columns: '
                    f'{list(checked_submission.columns)} != {expected_columns}'
                )
            if checked_submission.empty:
                raise RuntimeError('Submission artifact contains no rows')
            if checked_submission['row_id'].isna().any():
                raise RuntimeError('Submission contains null row_id values')
            if checked_submission['row_id'].duplicated().any():
                raise RuntimeError('Submission contains duplicate row_id values')

            is_rerun = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))
            if is_rerun and len(checked_submission) <= 1:
                raise RuntimeError(
                    'Competition rerun produced only the one-row commit dummy; '
                    'refusing to report a successful scored run'
                )
            artifact_kind = 'scored' if is_rerun else 'commit_dummy'
            print(
                f'SUBMISSION_VALIDATION=PASS kind={artifact_kind} '
                f'rows={len(checked_submission)} '
                f'games={checked_submission["game_id"].nunique()}'
            )
            """
        )
    )

    if ACCELERATOR not in _ACCELERATORS:
        raise SystemExit(
            f"Unknown ACCELERATOR={ACCELERATOR!r}. Pick one of: "
            f"{sorted(_ACCELERATORS)}"
        )
    accel = _ACCELERATORS[ACCELERATOR]

    notebook = {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "kaggle": {
                "accelerator": accel["name"],
                "isInternetEnabled": False,
                "isGpuEnabled": accel["gpu"],
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                "# ARC Prize 2026 — ARC-AGI-3 Submission\n\n"
                "Entrypoint: `WorldModelAgent` (internal-world loop with explicit "
                "baseline fallback). Built from the `agent/` package via "
                "`scripts/build_notebook.py`. Do not edit cells directly — edit the "
                "source modules and re-run `make submit`."
            ),
            install_cell,
            write_agent_cell,
            run_cell,
            dummy_submission_cell,
            validation_cell,
        ],
    }
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"[build_notebook] Wrote {NOTEBOOK_PATH.relative_to(ROOT)}  "
          f"(accelerator: {ACCELERATOR})")

    # Keep notebooks/kernel-metadata.json in sync so the user never has to
    # edit it just to flip CPU ↔ GPU.
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        wanted = _ACCELERATORS[ACCELERATOR]["gpu"]
        if meta.get("enable_gpu") != wanted:
            meta["enable_gpu"] = wanted
            METADATA_PATH.write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            )
            print(f"[build_notebook] Synced enable_gpu={wanted} in "
                  f"{METADATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
