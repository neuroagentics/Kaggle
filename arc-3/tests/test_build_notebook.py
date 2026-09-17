from __future__ import annotations

import json
from pathlib import Path

from scripts import build_notebook


ROOT = Path(__file__).resolve().parents[1]


def _code_sources(notebook: dict) -> list[str]:
    return [
        cell["source"]
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def test_arc_packages_are_process_scoped_and_pinned() -> None:
    sources = _code_sources(build_notebook.build())
    install = sources[0]

    assert "--target', RUNTIME_DIR" in install
    assert "--no-deps" in install
    assert "arc-agi==0.9.8" in install
    assert "arcengine==0.9.3" in install
    assert "pillow==12.2.0" in install
    assert "RUNTIME_DIR = '/tmp/arc3_runtime'" in install
    assert "shutil.rmtree(runtime_dir)" in install
    assert "Missing required offline wheels" in install
    assert "'--quiet'" in install
    assert "'--upgrade'" not in install
    assert "timeout=300" in install
    assert "/kaggle/working/arc3_runtime" not in "\n".join(sources)
    assert "!pip install" not in install


def test_rerun_and_commit_modes_are_unambiguous() -> None:
    sources = _code_sources(build_notebook.build())
    combined = "\n".join(sources)

    assert "RUN_MODE=competition_rerun" in combined
    assert "RUN_MODE=commit_smoke_test" in combined
    assert "COMMIT_ARTIFACT=dummy" in combined
    assert "PYTHONPATH" in combined
    assert "check=True" in combined
    assert "SUBMISSION_VALIDATION=PASS" in combined
    assert "Competition rerun produced only the one-row commit dummy" in combined


def test_runtime_and_framework_do_not_pollute_submission_outputs() -> None:
    sources = _code_sources(build_notebook.build())
    combined = "\n".join(sources)

    assert "framework_dir = Path('/tmp/ARC-AGI-3-Agents')" in combined
    assert "RECORDINGS_DIR=/tmp/arc3_recordings" in combined
    assert "/kaggle/working/ARC-AGI-3-Agents" not in combined
    assert "/kaggle/working/server_recording" not in combined
    # The agent package is reconstructed under /tmp (never /kaggle/working) so
    # its source files are not offered as submission candidates.
    assert "_PKG_ROOT = Path('/tmp/arc3_agent_pkg')" in combined
    assert "/kaggle/working/arc3_agent_pkg" not in combined
    # The framework template imports the bundled WorldModelAgent entrypoint.
    assert "from agent.world_model_agent import WorldModelAgent as MyAgent" in combined


def test_all_runtime_agent_modules_are_bundled() -> None:
    """The whole agent package must be embedded so WorldModelAgent imports resolve."""
    sources = _code_sources(build_notebook.build())
    combined = "\n".join(sources)
    for module in build_notebook._RUNTIME_AGENT_MODULES:
        assert module in combined, f"agent module not bundled: {module}"


def test_child_python_processes_disable_frozen_module_debugger_noise() -> None:
    sources = _code_sources(build_notebook.build())
    combined = "\n".join(sources)

    assert combined.count("'frozen_modules=off'") == 2
    assert combined.count("PYDEVD_DISABLE_FILE_VALIDATION") == 2
    assert combined.count("PYTHONDONTWRITEBYTECODE") == 2


def test_generated_python_cells_compile_without_syntax_warnings() -> None:
    sources = _code_sources(build_notebook.build())

    for index, source in enumerate(sources):
        if source.startswith("%%writefile"):
            continue
        compile(source, f"generated-cell-{index}", "exec")


def test_generated_notebook_matches_builder() -> None:
    generated = json.loads(
        (ROOT / "notebooks" / "submission.ipynb").read_text()
    )

    assert generated == build_notebook.build()
