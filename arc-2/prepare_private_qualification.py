"""Prepare a private, synthetic-only GPU qualification notebook; never submit.

Embeds the exact active small code bundle, avoiding updates to existing Kaggle
datasets. Weights and hash-verified wheels remain the existing attachments.
"""
import ast
import argparse
import base64
import hashlib
import json
from pathlib import Path
from release_policy import MODEL_ATTACHMENT
from hyper_arc.contender.data_protocol import load_verified_split

ROOT = Path(__file__).resolve().parent


def prepare(*, attach_competition=False, benchmark=False, benchmark_output_tokens=None):
    release = json.loads((ROOT / "release/current/release.json").read_text())
    bundle = ROOT / "release/current" / release["bundle"]
    payload = bundle.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if bundle.name != f"hyper_arc_solver_bundle.{digest}.zip":
        raise ValueError("Active bundle digest mismatch")
    tree = ast.parse((ROOT / "kaggle_launcher.py").read_text())
    wheels = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "EXPECTED_WHEELS" for t in node.targets))
    if benchmark:
        development, _ = load_verified_split(
            ROOT / 'data/arc-agi-2/arc-agi_training_challenges.json',
            ROOT / 'config/arc2_split_v1.json')
        task_id = next(iter(development))
        challenge = development[task_id]
        answers = json.loads((ROOT / 'data/arc-agi-2/arc-agi_training_solutions.json').read_text())[task_id]
    else:
        task_id, challenge, answers = None, None, None
    source = (f"BUNDLE = {base64.b64encode(payload).decode()!r}\nEXPECTED = {digest!r}\n"
              f"RELEASE = {release!r}\nWHEELS = {wheels!r}\n"
              f"BENCH_TASK_ID = {task_id!r}\nBENCH_CHALLENGE = {challenge!r}\n"
              f"BENCH_TARGETS = {answers!r}\nBENCH_OUTPUT_TOKENS = {benchmark_output_tokens!r}\n") + r'''
import base64, hashlib, io, json, os, pathlib, subprocess, sys, tempfile, traceback, zipfile
import torch
report_path = pathlib.Path('/kaggle/working/qualification.json')
try:
    if not torch.cuda.is_available():
        raise RuntimeError('GPU allocation missing; qualification cannot use CPU fallback')
    print('GPUS:', [{'name': torch.cuda.get_device_name(i),
                    'total_bytes': torch.cuda.get_device_properties(i).total_memory}
                   for i in range(torch.cuda.device_count())], flush=True)
    root = pathlib.Path(tempfile.mkdtemp(prefix='hyper_arc_qualification_'))
    payload = base64.b64decode(BUNDLE)
    assert hashlib.sha256(payload).hexdigest() == EXPECTED
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        manifest = json.loads(archive.read('bundle_manifest.json'))
        assert manifest['source_tree_sha256'] == RELEASE['source_tree_sha256']
        assert set(archive.namelist()) == set(manifest['files']) | {'bundle_manifest.json'}
        for name, meta in manifest['files'].items():
            path = root / name
            if not path.resolve().is_relative_to(root.resolve()):
                raise RuntimeError('Unsafe bundle member')
            content = archive.read(name)
            assert len(content) == meta['size'] and hashlib.sha256(content).hexdigest() == meta['sha256']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    wheels = []
    for name, digest in WHEELS.items():
        matches = list(pathlib.Path('/kaggle/input').rglob(name))
        if len(matches) != 1 or hashlib.sha256(matches[0].read_bytes()).hexdigest() != digest:
            raise RuntimeError('Missing or mismatched wheel: ' + name)
        wheels.append(str(matches[0]))
    deps = root / 'deps'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                    '--target', str(deps), *wheels], check=True, timeout=120)
    code = (f"RELEASE = {RELEASE!r}\n"
            f"BENCH_TASK_ID = {BENCH_TASK_ID!r}\n"
            f"BENCH_CHALLENGE = {BENCH_CHALLENGE!r}\n"
            f"BENCH_TARGETS = {BENCH_TARGETS!r}\n"
            f"BENCH_OUTPUT_TOKENS = {BENCH_OUTPUT_TOKENS!r}\n") + """
import json, pathlib, torch
from release_policy import model_preflight, MODEL_ID, RELEASE_ID, MODEL_MIN_GPU_MEMORY_GIB, MODEL_MIN_GPUS
from hyper_arc.contender.offline_model import OfflineTransformersTransport
from hyper_arc.contender.agentic_reasoner import AgenticReasoner
import time
models = [p.parent for p in pathlib.Path('/kaggle/input').rglob('config.json')
          if 'gemma-4-e4b-it' in str(p).lower()]
if len(models) != 1:
    raise RuntimeError('Expected exactly one attached model')
transport = OfflineTransformersTransport(models[0], device='cuda',
    minimum_gpu_memory_gib=MODEL_MIN_GPU_MEMORY_GIB, minimum_gpu_count=MODEL_MIN_GPUS)
report = {'passed': False, 'release_id': RELEASE_ID, 'model': MODEL_ID,
          'device': torch.cuda.get_device_name(0), 'submission_created': False,
          'scope': 'frozen-builder-task' if BENCH_TASK_ID is not None else 'synthetic-live-model-only',
          'stage': 'model_loaded'}
pathlib.Path('/kaggle/working/qualification.json').write_text(json.dumps(report, indent=2))
if BENCH_TASK_ID is not None:
    # This stage isolates task reasoning; synthetic preflight passed in v3 with
    # the same immutable bundle, attachment and L4 hardware.
    report['stage'] = 'builder_task_started'
    report['builder_benchmark'] = {'task_id': BENCH_TASK_ID,
                                   'selection': 'first frozen development task',
                                   'labels_passed_to_model': False}
    pathlib.Path('/kaggle/working/qualification.json').write_text(json.dumps(report, indent=2))
    result = AgenticReasoner(transport, model=MODEL_ID,
        rounds=RELEASE['rounds'], candidates_per_round=RELEASE['candidates'],
        max_output_tokens=BENCH_OUTPUT_TOKENS or RELEASE['output_tokens']).solve(
            BENCH_TASK_ID, BENCH_CHALLENGE, deadline=time.monotonic() + RELEASE['task_seconds'])
    expected = [tuple(tuple(row) for row in grid) for grid in BENCH_TARGETS]
    per_output_exact = [index < len(result.test_predictions)
                        and target in result.test_predictions[index]
                        for index, target in enumerate(expected)]
    report['builder_benchmark'] = {
        'task_id': BENCH_TASK_ID, 'rounds': result.rounds_executed,
        'generated': result.candidates_generated, 'executed': result.candidates_executed,
        'per_output_exact_pass_at_2': per_output_exact,
        'full_task_exact_pass_at_2': bool(per_output_exact) and all(per_output_exact),
        'failures': list(result.failures), 'selection': 'first frozen development task',
        'labels_passed_to_model': False,
        'output_token_cap': BENCH_OUTPUT_TOKENS or RELEASE['output_tokens']}
else:
    report.update(model_preflight(transport, MODEL_ID, seconds=90))
report.update({'passed': (result.candidates_executed > 0 if BENCH_TASK_ID is not None else True),
               'stage': 'completed'})
pathlib.Path('/kaggle/working/qualification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
"""
    environment = dict(os.environ)
    environment.update(PYTHONPATH=str(deps) + os.pathsep + str(root),
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    with open('/kaggle/working/model_runtime.log', 'w', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, '-u', '-c', code], cwd=root, env=environment,
                                stdout=log, stderr=subprocess.STDOUT,
                                timeout=300 if BENCH_TASK_ID is not None else 420)
    print(pathlib.Path('/kaggle/working/model_runtime.log').read_text(encoding='utf-8'), flush=True)
    result.check_returncode()
except Exception as exc:
    partial = json.loads(report_path.read_text()) if report_path.is_file() else {}
    detail = (f'timed out after {exc.timeout}s' if isinstance(exc, subprocess.TimeoutExpired)
              else f'exit status {exc.returncode}' if isinstance(exc, subprocess.CalledProcessError)
              else str(exc))
    partial.update({'passed': False, 'release_id': RELEASE['id'],
        'source_tree_sha256': RELEASE['source_tree_sha256'],
        'error': f'{type(exc).__name__}: {detail}',
        'traceback': traceback.format_exc(limit=5), 'submission_created': False})
    report_path.write_text(json.dumps(partial, indent=2))
    raise
print('Qualification only: no submission.json was created.', flush=True)
'''
    compile(source, "private-qualification", "exec")
    output = ROOT / "release/private_qualification"
    output.mkdir(parents=True, exist_ok=True)
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": [
        {"cell_type": "code", "id": "qualification", "metadata": {}, "outputs": [],
         "execution_count": None, "source": source.splitlines(keepends=True)}]}
    (output / "qualification.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8", newline="\n")
    metadata = {"id": "jthomaslockhart/hyper-arc-private-qualification", "title": "Hyper-ARC private qualification",
        "code_file": "qualification.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": False,
        "dataset_sources": ["jthomaslockhart/hyper-arc-gemma4-runtime"],
        "model_sources": [MODEL_ATTACHMENT],
        "competition_sources": ["arc-prize-2026-arc-agi-2"] if attach_competition else [],
        "machine_shape": "NvidiaL4"}
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8", newline="\n")
    print(f"Prepared private qualification at {output}; bundle={digest}; benchmark={benchmark}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attach-competition', action='store_true',
                        help='Attach ARC-2 for hardware eligibility only; no competition tasks are read.')
    parser.add_argument('--benchmark', action='store_true',
                        help='Also score one frozen builder task after synthetic preflight.')
    parser.add_argument('--benchmark-output-tokens', type=int,
                        help='Diagnostic output cap override; does not change release settings.')
    args = parser.parse_args()
    prepare(attach_competition=args.attach_competition, benchmark=args.benchmark,
            benchmark_output_tokens=args.benchmark_output_tokens)
