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

ROOT = Path(__file__).resolve().parent


def prepare(*, attach_competition=False):
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
    source = f"BUNDLE = {base64.b64encode(payload).decode()!r}\nEXPECTED = {digest!r}\nRELEASE = {release!r}\nWHEELS = {wheels!r}\n" + r'''
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
    code = """
import json, pathlib, torch
from release_policy import model_preflight, MODEL_ID, RELEASE_ID, MODEL_MIN_GPU_MEMORY_GIB
from hyper_arc.contender.offline_model import OfflineTransformersTransport
models = [p.parent for p in pathlib.Path('/kaggle/input').rglob('config.json')
          if 'gemma-4-e4b-it' in str(p).lower()]
if len(models) != 1:
    raise RuntimeError('Expected exactly one attached model')
transport = OfflineTransformersTransport(models[0], device='cuda',
    minimum_gpu_memory_gib=MODEL_MIN_GPU_MEMORY_GIB)
report = model_preflight(transport, MODEL_ID, seconds=90)
report.update({'release_id': RELEASE_ID, 'model': MODEL_ID,
               'device': torch.cuda.get_device_name(0), 'scope': 'synthetic-live-model-only',
               'submission_created': False})
pathlib.Path('/kaggle/working/qualification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
"""
    environment = dict(os.environ)
    environment.update(PYTHONPATH=str(deps) + os.pathsep + str(root),
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    with open('/kaggle/working/model_runtime.log', 'w', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, '-u', '-c', code], cwd=root, env=environment,
                                stdout=log, stderr=subprocess.STDOUT, timeout=420)
    print(pathlib.Path('/kaggle/working/model_runtime.log').read_text(encoding='utf-8'), flush=True)
    result.check_returncode()
except Exception as exc:
    report_path.write_text(json.dumps({'passed': False, 'release_id': RELEASE['id'],
        'source_tree_sha256': RELEASE['source_tree_sha256'], 'scope': 'synthetic-live-model-only',
        'error': str(exc), 'traceback': traceback.format_exc(), 'submission_created': False}, indent=2))
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
    print(f"Prepared synthetic-only private qualification at {output}; bundle={digest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attach-competition', action='store_true',
                        help='Attach ARC-2 for hardware eligibility only; no competition tasks are read.')
    prepare(attach_competition=parser.parse_args().attach_competition)
