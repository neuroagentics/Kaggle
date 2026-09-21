from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
import zipfile

# Build-time identity is injected by build_release.py, never edited in notebooks.
RELEASE = None

INPUT_ROOT = Path('/kaggle/input')
WORK_ROOT = Path('/kaggle/working')
RUNTIME_ROOT = Path('/tmp/hyper_arc_runtime')
OUTPUT_PATH = WORK_ROOT / 'submission.json'
CANDIDATE_PATH = WORK_ROOT / 'submission.candidate.json'
RUN_MANIFEST_PATH = WORK_ROOT / 'run_manifest.json'
STATUS_PATH = WORK_ROOT / 'deployment_status.json'
SOLVER_LOG_PATH = WORK_ROOT / 'hyper_arc_solver.log'
CHECKPOINT_PATH = WORK_ROOT / 'checkpoint.pt'
DEPS_ROOT = Path('/tmp/hyper_arc_model_deps')
EXPECTED_WHEELS = {
    'transformers-5.7.0-py3-none-any.whl': '869660cd8fc92badc041f5551bf755a42f4b9558c93341bf3fa3eeed7065079c',
    'huggingface_hub-1.5.0-py3-none-any.whl': 'c9c0b3ab95a777fc91666111f3b3ede71c0cdced3614c553a64e98920585c4ee',
    'tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl': '369cc9fc8cc10cb24143873a0d95438bb8ee257bb80c71989e3ee290e8d72c67',
    'safetensors-0.7.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl': 'dac7252938f0696ddea46f5e855dd3138444e82236e3be475f54929f0c510d48',
}

def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, separators=(',', ':')), encoding='utf-8')
    temporary.replace(path)

def prepare_model_dependencies():
    wheels = []
    for name, expected_digest in EXPECTED_WHEELS.items():
        matches = list(INPUT_ROOT.rglob(name))
        if len(matches) != 1:
            raise RuntimeError(f'Expected one offline wheel {name}, found {len(matches)}')
        actual_digest = hashlib.sha256(matches[0].read_bytes()).hexdigest()
        if actual_digest != expected_digest:
            raise RuntimeError(f'Offline wheel digest mismatch: {name}')
        wheels.append(matches[0])
    if DEPS_ROOT.exists():
        shutil.rmtree(DEPS_ROOT)
    DEPS_ROOT.mkdir(parents=True)
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
         '--target', str(DEPS_ROOT), *[str(path) for path in wheels]],
        check=True, capture_output=True, text=True, timeout=180,
    )
    sys.path.insert(0, str(DEPS_ROOT))

def locate_challenges():
    candidates = sorted(
        path for path in INPUT_ROOT.rglob('arc-agi_test_challenges.json')
        if 'arc-prize-2026-arc-agi-2' in str(path).lower()
    )
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one ARC Prize 2026 challenge file, found {len(candidates)}')
    return candidates[0]

def locate_model():
    candidates = sorted(
        path.parent for path in INPUT_ROOT.rglob('config.json')
        if 'gemma-4-e4b-it' in str(path.parent).lower()
    )
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one attached Gemma 4 E4B IT model, found {len(candidates)}')
    return candidates[0]

def prepare_runtime():
    print('=== 2. LOCATING AND VERIFYING IMMUTABLE SOLVER BUNDLE ===', flush=True)
    bundles = sorted(INPUT_ROOT.rglob('hyper_arc_solver_bundle.*.zip'))
    expanded_manifests = sorted(
        path for path in INPUT_ROOT.rglob('bundle_manifest.json')
        if 'arc-solver-code' in str(path).lower()
    )
    if len(bundles) + len(expanded_manifests) != 1:
        raise RuntimeError(
            f'Expected one archived or expanded Hyper-ARC bundle; '
            f'found archives={len(bundles)} manifests={len(expanded_manifests)}'
        )
    if RUNTIME_ROOT.exists():
        shutil.rmtree(RUNTIME_ROOT)
    RUNTIME_ROOT.mkdir(parents=True)
    if bundles:
        bundle_path = bundles[0]
        expected_digest = bundle_path.name.removeprefix('hyper_arc_solver_bundle.').removesuffix('.zip')
        actual_digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
        if len(expected_digest) != 64 or actual_digest != expected_digest:
            raise RuntimeError(f'Bundle digest mismatch: expected={expected_digest} actual={actual_digest}')
        with zipfile.ZipFile(bundle_path) as archive:
            names = set(archive.namelist())
            if 'bundle_manifest.json' not in names:
                raise RuntimeError('Bundle manifest is missing')
            manifest = json.loads(archive.read('bundle_manifest.json'))
            declared = set(manifest.get('files', {}))
            if names != declared | {'bundle_manifest.json'}:
                raise RuntimeError('Archive contents do not exactly match the manifest')
            for name, metadata in manifest['files'].items():
                member = Path(name)
                if member.is_absolute() or '..' in member.parts:
                    raise RuntimeError(f'Unsafe archive member: {name}')
                payload = archive.read(name)
                if len(payload) != metadata['size'] or hashlib.sha256(payload).hexdigest() != metadata['sha256']:
                    raise RuntimeError(f'Bundle member verification failed: {name}')
            archive.extractall(RUNTIME_ROOT)
    else:
        source_root = expanded_manifests[0].parent
        manifest = json.loads(expanded_manifests[0].read_text(encoding='utf-8'))
        declared = set(manifest.get('files', {}))
        actual_files = {
            path.relative_to(source_root).as_posix()
            for path in source_root.rglob('*') if path.is_file()
        }
        if actual_files != declared | {'bundle_manifest.json'}:
            raise RuntimeError('Expanded dataset contents do not exactly match the manifest')
        for name, metadata in manifest['files'].items():
            member = Path(name)
            if member.is_absolute() or '..' in member.parts:
                raise RuntimeError(f'Unsafe expanded member: {name}')
            source = source_root / member
            payload = source.read_bytes()
            if len(payload) != metadata['size'] or hashlib.sha256(payload).hexdigest() != metadata['sha256']:
                raise RuntimeError(f'Expanded member verification failed: {name}')
            destination = RUNTIME_ROOT / member
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    if manifest.get('schema_version') != 1 or manifest.get('project') != 'Hyper-ARC':
        raise RuntimeError('Unsupported or foreign bundle manifest')
    if RELEASE is None or manifest.get('release_id') != RELEASE['id']:
        raise RuntimeError('Notebook/bundle release identity mismatch')
    if manifest.get('source_tree_sha256') != RELEASE['source_tree_sha256']:
        raise RuntimeError('Notebook/bundle source tree mismatch')
    print(f"Verified and prepared {len(manifest['files'])} active files", flush=True)
    return manifest

def run_solver():
    print('=== 1. VERIFYING COMPETITION DATA AND OFFLINE MODEL ===', flush=True)
    challenge_path = locate_challenges()
    challenges = json.loads(challenge_path.read_text(encoding='utf-8'))
    if not isinstance(challenges, dict) or not challenges:
        raise RuntimeError('Competition challenge file is empty or malformed')
    model_path = locate_model()
    manifest = prepare_runtime()
    prepare_model_dependencies()
    print('=== 3. CHECKING KAGGLE RUNTIME AND ACTIVE CHANNELS ===', flush=True)
    import numpy as np
    import torch
    import transformers
    if not torch.cuda.is_available():
        raise RuntimeError('Release candidate requires a Kaggle GPU; CPU fallback is forbidden')
    print(json.dumps({
        'python': sys.version.split()[0], 'numpy': np.__version__,
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
    }, sort_keys=True), flush=True)
    sentinel = subprocess.run(
        [sys.executable, str(RUNTIME_ROOT / 'bundle_self_test.py')],
        cwd=RUNTIME_ROOT, check=True, capture_output=True, text=True, timeout=120,
    )
    sentinel_lines = [line for line in sentinel.stdout.splitlines() if line.strip()]
    print(sentinel_lines[-1] if sentinel_lines else '{"status":"PASS"}', flush=True)

    print('=== 4. RUNNING BOUNDED SOLVER ===', flush=True)
    command = [
        sys.executable, str(RUNTIME_ROOT / 'main.py'),
        '--challenge-file', str(challenge_path), '--output', str(CANDIDATE_PATH),
        '--run-manifest', str(RUN_MANIFEST_PATH), '--checkpoint', str(CHECKPOINT_PATH),
        '--procedural-memory', str(RUNTIME_ROOT / 'hyper_arc' / 'procedural_memory_v1.json'),
        '--agentic-memory', str(RUNTIME_ROOT / 'hyper_arc' / 'agentic_memory_builder_v2.json'),
        '--model-path', str(model_path), '--agentic-model-name', RELEASE['model'],
        '--enable-world-model', '--enable-procedural-memory',
        '--competition-run', '--enable-agentic-ai', '--enable-agentic-memory', '--agentic-device', 'cuda',
        '--agentic-rounds', str(RELEASE['rounds']), '--agentic-candidates', str(RELEASE['candidates']),
        '--agentic-max-output-tokens', str(RELEASE['output_tokens']),
        '--task-timeout', str(RELEASE['task_seconds']), '--global-timeout', str(RELEASE['run_seconds']),
        '--no-resume', '--enable-failure-memory',
        '--failure-memory', str(RUNTIME_ROOT / 'hyper_arc' / 'failure_memory_builder_v2.json'),
    ]
    solver_environment = dict(os.environ)
    solver_environment['PYTHONPATH'] = str(DEPS_ROOT) + os.pathsep + str(RUNTIME_ROOT)
    with SOLVER_LOG_PATH.open('w', encoding='utf-8') as solver_log:
        completed = subprocess.run(
            command, cwd=RUNTIME_ROOT, stdout=solver_log,
            stderr=subprocess.STDOUT, timeout=RELEASE['run_seconds'] + 180, env=solver_environment,
        )
    if completed.returncode != 0:
        raise RuntimeError(f'Solver exited with code {completed.returncode}')

    print('=== 5. POST-FLIGHT AUDIT AND ATOMIC PROMOTION ===', flush=True)
    sys.path.insert(0, str(RUNTIME_ROOT))
    from main import validate_submission
    candidate = json.loads(CANDIDATE_PATH.read_text(encoding='utf-8'))
    validate_submission(candidate, challenges)
    run_manifest = json.loads(RUN_MANIFEST_PATH.read_text(encoding='utf-8'))
    candidate_digest = hashlib.sha256(CANDIDATE_PATH.read_bytes()).hexdigest()
    if run_manifest['submission']['sha256'] != candidate_digest:
        raise RuntimeError('Run manifest does not match candidate submission')
    from release_policy import qualify_run
    qualify_run(run_manifest)
    CANDIDATE_PATH.replace(OUTPUT_PATH)
    atomic_json(STATUS_PATH, {
        'schema_version': 1, 'status': 'RUNTIME_QUALIFIED', 'tasks': len(candidate),
        'release_id': RELEASE['id'],
        'submission_sha256': candidate_digest,
        'bundle_source_tree_sha256': manifest['source_tree_sha256'],
    })
    print(f'RUNTIME QUALIFIED: {len(candidate)} task entries; hidden correctness is unknown', flush=True)
    print(json.dumps(run_manifest, indent=2, sort_keys=True), flush=True)

try:
    run_solver()
except Exception as exc:
    for unsafe_artifact in (OUTPUT_PATH, CANDIDATE_PATH):
        if unsafe_artifact.exists():
            unsafe_artifact.unlink()
    failure = {
        'schema_version': 1, 'status': 'FAILED_NO_SUBMISSION',
        'error_type': type(exc).__name__, 'error': str(exc)[:500],
        'traceback': traceback.format_exc(limit=20),
    }
    atomic_json(STATUS_PATH, failure)
    print(f'FAILED CLOSED: no submission after {type(exc).__name__}: {exc}', flush=True)
    if SOLVER_LOG_PATH.is_file():
        tail = SOLVER_LOG_PATH.read_text(encoding='utf-8', errors='replace').splitlines()[-40:]
        print('=== SOLVER LOG TAIL ===', flush=True)
        print('\n'.join(tail), flush=True)

if not OUTPUT_PATH.is_file():
    raise RuntimeError('No submission artifact: release candidate did not qualify')
print(f'FINAL ARTIFACT: {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)', flush=True)
