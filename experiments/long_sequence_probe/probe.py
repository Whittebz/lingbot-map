#!/usr/bin/env python3
"""Long3D probe: CPU manifest preparation, isolated GPU runs, artifact validation.

Run --help without torch. GPU stages must be launched through the shared runner.
Frame indices are zero based; initial [0,8) is ONE observation, never eight samples.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shlex
import socket
import subprocess
import sys
import time
import traceback

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))
from metrics import (L1, L2, Recorder, Instrumentation, cache_snapshot, describe,
                     frame_groups, tensor_tree, unique_storage_bytes)

FIXED = {
    'schema_version': 1, 'dataset_snapshot': '8023267a9e0b472dcb563599db010ff76aab4fd9',
    'sequence': 'Lecture hall', 'total_frames': 9545, 'short_frames': 320,
    'smoke_frames': 80, 'warmup_frames': 20, 'image_size': 518, 'patch_size': 14,
    'input_height': 392, 'input_width': 518, 'raw_width': 800, 'raw_height': 600,
    'scale_frames': 8, 'window': 64, 'keyframe_interval': 1, 'max_frame_num': 9545,
    'dtype': 'bfloat16', 'use_sdpa': False, 'compile': False,
    'camera_num_iterations': 4, 'enable_3d_rope': True, 'enable_camera_3d_rope': False,
    'seed': 0, 'checkpoint_basename': 'lingbot-map.pt', 'model_cpu_offload': False,
    'postprocess': 'raw_pose_depth_conf_to_cpu_npy', 'unproject': False,
    'merge': False, 'voxel_downsample': False, 'fsync_predictions': False,
    'profiler_starts': [100, 500, 1000, 3000, 6000, 9000],
    'profiler_wait': 2, 'profiler_warmup': 2, 'profiler_active': 6,
    'telemetry_every': 100, 'misc_fraction_limit': 0.02,
    'profiler_slowdown_limit': 2.0,
    'filename_sha256': '6f3ca8fc20530781aaeafa1e0ff02abfbe8a48a968067563ba8bcc34182bb155',
}
PREDICTIONS = ('pose_enc', 'depth', 'depth_conf')


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def git_info(path):
    path = Path(path).resolve()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()
    if Path(git('rev-parse', '--show-toplevel')).resolve() != path:
        raise ValueError(f'Not a repository root: {path}')
    dirty = git('status', '--porcelain', '--untracked-files=all')
    if dirty:
        raise ValueError(f'Requires clean fixed checkout: {path}\n{dirty}')
    return dict(path=str(path), commit=git('rev-parse', 'HEAD'),
                branch=git('rev-parse', '--abbrev-ref', 'HEAD'),
                remote=git('remote', 'get-url', 'origin'), dirty=False)


def get_versions(args):
    return {'model': git_info(MODEL_ROOT), 'research': git_info(args.research_repo),
            'workspace': git_info(args.workspace_repo)}


def version_identity(versions):
    return {name: value['commit'] for name, value in versions.items()}


def config_load(path):
    config = read_json(path)
    if config != FIXED:
        differences = [key for key in config.keys() | FIXED.keys() if config.get(key) != FIXED.get(key)]
        raise ValueError(f'Unsupported probe configuration fields/values: {differences}')
    return config


def protect_output(path, versions):
    path = Path(path).resolve()
    for repo in versions.values():
        if path.is_relative_to(Path(repo['path'])):
            raise ValueError('Output must live outside all execution checkouts')
    return path


def list_images(directory):
    files = sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'} and p.is_file())
    if len(files) != FIXED['total_frames']:
        raise ValueError(f'Expected 9545 direct image files, found {len(files)} in {directory}')
    return files


def filename_hash(files):
    # Canonical recipe: UTF-8 basename per line, sorted lexically, final LF.
    digest = hashlib.sha256(('\n'.join(p.name for p in files) + '\n').encode()).hexdigest()
    if digest != FIXED['filename_sha256']:
        raise ValueError('Sorted filename manifest differs from the canonical Lecture hall list')
    return digest, True


def prepare(args):
    """CPU-only: hash checkpoint and every image, decode all images, record versions."""
    from PIL import Image, ImageOps
    config = config_load(args.config)
    versions = get_versions(args)
    output = protect_output(args.output, versions)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'preparing', 'created_at': utcnow(), 'versions': versions,
                'config': config, 'config_path': str(Path(args.config).resolve()),
                'config_sha256': sha256(args.config), 'command': shlex.join(sys.argv),
                'host': socket.gethostname()}
    write_json(output / 'prepare_status.json', manifest)
    try:
        checkpoint = Path(args.checkpoint).resolve(strict=True)
        if checkpoint.name != config['checkpoint_basename']:
            raise ValueError('This configuration uses the standard lingbot-map.pt checkpoint')
        manifest['checkpoint'] = {'path': str(checkpoint), 'sha256': sha256(checkpoint),
                                  'size': checkpoint.stat().st_size}
        image_dir = Path(args.images_dir).resolve(strict=True)
        files = list_images(image_dir)
        name_hash, trailing_lf = filename_hash(files)
        reference_dir = Path(args.reference_images_dir).resolve(strict=True)
        reference_files = list_images(reference_dir)
        if [p.name for p in reference_files] != [p.name for p in files]:
            raise ValueError('Copied dataset filename list differs from historical source')
        entries = []
        for idx, path in enumerate(files):
            with Image.open(path) as raw:
                image = ImageOps.exif_transpose(raw)
                image.load()
                if image.size != (800, 600):
                    raise ValueError(f'Unexpected image size: {path}: {image.size}')
            digest = sha256(path)
            if digest != sha256(reference_files[idx]):
                raise ValueError(f'Copied image differs from historical source: {path}')
            entries.append({'name': path.name, 'sha256': digest, 'size': path.stat().st_size})
            if (idx + 1) % 1000 == 0:
                print(f'Verified {idx + 1}/{len(files)} images', flush=True)
        manifest.update(status='prepared', image_dir=str(image_dir), frames=entries,
                        filename_sha256=name_hash, filename_manifest_trailing_lf=trailing_lf,
                        reference_image_dir=str(reference_dir),
                        data_verification='all images decoded at 800x600; content matched historical source; canonical basename+LF hash matches')
        write_json(output / 'manifest.json', manifest)
        write_json(output / 'prepare_status.json', {'status': 'passed', 'manifest_sha256': sha256(output / 'manifest.json'), 'finished_at': utcnow()})
        print(f'PREPARE_OK {output / "manifest.json"}', flush=True)
    except BaseException as exc:
        write_json(output / 'prepare_status.json', {'status': 'failed', 'error': str(exc), 'finished_at': utcnow()})
        raise


def validate_manifest(manifest, versions, config_path, verify_content=True):
    if manifest['status'] != 'prepared' or manifest['config'] != FIXED:
        raise ValueError('Invalid prepared manifest/configuration')
    if version_identity(manifest['versions']) != version_identity(versions):
        raise ValueError('Checkout commits changed since preparation')
    if manifest['config_sha256'] != sha256(config_path):
        raise ValueError('Config changed since preparation')
    checkpoint = Path(manifest['checkpoint']['path'])
    if verify_content and sha256(checkpoint) != manifest['checkpoint']['sha256']:
        raise ValueError('Checkpoint changed since preparation')
    files = list_images(manifest['image_dir'])
    filename_hash(files)
    if [p.name for p in files] != [e['name'] for e in manifest['frames']]:
        raise ValueError('Input frame order changed')
    if verify_content:
        for path, entry in zip(files, manifest['frames']):
            if path.stat().st_size != entry['size'] or sha256(path) != entry['sha256']:
                raise ValueError(f'Input changed: {path}')
    return files


def run_identity(metadata):
    return (metadata['manifest_sha256'], metadata['config_sha256'],
            tuple(sorted(version_identity(metadata['versions']).items())))


def validate_run(directory, expected_stage=None, deep=True):
    directory = Path(directory)
    meta = read_json(directory / 'run.json')
    summary = read_json(directory / 'summary.json')
    if meta['status'] != 'completed' or summary['validation'] != 'passed':
        raise ValueError(f'Run did not pass: {directory}')
    if expected_stage and meta['stage'] != expected_stage:
        raise ValueError(f'Expected {expected_stage}: {directory}')
    expected_count = {'smoke': 80, 'short': 320, 'long': 9545}[meta['stage']]
    if meta['expected_frames'] != expected_count or summary['frames'] != expected_count:
        raise ValueError('Frame count mismatch')
    if not meta.get('equivalence_passed'):
        raise ValueError('Instrumented/reference comparison missing')
    if summary['csv_sha256'] != sha256(directory / 'frames.csv'):
        raise ValueError('Metrics CSV hash mismatch')
    with open(directory / 'frames.csv', newline='') as stream:
        rows = list(csv.DictReader(stream))
    groups = list(frame_groups(expected_count))
    if [(int(r['frame_start']), int(r['frame_end'])) for r in rows] != groups:
        raise ValueError('Missing, duplicated or reordered observations')
    for row in rows:
        for key in ('model_wall_ms', 'model_cuda_ms', 'e2e_ms'):
            if not math.isfinite(float(row[key])) or float(row[key]) <= 0:
                raise ValueError(f'Invalid timing {key}')
        if row['finite_output'] != 'True':
            raise ValueError('Non-finite or unverified prediction')
        if int(row['memory_peak_allocated']) < int(row['memory_allocated']):
            raise ValueError('Frame peak smaller than end allocation')
    if summary['prediction_index_sha256'] != sha256(directory / 'predictions.jsonl'):
        raise ValueError('Prediction index hash mismatch')
    entries = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text().splitlines()]
    if [e['frame'] for e in entries] != list(range(expected_count)):
        raise ValueError('Prediction frame index mismatch')
    for entry in entries:
        if set(entry['files']) != set(PREDICTIONS):
            raise ValueError('Missing prediction fields')
        for name, record in entry['files'].items():
            path = directory / record['path']
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError('Artifact path escapes run directory')
            if not path.is_file() or path.stat().st_size != record['bytes']:
                raise ValueError(f'Missing/truncated prediction: {path}')
            if deep and sha256(path) != record['sha256']:
                raise ValueError(f'Prediction hash mismatch: {path}')
    for trace in summary['traces']:
        if not (directory / trace['path']).is_file():
            raise ValueError('Profiler trace missing')
    return meta, summary


def check_prerequisites(args, identity):
    required = {'smoke': (None, 0), 'short': ('smoke', 1), 'long': ('short', 3)}
    stage, count = required[args.stage]
    parents = [Path(p).resolve() for p in args.prerequisite]
    if len(parents) != count or len(set(parents)) != count:
        raise ValueError(f'{args.stage} requires {count} distinct {stage} run directories')
    run_ids = set()
    for parent in parents:
        meta, _ = validate_run(parent, stage)
        if run_identity(meta) != identity:
            raise ValueError(f'Incompatible prerequisite versions/input/config: {parent}')
        if meta['run_id'] in run_ids:
            raise ValueError('Copies of the same run cannot count as independent repetitions')
        run_ids.add(meta['run_id'])
    return [str(p) for p in parents]


class Telemetry:
    def __init__(self):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        physical = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not physical.isdecimal():
            raise RuntimeError('Expected one physical GPU index from the shared GPU runner')
        self.physical = int(physical)
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.physical)
        self.uuid = pynvml.nvmlDeviceGetUUID(self.handle)
        if isinstance(self.uuid, bytes):
            self.uuid = self.uuid.decode()

    def sample(self):
        n, h = self.nvml, self.handle
        return dict(nvml_used_bytes=n.nvmlDeviceGetMemoryInfo(h).used,
                    gpu_temperature_c=n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU),
                    gpu_sm_clock_mhz=n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM),
                    gpu_memory_clock_mhz=n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_MEM))

    def close(self):
        self.nvml.nvmlShutdown()


def assert_input(images, torch, count, device):
    if tuple(images.shape) != (count, 3, 392, 518) or images.dtype != torch.float32 or images.device.type != device:
        raise RuntimeError(f'Unexpected loader tensor: {images.shape}, {images.dtype}, {images.device}')
    if not bool(torch.isfinite(images).all()) or float(images.min()) < 0 or float(images.max()) > 1:
        raise RuntimeError('Invalid input range')


def load_images(files, start, stop, torch):
    from lingbot_map.utils.load_fn import load_and_preprocess_images
    images = load_and_preprocess_images([str(p) for p in files[start:stop]], mode='crop', image_size=518, patch_size=14)
    assert_input(images, torch, stop - start, 'cpu')
    return images.to('cuda').unsqueeze(0)


def forward(model, images, count, torch):
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        return model(images, num_frame_for_scale=8, num_frame_per_block=count, causal_inference=True)


def output_cpu(prediction, torch, count):
    expected = {'pose_enc': (1, count, 9), 'depth': (1, count, 392, 518, 1),
                'depth_conf': (1, count, 392, 518)}
    result = {}
    for key in PREDICTIONS:
        value = prediction[key]
        if tuple(value.shape) != expected[key]:
            raise RuntimeError(f'Invalid output shape {key}: {value.shape}')
        value = value.detach().cpu()
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f'Non-finite output: {key}')
        result[key] = value
    return result


def reset_model(model):
    model.clean_kv_cache()
    model._set_skip_append(False)
    if model.camera_head.frame_idx != 0 or model.aggregator.total_frames_processed != 0:
        raise RuntimeError('Cache reset did not reset frame counters')
    manager = model.aggregator.kv_cache_manager
    if manager is not None and any(manager.frame_count):
        raise RuntimeError('Cache reset left frames in paged memory')


def warmup_reference(model, files, torch):
    reset_model(model)
    reference = []
    for start, stop in frame_groups(20):
        images = load_images(files, start, stop, torch)
        prediction = forward(model, images, stop - start, torch)
        reference.append(output_cpu(prediction, torch, stop - start))
        del prediction, images
    torch.cuda.synchronize()
    cache_snapshot(model, torch, 20)
    reset_model(model)
    return reference


def check_equivalence(model, files, reference, recorder, torch):
    for idx, (start, stop) in enumerate(frame_groups(20)):
        images = load_images(files, start, stop, torch)
        torch.cuda.synchronize()
        recorder.begin()
        prediction = forward(model, images, stop - start, torch)
        end = recorder.end()
        torch.cuda.synchronize()
        recorder.results(end)
        actual = output_cpu(prediction, torch, stop - start)
        for key in PREDICTIONS:
            torch.testing.assert_close(actual[key], reference[idx][key], rtol=1e-4, atol=1e-5)
        del actual, prediction, images
    cache_snapshot(model, torch, 20)
    reset_model(model)


def resident_snapshot(model, prediction, images, torch, cache):
    seen = set()
    groups = {'vit_weights_bytes': model.aggregator.patch_embed.parameters(),
              'frame_weights_bytes': model.aggregator.frame_blocks.parameters(),
              'gca_weights_bytes': model.aggregator.global_blocks.parameters(),
              'heads_weights_bytes': list(model.camera_head.parameters()) + list(model.depth_head.parameters())}
    result = {name: unique_storage_bytes(values, seen) for name, values in groups.items()}
    result['other_weights_bytes'] = unique_storage_bytes(model.parameters(), seen)
    result['registered_buffers_bytes'] = unique_storage_bytes(model.buffers(), seen)
    manager = model.aggregator.kv_cache_manager
    result['kv_pool_storage_bytes'] = unique_storage_bytes(manager.kv_caches, seen)
    result['camera_cache_storage_bytes'] = unique_storage_bytes(tensor_tree(model.camera_head.kv_cache, torch), seen)
    result['flashinfer_workspace_storage_bytes'] = unique_storage_bytes([manager.workspace_buffer], seen)
    # Include unregistered position caches/manager tensors without traversing Modules or wrapper internals.
    positions = [model.aggregator._cached_pos3d, manager._qo_indptr]
    for module in model.modules():
        for key, value in vars(module).items():
            if key in ('_parameters', '_buffers', '_modules'):
                continue
            positions.extend(tensor_tree(value, torch))
    getter = model.aggregator.position_getter
    if getter is not None:
        positions.extend(tensor_tree(getter.position_cache, torch))
    result['other_cached_tensor_bytes'] = unique_storage_bytes(positions, seen)
    result['current_input_output_bytes'] = unique_storage_bytes([images, *tensor_tree(prediction, torch)], seen)
    allocated = torch.cuda.memory_allocated()
    result['shared_unattributed_bytes'] = allocated - sum(result.values())
    if result['shared_unattributed_bytes'] < 0:
        raise RuntimeError('Resident storage accounting exceeds allocated memory')
    return result


class ProfilerWindows:
    def __init__(self, torch, output, count):
        self.torch, self.output, self.count = torch, output, count
        self.profiler = None
        self.start = None
        self.traces = []
        self.skipped = [s for s in FIXED['profiler_starts'] if s + 10 > count]

    def before(self, frame):
        if frame in FIXED['profiler_starts'] and frame + 10 <= self.count:
            self.start = frame
            self.profiler = self.torch.profiler.profile(
                activities=[self.torch.profiler.ProfilerActivity.CPU, self.torch.profiler.ProfilerActivity.CUDA],
                schedule=self.torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
                on_trace_ready=self.save, record_shapes=False, profile_memory=False, with_stack=False)
            self.profiler.__enter__()
        return self.profiler is not None

    def save(self, profiler):
        name = f'traces/frames_{self.start}_{self.start + 10}.json'
        (self.output / 'traces').mkdir(exist_ok=True)
        profiler.export_chrome_trace(str(self.output / name))
        self.traces.append({'start': self.start, 'end': self.start + 10, 'path': name})

    def after(self, frame):
        if self.profiler is not None:
            self.profiler.step()
            if frame + 1 == self.start + 10:
                self.close()

    def close(self):
        if self.profiler is not None:
            profiler, self.profiler = self.profiler, None
            profiler.__exit__(None, None, None)


def summarize(rows, elapsed_s, traces, skipped):
    clean = [r for r in rows if r['phase'] == 'stream' and not r['profiler_window']]
    metrics = ['model_wall_ms', 'model_cuda_ms', 'e2e_ms'] + ['L1_' + n + '_ms' for n in L1] + ['L2_' + n + '_ms' for n in L2]
    frames = sum(r['frame_count'] for r in rows)
    windows = {}
    for start, stop in ((500, 600), (2000, 2100), (5000, 5100)):
        subset = [r for r in rows if start <= r['frame_start'] < stop]
        windows[f'{start}:{stop}'] = {key: describe([r[key] for r in subset]) for key in metrics}
        windows[f'{start}:{stop}']['profiler_observations'] = sum(r['profiler_window'] for r in subset)
    return {'validation': 'pending', 'frames': frames, 'observations': len(rows),
            'sequence_wall_seconds': elapsed_s, 'sequence_fps': frames / elapsed_s,
            'fps_model_aggregate': frames * 1000 / sum(r['model_wall_ms'] for r in rows),
            'fps_e2e_aggregate': frames * 1000 / sum(r['e2e_ms'] for r in rows),
            'steady_without_profiler': {key: describe([r[key] for r in clean]) for key in metrics},
            'initialization': rows[0], 'windows': windows,
            'milestones': {str(n): next((r for r in rows if r['frame_end'] == n), None) for n in (100, 320, 1000, 3000, 9545)},
            'traces': traces, 'skipped_profiler_starts': skipped,
            'peak_allocated_bytes': max(r['memory_peak_allocated'] for r in rows)}


def gpu_run(args):
    import uuid
    config = config_load(args.config)
    versions = get_versions(args)
    output = protect_output(args.output, versions)
    manifest_path = Path(args.manifest).resolve()
    manifest = read_json(manifest_path)
    metadata = {'schema_version': 1, 'run_id': str(uuid.uuid4()), 'status': 'starting',
                'stage': args.stage, 'start_time': utcnow(), 'command': shlex.join(sys.argv),
                'host': socket.gethostname(), 'cwd': str(Path.cwd()),
                'output_path': str(output), 'log_file': os.environ.get('PROBE_LOG_FILE'),
                'versions': versions, 'config': config, 'config_file': str(Path(args.config).resolve()),
                'config_sha256': sha256(args.config), 'manifest_path': str(manifest_path),
                'manifest_sha256': sha256(manifest_path), 'checkpoint': manifest['checkpoint'],
                'expected_frames': {'smoke': 80, 'short': 320, 'long': 9545}[args.stage],
                'core_idea_version': 'v1.0 (observation only)',
                'preprocessing': 'lingbot_map.utils.load_fn.load_and_preprocess_images(mode=crop,image_size=518,patch_size=14)',
                'model_call': 'GCTStream.forward(causal_inference=True), initial 8 jointly then one at a time',
                'postprocess': 'raw pose_enc/depth/depth_conf CPU .npy writes, no 3D reconstruction; writes buffered, no fsync',
                'timing': 'host perf_counter seconds; same-stream CUDA interval ms includes host-starvation/observer effects; per-initialization block, then per-frame',
                'misc_definition': 'model CUDA interval minus disjoint L1 intervals; includes glue and observer gaps, not a pure kernel estimate',
                'runner': {'path': str(Path(args.workspace_repo).resolve() / 'utils/gpu_runner.sh'),
                           'sha256': sha256(Path(args.workspace_repo) / 'utils/gpu_runner.sh'),
                           'idle_rule': '0 MiB and no pmon process; runner selection evidence in launch log'},
                'environment': {key: os.environ.get(key) for key in ('CONDA_PREFIX', 'LIBRARY_PATH', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                'equivalence_passed': False}
    metadata['prerequisites'] = check_prerequisites(args, run_identity(metadata))
    files = validate_manifest(manifest, versions, args.config)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'run.json', metadata)
    write_json(output / 'config.json', config)
    telemetry = windows = recorder = None
    completed = 0
    try:
        import numpy as np
        import torch
        from lingbot_map.models.gct_stream import GCTStream
        import lingbot_map.models.gct_stream as stream_module
        metadata['loaded_model_module'] = str(Path(stream_module.__file__).resolve())
        if not Path(stream_module.__file__).resolve().is_relative_to(MODEL_ROOT):
            raise RuntimeError('Imported old editable checkout instead of fixed model')
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Shared GPU runner must expose exactly one available CUDA device')
        telemetry = Telemetry()
        metadata['nvml_before_model'] = telemetry.sample()
        driver = telemetry.nvml.nvmlSystemGetDriverVersion()
        metadata['nvidia_driver'] = driver.decode() if isinstance(driver, bytes) else driver
        metadata['gpu_selection_observed_at'] = utcnow()
        metadata['cpu_count'] = os.cpu_count()
        metadata['cpu_model'] = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), platform.processor())
        metadata['ram_memtotal'] = Path('/proc/meminfo').read_text().splitlines()[0]
        properties = torch.cuda.get_device_properties(0)
        metadata['gpu'] = {'physical_index': telemetry.physical, 'uuid': telemetry.uuid,
                           'name': properties.name, 'total_memory': properties.total_memory,
                           'capability': list(torch.cuda.get_device_capability(0))}
        metadata['software'] = {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                                'platform': platform.platform(),
                                'packages': {name: importlib.metadata.version(name) for name in ('flashinfer-python', 'numpy', 'Pillow')}}
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        model = GCTStream(img_size=518, patch_size=14, enable_3d_rope=True,
                          enable_camera_3d_rope=False, max_frame_num=9545,
                          kv_cache_sliding_window=64, kv_cache_scale_frames=8,
                          kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
                          use_sdpa=False, camera_num_iterations=4,
                          enable_camera=True, enable_depth=True, enable_point=False,
                          enable_local_point=False, use_gradient_checkpoint=False)
        checkpoint = torch.load(manifest['checkpoint']['path'], map_location='cpu', weights_only=True)
        state = checkpoint.get('model', checkpoint)
        incompatible = model.load_state_dict(state, strict=False)
        metadata['checkpoint_keys'] = {'missing': incompatible.missing_keys, 'unexpected': incompatible.unexpected_keys}
        if incompatible.missing_keys or any(not k.startswith(('point_head.', 'local_point_head.')) for k in incompatible.unexpected_keys):
            raise RuntimeError(f'Checkpoint does not cover configured model: {metadata["checkpoint_keys"]}')
        del state, checkpoint
        model = model.to('cuda').eval()
        metadata['weights_dtype'] = str(next(model.parameters()).dtype)
        metadata['heads_dtype'] = 'FP32 (source autocast disabled)'
        first = load_images(files, 0, 8, torch)
        assert_input(first[0], torch, 8, 'cuda')
        metadata['first_input_actual'] = {'shape': list(first.shape), 'dtype': str(first.dtype),
            'device': str(first.device), 'min': first.min().item(), 'max': first.max().item(), 'patch_tokens': 1036}
        del first
        def capture_encoder_input(module, inputs):
            if 'first_encoder_input_actual' not in metadata:
                value = inputs[0]
                if tuple(value.shape) != (8, 3, 392, 518):
                    raise RuntimeError(f'Unexpected encoder input: {value.shape}')
                metadata['first_encoder_input_actual'] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                    'device': str(value.device), 'min': value.min().item(), 'max': value.max().item(),
                    'normalization': 'ImageNet mean/std in aggregator, once'}
        capture = model.aggregator.patch_embed.register_forward_pre_hook(capture_encoder_input)
        print('20-frame reference warmup (unmeasured)', flush=True)
        try:
            reference = warmup_reference(model, files, torch)
        finally:
            capture.remove()
        recorder = Recorder(torch)
        metadata['status'] = 'validating_instrumentation'
        write_json(output / 'run.json', metadata)
        with Instrumentation(model, recorder):
            print('20-frame instrumented equivalence replay (unmeasured)', flush=True)
            check_equivalence(model, files, reference, recorder, torch)
            del reference
            metadata['equivalence_passed'] = True
            metadata['status'] = 'running'
            metadata['input'] = {'shape_initial': [1, 8, 3, 392, 518], 'shape_stream': [1, 1, 3, 392, 518],
                                 'loader_dtype': 'torch.float32', 'patch_tokens': 1036, 'special_tokens': 6}
            write_json(output / 'run.json', metadata)
            count = metadata['expected_frames']
            windows = ProfilerWindows(torch, output, count)
            (output / 'predictions').mkdir()
            rows = []
            started = time.perf_counter()
            with open(output / 'frames.csv', 'x', newline='') as csv_file, open(output / 'predictions.jsonl', 'x') as predictions_file:
                writer = None
                for start, stop in frame_groups(count):
                    loop_start = time.perf_counter()
                    profiled = windows.before(start)
                    recorder.trace = profiled
                    t0 = time.perf_counter()
                    images = load_images(files, start, stop, torch)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    recorder.begin()
                    prediction = forward(model, images, stop - start, torch)
                    event_end = recorder.end()
                    before_sync = time.perf_counter()
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    # Read allocator counters before any validation/postprocessing allocations.
                    allocated = torch.cuda.memory_allocated()
                    reserved = torch.cuda.memory_reserved()
                    measurement_start = time.perf_counter()
                    row = recorder.results(event_end)
                    row['memory_allocated'] = allocated
                    row['memory_reserved'] = reserved
                    row.update(cache_snapshot(model, torch, stop))
                    row.update(resident_snapshot(model, prediction, images, torch, row))
                    sample = telemetry.sample() if start == 0 or stop % 100 == 0 or stop == count else {}
                    row.update({key: sample.get(key, '') for key in ('nvml_used_bytes', 'gpu_temperature_c', 'gpu_sm_clock_mhz', 'gpu_memory_clock_mhz')})
                    measurement_ms = (time.perf_counter() - measurement_start) * 1000
                    cpu = output_cpu(prediction, torch, stop - start)
                    for offset in range(stop - start):
                        entry = {'frame': start + offset, 'files': {}}
                        for key in PREDICTIONS:
                            relative = f'predictions/{start + offset:06d}_{key}.npy'
                            path = output / relative
                            np.save(path, cpu[key][0, offset].numpy(), allow_pickle=False)
                            entry['files'][key] = {'path': relative, 'bytes': path.stat().st_size, 'sha256': sha256(path)}
                        predictions_file.write(json.dumps(entry) + '\n')
                    predictions_file.flush()
                    del cpu, prediction, images
                    t3 = time.perf_counter()
                    model_wall = (t2 - t1) * 1000
                    row.update(frame_start=start, frame_end=stop, frame_count=stop - start,
                               phase='init' if start == 0 else 'stream', profiler_window=profiled,
                               t_0=t0, t_1=t1, t_2=t2, t_3=t3,
                               preprocess_ms=(t1 - t0) * 1000, model_wall_ms=model_wall,
                               e2e_ms=(t3 - t0) * 1000, postprocess_ms=(t3 - t2) * 1000,
                               observation_after_t2_ms=measurement_ms,
                               sync_wait_ms=(t2 - before_sync) * 1000,
                               host_minus_cuda_ms=model_wall - row['model_cuda_ms'],
                               fps_model=(stop - start) * 1000 / model_wall,
                               fps_e2e=(stop - start) / (t3 - t0), finite_output=True,
                               input_patch_tokens=1036, anomalous_misc=row['L1_Misc_Glue_ms'] / model_wall >= 0.02)
                    for name in L1:
                        row['L1_' + name + '_fraction_model_wall'] = row['L1_' + name + '_ms'] / model_wall
                    for name in L2:
                        row['L2_' + name + '_fraction_model_wall'] = row['L2_' + name + '_ms'] / model_wall
                    # Stable CSV schema: host spans absent in init/stream are explicit zeros.
                    for level, names in (('L1', L1[:-1]), ('L2', L2)):
                        for name in names:
                            row.setdefault(level + '_' + name + '_host_ms', 0.0)
                    windows.after(start)
                    row['loop_before_csv_ms'] = (time.perf_counter() - loop_start) * 1000
                    if writer is None:
                        writer = csv.DictWriter(csv_file, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    csv_file.flush()
                    rows.append(row)
                    completed = stop
                    if stop % 100 == 0 or start == 0 or stop == count:
                        print(f'{stop}/{count}: model={model_wall:.2f} ms, allocated={allocated / 2**30:.2f} GiB', flush=True)
                    # Compare profiler windows to the preceding 20 frames; descriptive, not causal.
                    if start in [s + 9 for s in FIXED['profiler_starts']]:
                        window_start = start - 9
                        base = [r['loop_before_csv_ms'] for r in rows if window_start - 20 <= r['frame_start'] < window_start]
                        active = [r['loop_before_csv_ms'] for r in rows if window_start <= r['frame_start'] <= start]
                        ratio = describe(active)['median'] / describe(base)['median']
                        if ratio > config['profiler_slowdown_limit']:
                            raise RuntimeError(f'Profiler window {window_start} slowdown {ratio:.2f} exceeds fixed limit 2.0')
            elapsed = time.perf_counter() - started
            summary = summarize(rows, elapsed, windows.traces, windows.skipped)
            summary['misc_boundary_review_required'] = any(r['anomalous_misc'] for r in rows)
            summary['validation'] = 'needs_boundary_review' if summary['misc_boundary_review_required'] else 'passed'
            summary['csv_sha256'] = sha256(output / 'frames.csv')
            summary['prediction_index_sha256'] = sha256(output / 'predictions.jsonl')
            summary['profiler_perturbation'] = []
            for trace in windows.traces:
                base = [r['loop_before_csv_ms'] for r in rows if trace['start'] - 20 <= r['frame_start'] < trace['start']]
                active = [r['loop_before_csv_ms'] for r in rows if trace['start'] <= r['frame_start'] < trace['end']]
                summary['profiler_perturbation'].append({'start': trace['start'], 'median_ratio_to_previous_20': describe(active)['median'] / describe(base)['median']})
            write_json(output / 'summary.json', summary)
        metadata.update(status='completed', end_time=utcnow(), processed_frames=completed, exit_status=0)
        write_json(output / 'run.json', metadata)
        if summary['validation'] != 'passed':
            raise RuntimeError('Misc_Glue >= 2%: run retained but requires boundary review; subsequent stages blocked')
        validate_run(output)
        print(f'PROBE_OK {output}', flush=True)
    except BaseException as exc:
        if recorder:
            recorder.enabled = False
        metadata.update(status='failed', end_time=utcnow(), processed_frames=completed,
                        exit_status=1, error=str(exc), traceback=traceback.format_exc())
        write_json(output / 'run.json', metadata)
        raise
    finally:
        if windows:
            windows.close()
        if telemetry:
            telemetry.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    for stage in ('prepare', 'smoke', 'short', 'long'):
        p = sub.add_parser(stage)
        p.add_argument('--config', type=Path, required=True)
        p.add_argument('--research-repo', type=Path, required=True)
        p.add_argument('--workspace-repo', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        if stage == 'prepare':
            p.add_argument('--images-dir', type=Path, required=True)
            p.add_argument('--checkpoint', type=Path, required=True)
            p.add_argument('--reference-images-dir', type=Path, required=True)
        else:
            p.add_argument('--manifest', type=Path, required=True)
            p.add_argument('--prerequisite', type=Path, action='append', default=[])
    p = sub.add_parser('validate')
    p.add_argument('run', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args)
    elif args.stage == 'validate':
        validate_run(args.run)
        print('VALIDATION_OK')
    else:
        gpu_run(args)


if __name__ == '__main__':
    main()
