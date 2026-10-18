"""Probe accounting and instrumentation. Importable without torch/CUDA."""
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from functools import wraps
import math
import statistics
import time

L1 = ('ViT_Encoder', 'Frame_Attn_Blocks', 'GCA_Blocks', 'Task_Heads', 'Misc_Glue')
L2 = ('Normalize_Images', 'PatchEmbed_PosEnc', 'DINOv2_Blocks', 'DINO_Final_Norm',
      'Special_Token_Concat', 'ViT_Other', 'Frame_Attn_Op', 'Frame_MLP_Norm',
      'GCA_QKV_RoPE', 'GCA_KV_Schedule', 'GCA_FlashInfer_Kernel', 'GCA_Scale_SDPA',
      'GCA_Out_Projection', 'GCA_MLP_Norm', 'GCA_Other_Norm_Residual',
      'Camera_Head', 'Depth_Head')


def frame_groups(count, scale=8):
    if count < scale:
        raise ValueError('Input must contain all scale frames')
    yield 0, scale
    for start in range(scale, count):
        yield start, start + 1


def kv_accounting(frames, scale, window, patches, specials, layers, heads,
                  head_dim, element_size, page_size, physical_pages):
    anchor = min(frames, scale)
    local = min(max(0, frames - scale), window)
    trajectory = max(0, frames - anchor - local)
    counts = {'anchor': anchor * (patches + specials),
              'local': local * (patches + specials),
              'trajectory': trajectory * specials}
    per_token = layers * 2 * heads * head_dim * element_size
    occupied_pages = anchor + local + math.ceil(frames * specials / page_size)
    physical = physical_pages * page_size * per_token
    active = sum(counts.values()) * per_token
    if occupied_pages > physical_pages or active > physical:
        raise ValueError('KV capacity exhausted or invalid accounting')
    result = {f'kv_{key}_tokens_per_layer': value for key, value in counts.items()}
    result.update({f'kv_{key}_bytes': value * per_token for key, value in counts.items()})
    result.update(kv_active_bytes=active, kv_pool_bytes=physical,
                  kv_unused_page_bytes=(physical_pages - occupied_pages) * page_size * per_token,
                  kv_page_slack_bytes=occupied_pages * page_size * per_token - active,
                  kv_special_tokens_per_layer=frames * specials)
    return result


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * p / 100
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def describe(values):
    return {'count': len(values), 'mean': statistics.mean(values) if values else None,
            'std': statistics.stdev(values) if len(values) > 1 else None,
            'median': percentile(values, 50), 'p95': percentile(values, 95),
            'p99': percentile(values, 99)}


class Recorder:
    """Same-stream Events, paired intervals and frame-wide peak preservation.

    No synchronization inside spans. L1 scopes are disjoint; L2 is nested in L1.
    Misc is the complement on the model Event timeline, NOT host synchronization.
    Module peak increments are non-additive high-water marks.
    """
    def __init__(self, torch):
        self.torch = torch
        self.cuda = torch.cuda
        self.enabled = False
        self.trace = False
        self.pool = []
        self.cursor = 0
        self.intervals = []
        self.active_l1 = None

    def event(self):
        if self.cursor == len(self.pool):
            self.pool.append(self.cuda.Event(enable_timing=True))
        event = self.pool[self.cursor]
        self.cursor += 1
        event.record()
        return event

    def collect_peak(self):
        self.frame_peak = max(self.frame_peak, self.cuda.max_memory_allocated())

    def begin(self):
        if self.enabled or self.active_l1:
            raise RuntimeError('Unclosed recorder frame')
        self.cursor = 0
        self.intervals = []
        self.peaks = defaultdict(int)
        self.gap_peak = 0
        self.gap_allocated = self.cuda.memory_allocated()
        self.cuda.reset_peak_memory_stats()
        self.frame_peak = self.gap_allocated
        self.start = self.event()
        self.enabled = True

    @contextmanager
    def span(self, level, name):
        if not self.enabled:
            yield
            return
        if level == 'L1':
            if self.active_l1 is not None:
                raise RuntimeError(f'Overlapping L1: {self.active_l1}, {name}')
            self.active_l1 = name
            self.collect_peak()
            self.gap_peak = max(self.gap_peak, self.cuda.max_memory_allocated() - self.gap_allocated)
            allocated = self.cuda.memory_allocated()
            self.cuda.reset_peak_memory_stats()
        start = self.event()
        host_start = time.perf_counter()
        trace = self.torch.profiler.record_function(f'{level}/{name}') if self.trace else nullcontext()
        try:
            with trace:
                yield
        finally:
            end = self.event()
            self.intervals.append((level, name, start, end, (time.perf_counter() - host_start) * 1000))
            if level == 'L1':
                self.collect_peak()
                self.peaks[name] = max(self.peaks[name], self.cuda.max_memory_allocated() - allocated)
                self.active_l1 = None
                self.gap_allocated = self.cuda.memory_allocated()
                self.cuda.reset_peak_memory_stats()

    def end(self):
        end = self.event()
        self.enabled = False
        if self.active_l1:
            raise RuntimeError('Unclosed L1 scope')
        self.collect_peak()
        self.gap_peak = max(self.gap_peak, self.cuda.max_memory_allocated() - self.gap_allocated)
        return end

    def results(self, end):
        # Caller must synchronize before resolving Events.
        totals = defaultdict(float)
        host = defaultdict(float)
        for level, name, start, stop, host_ms in self.intervals:
            totals[level + '_' + name] += start.elapsed_time(stop)
            host[level + '_' + name] += host_ms
        model_ms = self.start.elapsed_time(end)
        covered = sum(totals['L1_' + key] for key in L1[:-1])
        residual = model_ms - covered
        if residual < -0.05:
            raise RuntimeError(f'L1 intervals overlap: residual={residual} ms')
        totals['L1_Misc_Glue'] = max(0.0, residual)
        for name in L2:
            totals['L2_' + name] += 0.0
        deductions = {
            'ViT_Other': ('ViT_Encoder', ['Normalize_Images', 'PatchEmbed_PosEnc',
                'DINOv2_Blocks', 'DINO_Final_Norm', 'Special_Token_Concat']),
            'Frame_MLP_Norm': ('Frame_Attn_Blocks', ['Frame_Attn_Op']),
            'GCA_Other_Norm_Residual': ('GCA_Blocks', ['GCA_QKV_RoPE', 'GCA_KV_Schedule',
                'GCA_FlashInfer_Kernel', 'GCA_Scale_SDPA', 'GCA_Out_Projection', 'GCA_MLP_Norm'])}
        for name, (parent, children) in deductions.items():
            delta = totals['L1_' + parent] - sum(totals['L2_' + key] for key in children)
            if delta < -0.05:
                raise RuntimeError(f'L2 intervals overlap in {parent}: {delta}')
            totals['L2_' + name] = max(0.0, delta)
            host['L2_' + name] = max(0.0, host['L1_' + parent] - sum(host['L2_' + key] for key in children))
        result = {key + '_ms': value for key, value in totals.items()}
        result.update({key + '_host_ms': value for key, value in host.items()})
        result.update({f'L1_{key}_peak_delta_bytes': self.peaks[key] for key in L1[:-1]})
        result.update(model_cuda_ms=model_ms, memory_peak_allocated=self.frame_peak,
                      L1_Misc_Glue_peak_delta_bytes=self.gap_peak,
                      l1_event_closure_error_ms=model_ms - sum(totals['L1_' + key] for key in L1))
        return result


class Instrumentation:
    """Instance-local wrappers; every original callable restored on exit."""
    def __init__(self, model, recorder):
        self.model = model
        self.recorder = recorder
        self.restore = []
        self.managers = set()

    def wrap(self, obj, method, level, name, child=None):
        original = getattr(obj, method)
        owned = method in vars(obj)
        previous = vars(obj).get(method)
        @wraps(original)
        def wrapped(*args, **kwargs):
            with self.recorder.span(level, name):
                with self.recorder.span('L2', child) if child else nullcontext():
                    return original(*args, **kwargs)
        setattr(obj, method, wrapped)
        self.restore.append((obj, method, owned, previous))

    def attach_manager(self, manager):
        if id(manager) in self.managers:
            return
        self.managers.add(id(manager))
        for method in ('append_frame', 'evict_frames'):
            self.wrap(manager, method, 'L2', 'GCA_KV_Schedule')
        self.wrap(manager.prefill_wrapper, 'plan', 'L2', 'GCA_KV_Schedule')
        self.wrap(manager.prefill_wrapper, 'run', 'L2', 'GCA_FlashInfer_Kernel')

    def __enter__(self):
        from lingbot_map.probe_scope import _recorder
        self.token = _recorder.set(self.recorder)
        agg = self.model.aggregator
        try:
            self.wrap(agg, '_embed_images', 'L1', 'ViT_Encoder')
            dino = agg.patch_embed
            self.wrap(dino, 'prepare_tokens_with_masks', 'L2', 'PatchEmbed_PosEnc')
            self.wrap(dino.norm, 'forward', 'L2', 'DINO_Final_Norm')
            for block in dino.blocks:
                self.wrap(block, 'forward', 'L2', 'DINOv2_Blocks')
            for block in agg.frame_blocks:
                self.wrap(block, 'forward', 'L1', 'Frame_Attn_Blocks')
                self.wrap(block.attn, 'forward', 'L2', 'Frame_Attn_Op')
            for block in agg.global_blocks:
                self.wrap(block, 'forward', 'L1', 'GCA_Blocks')
                self.wrap(block, 'attn_pre', 'L2', 'GCA_QKV_RoPE')
                self.wrap(block, 'ffn_residual', 'L2', 'GCA_MLP_Norm')
                self.wrap(block.attn.proj, 'forward', 'L2', 'GCA_Out_Projection')
            for method, name in (('_predict_camera', 'Camera_Head'), ('_predict_depth', 'Depth_Head')):
                self.wrap(self.model, method, 'L1', 'Task_Heads', child=name)
            # The mandatory reference warmup constructs the manager before attaching.
            if agg.kv_cache_manager is None:
                raise RuntimeError('Warmup must construct the paged cache first')
            self.attach_manager(agg.kv_cache_manager)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        from lingbot_map.probe_scope import _recorder
        for obj, method, owned, previous in reversed(self.restore):
            if owned:
                setattr(obj, method, previous)
            else:
                delattr(obj, method)
        self.restore.clear()
        _recorder.reset(self.token)


def unique_storage_bytes(tensors, seen=None):
    """Count full CUDA storages once, including views sharing a backing allocation."""
    seen = set() if seen is None else seen
    total = 0
    for tensor in tensors:
        if tensor.device.type != 'cuda':
            continue
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total


def tensor_tree(value, torch):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensor_tree(child, torch)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from tensor_tree(child, torch)


def cache_snapshot(model, torch, expected_frames):
    agg = model.aggregator
    manager = agg.kv_cache_manager
    if manager is None or manager.force_fp32:
        raise RuntimeError('Expected active BF16 FlashInfer cache')
    for idx in range(manager.num_blocks):
        stats = manager.get_cache_stats(idx)
        expected = dict(frame_count=expected_frames, scale_pages=min(expected_frames, 8),
                        live_pages=min(max(expected_frames - 8, 0), 64),
                        free_pages=manager.max_patch_pages - min(expected_frames, 8) - min(max(expected_frames - 8, 0), 64),
                        special_tokens=6 * expected_frames)
        if stats != expected:
            raise RuntimeError(f'KV invariant failed in block {idx}: {stats} != {expected}')
    if model.camera_head.frame_idx != expected_frames or agg.total_frames_processed != expected_frames:
        raise RuntimeError('Camera/aggregator frame counters disagree')
    if manager.kv_caches[0].dtype != torch.bfloat16:
        raise RuntimeError('KV cache must remain BF16')
    result = kv_accounting(expected_frames, 8, 64, manager.patches_per_frame, 6,
                           manager.num_blocks, manager.num_heads, manager.head_dim,
                           manager.kv_caches[0].element_size(), manager.page_size,
                           manager.max_num_pages)
    pool_actual = unique_storage_bytes(manager.kv_caches)
    if result['kv_pool_bytes'] != pool_actual:
        raise RuntimeError('Paged allocation does not match capacity accounting')
    result['camera_cache_bytes'] = unique_storage_bytes(tensor_tree(model.camera_head.kv_cache, torch))
    result['flashinfer_workspace_bytes'] = unique_storage_bytes([manager.workspace_buffer])
    return result
