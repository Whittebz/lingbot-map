"""CPU-only contracts. No torch import, no weights/data download, no GPU usage."""
import ast
import copy
import csv
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from metrics import Recorder, Instrumentation, frame_groups, kv_accounting, percentile, unique_storage_bytes
from probe import (FIXED, MODEL_ROOT, check_prerequisites, config_load, run_identity,
                   sha256, validate_run, write_json)


class Event:
    def __init__(self, cuda):
        self.cuda = cuda

    def record(self):
        self.time = self.cuda.time
        self.cuda.time += 0.01

    def elapsed_time(self, other):
        return other.time - self.time


class CUDA:
    def __init__(self):
        self.time = 0.0
        self.allocated = self.peak = 100

    def Event(self, enable_timing):
        return Event(self)

    def memory_allocated(self):
        return self.allocated

    def max_memory_allocated(self):
        return self.peak

    def reset_peak_memory_stats(self):
        self.peak = self.allocated

    def allocate(self, size):
        self.allocated = size
        self.peak = max(self.peak, size)
        self.time += 1


class AccountingTests(unittest.TestCase):
    def test_initialization_is_one_observation(self):
        groups = list(frame_groups(320))
        self.assertEqual(groups[0], (0, 8))
        self.assertEqual(len(groups), 313)
        self.assertEqual(sum(b - a for a, b in groups), 320)
        self.assertEqual(groups[-1], (319, 320))
        with self.assertRaises(ValueError):
            list(frame_groups(7))

    def test_kv_boundary_and_physical_allocation(self):
        physical = []
        for frames, trajectory in ((8, 0), (72, 0), (73, 6), (9545, 56838)):
            m = kv_accounting(frames, 8, 64, 1036, 6, 24, 16, 64, 2, 1036, 160)
            self.assertEqual(m['kv_trajectory_tokens_per_layer'], trajectory)
            self.assertEqual(m['kv_special_tokens_per_layer'], frames * 6)
            self.assertEqual(m['kv_pool_bytes'], m['kv_active_bytes'] + m['kv_unused_page_bytes'] + m['kv_page_slack_bytes'])
            physical.append(m['kv_pool_bytes'])
        self.assertEqual(len(set(physical)), 1)
        with self.assertRaises(ValueError):
            kv_accounting(9545, 8, 64, 1036, 6, 24, 16, 64, 2, 1036, 80)

    def test_frame_peak_survives_module_resets(self):
        cuda = CUDA()
        r = Recorder(SimpleNamespace(cuda=cuda))
        r.begin()
        cuda.allocate(180)  # gap before first L1
        with r.span('L1', 'ViT_Encoder'):
            with r.span('L2', 'Normalize_Images'):
                cuda.allocate(800)
                cuda.allocate(200)
        with r.span('L1', 'Task_Heads'):
            cuda.allocate(400)
            cuda.allocate(150)
        cuda.allocate(220)  # gap after last L1
        result = r.results(r.end())
        self.assertEqual(result['memory_peak_allocated'], 800)
        self.assertEqual(result['L1_ViT_Encoder_peak_delta_bytes'], 620)
        self.assertEqual(result['L1_Task_Heads_peak_delta_bytes'], 200)
        self.assertEqual(result['L1_Misc_Glue_peak_delta_bytes'], 80)
        self.assertAlmostEqual(result['l1_event_closure_error_ms'], 0)
        # Reused Event pool starts a clean frame, not a lifetime maximum.
        r.begin()
        with r.span('L1', 'Task_Heads'):
            cuda.allocate(300)
        self.assertEqual(r.results(r.end())['memory_peak_allocated'], 300)

    def test_l1_overlap_rejected_and_exception_unwinds(self):
        r = Recorder(SimpleNamespace(cuda=CUDA()))
        r.begin()
        with self.assertRaisesRegex(RuntimeError, 'Overlapping L1'):
            with r.span('L1', 'ViT_Encoder'):
                with r.span('L1', 'Task_Heads'):
                    pass
        self.assertIsNone(r.active_l1)
        r.end()

    def test_disabled_recorder_never_calls_cuda(self):
        r = Recorder(SimpleNamespace(cuda=None))
        with r.span('L1', 'ViT_Encoder'):
            pass

    def test_storage_views_count_once(self):
        def tensor(ptr, size, device='cuda'):
            return SimpleNamespace(device=SimpleNamespace(type=device),
                untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: ptr, nbytes=lambda: size))
        a, b, c = tensor(1, 100), tensor(1, 100), tensor(2, 200)
        # Same device representation must identify the same physical device.
        b.device = a.device
        self.assertEqual(unique_storage_bytes([a, b, c]), 300)
        self.assertEqual(unique_storage_bytes([tensor(3, 500, 'cpu')]), 0)

    def test_percentiles(self):
        self.assertIsNone(percentile([], 95))
        self.assertEqual(percentile([10, 20], 50), 15)


class SourceTests(unittest.TestCase):
    def test_scope_additions_preserve_model_ast(self):
        class StripProbe(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return None if node.module == 'lingbot_map.probe_scope' else node

            def visit_With(self, node):
                self.generic_visit(node)
                if len(node.items) == 1 and isinstance(node.items[0].context_expr, ast.Call):
                    func = node.items[0].context_expr.func
                    if isinstance(func, ast.Name) and func.id == 'probe_scope':
                        return node.body
                return node
        for path in ('lingbot_map/aggregator/base.py', 'lingbot_map/layers/attention.py'):
            # Fixed predecessor, not HEAD: still useful after committing instrumentation.
            original = subprocess.check_output(['git', '-C', str(MODEL_ROOT), 'show',
                'cab151991e16afb10e560a7aeea395f5efc27cd4:' + path], text=True)
            current = (MODEL_ROOT / path).read_text()
            self.assertEqual(ast.dump(StripProbe().visit(ast.parse(current))), ast.dump(ast.parse(original)))

    def test_wrapper_keeps_arguments_return_and_exception(self):
        class Obj:
            def call(self, x, factor=2):
                if x < 0:
                    raise ValueError('original error')
                return x * factor
        obj = Obj()
        r = Recorder(SimpleNamespace(cuda=CUDA()))
        inst = Instrumentation(None, r)
        inst.wrap(obj, 'call', 'L2', 'example')
        self.assertEqual(obj.call(4, factor=3), 12)
        with self.assertRaisesRegex(ValueError, 'original error'):
            obj.call(-1)
        self.assertEqual(len(inst.restore), 1)


    def test_full_instrumentation_stream_path_and_restoration(self):
        from lingbot_map.probe_scope import scope, _recorder
        cuda = CUDA()
        def op(*args, **kwargs):
            cuda.time += 1
            return 7
        class Op:
            def forward(self, *args, **kwargs):
                return op(*args, **kwargs)
        class DINO:
            def __init__(self):
                self.norm, self.blocks = Op(), [Op()]
            def prepare_tokens_with_masks(self, value):
                return op(value)
        manager = SimpleNamespace(append_frame=op, evict_frames=op,
                                  prefill_wrapper=SimpleNamespace(plan=op, run=op))
        class Global:
            def __init__(self):
                self.attn = SimpleNamespace(proj=Op())
            def attn_pre(self, value):
                return op(value)
            def ffn_residual(self, value):
                return op(value)
            def forward(self, value):
                self.attn_pre(value)
                manager.append_frame(value)
                manager.evict_frames(value)
                manager.prefill_wrapper.plan(value)
                manager.prefill_wrapper.run(value)
                self.attn.proj.forward(value)
                return self.ffn_residual(value)
        class Frame:
            def __init__(self):
                self.attn = Op()
            def forward(self, value):
                return self.attn.forward(value)
        class Agg:
            def __init__(self):
                self.patch_embed = DINO()
                self.frame_blocks, self.global_blocks = [Frame()], [Global()]
                self.kv_cache_manager = manager
            def _embed_images(self, value):
                with scope('Normalize_Images'):
                    op(value)
                self.patch_embed.prepare_tokens_with_masks(value)
                self.patch_embed.blocks[0].forward(value)
                self.patch_embed.norm.forward(value)
                with scope('Special_Token_Concat'):
                    return op(value)
        class Model:
            def __init__(self):
                self.aggregator = Agg()
            def _predict_camera(self, value):
                return op(value)
            def _predict_depth(self, value):
                return op(value)
        model = Model()
        recorder = Recorder(SimpleNamespace(cuda=cuda))
        with Instrumentation(model, recorder):
            recorder.begin()
            model.aggregator._embed_images(0)
            model.aggregator.frame_blocks[0].forward(0)
            model.aggregator.global_blocks[0].forward(0)
            model._predict_camera(0)
            model._predict_depth(0)
            result = recorder.results(recorder.end())
            for name in ('ViT_Encoder', 'Frame_Attn_Blocks', 'GCA_Blocks', 'Task_Heads'):
                self.assertGreater(result['L1_' + name + '_ms'], 0)
            self.assertGreater(result['L2_GCA_KV_Schedule_ms'], 3)
            self.assertGreater(result['L2_GCA_FlashInfer_Kernel_ms'], 1)
            self.assertAlmostEqual(result['l1_event_closure_error_ms'], 0)
        self.assertIsNone(_recorder.get())
        self.assertNotIn('_predict_camera', vars(model))
        self.assertIs(manager.prefill_wrapper.run, op)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.versions = {key: {'commit': key + '_commit'} for key in ('model', 'research', 'workspace')}

    def tearDown(self):
        self.temp.cleanup()

    def make_run(self, name, stage='short'):
        root = self.root / name
        root.mkdir()
        count = {'smoke': 80, 'short': 320}[stage]
        meta = dict(status='completed', stage=stage, expected_frames=count,
                    equivalence_passed=True, versions=self.versions, run_id=name,
                    config_sha256='config', manifest_sha256='manifest')
        write_json(root / 'run.json', meta)
        fields = ['frame_start', 'frame_end', 'model_wall_ms', 'model_cuda_ms', 'e2e_ms',
                  'finite_output', 'memory_peak_allocated', 'memory_allocated']
        with open(root / 'frames.csv', 'w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for start, stop in frame_groups(count):
                writer.writerow(dict(frame_start=start, frame_end=stop, model_wall_ms=2,
                                     model_cuda_ms=1, e2e_ms=3, finite_output=True,
                                     memory_peak_allocated=200, memory_allocated=100))
        # Shared fixture bytes are enough to exercise index and content integrity.
        artifact = root / 'prediction.npy'
        artifact.write_bytes(b'prediction data')
        with open(root / 'predictions.jsonl', 'w') as stream:
            for frame in range(count):
                record = {'path': artifact.name, 'bytes': artifact.stat().st_size, 'sha256': sha256(artifact)}
                stream.write(json.dumps({'frame': frame, 'files': {k: record for k in ('pose_enc', 'depth', 'depth_conf')}}) + '\n')
        write_json(root / 'summary.json', dict(validation='passed', frames=count,
                   csv_sha256=sha256(root / 'frames.csv'),
                   prediction_index_sha256=sha256(root / 'predictions.jsonl'), traces=[]))
        return root, meta

    def test_valid_run_and_corrupted_prediction(self):
        root, meta = self.make_run('short1')
        self.assertEqual(validate_run(root)[0], meta)
        (root / 'prediction.npy').write_bytes(b'broken')
        with self.assertRaises(ValueError):
            validate_run(root)

    def test_failed_run_cannot_unlock_long(self):
        root, meta = self.make_run('short1')
        meta['status'] = 'failed'
        write_json(root / 'run.json', meta)
        with self.assertRaises(ValueError):
            validate_run(root)

    def test_long_needs_three_distinct_compatible_runs(self):
        roots = [self.make_run(f'short{i}') for i in range(3)]
        args = SimpleNamespace(stage='long', prerequisite=[p for p, _ in roots])
        self.assertEqual(len(check_prerequisites(args, run_identity(roots[0][1]))), 3)
        args.prerequisite = [roots[0][0]] * 3
        with self.assertRaises(ValueError):
            check_prerequisites(args, run_identity(roots[0][1]))
        args.prerequisite = [p for p, _ in roots]
        changed = copy.deepcopy(roots[1][1])
        changed['versions']['model']['commit'] = 'different'
        write_json(roots[1][0] / 'run.json', changed)
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            check_prerequisites(args, run_identity(roots[0][1]))

    def test_wrong_stage_and_csv_tamper(self):
        root, _ = self.make_run('smoke', 'smoke')
        with self.assertRaises(ValueError):
            validate_run(root, 'short')
        with open(root / 'frames.csv', 'a') as stream:
            stream.write('corrupted\n')
        with self.assertRaisesRegex(ValueError, 'CSV hash'):
            validate_run(root)

    def test_unknown_config_cannot_silently_change_variable(self):
        config = self.root / 'config.json'
        write_json(config, FIXED)
        self.assertEqual(config_load(config), FIXED)
        altered = dict(FIXED, max_frame_num=1024)
        write_json(config, altered)
        with self.assertRaises(ValueError):
            config_load(config)


if __name__ == '__main__':
    unittest.main()
