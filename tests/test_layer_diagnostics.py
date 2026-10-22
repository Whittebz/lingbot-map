import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from lingbot_map.diagnostics import LayerDiagnostics

class Block(torch.nn.Module):
    def forward(self, x):
        return x + 1

class DiagnosticsTests(unittest.TestCase):
    def test_exact_comparison_and_hook_cleanup(self):
        model = SimpleNamespace(aggregator=SimpleNamespace(
            patch_start_idx=3, frame_blocks=[Block()], global_blocks=[Block()]))
        value = torch.arange(30, dtype=torch.float32).reshape(1, 6, 5)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)/"baseline"
            observer = LayerDiagnostics(model, base, sample_frames=(159,))
            observer.set_frame(159)
            for block in model.aggregator.frame_blocks+model.aggregator.global_blocks:
                block(value)
            observer.close()
            observer = LayerDiagnostics(model, Path(directory)/"candidate", base, sample_frames=(159,))
            observer.set_frame(159)
            for block in model.aggregator.frame_blocks+model.aggregator.global_blocks:
                block(value + 0.5)
            summary = observer.close()
            self.assertEqual(summary["observed_rows"], 2)
            for row in summary["rows"]:
                self.assertAlmostEqual(row["groups"]["image"]["rmse"], .5)
            for block in model.aggregator.frame_blocks+model.aggregator.global_blocks:
                self.assertEqual(len(block._forward_hooks), 0)

if __name__ == "__main__":
    unittest.main()
