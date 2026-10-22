"""Opt-in exact feature comparisons for single-frame intervention experiments.

Hooks synchronize and copy tensors: timings from enabled runs are diagnostic only.
Baseline features are persisted one layer at a time; candidate features are not kept.
"""
from pathlib import Path
import json
import numpy as np
import torch


class LayerDiagnostics:
    def __init__(self, model, output_dir, baseline_dir=None,
                 sample_frames=(159, 160, 161, 167, 191, 223, 255, 319)):
        self.output = Path(output_dir) / "layer_diagnostics"
        self.output.mkdir(parents=True, exist_ok=True)
        self.baseline = Path(baseline_dir) / "layer_diagnostics" if baseline_dir else None
        self.samples = set(sample_frames)
        self.frame = -1
        self.special = model.aggregator.patch_start_idx
        self.rows = []
        self.handles = []
        for family, blocks in (("frame", model.aggregator.frame_blocks),
                               ("global", model.aggregator.global_blocks)):
            for index, block in enumerate(blocks):
                self.handles.append(block.register_forward_hook(self._hook(family, index)))

    def set_frame(self, frame):
        self.frame = int(frame)

    @staticmethod
    def _stats(value, reference=None):
        result = {"rms": float(np.sqrt(np.mean(value.astype(np.float64) ** 2)))}
        if reference is None:
            return result
        a, b = value.astype(np.float64), reference.astype(np.float64)
        delta = a - b
        result["relative_l2"] = float(np.linalg.norm(delta) / max(np.linalg.norm(b), 1e-12))
        result["rmse"] = float(np.sqrt(np.mean(delta ** 2)))
        numerator = np.sum(a * b, axis=-1)
        denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
        valid = denominator > 1e-12
        result["cosine_valid_tokens"] = int(valid.sum())
        result["mean_cosine_distance"] = float(np.mean(1 - numerator[valid] / denominator[valid])) if valid.any() else None
        return result

    def _hook(self, family, index):
        def hook(module, args, output):
            if self.frame not in self.samples:
                return
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"{family}.{index} output is not a tensor")
            # Each sampled frame is a single-frame forward with complete spatial Q.
            feature = output.detach().float().cpu().numpy()
            feature = feature.reshape(-1, feature.shape[-2], feature.shape[-1])
            if feature.shape[0] != 1:
                raise ValueError("diagnostics require a single sampled frame")
            feature = feature[0]
            stem = f"frame_{self.frame:06d}_{family}_{index:02d}"
            reference = None
            if self.baseline is None:
                np.save(self.output / (stem + ".npy"), feature, allow_pickle=False)
            else:
                reference = np.load(self.baseline / (stem + ".npy"), mmap_mode="r", allow_pickle=False)
                if reference.shape != feature.shape:
                    raise ValueError(f"baseline feature shape mismatch: {stem}")
            row = {"frame": self.frame, "family": family, "layer": index,
                   "shape": list(feature.shape), "groups": {}}
            groups = {"camera": slice(0, 1), "register": slice(1, self.special - 1),
                      "scale": slice(self.special - 1, self.special),
                      "image": slice(self.special, None)}
            for name, region in groups.items():
                if feature[region].size:
                    row["groups"][name] = self._stats(feature[region], reference[region] if reference is not None else None)
            if args and isinstance(args[0], torch.Tensor) and args[0].numel() == output.numel():
                incoming = args[0].detach().float().cpu().numpy().reshape(feature.shape)
                row["block_update_relative_l2"] = float(np.linalg.norm(feature - incoming) / max(np.linalg.norm(incoming), 1e-12))
            if reference is not None:
                token_rmse = np.sqrt(np.mean((feature.astype(np.float64) - reference) ** 2, axis=-1))
                np.save(self.output / (stem + "_token_rmse.npy"), token_rmse.astype(np.float32), allow_pickle=False)
            self.rows.append(row)
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        payload = {"schema_version": 1, "comparison": "exact_full_channel_features",
                   "baseline_dir": str(self.baseline) if self.baseline else None,
                   "sample_frames": sorted(self.samples), "timing_contaminated_by_hooks": True,
                   "expected_rows": len(self.samples) * 48, "observed_rows": len(self.rows),
                   "rows": self.rows,
                   "not_collected": ["attention_matrices", "separate_attention_mlp_residual_updates"]}
        path = self.output / "summary.json"
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        return payload
