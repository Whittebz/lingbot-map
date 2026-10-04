#!/usr/bin/env python3
"""Experiment 1: probe image-token information with an RGB decoder.

This file is additive. It does not edit the LingBot-Map implementation.

Stages:
  extract  Run a frozen windowed model and save patch tokens plus RGB targets.
  train    Train a small RGB decoder on extracted token files.
  eval     Evaluate a trained decoder on extracted token files.

The model uses the official windowed GCT implementation. ``local`` tokens are
captured from a frame block before global attention. ``global_*`` tokens are
the token outputs returned by the official aggregator.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_image_paths(image_folder: str, first_k: int | None, stride: int) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    paths: List[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(image_folder, ext)))
    paths = sorted(set(paths))
    if stride > 1:
        paths = paths[::stride]
    if first_k is not None and first_k > 0:
        paths = paths[:first_k]
    if not paths:
        raise FileNotFoundError(f"No images found in {image_folder}")
    return paths


def load_images(paths: Sequence[str], image_size: int, patch_size: int) -> torch.Tensor:
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    images = load_and_preprocess_images(
        list(paths), mode="crop", image_size=image_size, patch_size=patch_size
    )
    if images.ndim != 4:
        raise ValueError(f"Expected [S,3,H,W] images, got {tuple(images.shape)}")
    return images.float().contiguous()


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    from lingbot_map.models.gct_stream_window import GCTStream

    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_camera=False,
        enable_point=False,
        enable_local_point=False,
        enable_depth=False,
        enable_track=False,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
        max_frame_num=args.max_frame_num,
    )
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"checkpoint={args.model_path}")
    print(f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    model.to(device).eval()
    return model


class RGBDecoder(nn.Module):
    """Small decoder for a token information probe."""

    def __init__(self, input_dim: int, patch_h: int, patch_w: int) -> None:
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.project = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=1),
            nn.GELU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4),
            nn.GELU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )

    def forward(self, tokens: torch.Tensor, output_hw: tuple[int, int]) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected [B,P,C] tokens, got {tuple(tokens.shape)}")
        b, p, c = tokens.shape
        expected = self.patch_h * self.patch_w
        if p != expected:
            raise ValueError(f"Expected {expected} patch tokens, got {p}")
        x = tokens.float().transpose(1, 2).reshape(b, c, self.patch_h, self.patch_w)
        x = self.project(x)
        x = self.up(x)
        x = F.interpolate(x, size=output_hw, mode="bilinear", align_corners=False)
        return torch.sigmoid(x)


class TokenFrameDataset(Dataset):
    def __init__(self, files: Sequence[Path]) -> None:
        if not files:
            raise ValueError("No token files were found")
        self.files = list(files)
        self.index: list[tuple[int, int]] = []
        self._metadata = []
        for file_idx, path in enumerate(self.files):
            item = torch.load(path, map_location="cpu", weights_only=False)
            tokens = item["tokens"]
            images = item["images"]
            if tokens.ndim != 3 or images.ndim != 4:
                raise ValueError(f"Invalid shapes in {path}")
            if tokens.shape[0] != images.shape[0]:
                raise ValueError(f"Token/image frame count mismatch in {path}")
            self._metadata.append((tuple(tokens.shape), tuple(images.shape)))
            self.index.extend((file_idx, frame_idx) for frame_idx in range(tokens.shape[0]))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_idx, frame_idx = self.index[idx]
        item = torch.load(self.files[file_idx], map_location="cpu", weights_only=False)
        return item["tokens"][frame_idx], item["images"][frame_idx]

    @property
    def token_shape(self) -> tuple[int, ...]:
        return self._metadata[0][0]

    @property
    def image_shape(self) -> tuple[int, ...]:
        return self._metadata[0][1]


def apply_control(tokens: torch.Tensor, control: str, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if control == "none":
        return tokens
    if control == "zero":
        return torch.zeros_like(tokens)
    if control == "frame_shuffle":
        if tokens.shape[0] < 2:
            return tokens
        permutation = torch.randperm(tokens.shape[0], generator=generator)
        return tokens[permutation]
    if control != "shuffle":
        raise ValueError(f"Unknown control: {control}")
    shuffled = tokens.clone()
    for frame_idx in range(tokens.shape[0]):
        permutation = torch.randperm(tokens.shape[1], generator=generator)
        shuffled[frame_idx] = tokens[frame_idx, permutation]
    return shuffled


def capture_local_tokens(model: nn.Module, layer: int):
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["tokens"] = output.detach()

    handle = model.aggregator.frame_blocks[layer].register_forward_hook(hook)
    return captured, handle


@torch.no_grad()
def extract(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = load_image_paths(args.image_folder, args.first_k, args.stride)
    images = load_images(paths, args.image_size, args.patch_size)
    model = build_model(args, device)

    if args.source == "local":
        if args.layer < 0 or args.layer >= len(model.aggregator.frame_blocks):
            raise ValueError(f"layer must be in [0, {len(model.aggregator.frame_blocks) - 1}]")
        captured, hook_handle = capture_local_tokens(model, args.layer)
    else:
        captured, hook_handle = None, None

    step = args.window_size - args.overlap
    if step <= 0:
        raise ValueError("window_size must be greater than overlap")
    files = []
    started = time.perf_counter()
    for window_idx, start in enumerate(range(0, len(paths), step)):
        end = min(start + args.window_size, len(paths))
        if end - start < args.min_window_frames:
            break
        model.clean_kv_cache()
        window_images = images[start:end].unsqueeze(0).to(device, non_blocking=True)
        num_scale = min(args.num_scale_frames, end - start)
        if captured is not None:
            captured.clear()
        aggregated, patch_start_idx = model._aggregate_features(
            window_images,
            num_frame_for_scale=num_scale,
            num_frame_per_block=end - start,
        )
        if args.source == "local":
            if "tokens" not in captured:
                raise RuntimeError("The local frame-block hook did not capture tokens")
            source_tokens = captured["tokens"]
            source_tokens = source_tokens.view(1, end - start, source_tokens.shape[1], source_tokens.shape[2])
            source_tokens = source_tokens[0, :, patch_start_idx:]
        else:
            source_index = int(args.source.split("_")[1])
            if source_index >= len(aggregated):
                raise RuntimeError(f"source index {source_index} is not available")
            source_tokens = aggregated[source_index][0, :, patch_start_idx:]

        source_tokens = apply_control(source_tokens.float().cpu(), args.control, args.seed + window_idx)
        target_images = images[start:end].cpu()
        patch_h = target_images.shape[-2] // args.patch_size
        patch_w = target_images.shape[-1] // args.patch_size
        if patch_h * patch_w != source_tokens.shape[1]:
            raise RuntimeError(
                f"Patch grid mismatch: target={patch_h}x{patch_w}, "
                f"tokens={source_tokens.shape[1]}"
            )
        file_path = output_dir / f"window_{window_idx:04d}.pt"
        torch.save(
            {
                "tokens": source_tokens.to(torch.float16),
                "images": target_images.to(torch.float16),
                "start": start,
                "end": end,
                "source": args.source,
                "control": args.control,
                "patch_h": patch_h,
                "patch_w": patch_w,
                "patch_start_idx": int(patch_start_idx),
                "image_paths": paths[start:end],
            },
            file_path,
        )
        files.append(str(file_path))
        print(f"saved {file_path} frames={end - start} tokens={tuple(source_tokens.shape)}")
        del window_images, aggregated, source_tokens
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if hook_handle is not None:
        hook_handle.remove()
    manifest = vars(args).copy()
    manifest.update({
        "num_input_frames": len(paths),
        "num_windows": len(files),
        "files": files,
        "elapsed_sec": time.perf_counter() - started,
        "token_dtype": "float16",
        "target_dtype": "float16",
    })
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"wrote {output_dir / 'manifest.json'}")


def ssim_global(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # A lightweight global SSIM estimate. It avoids an extra package dependency.
    dims = (1, 2, 3)
    mu_x, mu_y = pred.mean(dims, keepdim=True), target.mean(dims, keepdim=True)
    var_x = ((pred - mu_x) ** 2).mean(dims, keepdim=True)
    var_y = ((target - mu_y) ** 2).mean(dims, keepdim=True)
    cov = ((pred - mu_x) * (target - mu_y)).mean(dims, keepdim=True)
    c1, c2 = 0.01**2, 0.03**2
    value = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return value.mean()


def metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mse = F.mse_loss(pred, target).clamp_min(1e-12)
    return {
        "l1": float(F.l1_loss(pred, target).item()),
        "psnr": float((-10.0 * torch.log10(mse)).item()),
        "ssim_global": float(ssim_global(pred, target).item()),
    }


def split_files(data_dir: str, val_fraction: float) -> tuple[list[Path], list[Path]]:
    files = sorted(Path(data_dir).glob("window_*.pt"))
    if len(files) < 2:
        raise ValueError("At least two window files are required for train/validation split")
    split = max(1, min(len(files) - 1, round(len(files) * (1.0 - val_fraction))))
    return files[:split], files[split:]


def make_decoder(dataset: TokenFrameDataset, patch_size: int) -> tuple[RGBDecoder, tuple[int, int]]:
    _, patch_count, input_dim = dataset.token_shape
    _, _, height, width = dataset.image_shape
    patch_h, patch_w = height // patch_size, width // patch_size
    if patch_h * patch_w != patch_count:
        raise ValueError(
            f"Patch grid mismatch: image={height}x{width}, patch_size={patch_size}, "
            f"tokens={patch_count}"
        )
    return RGBDecoder(input_dim, patch_h, patch_w), (height, width)


def run_epoch(
    decoder: RGBDecoder,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_steps: int | None,
) -> dict[str, float]:
    training = optimizer is not None
    decoder.train(training)
    sums = {"l1": 0.0, "psnr": 0.0, "ssim_global": 0.0}
    count = 0
    for step, (tokens, target) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        tokens, target = tokens.to(device), target.to(device)
        with torch.set_grad_enabled(training):
            pred = decoder(tokens, target.shape[-2:])
            l1 = F.l1_loss(pred, target.float())
            edge_pred = pred[..., 1:, :] - pred[..., :-1, :]
            edge_target = target.float()[..., 1:, :] - target.float()[..., :-1, :]
            loss = l1 + 0.05 * F.l1_loss(edge_pred, edge_target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_metrics = metrics(pred.detach(), target.float())
        batch_size = tokens.shape[0]
        for key in sums:
            sums[key] += batch_metrics[key] * batch_size
        count += batch_size
    if count == 0:
        raise RuntimeError("No batches were processed")
    return {key: value / count for key, value in sums.items()}


def train_or_eval(args: argparse.Namespace, evaluate_only: bool) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    train_files, val_files = split_files(args.data_dir, args.val_fraction)
    train_set = TokenFrameDataset(train_files)
    val_set = TokenFrameDataset(val_files)
    decoder, _ = make_decoder(train_set, args.patch_size)
    decoder.to(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / "rgb_decoder.pt"

    if evaluate_only:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        decoder.load_state_dict(state["decoder"])
        loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        result = run_epoch(decoder, loader, device, None, args.max_steps)
        print(json.dumps({"split": "val", **result}, indent=2))
        return

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history = []
    for epoch in range(args.epochs):
        train_result = run_epoch(decoder, train_loader, device, optimizer, args.max_steps)
        val_result = run_epoch(decoder, val_loader, device, None, args.max_steps)
        record = {"epoch": epoch + 1, "train": train_result, "val": val_result}
        history.append(record)
        print(json.dumps(record, indent=2))

    torch.save(
        {
            "decoder": decoder.state_dict(),
            "input_dim": train_set.token_shape[-1],
            "patch_count": train_set.token_shape[-2],
            "history": history,
            "args": vars(args),
        },
        checkpoint_path,
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(f"saved {checkpoint_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    extract = sub.add_parser("extract")
    extract.add_argument("--model_path", required=True)
    extract.add_argument("--image_folder", required=True)
    extract.add_argument("--output_dir", required=True)
    extract.add_argument("--source", choices=("local", "global_0", "global_1", "global_2", "global_3"), default="local")
    extract.add_argument("--control", choices=("none", "zero", "shuffle", "frame_shuffle"), default="none")
    extract.add_argument("--layer", type=int, default=23)
    extract.add_argument("--first_k", type=int, default=None)
    extract.add_argument("--stride", type=int, default=1)
    extract.add_argument("--window_size", type=int, default=8)
    extract.add_argument("--overlap", type=int, default=0)
    extract.add_argument("--min_window_frames", type=int, default=2)
    extract.add_argument("--num_scale_frames", type=int, default=4)
    extract.add_argument("--image_size", type=int, default=518)
    extract.add_argument("--patch_size", type=int, default=14)
    extract.add_argument("--kv_cache_sliding_window", type=int, default=64)
    extract.add_argument("--max_frame_num", type=int, default=1024)
    extract.add_argument("--use_sdpa", action="store_true")
    extract.add_argument("--device", default="cuda:0")
    extract.add_argument("--seed", type=int, default=20260930)

    for name, help_text in (("train", "Train an RGB decoder"), ("eval", "Evaluate an RGB decoder")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--data_dir", required=True)
        command.add_argument("--output_dir", required=True)
        command.add_argument("--checkpoint", default=None)
        command.add_argument("--val_fraction", type=float, default=0.25)
        command.add_argument("--batch_size", type=int, default=1)
        command.add_argument("--patch_size", type=int, default=14)
        command.add_argument("--epochs", type=int, default=10)
        command.add_argument("--learning_rate", type=float, default=1e-4)
        command.add_argument("--max_steps", type=int, default=None)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--seed", type=int, default=20260930)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "extract":
        extract(args)
    elif args.stage == "train":
        train_or_eval(args, evaluate_only=False)
    elif args.stage == "eval":
        train_or_eval(args, evaluate_only=True)
    else:
        raise AssertionError(args.stage)


if __name__ == "__main__":
    main()
