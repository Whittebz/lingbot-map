"""Opt-in token-level KV interventions for GCT attention experiments."""
from contextlib import contextmanager
from functools import lru_cache
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Sequence
import torch

_ACTIVE = ContextVar("gct_attention_intervention", default=None)
_SITE = ContextVar("gct_attention_intervention_site", default=None)

@lru_cache(maxsize=16)
def _uniform_order(grid_h: int, grid_w: int) -> list[int]:
    """Nested 2D farthest-point order in original raster coordinates."""
    count = grid_h * grid_w
    if count <= 0:
        return []
    points = [(i // grid_w, i % grid_w) for i in range(count)]
    center = min(range(count), key=lambda i: (
        (points[i][0] - (grid_h - 1) / 2) ** 2 + (points[i][1] - (grid_w - 1) / 2) ** 2, i))
    chosen = [center]
    remaining = set(range(count)) - {center}
    nearest = [abs(y - points[center][0]) + abs(x - points[center][1]) for y, x in points]
    while remaining:
        nxt = max(remaining, key=lambda i: (nearest[i], -i))
        chosen.append(nxt)
        remaining.remove(nxt)
        y, x = points[nxt]
        for i in remaining:
            py, px = points[i]
            nearest[i] = min(nearest[i], abs(py - y) + abs(px - x))
    return chosen

def select_image_tokens(count: int, keep_ratio: float, *, grid_size=None, device=None) -> torch.Tensor:
    """Return nested, near-uniform image-token indices."""
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
    keep_count = max(1, min(count, int(round(count * keep_ratio)))) if count else 0
    if grid_size is None:
        side = int(count ** 0.5)
        if side * side != count:
            raise ValueError("grid_size=(height, width) is required for non-square patch grids")
        grid_size = (side, side)
    if grid_size[0] * grid_size[1] != count:
        raise ValueError(f"grid {grid_size} does not match {count} image tokens")
    return torch.tensor(sorted(_uniform_order(*grid_size)[:keep_count]), dtype=torch.long, device=device)

@dataclass(frozen=True)
class InterventionSite:
    kind: str
    block_idx: int
    frame_indices: tuple[int, ...]
    num_special: int
    grid_size: tuple[int, int] | None = None

class AttentionKVIntervention:
    """Opt-in intervention for one attention module and absolute frame."""
    def __init__(self, kind: str, block_idx: int, target_frame: int = 159,
                 keep_ratio: float = 0.3, retain_in_cache: bool = True):
        if kind not in ("frame", "global"):
            raise ValueError("kind must be frame or global")
        if block_idx < 0 or target_frame < 0:
            raise ValueError("block_idx and target_frame must be non-negative")
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        self.kind, self.block_idx = kind, block_idx
        self.target_frame, self.keep_ratio = target_frame, keep_ratio
        self.retain_in_cache = retain_in_cache
        self.hit_count = 0
        self.original_token_count = 0
        self.retained_token_count = 0
        self.cache_retain_count = 0
        self.cache_drop_count = 0
        self.attention_read_token_count = 0
        self.target_attention_token_count = 0
        self.visible_kv_token_count = 0

    @property
    def mode(self):
        return "current_and_cache" if self.retain_in_cache else "current_only"

    @property
    def diagnostics(self):
        return {
            "hit_count": self.hit_count,
            "original_token_count": self.original_token_count,
            "retained_token_count": self.retained_token_count,
            "dropped_token_count": self.original_token_count - self.retained_token_count,
            "cache_retain_count": self.cache_retain_count,
            "cache_drop_count": self.cache_drop_count,
            "attention_read_token_count": self.attention_read_token_count,
            "target_attention_token_count": self.target_attention_token_count,
            "visible_kv_token_count": self.visible_kv_token_count,
            "mode": self.mode,
        }

    def __enter__(self):
        self._token = _ACTIVE.set(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        _ACTIVE.reset(self._token)

@contextmanager
def intervention_site(kind: str, block_idx: int, frame_indices: Sequence[int], num_special: int,
                      grid_size=None):
    token = _SITE.set(InterventionSite(kind, block_idx, tuple(frame_indices), num_special, grid_size))
    try:
        yield
    finally:
        _SITE.reset(token)

def _record_selection(config, num_special: int, patch_count: int, image: torch.Tensor):
    config.hit_count += 1
    config.original_token_count = patch_count
    config.retained_token_count = image.numel()
    config.cache_retain_count = image.numel() if config.retain_in_cache else patch_count
    config.cache_drop_count = patch_count - config.cache_retain_count
    config.target_attention_token_count = num_special + image.numel()
    config.attention_read_token_count = config.target_attention_token_count


def active_selection(kind: str, block_idx: int, frame_idx: int, num_special: int,
                     patch_count: int, device=None):
    """Return special indices, selected image indices, and config if site matches."""
    config, site = _ACTIVE.get(), _SITE.get()
    if (config is None or site is None or config.kind != kind or site.kind != kind
            or config.block_idx != block_idx or frame_idx != config.target_frame):
        return None
    num_special = site.num_special
    special = torch.arange(num_special, dtype=torch.long, device=device)
    image = select_image_tokens(patch_count, config.keep_ratio, grid_size=site.grid_size, device=device) + num_special
    _record_selection(config, num_special, patch_count, image)
    return special, image, config


def selection_for_site_row(kind: str, row: int, token_count: int, device=None):
    config, site = _ACTIVE.get(), _SITE.get()
    if config is None or site is None or row >= len(site.frame_indices):
        return None
    frame_idx = site.frame_indices[row]
    if (config.kind != kind or site.kind != kind or config.block_idx != site.block_idx
            or frame_idx != config.target_frame):
        return None
    num_special = site.num_special
    patch_count = token_count - num_special
    special = torch.arange(num_special, dtype=torch.long, device=device)
    image = select_image_tokens(patch_count, config.keep_ratio, grid_size=site.grid_size, device=device) + num_special
    _record_selection(config, num_special, patch_count, image)
    return special, image, config
