import collections
import unittest

import torch

from lingbot_map.layers.attention_intervention import (
    AttentionKVIntervention,
    intervention_site,
    select_image_tokens,
    selection_for_site_row,
)
from lingbot_map.layers.flashinfer_cache import FlashInferKVCacheManager


class AttentionInterventionTests(unittest.TestCase):
    def test_2d_selections_are_nested_and_keep_all_specials(self):
        grid = (4, 6)
        low = set(select_image_tokens(24, 0.3, grid_size=grid).tolist())
        mid = set(select_image_tokens(24, 0.5, grid_size=grid).tolist())
        high = set(select_image_tokens(24, 0.7, grid_size=grid).tolist())
        self.assertTrue(low < mid < high)
        self.assertEqual(len(low), round(24 * 0.3))
        self.assertEqual(len(mid), round(24 * 0.5))
        self.assertEqual(len(high), round(24 * 0.7))

        context = AttentionKVIntervention("frame", 3, target_frame=159, keep_ratio=0.5)
        with context, intervention_site("frame", 3, (159,), 3, grid):
            special, image, _ = selection_for_site_row("frame", 0, 27)
        self.assertEqual(special.tolist(), [0, 1, 2])
        self.assertEqual(image.numel(), 12)
        self.assertTrue(all(i >= 3 for i in image.tolist()))
        self.assertEqual(context.hit_count, 1)

    def test_keep_one_selects_every_image_token(self):
        got = select_image_tokens(24, 1.0, grid_size=(4, 6))
        self.assertEqual(got.tolist(), list(range(24)))

    def test_compaction_packs_variable_pages_and_specials(self):
        manager = FlashInferKVCacheManager.__new__(FlashInferKVCacheManager)
        manager.page_size = 4
        manager.patches_per_frame = 4
        manager.num_special_tokens = 2
        manager.num_heads = 1
        manager.head_dim = 1
        manager.dtype = torch.float32
        manager.device = torch.device("cpu")
        manager.scale_patch_pages = [collections.deque([0, 1])]
        manager.live_window_patch_pages = [collections.deque()]
        manager.patch_valid_counts = [{0: 4, 1: 2}]
        manager.all_special_pages = [[2]]
        manager.special_token_count = [3]
        manager._scratch_kv_caches = {}
        manager.last_attention_token_count = {}

        cache = torch.zeros((3, 2, 4, 1, 1))
        cache[0, 0, :, 0, 0] = torch.tensor([0, 1, 2, 3])
        cache[0, 1, :, 0, 0] = torch.tensor([10, 11, 12, 13])
        cache[1, 0, :, 0, 0] = torch.tensor([4, 5, 90, 91])
        cache[1, 1, :, 0, 0] = torch.tensor([14, 15, 90, 91])
        cache[2, 0, :3, 0, 0] = torch.tensor([20, 21, 22])
        cache[2, 1, :3, 0, 0] = torch.tensor([30, 31, 32])
        manager.kv_caches = [cache]

        pages, last_len, packed = manager._build_compacted_attention_pages(0)
        self.assertEqual((pages, last_len), (3, 1))
        self.assertEqual(manager.last_attention_token_count[0], 9)
        self.assertEqual(packed[:2, 0, :, 0, 0].flatten().tolist(),
                         [0, 1, 2, 3, 4, 5, 20, 21])
        self.assertEqual(packed[2, 0, 0, 0, 0].item(), 22)
        self.assertEqual(packed[:2, 1, :, 0, 0].flatten().tolist(),
                         [10, 11, 12, 13, 14, 15, 30, 31])

    def test_partial_page_keeps_normal_window_eviction(self):
        manager = FlashInferKVCacheManager.__new__(FlashInferKVCacheManager)
        manager.live_window_patch_pages = [collections.deque([7, 8])]
        manager.free_patch_pages = [[1]]
        manager.patch_valid_counts = [{7: 3, 8: 4}]
        manager._defer_eviction = False
        manager.evict_frames(0, scale_frames=2, sliding_window=1)
        self.assertEqual(list(manager.live_window_patch_pages[0]), [8])
        self.assertNotIn(7, manager.patch_valid_counts[0])
        self.assertEqual(manager.patch_valid_counts[0][8], 4)
        self.assertEqual(manager.free_patch_pages[0], [1, 7])


if __name__ == "__main__":
    unittest.main()
