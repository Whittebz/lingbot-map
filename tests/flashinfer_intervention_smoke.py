"""Real FlashInfer partial-page tests. Must run through shared gpu_runner.sh."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from lingbot_map.layers.flashinfer_cache import FlashInferKVCacheManager

def run(output):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("one runner-selected GPU required")
    torch.manual_seed(73)
    device=torch.device("cuda")
    dtype=torch.bfloat16
    frames=[(torch.randn(8,2,64,device=device,dtype=dtype),
             torch.randn(8,2,64,device=device,dtype=dtype)) for _ in range(7)]
    queries=[torch.randn(8,2,64,device=device,dtype=dtype) for _ in frames]
    selected=torch.tensor([0,2,4],device=device,dtype=torch.long)
    rows=[]
    for mode in ("identity","current_only","current_and_cache"):
        manager=FlashInferKVCacheManager(num_blocks=1,max_num_frames=8,tokens_per_frame=8,
            num_heads=2,head_dim=64,dtype=dtype,device=device,num_special_tokens=2,
            scale_frames=1,sliding_window=2,max_total_frames=16)
        for i,((k,v),q) in enumerate(zip(frames,queries)):
            subset=selected if i==2 and mode=="current_and_cache" else None
            pid=manager.append_frame(0,k,v,patch_indices=subset)
            if i==2 and mode=="current_only": manager.set_current_read_subset(0,pid,selected)
            manager.evict_frames(0,scale_frames=1,sliding_window=2)
            got=manager.compute_attention(0,q)
            visible=[0]+list(range(max(1,i-1),i+1))
            patch_k=[];patch_v=[]
            for j in visible:
                use_subset=(j==2 and (mode=="current_and_cache" or (mode=="current_only" and i==2)))
                ids=selected if use_subset else torch.arange(6,device=device)
                patch_k.append(frames[j][0][2:].index_select(0,ids))
                patch_v.append(frames[j][1][2:].index_select(0,ids))
            expected_k=torch.cat(patch_k+[frames[j][0][:2] for j in range(i+1)])
            expected_v=torch.cat(patch_v+[frames[j][1][:2] for j in range(i+1)])
            ref=F.scaled_dot_product_attention(q.float().transpose(0,1).unsqueeze(0),
                expected_k.float().transpose(0,1).unsqueeze(0),expected_v.float().transpose(0,1).unsqueeze(0))
            ref=ref[0].transpose(0,1)
            error=float((got.float()-ref).abs().max())
            torch.testing.assert_close(got.float(),ref,atol=.015,rtol=.04)
            assert manager.last_attention_token_count[0]==expected_k.shape[0]
            assert manager.special_token_count[0]==2*(i+1)
            assert len(manager.live_window_patch_pages[0])<=2
            rows.append({"mode":mode,"frame":i,"kv_tokens":expected_k.shape[0],"max_abs_error":error})
        manager.reset()
        assert not manager.patch_valid_counts[0] and not manager._scratch_kv_caches
    payload={"status":"passed","backend":"FlashInfer","gpu":torch.cuda.get_device_name(0),
             "torch":torch.__version__,"cuda":torch.version.cuda,"cases":rows}
    if output:
        Path(output).parent.mkdir(parents=True,exist_ok=True)
        Path(output).write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output");args=parser.parse_args()
    run(args.output)

