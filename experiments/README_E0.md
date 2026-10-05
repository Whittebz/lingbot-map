# E0 Baseline 执行指南

## 快速开始

### 在远程服务器运行

```bash
# SSH 到远程
ssh kt
cd /home/dengbaichuan/research/lingbot_map

# 激活环境
conda activate lingbot-map

# 运行 baseline（默认 loop 场景）
bash experiments/run_E0_baseline.sh

# 或指定场景
bash experiments/run_E0_baseline.sh courthouse
bash experiments/run_E0_baseline.sh university
```

### 可用场景

- `loop` - RGB 重建实验用过，便于对比
- `courthouse` - RGB 重建实验用过
- `university` - RGB 重建实验用过
- `official_long_examples/drive_frames.mp4` - 官方 demo 数据（需要修改脚本）

## 输出结构

```
experiments/results/E0_baseline/
└── loop/                          # 场景名称
    ├── metadata.yaml              # 完整的实验元信息
    ├── run_20261005_101234.log    # 推理日志（含时间戳）
    └── (demo.py 的可视化输出)
```

## Metadata 内容

`metadata.yaml` 包含：
- 实验配置（mode, keyframe_interval, window_size 等）
- Git 信息（commit, branch, diff）
- 模型和数据哈希
- 环境信息（CUDA, PyTorch, GPU）
- 运行时信息（timestamp, log_file）

## 提取指标

推理日志包含：
- 总推理时间
- GPU 显存峰值
- 每帧处理信息

需要手动从日志提取并记录到 `metrics.json`：

```json
{
  "timing": {
    "total_time_sec": 0.0,
    "fps": 0.0,
    "mean_latency_ms": 0.0
  },
  "memory": {
    "peak_gpu_mb": 0
  }
}
```

## 配置说明

### 核心 Streaming 参数

| 参数 | 值 | 说明 |
|---|---|---|
| `mode` | `streaming` | ✅ 真正的 Streaming 模式 |
| `num_scale_frames` | `8` | 锚点帧数量 |
| `kv_cache_sliding_window` | `64` | Local Window 大小 |
| `keyframe_interval` | `1` | 每帧都处理（baseline） |
| `camera_num_iterations` | `4` | 相机头迭代次数 |
| `use_sdpa` | flag | 使用 SDPA（不需要 flashinfer）|
| `offload_to_cpu` | flag | 降低 GPU 显存峰值 |

### 与 exp1 (RGB 重建) 的区别

| 维度 | E0 Baseline | exp1 (RGB 重建) |
|---|---|---|
| API | `inference_streaming` | `_aggregate_features` |
| KV Cache | 持续保持 64 帧 | 每 8 帧清空 |
| 处理方式 | 逐帧因果 | 批量（8 帧/批）|
| Local Window | 真正的 64 帧 | 实际只有 8 帧 |

## 下一步

E0 baseline 完成后：

1. **检查日志**：
   ```bash
   cat experiments/results/E0_baseline/loop/run_*.log
   ```

2. **查看 metadata**：
   ```bash
   cat experiments/results/E0_baseline/loop/metadata.yaml
   ```

3. **提取指标**：
   - 从日志中提取 timing 和 memory 指标
   - 记录到 `metrics.json`

4. **对比 RGB 重建**：
   - 如果需要，运行快速 Streaming token 提取验证

5. **开始 E1**：
   - 探索深度置信度提取
   - 实现 Patch 筛选
