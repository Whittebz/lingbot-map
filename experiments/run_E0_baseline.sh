#!/bin/bash
# E0 Baseline: Streaming 推理基线（无 Patch 筛选）
# 日期: 2026-10-05
# 目的: 固定可复现的 baseline，作为后续所有优化实验的参考点

set -e  # 遇到错误立即退出

# ==================== 配置 ====================
SCENE=${1:-loop}  # 默认 loop，可通过参数覆盖
OUTPUT_BASE=experiments/results/E0_baseline

# 模型和数据
MODEL_PATH=models/lingbot-map.pt
IMAGE_FOLDER=example/${SCENE}

# Streaming 配置
MODE=streaming
NUM_SCALE_FRAMES=8
KV_CACHE_SLIDING_WINDOW=64
KEYFRAME_INTERVAL=1
CAMERA_NUM_ITERATIONS=4
IMAGE_SIZE=518
PATCH_SIZE=14
MAX_FRAME_NUM=1024

# 其他
USE_SDPA=--use_sdpa
OFFLOAD_TO_CPU=--offload_to_cpu
SEED=20260930

# 输出目录
OUTPUT_DIR=${OUTPUT_BASE}/${SCENE}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=${OUTPUT_DIR}/run_${TIMESTAMP}.log

# ==================== 准备 ====================
echo "[E0 Baseline] Starting at $(date)"
echo "Scene: ${SCENE}"
echo "Output: ${OUTPUT_DIR}"

mkdir -p ${OUTPUT_DIR}

# 记录 git 信息
GIT_COMMIT=$(git rev-parse HEAD)
GIT_BRANCH=$(git branch --show-current)
GIT_DIFF_STAT=$(git diff --stat)

echo "Git commit: ${GIT_COMMIT}"
echo "Git branch: ${GIT_BRANCH}"

# 记录环境信息
CUDA_VERSION=$(nvcc --version | grep release | awk '{print $5}' | sed 's/,//')
PYTORCH_VERSION=$(python -c 'import torch; print(torch.__version__)')
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

echo "CUDA: ${CUDA_VERSION}"
echo "PyTorch: ${PYTORCH_VERSION}"
echo "GPU: ${GPU_MODEL}"

# ==================== 运行推理 ====================
echo ""
echo "[E0 Baseline] Running streaming inference..."

# 运行并记录日志
python demo.py \
  --model_path ${MODEL_PATH} \
  --image_folder ${IMAGE_FOLDER} \
  --mode ${MODE} \
  --num_scale_frames ${NUM_SCALE_FRAMES} \
  --kv_cache_sliding_window ${KV_CACHE_SLIDING_WINDOW} \
  --keyframe_interval ${KEYFRAME_INTERVAL} \
  --camera_num_iterations ${CAMERA_NUM_ITERATIONS} \
  --image_size ${IMAGE_SIZE} \
  --patch_size ${PATCH_SIZE} \
  --max_frame_num ${MAX_FRAME_NUM} \
  ${USE_SDPA} \
  ${OFFLOAD_TO_CPU} \
  2>&1 | tee ${LOG_FILE}

# ==================== 记录 Metadata ====================
echo ""
echo "[E0 Baseline] Recording metadata..."

# 计算哈希值
MODEL_HASH=$(sha256sum ${MODEL_PATH} | awk '{print $1}')
if [ -f ${IMAGE_FOLDER}/manifest.txt ]; then
  DATA_HASH=$(sha256sum ${IMAGE_FOLDER}/manifest.txt | awk '{print $1}')
else
  DATA_HASH="N/A"
fi

cat > ${OUTPUT_DIR}/metadata.yaml << METADATA
experiment: E0_baseline
scene: ${SCENE}
date: $(date -Iseconds)
mode: ${MODE}

# Git 信息
git_commit: ${GIT_COMMIT}
git_branch: ${GIT_BRANCH}
git_diff: |
${GIT_DIFF_STAT}

# 模型配置
model_path: ${MODEL_PATH}
model_hash: ${MODEL_HASH}

# 数据
image_folder: ${IMAGE_FOLDER}
data_hash: ${DATA_HASH}

# Streaming 配置
num_scale_frames: ${NUM_SCALE_FRAMES}
kv_cache_sliding_window: ${KV_CACHE_SLIDING_WINDOW}
keyframe_interval: ${KEYFRAME_INTERVAL}
camera_num_iterations: ${CAMERA_NUM_ITERATIONS}
image_size: ${IMAGE_SIZE}
patch_size: ${PATCH_SIZE}
max_frame_num: ${MAX_FRAME_NUM}
use_sdpa: true
offload_to_cpu: true

# 环境
cuda_version: ${CUDA_VERSION}
pytorch_version: ${PYTORCH_VERSION}
gpu_model: ${GPU_MODEL}

# 运行时
random_seed: ${SEED}
timestamp: ${TIMESTAMP}
log_file: ${LOG_FILE}
METADATA

echo "Metadata saved to: ${OUTPUT_DIR}/metadata.yaml"

# ==================== 完成 ====================
echo ""
echo "[E0 Baseline] Completed at $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Check log: cat ${LOG_FILE}"
echo "  2. View metadata: cat ${OUTPUT_DIR}/metadata.yaml"
echo "  3. Extract metrics from log"
