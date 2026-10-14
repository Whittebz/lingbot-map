#!/bin/bash
# Batch extraction for complete exp3 analysis (12 positions)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_RUNNER="${SCRIPT_DIR}/../utils/gpu_runner.sh"

# 激活环境
source /home/dengbaichuan/miniconda3/bin/activate lingbot-map
export PYTHONPATH=/home/dengbaichuan/research/lingbot_map/experiments:${PYTHONPATH}

# 参数
SCENE=${1:-loop}
FIRST_K=${2:-30}

cd /home/dengbaichuan/research/lingbot_map

# 需要提取的位置
declare -a SOURCES=(
    "frame_blocks_4"
    "global_blocks_4"
    "frame_blocks_11"
    "global_blocks_11"
    "frame_blocks_17"
    "aggregated_4"
    "aggregated_11"
    "aggregated_17"
    "aggregated_23"
)

echo "==================== 批量提取开始 ===================="
echo "Scene: ${SCENE}"
echo "First K: ${FIRST_K}"
echo "需要提取: ${#SOURCES[@]} 个位置"
echo "=================================================="

for SOURCE in "${SOURCES[@]}"; do
    echo ""
    echo "==== 提取: ${SOURCE} ===="

    OUTPUT_DIR="experiments/runs/exp3_${SCENE}_${SOURCE}_${FIRST_K}frames"

    # 检查是否已存在
    if [ -f "${OUTPUT_DIR}/window_0000.pt" ]; then
        echo "⏭️  跳过（已存在）: ${OUTPUT_DIR}"
        continue
    fi

    # 判断是 aggregated 还是普通层
    if [[ "$SOURCE" == aggregated_* ]]; then
        # 提取 aggregated tokens
        BLOCK_IDX="${SOURCE#aggregated_}"

        "${GPU_RUNNER}" python experiments/exp3_full_information_flow/extract_aggregated.py \
            --model_path models/lingbot-map.pt \
            --scene ${SCENE} \
            --block_idx ${BLOCK_IDX} \
            --image_folder example/${SCENE} \
            --output_dir ${OUTPUT_DIR} \
            --first_k ${FIRST_K} \
            --use_sdpa \
            --device cuda:0
    else
        # 提取普通层 tokens
        "${GPU_RUNNER}" python experiments/exp3_full_information_flow/extract_tokens.py \
            --model_path models/lingbot-map.pt \
            --scene ${SCENE} \
            --source ${SOURCE} \
            --image_folder example/${SCENE} \
            --output_dir ${OUTPUT_DIR} \
            --first_k ${FIRST_K} \
            --use_sdpa \
            --device cuda:0
    fi

    if [ $? -eq 0 ]; then
        echo "✅ 完成: ${SOURCE}"
    else
        echo "❌ 失败: ${SOURCE}"
        exit 1
    fi
done

echo ""
echo "==================== 提取完成 ===================="
echo "已提取 ${#SOURCES[@]} 个位置"
echo "开始训练解码器..."
echo "=================================================="

# 训练所有解码器（并行）
declare -a TRAIN_PIDS=()

for SOURCE in "${SOURCES[@]}"; do
    OUTPUT_DIR="experiments/runs/exp3_${SCENE}_${SOURCE}_${FIRST_K}frames"

    # 检查是否已训练
    if [ -f "${OUTPUT_DIR}/rgb_decoder.pt" ]; then
        echo "⏭️  跳过训练（已存在）: ${SOURCE}"
        continue
    fi

    echo "🔄 训练: ${SOURCE}"

    "${GPU_RUNNER}" python experiments/exp3_full_information_flow/train_simple.py \
        --data_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --epochs 10 \
        --batch_size 4 \
        --device cuda:0 \
        > ${OUTPUT_DIR}/train.log 2>&1 &

    TRAIN_PIDS+=($!)

    # 最多同时训练 2 个
    if [ ${#TRAIN_PIDS[@]} -ge 2 ]; then
        wait ${TRAIN_PIDS[0]}
        TRAIN_PIDS=("${TRAIN_PIDS[@]:1}")
    fi
done

# 等待所有训练完成
for pid in "${TRAIN_PIDS[@]}"; do
    wait $pid
done

echo ""
echo "==================== 全部完成 ===================="
echo "✅ 提取: ${#SOURCES[@]} 个位置"
echo "✅ 训练: ${#SOURCES[@]} 个解码器"
echo "=================================================="

# 汇总结果
echo ""
echo "==================== PSNR 结果 ===================="
for SOURCE in "${SOURCES[@]}"; do
    OUTPUT_DIR="experiments/runs/exp3_${SCENE}_${SOURCE}_${FIRST_K}frames"
    if [ -f "${OUTPUT_DIR}/metrics.json" ]; then
        PSNR=$(python -c "import json; print(json.load(open('${OUTPUT_DIR}/metrics.json'))['final_epoch']['psnr'])" 2>/dev/null || echo "N/A")
        printf "%-25s: %s dB\n" "${SOURCE}" "${PSNR}"
    fi
done
echo "=================================================="
