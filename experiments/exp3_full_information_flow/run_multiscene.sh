#!/bin/bash
# Multi-scene testing for exp3 (100 frames per scene)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_RUNNER="${SCRIPT_DIR}/../utils/gpu_runner.sh"

# 激活环境
source /home/dengbaichuan/miniconda3/bin/activate lingbot-map
export PYTHONPATH=/home/dengbaichuan/research/lingbot_map/experiments:${PYTHONPATH}

cd /home/dengbaichuan/research/lingbot_map

# 场景列表
SCENES=("loop" "courthouse" "university")
FIRST_K=100  # 每个场景 100 帧

# 测试位置（关键位置）
SOURCES=(
    "patch_embed"
    "frame_blocks_11"
    "aggregated_11"
    "aggregated_23"
)

echo "==================== 多场景测试 ===================="
echo "场景数: ${#SCENES[@]}"
echo "每场景帧数: ${FIRST_K}"
echo "测试位置: ${#SOURCES[@]}"
echo "总任务数: $((${#SCENES[@]} * ${#SOURCES[@]}))"
echo "=================================================="

# 遍历所有场景和位置
for SCENE in "${SCENES[@]}"; do
    echo ""
    echo "#################### 场景: ${SCENE} ####################"

    for SOURCE in "${SOURCES[@]}"; do
        echo ""
        echo "==== ${SCENE} - ${SOURCE} ===="

        OUTPUT_DIR="experiments/runs/exp3_${SCENE}_${SOURCE}_${FIRST_K}frames"

        # 检查是否已存在
        if [ -f "${OUTPUT_DIR}/window_0000.pt" ]; then
            echo "⏭️  跳过提取（已存在）"
        else
            # 判断是 aggregated 还是普通层
            if [[ "$SOURCE" == aggregated_* ]]; then
                BLOCK_IDX="${SOURCE#aggregated_}"

                echo "🔄 提取 aggregated[${BLOCK_IDX}]"
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
                echo "🔄 提取 ${SOURCE}"
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
                echo "✅ 提取完成"
            else
                echo "❌ 提取失败"
                exit 1
            fi
        fi

        # 训练解码器
        if [ -f "${OUTPUT_DIR}/rgb_decoder.pt" ]; then
            echo "⏭️  跳过训练（已存在）"
        else
            echo "🔄 训练解码器"

            "${GPU_RUNNER}" python experiments/exp3_full_information_flow/train_simple.py \
                --data_dir ${OUTPUT_DIR} \
                --output_dir ${OUTPUT_DIR} \
                --epochs 10 \
                --batch_size 4 \
                --device cuda:0

            if [ $? -eq 0 ]; then
                echo "✅ 训练完成"
            else
                echo "❌ 训练失败"
                exit 1
            fi
        fi

        # 显示 PSNR
        if [ -f "${OUTPUT_DIR}/metrics.json" ]; then
            PSNR=$(python -c "import json; print(f'{json.load(open(\"${OUTPUT_DIR}/metrics.json\"))[\"final_epoch\"][\"psnr\"]:.2f}')" 2>/dev/null || echo "N/A")
            echo "📊 PSNR: ${PSNR} dB"
        fi
    done
done

echo ""
echo "==================== 全部完成 ===================="
echo ""
echo "==================== 结果汇总 ===================="

# 生成表格
printf "\n%-15s" "位置"
for SCENE in "${SCENES[@]}"; do
    printf " | %-12s" "${SCENE}"
done
printf "\n"

printf "%-15s" "---------------"
for SCENE in "${SCENES[@]}"; do
    printf "-+-%-12s" "------------"
done
printf "\n"

for SOURCE in "${SOURCES[@]}"; do
    printf "%-15s" "${SOURCE}"
    for SCENE in "${SCENES[@]}"; do
        OUTPUT_DIR="experiments/runs/exp3_${SCENE}_${SOURCE}_${FIRST_K}frames"
        if [ -f "${OUTPUT_DIR}/metrics.json" ]; then
            PSNR=$(python -c "import json; print(f'{json.load(open(\"${OUTPUT_DIR}/metrics.json\"))[\"final_epoch\"][\"psnr\"]:.2f}')" 2>/dev/null || echo "N/A")
            printf " | %10s dB" "${PSNR}"
        else
            printf " | %12s" "N/A"
        fi
    done
    printf "\n"
done

echo ""
echo "=================================================="
