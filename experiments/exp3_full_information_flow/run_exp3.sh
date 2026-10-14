#!/bin/bash
# exp3 运行脚本 - 完整信息流分析
# 使用智能 GPU 选择

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_RUNNER="${SCRIPT_DIR}/../utils/gpu_runner.sh"

# 激活环境
source /home/dengbaichuan/miniconda3/bin/activate lingbot-map
export PYTHONPATH=/home/dengbaichuan/research/lingbot_map/experiments:${PYTHONPATH}

# 参数解析
STAGE=${1:-extract}  # extract, train, eval
SCENE=${2:-loop}
SOURCE=${3:-patch_embed}
FIRST_K=${4:-30}

# 输出目录
OUTPUT_DIR="experiments/runs/exp3_${SCENE}_${SOURCE}_${FIRST_K}frames"

echo "==================== exp3 配置 ===================="
echo "Stage: ${STAGE}"
echo "Scene: ${SCENE}"
echo "Source: ${SOURCE}"
echo "First K: ${FIRST_K}"
echo "Output: ${OUTPUT_DIR}"
echo "=================================================="

cd /home/dengbaichuan/research/lingbot_map

case $STAGE in
    extract)
        "${GPU_RUNNER}" python experiments/exp3_full_information_flow/extract_tokens.py \
            --model_path models/lingbot-map.pt \
            --scene ${SCENE} \
            --source ${SOURCE} \
            --image_folder example/${SCENE} \
            --output_dir ${OUTPUT_DIR} \
            --first_k ${FIRST_K} \
            --use_sdpa \
            --device cuda:0
        ;;

    train)
        # 复用 exp2 的训练脚本
        "${GPU_RUNNER}" python experiments/rgb_reconstruction/exp2_streaming_token_reconstruction.py train \
            --data_dir ${OUTPUT_DIR} \
            --output_dir ${OUTPUT_DIR} \
            --epochs 10 \
            --batch_size 4 \
            --device cuda:0
        ;;

    eval)
        # 复用 exp2 的评估脚本
        "${GPU_RUNNER}" python experiments/rgb_reconstruction/exp2_streaming_token_reconstruction.py eval \
            --data_dir ${OUTPUT_DIR} \
            --output_dir ${OUTPUT_DIR} \
            --device cuda:0
        ;;

    *)
        echo "ERROR: Unknown stage '${STAGE}'"
        echo "Usage: bash run_exp3.sh [extract|train|eval] [scene] [source] [first_k]"
        echo ""
        echo "Sources:"
        echo "  - patch_embed"
        echo "  - frame_blocks_0_input"
        echo "  - frame_blocks_0"
        echo "  - frame_blocks_12"
        exit 1
        ;;
esac

echo "✅ 完成！"
