#!/bin/bash

# ================= 配置区 =================
# 输出的 CSV 文件名
CSV_STAGE1="results_stage1_structure.csv"
CSV_STAGE2="results_stage2_dynamics.csv"

# 默认参数
DEFAULT_EPOCHS=50
DEFAULT_SEED=24

# Stage 1 使用固定的 Patience
STAGE1_PATIENCE=10

# ================= 辅助函数 (已修改) =================
extract_metrics() {
    local output="$1"
    
    # 【修改点 1】提取 Accuracy
    # grep 会匹配到所有的 "Epoch X Validation Accuracy: ..." 和最后的 "Accuracy: ..."
    # 使用 'tail -n 1' 强制只取最后一行（即最终测试集结果）
    local acc=$(echo "$output" | grep "Accuracy:" | tail -n 1 | sed -n 's/.*Accuracy: \([0-9.]*\).*/\1/p')
    
    # 【修改点 2】提取 GP, GR, FS, OE
    # 同样加上 'tail -n 1' 以防万一，确保只取最后一次输出
    local gp=$(echo "$output" | grep "test GP" | tail -n 1 | sed -n 's/.*GP: \([0-9.]*\).*/\1/p')
    local gr=$(echo "$output" | grep "test GP" | tail -n 1 | sed -n 's/.*GR: \([0-9.]*\).*/\1/p')
    local fs=$(echo "$output" | grep "test GP" | tail -n 1 | sed -n 's/.*FS: \([0-9.]*\).*/\1/p')
    local oe=$(echo "$output" | grep "test GP" | tail -n 1 | sed -n 's/.*OE: \([0-9.]*\).*/\1/p')

    # 如果提取失败（比如训练出错），设为 -1
    echo "${acc:--1},${gp:--1},${gr:--1},${fs:--1},${oe:--1}"
}

# ============================================================
# 阶段一: 结构参数搜索 (Structure Search)
# ============================================================
echo "------------------------------------------------"
echo "STAGE 1: Searching for Optimal DCA Structure"
echo "Results will be saved to: $CSV_STAGE1"
echo "------------------------------------------------"

# Stage 1 表头
echo "Heads,Dim,LR,BatchSize,MaxLen,Patience,Accuracy,GP,GR,FS,OE" > "$CSV_STAGE1"

HEADS_LIST=(2 4 8)
D_MODEL_LIST=(64 128 256)

FIXED_LR=1e-4
FIXED_BS=16
# FIXED_LEN=200
# FIXED_LEN=5   # bigdata 数据集使用较短的 max_len  四分类数据集
FIXED_LEN=300   #sigir  2分类数据集

# 数据集配置
DATASET="sigir"  # 可选: "reddit", "bigdata", "sigir"

if [ "$DATASET" = "sigir" ]; then
    CLASSNUM=2
    DATA_EMBEDDINGS="../data/sigir_bert_embeddings.pkl"
    DATA_FEATURES="../data_analy/feature_sigir.csv"
elif [ "$DATASET" = "bigdata" ]; then
    CLASSNUM=4
    DATA_EMBEDDINGS="../data/bigdata_bert_embeddings.pkl"
    DATA_FEATURES="../data_analy/feature_bigdata.csv"
else  # reddit
    CLASSNUM=5
    DATA_EMBEDDINGS="../data/bert_embeddings.pkl"
    DATA_FEATURES="../data_analy/feature_reddit_500.csv"
fi

echo "Dataset: $DATASET (Classes: $CLASSNUM)"

for head in "${HEADS_LIST[@]}"; do
    for dim in "${D_MODEL_LIST[@]}"; do
        if (( dim % head == 0 )); then
            echo "========================================"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running Stage 1: Heads=${head}, Dim=${dim}"
            echo "========================================"
            
            # 使用 tee 同时显示输出和捕获到变量
            OUTPUT=$(python dual_v3.py \
                --cv_heads $head \
                --cv_d_model $dim \
                --lr $FIXED_LR \
                --batch_size $FIXED_BS \
                --max_len $FIXED_LEN \
                --epochs $DEFAULT_EPOCHS \
                --patience $STAGE1_PATIENCE \
                --seed $DEFAULT_SEED \
                --classnum $CLASSNUM \
                --data_embeddings "$DATA_EMBEDDINGS" \
                --data_features "$DATA_FEATURES" 2>&1 | tee /dev/tty)
            
            # 提取指标
            METRICS=$(extract_metrics "$OUTPUT")
            
            # 写入 CSV
            echo "$head,$dim,$FIXED_LR,$FIXED_BS,$FIXED_LEN,$STAGE1_PATIENCE,$METRICS" >> "$CSV_STAGE1"
            
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done. Metrics: $METRICS"
            echo ""
        fi
    done
done

# ============================================================
# 阶段二: 训练参数搜索 (Training Dynamics Search)
# ============================================================
# 【重要】请根据 Stage 1 的结果，在这里手动修改最佳结构参数
BEST_HEAD=4
BEST_DIM=128
# BEST_HEAD=8 #bigdata
# BEST_DIM=256

echo "------------------------------------------------"
echo "STAGE 2: Searching for Batch Size, LR, Hidden Size AND Patience"
echo "Using Structure: Heads=${BEST_HEAD}, Dim=${BEST_DIM}"
echo "Results will be saved to: $CSV_STAGE2"
echo "------------------------------------------------"

# Stage 2 表头
echo "Heads,Dim,LR,BatchSize,HiddenSize,Patience,Accuracy,GP,GR,FS,OE" > "$CSV_STAGE2"

LR_LIST=(1e-4 1e-5 5e-5 )
BATCH_LIST=(4 8 16 32)
HIDDEN_SIZE=(128 256)
PATIENCE_LIST=(10)

for lr in "${LR_LIST[@]}"; do
    for bs in "${BATCH_LIST[@]}"; do
        for hs in "${HIDDEN_SIZE[@]}"; do
            for pat in "${PATIENCE_LIST[@]}"; do
                echo "========================================"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running Stage 2: LR=${lr}, BS=${bs}, Hidden=${hs}, Patience=${pat}"
                echo "========================================"
                
                # 使用 tee 同时显示输出和捕获到变量
                OUTPUT=$(python dual_v3.py \
                    --cv_heads $BEST_HEAD \
                    --cv_d_model $BEST_DIM \
                    --lr $lr \
                    --batch_size $bs \
                    --max_len $FIXED_LEN \
                    --hidden_size $hs \
                    --epochs $DEFAULT_EPOCHS \
                    --patience $pat \
                    --seed $DEFAULT_SEED \
                    --classnum $CLASSNUM \
                    --data_embeddings "$DATA_EMBEDDINGS" \
                    --data_features "$DATA_FEATURES" 2>&1 | tee /dev/tty)
                
                METRICS=$(extract_metrics "$OUTPUT")
                
                echo "$BEST_HEAD,$BEST_DIM,$lr,$bs,$hs,$pat,$METRICS" >> "$CSV_STAGE2"
                
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done. Metrics: $METRICS"
                echo ""
            done
        done
    done
done

echo "================================================"
echo "All experiments completed at $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"
echo "Results saved to:"
echo "  - $CSV_STAGE1"
echo "  - $CSV_STAGE2"