#!/bin/bash

# ================= 配置区 =================
# 输出的 CSV 文件名
CSV_STAGE1="weibo_results_stage1_structure.csv"
CSV_STAGE2="weibo_results_stage2_dynamics.csv"

# 默认参数
DEFAULT_EPOCHS=50
DEFAULT_SEED=24

# Stage 1 使用固定的 Patience
STAGE1_PATIENCE=10

# ================= 辅助函数 =================
extract_metrics() {
    local output="$1"
    
    # 提取测试集最终 Accuracy
    local acc=$(echo "$output" | grep "Accuracy:" | tail -n 1 | sed -n 's/.*Accuracy: \([0-9.]*\).*/\1/p')
    
    # 提取 F1-Score (Macro)
    local f1=$(echo "$output" | grep "F1-Score:" | tail -n 1 | sed -n 's/.*F1-Score: \([0-9.]*\).*/\1/p')
    
    # 提取 Precision (Macro)
    local pre=$(echo "$output" | grep "Precision:" | tail -n 1 | sed -n 's/.*Precision: \([0-9.]*\).*/\1/p')
    
    # 提取 Recall (Macro)
    local rec=$(echo "$output" | grep "Recall:" | tail -n 1 | sed -n 's/.*Recall: \([0-9.]*\).*/\1/p')

    # 如果提取失败，设为 -1
    echo "${acc:--1},${f1:--1},${pre:--1},${rec:--1}"
}

# ============================================================
# 阶段一: 结构参数搜索 (DCA Structure Search)
# ============================================================
echo "------------------------------------------------"
echo "STAGE 1: Searching for Optimal DCA Structure (Weibo)"
echo "Results will be saved to: $CSV_STAGE1"
echo "------------------------------------------------"

# Weibo Stage 1 表头
echo "Heads,Dim,LR,BatchSize,MaxLen,Accuracy,F1_Macro,Precision,Recall" > "$CSV_STAGE1"

# 变量维度注意力头数与模型维度列表
HEADS_LIST=( 2)
D_MODEL_LIST=(128)
# D_MODEL_LIST=(64 128 256 512)

FIXED_LR=1e-5     # 这里的初始学习率参考了 train.py 的默认值
FIXED_BS=8
FIXED_LEN=100     # Weibo 数据集典型长度

for head in "${HEADS_LIST[@]}"; do
    for dim in "${D_MODEL_LIST[@]}"; do
        if (( dim % head == 0 )); then
            echo -n "Running Stage 1: Heads=${head}, Dim=${dim} ... "
            
            # 运行 train.py 并捕获输出
            OUTPUT=$(python weibo_train.py \
                --cv_heads $head \
                --cv_d_model $dim \
                --lr $FIXED_LR \
                --batch_size $FIXED_BS \
                --max_len $FIXED_LEN \
                --epochs $DEFAULT_EPOCHS \
                --patience $STAGE1_PATIENCE \
                --seed $DEFAULT_SEED \
                --classnum 2 \
                --data_embeddings "../data/user_post_embeddings_bert_wwm.pkl" \
                --data_features "../data_analy/feature_weibo.csv" 2>&1)
            
            METRICS=$(extract_metrics "$OUTPUT")
            echo "$head,$dim,$FIXED_LR,$FIXED_BS,$FIXED_LEN,$METRICS" >> "$CSV_STAGE1"
            echo "Done. Metrics: $METRICS"
        fi
    done
done

# ============================================================
# 阶段二: 训练动力学搜索 (Training Dynamics Search)
# ============================================================
# 请在运行完阶段一后，根据 CSV 结果填入最佳结构参数
BEST_HEAD=2
BEST_DIM=128

echo "------------------------------------------------"
echo "STAGE 2: Searching for LR, Batch Size, Hidden Size and Patience (Weibo)"
echo "Using Structure: Heads=${BEST_HEAD}, Dim=${BEST_DIM}"
echo "------------------------------------------------"

echo "Heads,Dim,LR,BatchSize,HiddenSize,Patience,Accuracy,F1_Macro,Precision,Recall" > "$CSV_STAGE2"

# 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5
LR_LIST=(5e-3 1e-3 5e-4 1e-4 5e-5 1e-5)
BATCH_LIST=(8 16 32)
HIDDEN_SIZE_LIST=(64 128 256)
PATIENCE_LIST=(10)

for lr in "${LR_LIST[@]}"; do
    for bs in "${BATCH_LIST[@]}"; do
        for hs in "${HIDDEN_SIZE_LIST[@]}"; do
            for pat in "${PATIENCE_LIST[@]}"; do
            
                echo -n "Running Stage 2: LR=${lr}, BS=${bs}, Hidden=${hs}, Patience=${pat} ... "
                
                OUTPUT=$(python weibo_train.py \
                    --cv_heads $BEST_HEAD \
                    --cv_d_model $BEST_DIM \
                    --lr $lr \
                    --batch_size $bs \
                    --max_len $FIXED_LEN \
                    --hidden_size $hs \
                    --epochs $DEFAULT_EPOCHS \
                    --patience $pat \
                    --seed $DEFAULT_SEED \
                    --classnum 2 \
                    --data_embeddings "../data/user_post_embeddings_bert_wwm.pkl" \
                    --data_features "../data_analy/feature_weibo.csv" 2>&1)
                
                METRICS=$(extract_metrics "$OUTPUT")
                echo "$BEST_HEAD,$BEST_DIM,$lr,$bs,$hs,$pat,$METRICS" >> "$CSV_STAGE2"
                echo "Done. Metrics: $METRICS"
                
            done
        done
    done
done

echo "All experiments for Weibo completed."